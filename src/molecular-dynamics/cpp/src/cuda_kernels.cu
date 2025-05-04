// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "molecular_dynamics/cuda_kernels.cuh"
#include "molecular_dynamics/common.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>

namespace molecular_dynamics {

// Helper functions for CUDA kernels
__device__ Vec3 minimum_image_vector_device(const Vec3& r1, const Vec3& r2, const Vec3& box_dims, bool use_pbc) {
    if (!use_pbc) {
        return r2 - r1;
    }
    
    Vec3 dr = r2 - r1;
    
    // Apply periodic boundary conditions
    if (dr.x > 0.5 * box_dims.x) dr.x -= box_dims.x;
    else if (dr.x < -0.5 * box_dims.x) dr.x += box_dims.x;
    
    if (dr.y > 0.5 * box_dims.y) dr.y -= box_dims.y;
    else if (dr.y < -0.5 * box_dims.y) dr.y += box_dims.y;
    
    if (dr.z > 0.5 * box_dims.z) dr.z -= box_dims.z;
    else if (dr.z < -0.5 * box_dims.z) dr.z += box_dims.z;
    
    return dr;
}

__device__ scalar_t lennard_jones_potential(scalar_t r_squared, scalar_t sigma, scalar_t epsilon) {
    scalar_t sigma_r_6 = pow(sigma * sigma / r_squared, 3);
    scalar_t sigma_r_12 = sigma_r_6 * sigma_r_6;
    return 4.0 * epsilon * (sigma_r_12 - sigma_r_6);
}

__device__ Vec3 lennard_jones_force(const Vec3& dr, scalar_t r_squared, scalar_t sigma, scalar_t epsilon) {
    scalar_t sigma_r_6 = pow(sigma * sigma / r_squared, 3);
    scalar_t sigma_r_12 = sigma_r_6 * sigma_r_6;
    scalar_t force_magnitude = 24.0 * epsilon * (2.0 * sigma_r_12 - sigma_r_6) / r_squared;
    return dr * force_magnitude;
}

__device__ Vec3 coulomb_force(const Vec3& dr, scalar_t r_squared, scalar_t charge_i, scalar_t charge_j) {
    const scalar_t COULOMB_CONSTANT = 1389.35457; // in kJ/(mol*Å) for charges in elementary charge units
    scalar_t r = sqrt(r_squared);
    scalar_t force_magnitude = COULOMB_CONSTANT * charge_i * charge_j / (r_squared * r);
    return dr * force_magnitude;
}

// Basic Lennard-Jones force kernel
__global__ void lj_force_kernel(
    const Vec3* positions,
    const scalar_t* charges,
    const int* types,
    Vec3* forces,
    const scalar_t* lj_params,
    Vec3 box_dims,
    bool use_pbc,
    scalar_t cutoff_squared,
    int num_atoms
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= num_atoms) return;
    
    Vec3 force_i = Vec3(0.0, 0.0, 0.0);
    Vec3 pos_i = positions[i];
    scalar_t charge_i = charges[i];
    int type_i = types[i];
    
    for (int j = 0; j < num_atoms; j++) {
        if (i == j) continue;
        
        Vec3 pos_j = positions[j];
        Vec3 dr = minimum_image_vector_device(pos_i, pos_j, box_dims, use_pbc);
        scalar_t r_squared = dr.length_squared();
        
        if (r_squared < cutoff_squared) {
            // Get LJ parameters for this atom pair
            int type_j = types[j];
            scalar_t sigma = 0.5 * (lj_params[type_i*2] + lj_params[type_j*2]);  // Lorentz-Berthelot mixing rule
            scalar_t epsilon = sqrt(lj_params[type_i*2+1] * lj_params[type_j*2+1]);
            
            // Add LJ force
            force_i += lennard_jones_force(dr, r_squared, sigma, epsilon);
            
            // Add Coulomb force
            scalar_t charge_j = charges[j];
            force_i += coulomb_force(dr, r_squared, charge_i, charge_j);
        }
    }
    
    forces[i] = force_i;
}

// Optimized tiled kernel for better performance
template<int TILE_SIZE>
__global__ void tiled_force_kernel(
    const Vec3* positions,
    const scalar_t* charges,
    const int* types,
    Vec3* forces,
    const scalar_t* lj_params,
    Vec3 box_dims,
    bool use_pbc,
    scalar_t cutoff_squared,
    int num_atoms
) {
    extern __shared__ char shared_memory[];
    
    Vec3* shared_positions = reinterpret_cast<Vec3*>(shared_memory);
    scalar_t* shared_charges = reinterpret_cast<scalar_t*>(shared_memory + TILE_SIZE * sizeof(Vec3));
    int* shared_types = reinterpret_cast<int*>(shared_charges + TILE_SIZE);
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    Vec3 pos_i = i < num_atoms ? positions[i] : Vec3(0, 0, 0);
    scalar_t charge_i = i < num_atoms ? charges[i] : 0.0;
    int type_i = i < num_atoms ? types[i] : 0;
    Vec3 force_i = Vec3(0, 0, 0);
    
    // Loop over tiles
    for (int tile = 0; tile < (num_atoms + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        int j = tile * TILE_SIZE + threadIdx.x;
        if (j < num_atoms) {
            shared_positions[threadIdx.x] = positions[j];
            shared_charges[threadIdx.x] = charges[j];
            shared_types[threadIdx.x] = types[j];
        }
        __syncthreads();
        
        // Compute forces between particles in this tile
        if (i < num_atoms) {
            for (int k = 0; k < TILE_SIZE && tile * TILE_SIZE + k < num_atoms; k++) {
                int j = tile * TILE_SIZE + k;
                if (i != j) {
                    Vec3 pos_j = shared_positions[k];
                    Vec3 dr = minimum_image_vector_device(pos_i, pos_j, box_dims, use_pbc);
                    scalar_t r_squared = dr.length_squared();
                    
                    if (r_squared < cutoff_squared) {
                        // Get LJ parameters for this atom pair
                        int type_j = shared_types[k];
                        scalar_t sigma = 0.5 * (lj_params[type_i*2] + lj_params[type_j*2]);
                        scalar_t epsilon = sqrt(lj_params[type_i*2+1] * lj_params[type_j*2+1]);
                        
                        // Add LJ force
                        force_i += lennard_jones_force(dr, r_squared, sigma, epsilon);
                        
                        // Add Coulomb force
                        scalar_t charge_j = shared_charges[k];
                        force_i += coulomb_force(dr, r_squared, charge_i, charge_j);
                    }
                }
            }
        }
        __syncthreads();
    }
    
    if (i < num_atoms) {
        forces[i] = force_i;
    }
}

// Kernel for velocity Verlet position update
__global__ void velocity_verlet_kernel(
    Vec3* positions,
    Vec3* velocities,
    const Vec3* forces,
    const scalar_t* masses,
    scalar_t dt,
    int num_atoms,
    Vec3 box_dims,
    bool use_pbc
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= num_atoms) return;
    
    Vec3 pos = positions[i];
    Vec3 vel = velocities[i];
    Vec3 force = forces[i];
    scalar_t mass = masses[i];
    
    // Update position: r(t+dt) = r(t) + v(t)*dt + 0.5*f(t)/m*dt^2
    Vec3 acceleration = force / mass;
    pos += vel * dt + acceleration * (0.5 * dt * dt);
    
    // Apply periodic boundary conditions
    if (use_pbc) {
        if (pos.x < 0) pos.x += box_dims.x;
        else if (pos.x >= box_dims.x) pos.x -= box_dims.x;
        
        if (pos.y < 0) pos.y += box_dims.y;
        else if (pos.y >= box_dims.y) pos.y -= box_dims.y;
        
        if (pos.z < 0) pos.z += box_dims.z;
        else if (pos.z >= box_dims.z) pos.z -= box_dims.z;
    }
    
    // Update velocity: v(t+dt/2) = v(t) + 0.5*f(t)/m*dt
    vel += acceleration * (0.5 * dt);
    
    positions[i] = pos;
    velocities[i] = vel;
}

// Kernel for velocity update in second half of velocity Verlet
__global__ void velocity_update_kernel(
    Vec3* velocities,
    const Vec3* forces,
    const scalar_t* masses,
    scalar_t dt,
    int num_atoms
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= num_atoms) return;
    
    Vec3 vel = velocities[i];
    Vec3 force = forces[i];
    scalar_t mass = masses[i];
    
    // Update velocity: v(t+dt) = v(t+dt/2) + 0.5*f(t+dt)/m*dt
    Vec3 acceleration = force / mass;
    vel += acceleration * (0.5 * dt);
    
    velocities[i] = vel;
}

// Berendsen thermostat kernel
__global__ void berendsen_thermostat_kernel(
    Vec3* velocities,
    const scalar_t* masses,
    scalar_t lambda,
    int num_atoms
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= num_atoms) return;
    
    // Scale velocities by lambda
    velocities[i] *= sqrt(lambda);
}

// Kernel to calculate kinetic energy
__global__ void kinetic_energy_kernel(
    const Vec3* velocities,
    const scalar_t* masses,
    scalar_t* energies,
    int num_atoms
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= num_atoms) return;
    
    Vec3 vel = velocities[i];
    scalar_t mass = masses[i];
    
    // KE = 0.5 * m * v^2
    energies[i] = 0.5 * mass * vel.length_squared();
}

// Kernel for bond forces
__global__ void bond_force_kernel(
    const Vec3* positions,
    Vec3* forces,
    const int* atom1_indices,
    const int* atom2_indices,
    const scalar_t* eq_distances,
    const scalar_t* force_constants,
    Vec3 box_dims,
    bool use_pbc,
    int num_bonds
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= num_bonds) return;
    
    int idx1 = atom1_indices[i];
    int idx2 = atom2_indices[i];
    scalar_t r0 = eq_distances[i];
    scalar_t k = force_constants[i];
    
    Vec3 pos1 = positions[idx1];
    Vec3 pos2 = positions[idx2];
    
    Vec3 dr = minimum_image_vector_device(pos1, pos2, box_dims, use_pbc);
    scalar_t r = dr.length();
    
    // F = -k * (r - r0) * dr/r
    Vec3 force = dr * (-k * (r - r0) / r);
    
    // Atomic add since multiple bonds may involve the same atom
    atomicAdd(&forces[idx1].x, force.x);
    atomicAdd(&forces[idx1].y, force.y);
    atomicAdd(&forces[idx1].z, force.z);
    
    atomicAdd(&forces[idx2].x, -force.x);
    atomicAdd(&forces[idx2].y, -force.y);
    atomicAdd(&forces[idx2].z, -force.z);
}

// Host function implementations

void launch_lj_force_kernel(
    const Vec3* positions,
    const scalar_t* charges,
    const int* types,
    Vec3* forces,
    const scalar_t* lj_params,
    Vec3 box_dims,
    bool use_pbc,
    scalar_t cutoff,
    int num_atoms,
    cudaStream_t stream
) {
    scalar_t cutoff_squared = cutoff * cutoff;
    
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size = (num_atoms + block_size - 1) / block_size;
    
    lj_force_kernel<<<grid_size, block_size, 0, stream>>>(
        positions,
        charges,
        types,
        forces,
        lj_params,
        box_dims,
        use_pbc,
        cutoff_squared,
        num_atoms
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_tiled_force_kernel(
    const Vec3* positions,
    const scalar_t* charges,
    const int* types,
    Vec3* forces,
    const scalar_t* lj_params,
    Vec3 box_dims,
    bool use_pbc,
    scalar_t cutoff,
    int num_atoms,
    int block_size,
    int tile_size,
    cudaStream_t stream
) {
    scalar_t cutoff_squared = cutoff * cutoff;
    
    int grid_size = (num_atoms + block_size - 1) / block_size;
    size_t shared_mem_size = tile_size * (sizeof(Vec3) + sizeof(scalar_t) + sizeof(int));
    
    switch (tile_size) {
        case 8:
            tiled_force_kernel<8><<<grid_size, block_size, shared_mem_size, stream>>>(
                positions, charges, types, forces, lj_params, box_dims, use_pbc, cutoff_squared, num_atoms
            );
            break;
        case 16:
            tiled_force_kernel<16><<<grid_size, block_size, shared_mem_size, stream>>>(
                positions, charges, types, forces, lj_params, box_dims, use_pbc, cutoff_squared, num_atoms
            );
            break;
        case 32:
            tiled_force_kernel<32><<<grid_size, block_size, shared_mem_size, stream>>>(
                positions, charges, types, forces, lj_params, box_dims, use_pbc, cutoff_squared, num_atoms
            );
            break;
        default:
            tiled_force_kernel<24><<<grid_size, block_size, shared_mem_size, stream>>>(
                positions, charges, types, forces, lj_params, box_dims, use_pbc, cutoff_squared, num_atoms
            );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_velocity_verlet_kernel(
    Vec3* positions,
    Vec3* velocities,
    const Vec3* forces,
    const scalar_t* masses,
    scalar_t dt,
    int num_atoms,
    Vec3 box_dims,
    bool use_pbc,
    cudaStream_t stream
) {
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size = (num_atoms + block_size - 1) / block_size;
    
    velocity_verlet_kernel<<<grid_size, block_size, 0, stream>>>(
        positions, velocities, forces, masses, dt, num_atoms, box_dims, use_pbc
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_berendsen_thermostat_kernel(
    Vec3* velocities,
    const scalar_t* masses,
    scalar_t lambda,
    int num_atoms,
    cudaStream_t stream
) {
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size = (num_atoms + block_size - 1) / block_size;
    
    berendsen_thermostat_kernel<<<grid_size, block_size, 0, stream>>>(
        velocities, masses, lambda, num_atoms
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_kinetic_energy_kernel(
    const Vec3* velocities,
    const scalar_t* masses,
    scalar_t* energies,
    int num_atoms,
    cudaStream_t stream
) {
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size = (num_atoms + block_size - 1) / block_size;
    
    kinetic_energy_kernel<<<grid_size, block_size, 0, stream>>>(
        velocities, masses, energies, num_atoms
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_bond_force_kernel(
    const Vec3* positions,
    Vec3* forces,
    const int* atom1_indices,
    const int* atom2_indices,
    const scalar_t* eq_distances,
    const scalar_t* force_constants,
    Vec3 box_dims,
    bool use_pbc,
    int num_bonds,
    cudaStream_t stream
) {
    int block_size = DEFAULT_BLOCK_SIZE;
    int grid_size = (num_bonds + block_size - 1) / block_size;
    
    bond_force_kernel<<<grid_size, block_size, 0, stream>>>(
        positions, forces, atom1_indices, atom2_indices, eq_distances, force_constants,
        box_dims, use_pbc, num_bonds
    );
    
    CUDA_CHECK(cudaGetLastError());
}

void launch_optimized_force_kernel(
    const Vec3* positions,
    const scalar_t* charges,
    const int* types,
    Vec3* forces,
    const scalar_t* lj_params,
    Vec3 box_dims,
    bool use_pbc,
    scalar_t cutoff,
    int num_atoms,
    const DeviceCapabilities& capabilities,
    cudaStream_t stream
) {
    // Choose the appropriate kernel based on device capabilities
    if (capabilities.device_type == GPUDeviceType::CPU) {
        // CPU fallback - don't call GPU kernels
        return;
    }
    
    // For small systems, don't use tiling
    if (num_atoms < 256) {
        launch_lj_force_kernel(
            positions, charges, types, forces, lj_params, box_dims, use_pbc, cutoff, num_atoms, stream
        );
        return;
    }
    
    // For larger systems, use tiled kernel with optimal parameters for the device
    int block_size = capabilities.get_optimal_block_size();
    int tile_size = capabilities.get_optimal_tile_size();
    
    // Scale tile size based on available shared memory
    size_t shared_mem_per_block = capabilities.max_shared_memory_per_block;
    size_t required_shared_mem = tile_size * (sizeof(Vec3) + sizeof(scalar_t) + sizeof(int));
    
    // If required shared memory exceeds available, reduce tile size
    if (required_shared_mem > shared_mem_per_block) {
        // Calculate maximum possible tile size
        tile_size = shared_mem_per_block / (sizeof(Vec3) + sizeof(scalar_t) + sizeof(int));
        
        // Round down to power of 2 or 8, whichever is larger
        if (tile_size >= 32) tile_size = 32;
        else if (tile_size >= 16) tile_size = 16;
        else tile_size = 8;
    }
    
    launch_tiled_force_kernel(
        positions, charges, types, forces, lj_params, box_dims, use_pbc, cutoff,
        num_atoms, block_size, tile_size, stream
    );
}

// Device capabilities detection
DeviceCapabilities detect_device_capabilities() {
    DeviceCapabilities caps;
    
    // Default to CPU fallback
    caps.device_type = GPUDeviceType::CPU;
    caps.compute_capability_major = 0;
    caps.compute_capability_minor = 0;
    caps.global_memory_bytes = 0;
    caps.multiprocessor_count = 0;
    caps.max_threads_per_multiprocessor = 0;
    caps.max_threads_per_block = DEFAULT_BLOCK_SIZE;
    caps.max_shared_memory_per_block = 16 * 1024; // 16 KB default
    
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cout << "No CUDA-capable device found, using CPU fallback" << std::endl;
        return caps;
    }
    
    // Get properties of device 0 (we're not handling multi-GPU setups here)
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    
    // Set capabilities based on device properties
    caps.compute_capability_major = props.major;
    caps.compute_capability_minor = props.minor;
    caps.global_memory_bytes = props.totalGlobalMem;
    caps.multiprocessor_count = props.multiProcessorCount;
    caps.max_threads_per_multiprocessor = props.maxThreadsPerMultiProcessor;
    caps.max_threads_per_block = props.maxThreadsPerBlock;
    caps.max_shared_memory_per_block = props.sharedMemPerBlock;
    
    // Determine device type
    if (props.major == 8 && props.minor == 7) {
        // Jetson Orin (SM 8.7)
        caps.device_type = GPUDeviceType::JetsonOrin;
        std::cout << "Detected NVIDIA Jetson Orin (SM 8.7)" << std::endl;
    } else if (props.major == 7 && props.minor == 5) {
        // T4 (SM 7.5)
        caps.device_type = GPUDeviceType::T4;
        std::cout << "Detected NVIDIA T4 (SM 7.5)" << std::endl;
    } else if (props.major >= 8) {
        // Other high-end GPUs (SM >= 8.0)
        caps.device_type = GPUDeviceType::HighEnd;
        std::cout << "Detected high-end NVIDIA GPU (SM " << props.major << "." << props.minor << ")" << std::endl;
    } else {
        // Other GPUs
        caps.device_type = GPUDeviceType::Unknown;
        std::cout << "Detected NVIDIA GPU (SM " << props.major << "." << props.minor << ")" << std::endl;
    }
    
    std::cout << "GPU Memory: " << (caps.global_memory_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << caps.multiprocessor_count << std::endl;
    std::cout << "Max threads per block: " << caps.max_threads_per_block << std::endl;
    std::cout << "Shared memory per block: " << (caps.max_shared_memory_per_block / 1024) << " KB" << std::endl;
    
    return caps;
}

} // namespace molecular_dynamics