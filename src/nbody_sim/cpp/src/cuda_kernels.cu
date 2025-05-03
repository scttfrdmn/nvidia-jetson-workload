// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "nbody_sim/cuda_kernels.cuh"

namespace nbody_sim {
namespace cuda {

__global__ void compute_accelerations_kernel(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening
) {
    // Get thread ID
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within range
    if (i >= num_particles) {
        return;
    }
    
    // Get position of current particle
    const Vec3 pos_i = positions[i];
    
    // Initialize acceleration to zero
    Vec3 acc(0.0, 0.0, 0.0);
    
    // Compute gravitational interactions with all other particles
    for (index_t j = 0; j < num_particles; ++j) {
        // Skip self-interaction
        if (j == i) {
            continue;
        }
        
        // Get position and mass of other particle
        const Vec3 pos_j = positions[j];
        const scalar_t mass_j = masses[j];
        
        // Compute displacement vector
        Vec3 r = pos_j - pos_i;
        
        // Compute distance squared
        scalar_t r_squared = r.length_squared();
        
        // Add softening to prevent singularities
        r_squared += softening * softening;
        
        // Compute inverse distance cubed (for acceleration calculation)
        scalar_t inv_r_cubed = 1.0 / (sqrt(r_squared) * r_squared);
        
        // Accumulate acceleration: a = G * m * r / |r|^3
        acc += r * (G * mass_j * inv_r_cubed);
    }
    
    // Store computed acceleration
    accelerations[i] = acc;
}

void launch_compute_accelerations(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    index_t block_size,
    cudaStream_t stream
) {
    // Calculate grid size based on number of particles and block size
    index_t grid_size = (num_particles + block_size - 1) / block_size;
    
    // Launch kernel
    compute_accelerations_kernel<<<grid_size, block_size, 0, stream>>>(
        positions,
        masses,
        accelerations,
        num_particles,
        G,
        softening
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

__global__ void compute_accelerations_tiled_kernel(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening
) {
    // Shared memory for positions and masses
    extern __shared__ char shared_memory[];
    Vec3* shared_positions = reinterpret_cast<Vec3*>(shared_memory);
    scalar_t* shared_masses = reinterpret_cast<scalar_t*>(shared_memory + blockDim.x * sizeof(Vec3));
    
    // Get thread ID
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load current particle data
    Vec3 my_pos;
    if (i < num_particles) {
        my_pos = positions[i];
    }
    
    // Initialize acceleration to zero
    Vec3 acc(0.0, 0.0, 0.0);
    
    // Process tiles
    for (index_t tile_start = 0; tile_start < num_particles; tile_start += blockDim.x) {
        // Load tile into shared memory
        if (tile_start + threadIdx.x < num_particles) {
            shared_positions[threadIdx.x] = positions[tile_start + threadIdx.x];
            shared_masses[threadIdx.x] = masses[tile_start + threadIdx.x];
        }
        
        // Ensure all threads have loaded their data
        __syncthreads();
        
        // Compute interactions with particles in this tile
        if (i < num_particles) {
            for (index_t j = 0; j < blockDim.x && tile_start + j < num_particles; ++j) {
                // Skip self-interaction
                if (tile_start + j == i) {
                    continue;
                }
                
                // Get position and mass of other particle
                const Vec3& pos_j = shared_positions[j];
                const scalar_t mass_j = shared_masses[j];
                
                // Compute displacement vector
                Vec3 r = pos_j - my_pos;
                
                // Compute distance squared
                scalar_t r_squared = r.length_squared();
                
                // Add softening to prevent singularities
                r_squared += softening * softening;
                
                // Compute inverse distance cubed (for acceleration calculation)
                scalar_t inv_r_cubed = 1.0 / (sqrt(r_squared) * r_squared);
                
                // Accumulate acceleration: a = G * m * r / |r|^3
                acc += r * (G * mass_j * inv_r_cubed);
            }
        }
        
        // Ensure all threads are done with shared memory
        __syncthreads();
    }
    
    // Store computed acceleration
    if (i < num_particles) {
        accelerations[i] = acc;
    }
}

void launch_compute_accelerations_tiled(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    index_t block_size,
    cudaStream_t stream
) {
    // Calculate grid size based on number of particles and block size
    index_t grid_size = (num_particles + block_size - 1) / block_size;
    
    // Calculate shared memory size
    size_t shared_mem_size = block_size * (sizeof(Vec3) + sizeof(scalar_t));
    
    // Launch kernel
    compute_accelerations_tiled_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        positions,
        masses,
        accelerations,
        num_particles,
        G,
        softening
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

__global__ void update_positions_kernel(
    Vec3* positions,
    const Vec3* velocities,
    index_t num_particles,
    scalar_t dt
) {
    // Get thread ID
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within range
    if (i >= num_particles) {
        return;
    }
    
    // Update position: x = x + v * dt
    positions[i] = positions[i] + velocities[i] * dt;
}

void launch_update_positions(
    Vec3* positions,
    const Vec3* velocities,
    index_t num_particles,
    scalar_t dt,
    index_t block_size,
    cudaStream_t stream
) {
    // Calculate grid size based on number of particles and block size
    index_t grid_size = (num_particles + block_size - 1) / block_size;
    
    // Launch kernel
    update_positions_kernel<<<grid_size, block_size, 0, stream>>>(
        positions,
        velocities,
        num_particles,
        dt
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

__global__ void update_velocities_kernel(
    Vec3* velocities,
    const Vec3* accelerations,
    index_t num_particles,
    scalar_t dt
) {
    // Get thread ID
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within range
    if (i >= num_particles) {
        return;
    }
    
    // Update velocity: v = v + a * dt
    velocities[i] = velocities[i] + accelerations[i] * dt;
}

void launch_update_velocities(
    Vec3* velocities,
    const Vec3* accelerations,
    index_t num_particles,
    scalar_t dt,
    index_t block_size,
    cudaStream_t stream
) {
    // Calculate grid size based on number of particles and block size
    index_t grid_size = (num_particles + block_size - 1) / block_size;
    
    // Launch kernel
    update_velocities_kernel<<<grid_size, block_size, 0, stream>>>(
        velocities,
        accelerations,
        num_particles,
        dt
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

__global__ void update_velocities_half_step_kernel(
    Vec3* velocities,
    const Vec3* accelerations,
    index_t num_particles,
    scalar_t dt
) {
    // Get thread ID
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within range
    if (i >= num_particles) {
        return;
    }
    
    // Update velocity with half time step: v = v + a * (dt/2)
    velocities[i] = velocities[i] + accelerations[i] * (dt * 0.5);
}

void launch_update_velocities_half_step(
    Vec3* velocities,
    const Vec3* accelerations,
    index_t num_particles,
    scalar_t dt,
    index_t block_size,
    cudaStream_t stream
) {
    // Calculate grid size based on number of particles and block size
    index_t grid_size = (num_particles + block_size - 1) / block_size;
    
    // Launch kernel
    update_velocities_half_step_kernel<<<grid_size, block_size, 0, stream>>>(
        velocities,
        accelerations,
        num_particles,
        dt
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace nbody_sim