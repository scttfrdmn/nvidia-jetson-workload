// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "nbody_sim/cuda_kernels.cuh"
#include "nbody_sim/device_adaptor.hpp"
#include <vector>
#include <thread>
#include <algorithm>

namespace nbody_sim {
namespace cuda {

// Templated tiled kernel for different tile sizes
template<int TILE_SIZE>
__global__ void compute_accelerations_tiled_templated_kernel(
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
    scalar_t* shared_masses = reinterpret_cast<scalar_t*>(shared_memory + TILE_SIZE * sizeof(Vec3));
    
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
    for (index_t tile_start = 0; tile_start < num_particles; tile_start += TILE_SIZE) {
        // Load tile into shared memory
        if (tile_start + threadIdx.x < num_particles && threadIdx.x < TILE_SIZE) {
            shared_positions[threadIdx.x] = positions[tile_start + threadIdx.x];
            shared_masses[threadIdx.x] = masses[tile_start + threadIdx.x];
        }
        
        // Ensure all threads have loaded their data
        __syncthreads();
        
        // Compute interactions with particles in this tile
        if (i < num_particles) {
            for (index_t j = 0; j < TILE_SIZE && tile_start + j < num_particles; ++j) {
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

// Kernel that uses texture memory for better cache performance on T4 and similar GPUs
texture<float, 1, cudaReadModeElementType> tex_masses;
cudaTextureObject_t tex_positions;

__global__ void compute_accelerations_textured_kernel(
    const Vec3* positions,
    cudaTextureObject_t tex_positions,
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
        
        // Get position and mass of other particle from texture memory
        float4 pos_j_tex = tex1Dfetch<float4>(tex_positions, j);
        Vec3 pos_j(pos_j_tex.x, pos_j_tex.y, pos_j_tex.z);
        const scalar_t mass_j = tex1Dfetch<float>(tex_masses, j);
        
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

// Warp-optimized kernel for SM 7.0+ GPUs
__global__ void compute_accelerations_warp_optimized_kernel(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening
) {
    // Get thread ID
    const index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Define warp size
    const int WARP_SIZE = 32;
    
    // Get warp and lane IDs
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Shared memory for positions and masses (one set per warp)
    extern __shared__ char shared_memory[];
    Vec3* shared_positions = reinterpret_cast<Vec3*>(shared_memory) + warp_id * WARP_SIZE;
    scalar_t* shared_masses = reinterpret_cast<scalar_t*>(shared_memory + blockDim.x * sizeof(Vec3)) + warp_id * WARP_SIZE;
    
    // Load current particle data
    Vec3 my_pos;
    if (i < num_particles) {
        my_pos = positions[i];
    }
    
    // Initialize acceleration to zero
    Vec3 acc(0.0, 0.0, 0.0);
    
    // Process tiles
    for (index_t tile_start = 0; tile_start < num_particles; tile_start += WARP_SIZE) {
        // Load tile into shared memory
        if (tile_start + lane_id < num_particles) {
            shared_positions[lane_id] = positions[tile_start + lane_id];
            shared_masses[lane_id] = masses[tile_start + lane_id];
        }
        
        // Compute interactions with particles in this tile
        if (i < num_particles) {
            for (index_t j = 0; j < WARP_SIZE && tile_start + j < num_particles; ++j) {
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
    }
    
    // Store computed acceleration
    if (i < num_particles) {
        accelerations[i] = acc;
    }
}

// CPU implementation for hybrid workload distribution
void compute_accelerations_cpu_parallel(
    const Vec3* h_positions,
    const scalar_t* h_masses,
    Vec3* h_accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    index_t start_idx,
    index_t end_idx
) {
    for (index_t i = start_idx; i < end_idx; ++i) {
        const Vec3& pos_i = h_positions[i];
        Vec3 acc(0.0, 0.0, 0.0);
        
        for (index_t j = 0; j < num_particles; ++j) {
            if (j == i) {
                continue;
            }
            
            const Vec3& pos_j = h_positions[j];
            const scalar_t mass_j = h_masses[j];
            
            Vec3 r = pos_j - pos_i;
            scalar_t r_squared = r.length_squared();
            r_squared += softening * softening;
            
            scalar_t inv_r_cubed = 1.0 / (sqrt(r_squared) * r_squared);
            acc += r * (G * mass_j * inv_r_cubed);
        }
        
        h_accelerations[i] = acc;
    }
}

// Helper for full CPU implementation
void compute_accelerations_cpu(
    const Vec3* h_positions,
    const scalar_t* h_masses,
    Vec3* h_accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening
) {
    // Determine number of threads to use
    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4; // Fallback to 4 threads
    }
    
    // Don't use more threads than particles
    num_threads = std::min(num_threads, static_cast<unsigned int>(num_particles));
    
    // Compute work partition size
    index_t partition_size = num_particles / num_threads;
    
    // Launch worker threads
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        index_t start_idx = t * partition_size;
        index_t end_idx = (t == num_threads - 1) ? num_particles : (t + 1) * partition_size;
        
        threads.push_back(std::thread(
            compute_accelerations_cpu_parallel,
            h_positions,
            h_masses,
            h_accelerations,
            num_particles,
            G,
            softening,
            start_idx,
            end_idx
        ));
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

// Launch templated tiled kernel
void launch_compute_accelerations_tiled_templated(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    index_t block_size,
    index_t tile_size,
    cudaStream_t stream
) {
    index_t grid_size = (num_particles + block_size - 1) / block_size;
    size_t shared_mem_size = tile_size * (sizeof(Vec3) + sizeof(scalar_t));
    
    switch (tile_size) {
        case 8:
            compute_accelerations_tiled_templated_kernel<8><<<grid_size, block_size, shared_mem_size, stream>>>(
                positions, masses, accelerations, num_particles, G, softening
            );
            break;
        case 16:
            compute_accelerations_tiled_templated_kernel<16><<<grid_size, block_size, shared_mem_size, stream>>>(
                positions, masses, accelerations, num_particles, G, softening
            );
            break;
        case 32:
            compute_accelerations_tiled_templated_kernel<32><<<grid_size, block_size, shared_mem_size, stream>>>(
                positions, masses, accelerations, num_particles, G, softening
            );
            break;
        default:
            compute_accelerations_tiled_templated_kernel<16><<<grid_size, block_size, shared_mem_size, stream>>>(
                positions, masses, accelerations, num_particles, G, softening
            );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

// Launch textured memory kernel
void launch_compute_accelerations_textured(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    index_t block_size,
    bool use_texture_memory,
    cudaStream_t stream
) {
    // If texture memory is not requested, fall back to tiled kernel
    if (!use_texture_memory) {
        launch_compute_accelerations_tiled_templated(
            positions, masses, accelerations, num_particles, G, softening, block_size, 16, stream
        );
        return;
    }
    
    // Bind masses to texture
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaBindTexture(NULL, tex_masses, masses, channelDesc, num_particles * sizeof(float)));
    
    // Create texture object for positions
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = (void*)positions;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float4>();
    resDesc.res.linear.sizeInBytes = num_particles * sizeof(float4);
    
    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    
    CUDA_CHECK(cudaCreateTextureObject(&tex_positions, &resDesc, &texDesc, NULL));
    
    // Launch kernel
    index_t grid_size = (num_particles + block_size - 1) / block_size;
    compute_accelerations_textured_kernel<<<grid_size, block_size, 0, stream>>>(
        positions, tex_positions, accelerations, num_particles, G, softening
    );
    
    // Clean up
    CUDA_CHECK(cudaUnbindTexture(tex_masses));
    CUDA_CHECK(cudaDestroyTextureObject(tex_positions));
    
    CUDA_CHECK(cudaGetLastError());
}

// Launch warp-optimized kernel
void launch_compute_accelerations_warp_optimized(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    index_t block_size,
    cudaStream_t stream
) {
    index_t grid_size = (num_particles + block_size - 1) / block_size;
    
    // Calculate shared memory size (one set of positions and masses per warp)
    int warps_per_block = (block_size + 31) / 32;
    size_t shared_mem_size = warps_per_block * 32 * (sizeof(Vec3) + sizeof(scalar_t));
    
    compute_accelerations_warp_optimized_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        positions, masses, accelerations, num_particles, G, softening
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Adaptive launch function that chooses the best kernel for the device
void launch_compute_accelerations_adaptive(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    const DeviceCapabilities& capabilities,
    WorkloadStrategy strategy,
    cudaStream_t stream
) {
    // If CPU-only strategy is requested or we have no GPU
    if (strategy == WorkloadStrategy::CPUOnly || capabilities.device_type == DeviceType::CPU) {
        // Copy data to host if it's not already there
        Vec3* h_positions = new Vec3[num_particles];
        scalar_t* h_masses = new scalar_t[num_particles];
        Vec3* h_accelerations = new Vec3[num_particles];
        
        CUDA_CHECK(cudaMemcpy(h_positions, positions, num_particles * sizeof(Vec3), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_masses, masses, num_particles * sizeof(scalar_t), cudaMemcpyDeviceToHost));
        
        // Compute on CPU
        compute_accelerations_cpu(h_positions, h_masses, h_accelerations, num_particles, G, softening);
        
        // Copy results back to device
        CUDA_CHECK(cudaMemcpy(accelerations, h_accelerations, num_particles * sizeof(Vec3), cudaMemcpyHostToDevice));
        
        // Free host memory
        delete[] h_positions;
        delete[] h_masses;
        delete[] h_accelerations;
        
        return;
    }
    
    // Choose appropriate kernel based on device capabilities
    if (capabilities.device_type == DeviceType::JetsonOrin) {
        // Jetson Orin: Use tiled kernel with optimized parameters
        index_t block_size = 128;
        index_t tile_size = 16;
        launch_compute_accelerations_tiled_templated(
            positions, masses, accelerations, num_particles, G, softening, block_size, tile_size, stream
        );
    } 
    else if (capabilities.device_type == DeviceType::T4) {
        // T4: Use textured memory for better cache performance
        index_t block_size = 128;
        bool use_texture_memory = true;
        launch_compute_accelerations_textured(
            positions, masses, accelerations, num_particles, G, softening, block_size, use_texture_memory, stream
        );
    }
    else if (capabilities.compute_capability_major >= 7) {
        // SM 7.0+: Use warp-optimized kernel
        index_t block_size = 256;
        launch_compute_accelerations_warp_optimized(
            positions, masses, accelerations, num_particles, G, softening, block_size, stream
        );
    }
    else {
        // Fallback: Use basic tiled kernel
        index_t block_size = 256;
        index_t tile_size = 16;
        launch_compute_accelerations_tiled_templated(
            positions, masses, accelerations, num_particles, G, softening, block_size, tile_size, stream
        );
    }
    
    // For hybrid strategies, also compute part of the work on CPU
    if (strategy == WorkloadStrategy::Hybrid || strategy == WorkloadStrategy::AdaptiveHybrid) {
        // Determine CPU/GPU split ratio
        float cpu_ratio = 0.2f;  // Default: CPU handles 20% of particles
        
        if (strategy == WorkloadStrategy::AdaptiveHybrid) {
            // Adaptive strategy adjusts ratio based on relative performance
            // This would need benchmarking to determine optimal split
            // For now, we use a simplified heuristic based on device type
            if (capabilities.device_type == DeviceType::JetsonOrin) {
                cpu_ratio = 0.1f;  // Jetson Orin: CPU handles 10%
            } else if (capabilities.device_type == DeviceType::T4) {
                cpu_ratio = 0.05f; // T4: CPU handles 5%
            } else if (capabilities.device_type == DeviceType::HighEnd) {
                cpu_ratio = 0.02f; // High-end GPU: CPU handles 2%
            } else {
                cpu_ratio = 0.3f;  // Other GPUs: CPU handles 30%
            }
        }
        
        // Determine CPU portion size
        index_t cpu_start = 0;
        index_t cpu_end = static_cast<index_t>(num_particles * cpu_ratio);
        
        if (cpu_end > 0) {
            // Copy data for CPU portion
            Vec3* h_positions = new Vec3[num_particles];
            scalar_t* h_masses = new scalar_t[num_particles];
            Vec3* h_accelerations = new Vec3[cpu_end];
            
            CUDA_CHECK(cudaMemcpy(h_positions, positions, num_particles * sizeof(Vec3), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_masses, masses, num_particles * sizeof(scalar_t), cudaMemcpyDeviceToHost));
            
            // Compute CPU portion
            compute_accelerations_cpu_parallel(
                h_positions, h_masses, h_accelerations, num_particles, G, softening, cpu_start, cpu_end
            );
            
            // Copy CPU results back to device
            CUDA_CHECK(cudaMemcpy(accelerations + cpu_start, h_accelerations, 
                      (cpu_end - cpu_start) * sizeof(Vec3), cudaMemcpyHostToDevice));
            
            // Free host memory
            delete[] h_positions;
            delete[] h_masses;
            delete[] h_accelerations;
        }
    }
}

} // namespace cuda
} // namespace nbody_sim