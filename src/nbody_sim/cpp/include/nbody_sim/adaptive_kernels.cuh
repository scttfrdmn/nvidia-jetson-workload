// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "nbody_sim/common.hpp"
#include "nbody_sim/device_adaptor.hpp"

namespace nbody_sim {
namespace cuda {

/**
 * @brief Launch templated tiled kernel with specified tile size.
 * 
 * @param positions Array of particle positions
 * @param masses Array of particle masses
 * @param accelerations Output array for accelerations
 * @param num_particles Number of particles
 * @param G Gravitational constant
 * @param softening Softening parameter
 * @param block_size CUDA block size
 * @param tile_size Tile size (8, 16, or 32)
 * @param stream CUDA stream
 */
void launch_compute_accelerations_tiled_templated(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    index_t block_size,
    index_t tile_size,
    cudaStream_t stream = 0
);

/**
 * @brief Launch kernel that uses texture memory for better cache performance.
 * 
 * @param positions Array of particle positions
 * @param masses Array of particle masses
 * @param accelerations Output array for accelerations
 * @param num_particles Number of particles
 * @param G Gravitational constant
 * @param softening Softening parameter
 * @param block_size CUDA block size
 * @param use_texture_memory Whether to use texture memory
 * @param stream CUDA stream
 */
void launch_compute_accelerations_textured(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    index_t block_size,
    bool use_texture_memory = true,
    cudaStream_t stream = 0
);

/**
 * @brief Launch warp-optimized kernel for SM 7.0+ GPUs.
 * 
 * @param positions Array of particle positions
 * @param masses Array of particle masses
 * @param accelerations Output array for accelerations
 * @param num_particles Number of particles
 * @param G Gravitational constant
 * @param softening Softening parameter
 * @param block_size CUDA block size
 * @param stream CUDA stream
 */
void launch_compute_accelerations_warp_optimized(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    index_t block_size,
    cudaStream_t stream = 0
);

/**
 * @brief Compute accelerations on CPU with multithreading.
 * 
 * @param h_positions Host array of particle positions
 * @param h_masses Host array of particle masses
 * @param h_accelerations Host output array for accelerations
 * @param num_particles Number of particles
 * @param G Gravitational constant
 * @param softening Softening parameter
 */
void compute_accelerations_cpu(
    const Vec3* h_positions,
    const scalar_t* h_masses,
    Vec3* h_accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening
);

/**
 * @brief Compute accelerations on CPU for a specific range of particles.
 * 
 * @param h_positions Host array of particle positions
 * @param h_masses Host array of particle masses
 * @param h_accelerations Host output array for accelerations
 * @param num_particles Number of particles
 * @param G Gravitational constant
 * @param softening Softening parameter
 * @param start_idx Start index for the range
 * @param end_idx End index for the range
 */
void compute_accelerations_cpu_parallel(
    const Vec3* h_positions,
    const scalar_t* h_masses,
    Vec3* h_accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    index_t start_idx,
    index_t end_idx
);

/**
 * @brief Adaptive launch function that chooses the best kernel for the device.
 * 
 * This function selects the most appropriate implementation based on device
 * capabilities and workload strategy.
 * 
 * @param positions Array of particle positions
 * @param masses Array of particle masses
 * @param accelerations Output array for accelerations
 * @param num_particles Number of particles
 * @param G Gravitational constant
 * @param softening Softening parameter
 * @param capabilities Device capabilities
 * @param strategy Workload distribution strategy
 * @param stream CUDA stream
 */
void launch_compute_accelerations_adaptive(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    const DeviceCapabilities& capabilities,
    WorkloadStrategy strategy = WorkloadStrategy::GPUOnly,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace nbody_sim