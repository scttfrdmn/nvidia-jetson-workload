// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "nbody_sim/common.hpp"

namespace nbody_sim {
namespace cuda {

/**
 * @brief Compute accelerations for all particles using direct summation method.
 * 
 * This kernel calculates gravitational interactions between all pairs of particles
 * using the O(n²) direct summation approach. Each thread computes the acceleration
 * for one particle.
 * 
 * @param positions Array of particle positions
 * @param masses Array of particle masses
 * @param accelerations Output array for accelerations
 * @param num_particles Number of particles
 * @param G Gravitational constant
 * @param softening Softening parameter to prevent singularities
 */
__global__ void compute_accelerations_kernel(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening
);

/**
 * @brief Launch the acceleration computation kernel with appropriate parameters.
 * 
 * @param positions Array of particle positions
 * @param masses Array of particle masses
 * @param accelerations Output array for accelerations
 * @param num_particles Number of particles
 * @param G Gravitational constant
 * @param softening Softening parameter
 * @param block_size CUDA block size
 * @param stream CUDA stream (optional)
 */
void launch_compute_accelerations(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening = DEFAULT_SOFTENING,
    index_t block_size = DEFAULT_BLOCK_SIZE,
    cudaStream_t stream = 0
);

/**
 * @brief Compute accelerations using a tiled approach for better performance.
 * 
 * This kernel uses shared memory to reduce global memory accesses by loading
 * tiles of particle data into shared memory.
 * 
 * @param positions Array of particle positions
 * @param masses Array of particle masses
 * @param accelerations Output array for accelerations
 * @param num_particles Number of particles
 * @param G Gravitational constant
 * @param softening Softening parameter to prevent singularities
 */
__global__ void compute_accelerations_tiled_kernel(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening
);

/**
 * @brief Launch the tiled acceleration computation kernel.
 * 
 * @param positions Array of particle positions
 * @param masses Array of particle masses
 * @param accelerations Output array for accelerations
 * @param num_particles Number of particles
 * @param G Gravitational constant
 * @param softening Softening parameter
 * @param block_size CUDA block size
 * @param stream CUDA stream (optional)
 */
void launch_compute_accelerations_tiled(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening = DEFAULT_SOFTENING,
    index_t block_size = DEFAULT_BLOCK_SIZE,
    cudaStream_t stream = 0
);

/**
 * @brief Update particle positions using their velocities.
 * 
 * @param positions Array of particle positions
 * @param velocities Array of particle velocities
 * @param num_particles Number of particles
 * @param dt Time step
 */
__global__ void update_positions_kernel(
    Vec3* positions,
    const Vec3* velocities,
    index_t num_particles,
    scalar_t dt
);

/**
 * @brief Launch the position update kernel.
 * 
 * @param positions Array of particle positions
 * @param velocities Array of particle velocities
 * @param num_particles Number of particles
 * @param dt Time step
 * @param block_size CUDA block size
 * @param stream CUDA stream (optional)
 */
void launch_update_positions(
    Vec3* positions,
    const Vec3* velocities,
    index_t num_particles,
    scalar_t dt,
    index_t block_size = DEFAULT_BLOCK_SIZE,
    cudaStream_t stream = 0
);

/**
 * @brief Update particle velocities using their accelerations.
 * 
 * @param velocities Array of particle velocities
 * @param accelerations Array of particle accelerations
 * @param num_particles Number of particles
 * @param dt Time step
 */
__global__ void update_velocities_kernel(
    Vec3* velocities,
    const Vec3* accelerations,
    index_t num_particles,
    scalar_t dt
);

/**
 * @brief Launch the velocity update kernel.
 * 
 * @param velocities Array of particle velocities
 * @param accelerations Array of particle accelerations
 * @param num_particles Number of particles
 * @param dt Time step
 * @param block_size CUDA block size
 * @param stream CUDA stream (optional)
 */
void launch_update_velocities(
    Vec3* velocities,
    const Vec3* accelerations,
    index_t num_particles,
    scalar_t dt,
    index_t block_size = DEFAULT_BLOCK_SIZE,
    cudaStream_t stream = 0
);

/**
 * @brief Perform a half-step velocity update for Leapfrog integration.
 * 
 * @param velocities Array of particle velocities
 * @param accelerations Array of particle accelerations
 * @param num_particles Number of particles
 * @param dt Time step
 */
__global__ void update_velocities_half_step_kernel(
    Vec3* velocities,
    const Vec3* accelerations,
    index_t num_particles,
    scalar_t dt
);

/**
 * @brief Launch the half-step velocity update kernel.
 * 
 * @param velocities Array of particle velocities
 * @param accelerations Array of particle accelerations
 * @param num_particles Number of particles
 * @param dt Time step
 * @param block_size CUDA block size
 * @param stream CUDA stream (optional)
 */
void launch_update_velocities_half_step(
    Vec3* velocities,
    const Vec3* accelerations,
    index_t num_particles,
    scalar_t dt,
    index_t block_size = DEFAULT_BLOCK_SIZE,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace nbody_sim