// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "molecular_dynamics/common.hpp"
#include <cuda_runtime.h>

namespace molecular_dynamics {

/**
 * @brief Compute forces between atoms using the Lennard-Jones potential.
 * 
 * @param positions Array of atom positions
 * @param charges Array of atom charges
 * @param types Array of atom types (as integers)
 * @param forces Output array of forces
 * @param lj_params Flattened array of Lennard-Jones parameters (sigma, epsilon)
 * @param box_dims Box dimensions for periodic boundary conditions
 * @param use_pbc Whether to use periodic boundary conditions
 * @param cutoff Cutoff distance for interactions
 * @param num_atoms Number of atoms
 * @param stream CUDA stream
 */
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
);

/**
 * @brief Compute forces using a tiled approach for better performance.
 * 
 * @param positions Array of atom positions
 * @param charges Array of atom charges
 * @param types Array of atom types (as integers)
 * @param forces Output array of forces
 * @param lj_params Flattened array of Lennard-Jones parameters (sigma, epsilon)
 * @param box_dims Box dimensions for periodic boundary conditions
 * @param use_pbc Whether to use periodic boundary conditions
 * @param cutoff Cutoff distance for interactions
 * @param num_atoms Number of atoms
 * @param block_size CUDA block size (scaling parameter)
 * @param tile_size Tile size (scaling parameter)
 * @param stream CUDA stream
 */
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
);

/**
 * @brief Update positions using the velocity Verlet algorithm.
 * 
 * @param positions Array of atom positions
 * @param velocities Array of atom velocities
 * @param forces Array of atom forces
 * @param masses Array of atom masses
 * @param dt Time step in picoseconds
 * @param num_atoms Number of atoms
 * @param box_dims Box dimensions for periodic boundary conditions
 * @param use_pbc Whether to use periodic boundary conditions
 * @param stream CUDA stream
 */
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
);

/**
 * @brief Apply the Berendsen thermostat.
 * 
 * @param velocities Array of atom velocities
 * @param masses Array of atom masses
 * @param lambda Scaling factor
 * @param num_atoms Number of atoms
 * @param stream CUDA stream
 */
void launch_berendsen_thermostat_kernel(
    Vec3* velocities,
    const scalar_t* masses,
    scalar_t lambda,
    int num_atoms,
    cudaStream_t stream
);

/**
 * @brief Calculate the kinetic energy of all atoms.
 * 
 * @param velocities Array of atom velocities
 * @param masses Array of atom masses
 * @param energies Output array of per-atom energies
 * @param num_atoms Number of atoms
 * @param stream CUDA stream
 */
void launch_kinetic_energy_kernel(
    const Vec3* velocities,
    const scalar_t* masses,
    scalar_t* energies,
    int num_atoms,
    cudaStream_t stream
);

/**
 * @brief Calculate bond forces between atoms.
 * 
 * @param positions Array of atom positions
 * @param forces Output array of forces
 * @param atom1_indices Array of first atom indices for each bond
 * @param atom2_indices Array of second atom indices for each bond
 * @param eq_distances Array of equilibrium distances
 * @param force_constants Array of force constants
 * @param box_dims Box dimensions for periodic boundary conditions
 * @param use_pbc Whether to use periodic boundary conditions
 * @param num_bonds Number of bonds
 * @param stream CUDA stream
 */
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
);

/**
 * @brief Wrapper function to select the most appropriate kernel based on device capabilities.
 * 
 * @param positions Array of atom positions
 * @param charges Array of atom charges
 * @param types Array of atom types (as integers)
 * @param forces Output array of forces
 * @param lj_params Flattened array of Lennard-Jones parameters (sigma, epsilon)
 * @param box_dims Box dimensions for periodic boundary conditions
 * @param use_pbc Whether to use periodic boundary conditions
 * @param cutoff Cutoff distance for interactions
 * @param num_atoms Number of atoms
 * @param capabilities Device capabilities for scaling
 * @param stream CUDA stream
 */
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
);

} // namespace molecular_dynamics