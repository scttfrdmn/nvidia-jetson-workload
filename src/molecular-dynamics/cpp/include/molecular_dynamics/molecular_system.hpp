// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "molecular_dynamics/atom.hpp"
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <array>

namespace molecular_dynamics {

/**
 * @brief Class representing a molecular system (collection of atoms).
 */
class MolecularSystem {
public:
    /**
     * @brief Default constructor.
     */
    MolecularSystem();

    /**
     * @brief Constructor with atoms.
     * 
     * @param atoms Vector of atoms
     */
    explicit MolecularSystem(const std::vector<Atom>& atoms);

    /**
     * @brief Constructor with atoms and box dimensions.
     * 
     * @param atoms Vector of atoms
     * @param box_dimensions Box dimensions for periodic boundary conditions
     */
    MolecularSystem(
        const std::vector<Atom>& atoms,
        const Vec3& box_dimensions
    );

    /**
     * @brief Get the number of atoms in the system.
     * 
     * @return size_t Number of atoms
     */
    size_t size() const { return atoms_.size(); }

    /**
     * @brief Get a reference to an atom by index.
     * 
     * @param index Atom index
     * @return const Atom& Atom reference
     */
    const Atom& atom(size_t index) const;

    /**
     * @brief Get a mutable reference to an atom by index.
     * 
     * @param index Atom index
     * @return Atom& Mutable atom reference
     */
    Atom& atom(size_t index);

    /**
     * @brief Get all atoms in the system.
     * 
     * @return const std::vector<Atom>& Vector of atoms
     */
    const std::vector<Atom>& atoms() const { return atoms_; }

    /**
     * @brief Get a mutable reference to all atoms.
     * 
     * @return std::vector<Atom>& Mutable vector of atoms
     */
    std::vector<Atom>& atoms() { return atoms_; }

    /**
     * @brief Add an atom to the system.
     * 
     * @param atom Atom to add
     */
    void add_atom(const Atom& atom);

    /**
     * @brief Get the box dimensions.
     * 
     * @return Vec3 Box dimensions
     */
    const Vec3& box_dimensions() const { return box_dimensions_; }

    /**
     * @brief Set the box dimensions.
     * 
     * @param dimensions New box dimensions
     */
    void set_box_dimensions(const Vec3& dimensions) { box_dimensions_ = dimensions; }

    /**
     * @brief Check if periodic boundary conditions are enabled.
     * 
     * @return bool True if PBC is enabled
     */
    bool has_periodic_boundary() const { return use_periodic_boundary_; }

    /**
     * @brief Enable or disable periodic boundary conditions.
     * 
     * @param enable Whether to enable PBC
     */
    void set_periodic_boundary(bool enable) { use_periodic_boundary_ = enable; }

    /**
     * @brief Calculate total kinetic energy of the system.
     * 
     * @return scalar_t Total kinetic energy in kJ/mol
     */
    scalar_t total_kinetic_energy() const;

    /**
     * @brief Calculate total potential energy of the system.
     * 
     * @return scalar_t Total potential energy in kJ/mol
     */
    scalar_t total_potential_energy() const;

    /**
     * @brief Calculate total energy of the system.
     * 
     * @return scalar_t Total energy in kJ/mol
     */
    scalar_t total_energy() const;

    /**
     * @brief Calculate the temperature of the system.
     * 
     * @return scalar_t Temperature in Kelvin
     */
    scalar_t temperature() const;

    /**
     * @brief Apply minimum image convention for periodic boundary conditions.
     * 
     * @param r Vector to apply PBC to
     * @return Vec3 Vector with PBC applied
     */
    Vec3 apply_periodic_boundary(const Vec3& r) const;

    /**
     * @brief Calculate minimum image distance between two positions.
     * 
     * @param r1 First position
     * @param r2 Second position
     * @return Vec3 Minimum image vector from r1 to r2
     */
    Vec3 minimum_image_vector(const Vec3& r1, const Vec3& r2) const;

    /**
     * @brief Calculate distance matrix between all atoms.
     * 
     * @return std::vector<std::vector<scalar_t>> Distance matrix
     */
    std::vector<std::vector<scalar_t>> calculate_distance_matrix() const;

    /**
     * @brief Update forces on all atoms.
     */
    void update_forces();
    
    /**
     * @brief Update forces on all atoms using GPU.
     */
    void update_forces_gpu();
    
    /**
     * @brief Update forces on all atoms using CPU.
     */
    void update_forces_cpu();
    
    /**
     * @brief Create a deep copy of this molecular system.
     * 
     * @return std::unique_ptr<MolecularSystem> New molecular system
     */
    std::unique_ptr<MolecularSystem> clone() const;
    
    /**
     * @brief Load a molecular system from a PDB file.
     * 
     * @param filename Path to PDB file
     * @return std::unique_ptr<MolecularSystem> New molecular system
     */
    static std::unique_ptr<MolecularSystem> load_from_pdb(const std::string& filename);
    
    /**
     * @brief Load a molecular system from a PDB file with force field parameters.
     * 
     * @param pdb_filename Path to PDB file
     * @param topology_filename Path to topology file
     * @param parameter_filename Path to parameter file
     * @param force_field_type Type of force field
     * @return std::unique_ptr<MolecularSystem> New molecular system
     */
    static std::unique_ptr<MolecularSystem> load_with_forcefield(
        const std::string& pdb_filename,
        const std::string& topology_filename,
        const std::string& parameter_filename,
        ForceFieldType force_field_type = ForceFieldType::AMBER
    );
    
    /**
     * @brief Create a water box system.
     * 
     * @param box_size Size of the cubic box in Angstroms
     * @param density Density in g/cm^3 (default is 1.0 for water)
     * @return std::unique_ptr<MolecularSystem> New molecular system
     */
    static std::unique_ptr<MolecularSystem> create_water_box(
        scalar_t box_size,
        scalar_t density = 1.0
    );
    
    /**
     * @brief Create a Lennard-Jones fluid system.
     * 
     * @param num_particles Number of particles
     * @param box_size Size of the cubic box in Angstroms
     * @param temperature Temperature in Kelvin
     * @param seed Random seed
     * @return std::unique_ptr<MolecularSystem> New molecular system
     */
    static std::unique_ptr<MolecularSystem> create_lj_fluid(
        size_t num_particles,
        scalar_t box_size,
        scalar_t temperature = DEFAULT_TEMPERATURE,
        unsigned int seed = 0
    );

private:
    std::vector<Atom> atoms_;                // Atoms in the system
    Vec3 box_dimensions_;                    // Box dimensions for periodic boundary conditions
    bool use_periodic_boundary_ = false;     // Whether to use periodic boundary conditions
    
    // Lookup table for LJ parameters
    struct LJParams {
        scalar_t sigma;  // in Angstroms
        scalar_t epsilon;  // in kJ/mol
    };
    std::unordered_map<AtomType, LJParams> lj_params_;
    
    // Bond and angle information for force calculations
    struct Bond {
        atom_id_t atom1_id;
        atom_id_t atom2_id;
        scalar_t equilibrium_length;  // in Angstroms
        scalar_t force_constant;      // in kJ/(mol*A^2)
    };
    
    struct Angle {
        atom_id_t atom1_id;
        atom_id_t atom2_id;
        atom_id_t atom3_id;
        scalar_t equilibrium_angle;  // in radians
        scalar_t force_constant;     // in kJ/(mol*rad^2)
    };
    
    struct Dihedral {
        atom_id_t atom1_id;
        atom_id_t atom2_id;
        atom_id_t atom3_id;
        atom_id_t atom4_id;
        int periodicity;
        scalar_t phase;           // in radians
        scalar_t force_constant;  // in kJ/mol
    };
    
    std::vector<Bond> bonds_;
    std::vector<Angle> angles_;
    std::vector<Dihedral> dihedrals_;
    
    // Neighbor list for optimization
    struct NeighborList {
        std::vector<std::vector<size_t>> neighbors;
        scalar_t cutoff;
        scalar_t skin;
        Vec3 last_update_box_dimensions;
        bool valid = false;
    };
    
    NeighborList neighbor_list_;
    
    // Initialize LJ parameters
    void initialize_lj_parameters();
    
    // Update neighbor list
    void update_neighbor_list(scalar_t cutoff, scalar_t skin = 2.0);
    
    // Calculate non-bonded forces (LJ + electrostatic)
    void calculate_nonbonded_forces();
    
    // Calculate bonded forces (bonds, angles, dihedrals)
    void calculate_bonded_forces();
    
    // Apply force field to a loaded system
    void apply_force_field(
        const std::string& topology_filename,
        const std::string& parameter_filename,
        ForceFieldType force_field_type
    );
};

} // namespace molecular_dynamics