// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "molecular_dynamics/common.hpp"
#include <string>

namespace molecular_dynamics {

/**
 * @brief Class representing an atom in a molecular system.
 */
class Atom {
public:
    /**
     * @brief Default constructor.
     */
    Atom();

    /**
     * @brief Constructor with parameters.
     * 
     * @param position Initial position
     * @param velocity Initial velocity
     * @param mass Atomic mass in atomic mass units (amu)
     * @param charge Atomic charge in elementary charge units
     * @param atom_type Type of atom (element)
     * @param atom_id Unique atom identifier
     * @param residue_id Residue identifier
     * @param atom_name Atom name (e.g., "CA" for alpha carbon)
     * @param residue_name Residue name (e.g., "ALA" for alanine)
     */
    Atom(
        const Vec3& position,
        const Vec3& velocity,
        scalar_t mass,
        scalar_t charge,
        AtomType atom_type,
        atom_id_t atom_id,
        res_id_t residue_id,
        const std::string& atom_name = "",
        const std::string& residue_name = ""
    );

    /**
     * @brief Get the position of the atom.
     * 
     * @return Vec3 Position vector
     */
    const Vec3& position() const { return position_; }

    /**
     * @brief Set the position of the atom.
     * 
     * @param position New position
     */
    void set_position(const Vec3& position) { position_ = position; }

    /**
     * @brief Get the velocity of the atom.
     * 
     * @return Vec3 Velocity vector
     */
    const Vec3& velocity() const { return velocity_; }

    /**
     * @brief Set the velocity of the atom.
     * 
     * @param velocity New velocity
     */
    void set_velocity(const Vec3& velocity) { velocity_ = velocity; }

    /**
     * @brief Get the force on the atom.
     * 
     * @return Vec3 Force vector
     */
    const Vec3& force() const { return force_; }

    /**
     * @brief Set the force on the atom.
     * 
     * @param force New force
     */
    void set_force(const Vec3& force) { force_ = force; }

    /**
     * @brief Get the atom mass.
     * 
     * @return scalar_t Mass in amu
     */
    scalar_t mass() const { return mass_; }

    /**
     * @brief Set the atom mass.
     * 
     * @param mass New mass in amu
     */
    void set_mass(scalar_t mass) { mass_ = mass; }

    /**
     * @brief Get the atom charge.
     * 
     * @return scalar_t Charge in elementary charge units
     */
    scalar_t charge() const { return charge_; }

    /**
     * @brief Set the atom charge.
     * 
     * @param charge New charge
     */
    void set_charge(scalar_t charge) { charge_ = charge; }

    /**
     * @brief Get the atom type.
     * 
     * @return AtomType Type of the atom
     */
    AtomType type() const { return type_; }

    /**
     * @brief Set the atom type.
     * 
     * @param type New atom type
     */
    void set_type(AtomType type) { type_ = type; }

    /**
     * @brief Get the atom ID.
     * 
     * @return atom_id_t Atom identifier
     */
    atom_id_t id() const { return id_; }

    /**
     * @brief Get the residue ID.
     * 
     * @return res_id_t Residue identifier
     */
    res_id_t residue_id() const { return residue_id_; }

    /**
     * @brief Get the atom name.
     * 
     * @return std::string Atom name
     */
    const std::string& atom_name() const { return atom_name_; }

    /**
     * @brief Get the residue name.
     * 
     * @return std::string Residue name
     */
    const std::string& residue_name() const { return residue_name_; }

    /**
     * @brief Calculate the kinetic energy of the atom.
     * 
     * @return scalar_t Kinetic energy in kJ/mol
     */
    scalar_t kinetic_energy() const;

    /**
     * @brief Update the position based on velocity and force.
     * 
     * @param dt Time step in picoseconds
     */
    void update_position(scalar_t dt);

    /**
     * @brief Update the velocity based on force.
     * 
     * @param dt Time step in picoseconds
     */
    void update_velocity(scalar_t dt);

private:
    Vec3 position_;       // Position in Angstroms
    Vec3 velocity_;       // Velocity in Angstroms/ps
    Vec3 force_;          // Force in kJ/(mol*Angstrom)
    scalar_t mass_;       // Mass in amu
    scalar_t charge_;     // Charge in elementary charge units
    AtomType type_;       // Atom type (element)
    atom_id_t id_;        // Unique atom identifier
    res_id_t residue_id_; // Residue identifier
    std::string atom_name_;     // Atom name (e.g., "CA" for alpha carbon)
    std::string residue_name_;  // Residue name (e.g., "ALA" for alanine)
};

} // namespace molecular_dynamics