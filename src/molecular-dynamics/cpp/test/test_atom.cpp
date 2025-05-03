// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include <gtest/gtest.h>
#include "molecular_dynamics/atom.hpp"
#include <cmath>

namespace molecular_dynamics {
namespace test {

// Test atom construction and basic properties
TEST(AtomTest, Construction) {
    // Default constructor
    Atom atom1;
    EXPECT_DOUBLE_EQ(atom1.position().x, 0.0);
    EXPECT_DOUBLE_EQ(atom1.position().y, 0.0);
    EXPECT_DOUBLE_EQ(atom1.position().z, 0.0);
    EXPECT_DOUBLE_EQ(atom1.velocity().x, 0.0);
    EXPECT_DOUBLE_EQ(atom1.velocity().y, 0.0);
    EXPECT_DOUBLE_EQ(atom1.velocity().z, 0.0);
    EXPECT_DOUBLE_EQ(atom1.force().x, 0.0);
    EXPECT_DOUBLE_EQ(atom1.force().y, 0.0);
    EXPECT_DOUBLE_EQ(atom1.force().z, 0.0);
    EXPECT_DOUBLE_EQ(atom1.mass(), 0.0);
    EXPECT_DOUBLE_EQ(atom1.charge(), 0.0);
    EXPECT_EQ(atom1.type(), AtomType::Other);
    EXPECT_EQ(atom1.id(), 0);
    EXPECT_EQ(atom1.residue_id(), 0);
    EXPECT_EQ(atom1.atom_name(), "");
    EXPECT_EQ(atom1.residue_name(), "");
    
    // Parameterized constructor
    Vec3 pos(1.0, 2.0, 3.0);
    Vec3 vel(4.0, 5.0, 6.0);
    scalar_t mass = 12.0;
    scalar_t charge = -0.5;
    AtomType type = AtomType::Carbon;
    atom_id_t id = 42;
    res_id_t res_id = 10;
    std::string atom_name = "CA";
    std::string res_name = "ALA";
    
    Atom atom2(pos, vel, mass, charge, type, id, res_id, atom_name, res_name);
    
    EXPECT_DOUBLE_EQ(atom2.position().x, 1.0);
    EXPECT_DOUBLE_EQ(atom2.position().y, 2.0);
    EXPECT_DOUBLE_EQ(atom2.position().z, 3.0);
    EXPECT_DOUBLE_EQ(atom2.velocity().x, 4.0);
    EXPECT_DOUBLE_EQ(atom2.velocity().y, 5.0);
    EXPECT_DOUBLE_EQ(atom2.velocity().z, 6.0);
    EXPECT_DOUBLE_EQ(atom2.force().x, 0.0);
    EXPECT_DOUBLE_EQ(atom2.force().y, 0.0);
    EXPECT_DOUBLE_EQ(atom2.force().z, 0.0);
    EXPECT_DOUBLE_EQ(atom2.mass(), 12.0);
    EXPECT_DOUBLE_EQ(atom2.charge(), -0.5);
    EXPECT_EQ(atom2.type(), AtomType::Carbon);
    EXPECT_EQ(atom2.id(), 42);
    EXPECT_EQ(atom2.residue_id(), 10);
    EXPECT_EQ(atom2.atom_name(), "CA");
    EXPECT_EQ(atom2.residue_name(), "ALA");
}

// Test setting atom properties
TEST(AtomTest, SetProperties) {
    Atom atom;
    
    Vec3 pos(1.0, 2.0, 3.0);
    atom.set_position(pos);
    EXPECT_DOUBLE_EQ(atom.position().x, 1.0);
    EXPECT_DOUBLE_EQ(atom.position().y, 2.0);
    EXPECT_DOUBLE_EQ(atom.position().z, 3.0);
    
    Vec3 vel(4.0, 5.0, 6.0);
    atom.set_velocity(vel);
    EXPECT_DOUBLE_EQ(atom.velocity().x, 4.0);
    EXPECT_DOUBLE_EQ(atom.velocity().y, 5.0);
    EXPECT_DOUBLE_EQ(atom.velocity().z, 6.0);
    
    Vec3 force(7.0, 8.0, 9.0);
    atom.set_force(force);
    EXPECT_DOUBLE_EQ(atom.force().x, 7.0);
    EXPECT_DOUBLE_EQ(atom.force().y, 8.0);
    EXPECT_DOUBLE_EQ(atom.force().z, 9.0);
    
    atom.set_mass(12.0);
    EXPECT_DOUBLE_EQ(atom.mass(), 12.0);
    
    atom.set_charge(-0.5);
    EXPECT_DOUBLE_EQ(atom.charge(), -0.5);
    
    atom.set_type(AtomType::Oxygen);
    EXPECT_EQ(atom.type(), AtomType::Oxygen);
}

// Test kinetic energy calculation
TEST(AtomTest, KineticEnergy) {
    Vec3 pos(0.0, 0.0, 0.0);
    Vec3 vel(2.0, 3.0, 4.0);
    scalar_t mass = 2.0;
    
    Atom atom(pos, vel, mass, 0.0, AtomType::Other, 0, 0);
    
    // KE = 0.5 * m * v^2 = 0.5 * 2.0 * (2^2 + 3^2 + 4^2) = 0.5 * 2.0 * 29 = 29.0
    scalar_t expected_ke = 0.5 * mass * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
    scalar_t ke = atom.kinetic_energy();
    
    EXPECT_DOUBLE_EQ(ke, expected_ke);
}

// Test position update
TEST(AtomTest, UpdatePosition) {
    Vec3 pos(1.0, 2.0, 3.0);
    Vec3 vel(4.0, 5.0, 6.0);
    Vec3 force(7.0, 8.0, 9.0);
    scalar_t mass = 2.0;
    
    Atom atom(pos, vel, mass, 0.0, AtomType::Other, 0, 0);
    atom.set_force(force);
    
    scalar_t dt = 0.1;
    atom.update_position(dt);
    
    // New position = old position + velocity * dt + 0.5 * force/mass * dt^2
    Vec3 expected_pos = pos + vel * dt + (force / mass) * (0.5 * dt * dt);
    
    EXPECT_DOUBLE_EQ(atom.position().x, expected_pos.x);
    EXPECT_DOUBLE_EQ(atom.position().y, expected_pos.y);
    EXPECT_DOUBLE_EQ(atom.position().z, expected_pos.z);
}

// Test velocity update
TEST(AtomTest, UpdateVelocity) {
    Vec3 pos(1.0, 2.0, 3.0);
    Vec3 vel(4.0, 5.0, 6.0);
    Vec3 force(7.0, 8.0, 9.0);
    scalar_t mass = 2.0;
    
    Atom atom(pos, vel, mass, 0.0, AtomType::Other, 0, 0);
    atom.set_force(force);
    
    scalar_t dt = 0.1;
    atom.update_velocity(dt);
    
    // New velocity = old velocity + force/mass * dt
    Vec3 expected_vel = vel + (force / mass) * dt;
    
    EXPECT_DOUBLE_EQ(atom.velocity().x, expected_vel.x);
    EXPECT_DOUBLE_EQ(atom.velocity().y, expected_vel.y);
    EXPECT_DOUBLE_EQ(atom.velocity().z, expected_vel.z);
}

} // namespace test
} // namespace molecular_dynamics