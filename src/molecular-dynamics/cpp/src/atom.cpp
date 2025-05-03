// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "molecular_dynamics/atom.hpp"

namespace molecular_dynamics {

Atom::Atom()
    : position_(Vec3(0, 0, 0)),
      velocity_(Vec3(0, 0, 0)),
      force_(Vec3(0, 0, 0)),
      mass_(0.0),
      charge_(0.0),
      type_(AtomType::Other),
      id_(0),
      residue_id_(0),
      atom_name_(""),
      residue_name_("") {
}

Atom::Atom(
    const Vec3& position,
    const Vec3& velocity,
    scalar_t mass,
    scalar_t charge,
    AtomType atom_type,
    atom_id_t atom_id,
    res_id_t residue_id,
    const std::string& atom_name,
    const std::string& residue_name
)
    : position_(position),
      velocity_(velocity),
      force_(Vec3(0, 0, 0)),
      mass_(mass),
      charge_(charge),
      type_(atom_type),
      id_(atom_id),
      residue_id_(residue_id),
      atom_name_(atom_name),
      residue_name_(residue_name) {
}

scalar_t Atom::kinetic_energy() const {
    // KE = 0.5 * m * v^2
    return 0.5 * mass_ * velocity_.length_squared();
}

void Atom::update_position(scalar_t dt) {
    // Update position: r(t+dt) = r(t) + v(t)*dt + 0.5*f(t)/m*dt^2
    position_ += velocity_ * dt + (force_ / mass_) * (0.5 * dt * dt);
}

void Atom::update_velocity(scalar_t dt) {
    // Update velocity: v(t+dt) = v(t) + f(t)/m*dt
    velocity_ += (force_ / mass_) * dt;
}

} // namespace molecular_dynamics