// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "nbody_sim/common.hpp"
#include <vector>
#include <random>
#include <memory>

namespace nbody_sim {

/**
 * @brief Class representing a particle in the N-body simulation.
 */
class Particle {
public:
    /**
     * @brief Construct a new Particle object with default values.
     */
    Particle();

    /**
     * @brief Construct a new Particle object with specific values.
     * 
     * @param position Initial position vector
     * @param velocity Initial velocity vector
     * @param mass Particle mass
     * @param id Unique identifier for the particle
     */
    Particle(const Vec3& position, const Vec3& velocity, scalar_t mass, index_t id);

    /**
     * @brief Get the particle's position.
     * 
     * @return const Vec3& Position vector
     */
    const Vec3& position() const { return position_; }

    /**
     * @brief Get the particle's velocity.
     * 
     * @return const Vec3& Velocity vector
     */
    const Vec3& velocity() const { return velocity_; }

    /**
     * @brief Get the particle's acceleration.
     * 
     * @return const Vec3& Acceleration vector
     */
    const Vec3& acceleration() const { return acceleration_; }

    /**
     * @brief Get the particle's mass.
     * 
     * @return scalar_t Mass value
     */
    scalar_t mass() const { return mass_; }

    /**
     * @brief Get the particle's ID.
     * 
     * @return index_t Particle ID
     */
    index_t id() const { return id_; }

    /**
     * @brief Set the particle's position.
     * 
     * @param position New position vector
     */
    void set_position(const Vec3& position) { position_ = position; }

    /**
     * @brief Set the particle's velocity.
     * 
     * @param velocity New velocity vector
     */
    void set_velocity(const Vec3& velocity) { velocity_ = velocity; }

    /**
     * @brief Set the particle's acceleration.
     * 
     * @param acceleration New acceleration vector
     */
    void set_acceleration(const Vec3& acceleration) { acceleration_ = acceleration; }

    /**
     * @brief Set the particle's mass.
     * 
     * @param mass New mass value
     */
    void set_mass(scalar_t mass) { mass_ = mass; }

    /**
     * @brief Set the particle's ID.
     * 
     * @param id New particle ID
     */
    void set_id(index_t id) { id_ = id; }

    /**
     * @brief Update position based on velocity and time step.
     * 
     * @param dt Time step
     */
    void update_position(scalar_t dt);

    /**
     * @brief Update velocity based on acceleration and time step.
     * 
     * @param dt Time step
     */
    void update_velocity(scalar_t dt);

    /**
     * @brief Calculate the particle's kinetic energy.
     * 
     * @return scalar_t Kinetic energy value
     */
    scalar_t kinetic_energy() const;

private:
    Vec3 position_;      // Position vector
    Vec3 velocity_;      // Velocity vector
    Vec3 acceleration_;  // Acceleration vector
    scalar_t mass_;      // Mass
    index_t id_;         // Particle ID
};

/**
 * @brief Class representing a system of particles for N-body simulation.
 */
class ParticleSystem {
public:
    /**
     * @brief Construct a new ParticleSystem with default parameters.
     */
    ParticleSystem();

    /**
     * @brief Construct a new ParticleSystem with specified gravitational constant.
     * 
     * @param G Gravitational constant
     */
    explicit ParticleSystem(scalar_t G);

    /**
     * @brief Construct a new ParticleSystem with specified particles and gravitational constant.
     * 
     * @param particles Vector of particles
     * @param G Gravitational constant
     */
    ParticleSystem(const std::vector<Particle>& particles, scalar_t G = DEFAULT_G);

    /**
     * @brief Get the number of particles in the system.
     * 
     * @return size_t Number of particles
     */
    size_t size() const { return particles_.size(); }

    /**
     * @brief Get the gravitational constant.
     * 
     * @return scalar_t Gravitational constant
     */
    scalar_t gravitational_constant() const { return G_; }

    /**
     * @brief Set the gravitational constant.
     * 
     * @param G New gravitational constant
     */
    void set_gravitational_constant(scalar_t G) { G_ = G; }

    /**
     * @brief Add a particle to the system.
     * 
     * @param particle Particle to add
     */
    void add_particle(const Particle& particle);

    /**
     * @brief Get a particle by index.
     * 
     * @param index Index of the particle
     * @return const Particle& Reference to the particle
     */
    const Particle& particle(size_t index) const;

    /**
     * @brief Get a particle by index (mutable).
     * 
     * @param index Index of the particle
     * @return Particle& Reference to the particle
     */
    Particle& particle(size_t index);

    /**
     * @brief Get all particles.
     * 
     * @return const std::vector<Particle>& Vector of particles
     */
    const std::vector<Particle>& particles() const { return particles_; }

    /**
     * @brief Get all particles (mutable).
     * 
     * @return std::vector<Particle>& Vector of particles
     */
    std::vector<Particle>& particles() { return particles_; }

    /**
     * @brief Update accelerations of all particles based on gravitational interactions.
     * CPU implementation (O(n²) complexity).
     */
    void update_accelerations_cpu();

    /**
     * @brief Update accelerations of all particles based on gravitational interactions.
     * GPU implementation using CUDA.
     */
    void update_accelerations_gpu();

    /**
     * @brief Choose the best method to update accelerations based on system size and available hardware.
     */
    void update_accelerations();

    /**
     * @brief Calculate the total mass of the system.
     * 
     * @return scalar_t Total mass
     */
    scalar_t total_mass() const;

    /**
     * @brief Calculate the center of mass of the system.
     * 
     * @return Vec3 Center of mass position
     */
    Vec3 center_of_mass() const;

    /**
     * @brief Calculate the total momentum of the system.
     * 
     * @return Vec3 Total momentum vector
     */
    Vec3 total_momentum() const;

    /**
     * @brief Calculate the total angular momentum of the system.
     * 
     * @return Vec3 Total angular momentum vector
     */
    Vec3 total_angular_momentum() const;

    /**
     * @brief Calculate the total kinetic energy of the system.
     * 
     * @return scalar_t Total kinetic energy
     */
    scalar_t total_kinetic_energy() const;

    /**
     * @brief Calculate the total potential energy of the system.
     * 
     * @return scalar_t Total potential energy
     */
    scalar_t total_potential_energy() const;

    /**
     * @brief Calculate the total energy of the system (kinetic + potential).
     * 
     * @return scalar_t Total energy
     */
    scalar_t total_energy() const;

    /**
     * @brief Create a copy of this particle system.
     * 
     * @return std::unique_ptr<ParticleSystem> New particle system
     */
    std::unique_ptr<ParticleSystem> clone() const;

    /**
     * @brief Create a random particle system.
     * 
     * @param num_particles Number of particles
     * @param box_size Size of the box for random distribution
     * @param max_mass Maximum particle mass
     * @param max_velocity Maximum initial velocity
     * @param G Gravitational constant
     * @param seed Random seed for reproducibility
     * @return std::unique_ptr<ParticleSystem> New particle system
     */
    static std::unique_ptr<ParticleSystem> create_random_system(
        size_t num_particles,
        scalar_t box_size = 10.0,
        scalar_t max_mass = 1.0,
        scalar_t max_velocity = 0.1,
        scalar_t G = DEFAULT_G,
        unsigned int seed = 0
    );

    /**
     * @brief Create a solar system simulation.
     * 
     * @param scale_factor Factor to scale distances and velocities
     * @param G Gravitational constant
     * @return std::unique_ptr<ParticleSystem> New particle system
     */
    static std::unique_ptr<ParticleSystem> create_solar_system(
        scalar_t scale_factor = 1.0,
        scalar_t G = 4.0 * M_PI * M_PI  // In AU^3 / (year^2 * solar_mass)
    );

    /**
     * @brief Create a galaxy model.
     * 
     * @param num_particles Number of particles
     * @param radius Galaxy radius
     * @param height Galaxy height/thickness
     * @param min_mass Minimum particle mass
     * @param max_mass Maximum particle mass
     * @param G Gravitational constant
     * @param seed Random seed for reproducibility
     * @return std::unique_ptr<ParticleSystem> New particle system
     */
    static std::unique_ptr<ParticleSystem> create_galaxy_model(
        size_t num_particles = 1000,
        scalar_t radius = 10.0,
        scalar_t height = 1.0,
        scalar_t min_mass = 0.1,
        scalar_t max_mass = 1.0,
        scalar_t G = DEFAULT_G,
        unsigned int seed = 0
    );

private:
    std::vector<Particle> particles_;  // Vector of particles
    scalar_t G_;                       // Gravitational constant

    // CUDA specific members
    Vec3* d_positions_ = nullptr;      // Device memory for positions
    Vec3* d_velocities_ = nullptr;     // Device memory for velocities
    Vec3* d_accelerations_ = nullptr;  // Device memory for accelerations
    scalar_t* d_masses_ = nullptr;     // Device memory for masses
    bool gpu_initialized_ = false;     // Flag to track GPU memory initialization
    
    // Initialize GPU memory for particles
    void initialize_gpu_memory();
    
    // Free GPU memory
    void free_gpu_memory();
    
    // Copy data from host to device
    void copy_to_device();
    
    // Copy data from device to host
    void copy_from_device();
};

} // namespace nbody_sim