// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "nbody_sim/particle.hpp"
#include "nbody_sim/integrator.hpp"
#include <memory>
#include <functional>
#include <chrono>
#include <string>
#include <map>

namespace nbody_sim {

/**
 * @brief Class for running N-body simulations.
 */
class Simulation {
public:
    /**
     * @brief Callback function type for step events.
     */
    using StepCallback = std::function<void(const Simulation&)>;

    /**
     * @brief Construct a new Simulation with default parameters.
     */
    Simulation();

    /**
     * @brief Construct a new Simulation with specified parameters.
     * 
     * @param system Particle system
     * @param integrator Integration method
     * @param dt Time step size
     * @param duration Total simulation time
     */
    Simulation(
        std::unique_ptr<ParticleSystem> system,
        std::unique_ptr<Integrator> integrator,
        scalar_t dt = DEFAULT_TIMESTEP,
        scalar_t duration = 10.0
    );

    /**
     * @brief Get the current simulation time.
     * 
     * @return scalar_t Current time
     */
    scalar_t current_time() const { return current_time_; }

    /**
     * @brief Get the current simulation step.
     * 
     * @return index_t Current step
     */
    index_t current_step() const { return current_step_; }

    /**
     * @brief Get the total number of steps.
     * 
     * @return index_t Total steps
     */
    index_t total_steps() const { return total_steps_; }

    /**
     * @brief Get the time step size.
     * 
     * @return scalar_t Time step
     */
    scalar_t dt() const { return dt_; }

    /**
     * @brief Get the total simulation duration.
     * 
     * @return scalar_t Duration
     */
    scalar_t duration() const { return duration_; }

    /**
     * @brief Get the particle system.
     * 
     * @return const ParticleSystem& Particle system
     */
    const ParticleSystem& system() const { return *system_; }

    /**
     * @brief Get the particle system (mutable).
     * 
     * @return ParticleSystem& Particle system
     */
    ParticleSystem& system() { return *system_; }

    /**
     * @brief Get the integrator.
     * 
     * @return const Integrator& Integrator
     */
    const Integrator& integrator() const { return *integrator_; }

    /**
     * @brief Advance the simulation by one time step.
     */
    void step();

    /**
     * @brief Run the simulation for the specified duration.
     * 
     * @param callback Optional callback function to call after each step
     * @param callback_interval How often to call the callback (in steps)
     */
    void run(
        StepCallback callback = nullptr,
        index_t callback_interval = 1
    );

    /**
     * @brief Save the current simulation state to a file.
     * 
     * @param filename Path to the output file
     */
    void save_state(const std::string& filename) const;

    /**
     * @brief Load a simulation state from a file.
     * 
     * @param filename Path to the input file
     */
    void load_state(const std::string& filename);

    /**
     * @brief Get performance metrics from the simulation.
     * 
     * @return std::map<std::string, double> Map of metrics
     */
    std::map<std::string, double> get_performance_metrics() const;

    /**
     * @brief Create visualization data for the current state.
     * 
     * @param include_velocities Whether to include velocity data
     * @return std::map<std::string, std::vector<double>> Map of data arrays
     */
    std::map<std::string, std::vector<double>> create_visualization_data(
        bool include_velocities = true
    ) const;

    /**
     * @brief Create a simulation with a random particle system.
     * 
     * @param num_particles Number of particles
     * @param integration_type Integration method
     * @param dt Time step size
     * @param duration Total simulation time
     * @param seed Random seed
     * @return std::unique_ptr<Simulation> New simulation
     */
    static std::unique_ptr<Simulation> create_random_simulation(
        size_t num_particles,
        IntegrationType integration_type = IntegrationType::Leapfrog,
        scalar_t dt = DEFAULT_TIMESTEP,
        scalar_t duration = 10.0,
        unsigned int seed = 0
    );

    /**
     * @brief Create a simulation of the solar system.
     * 
     * @param integration_type Integration method
     * @param dt Time step size
     * @param duration Total simulation time
     * @param scale_factor Scale factor for distances and velocities
     * @return std::unique_ptr<Simulation> New simulation
     */
    static std::unique_ptr<Simulation> create_solar_system_simulation(
        IntegrationType integration_type = IntegrationType::Leapfrog,
        scalar_t dt = DEFAULT_TIMESTEP,
        scalar_t duration = 10.0,
        scalar_t scale_factor = 1.0
    );

    /**
     * @brief Create a simulation of a galaxy.
     * 
     * @param num_particles Number of particles
     * @param integration_type Integration method
     * @param dt Time step size
     * @param duration Total simulation time
     * @param seed Random seed
     * @return std::unique_ptr<Simulation> New simulation
     */
    static std::unique_ptr<Simulation> create_galaxy_simulation(
        size_t num_particles,
        IntegrationType integration_type = IntegrationType::Leapfrog,
        scalar_t dt = DEFAULT_TIMESTEP,
        scalar_t duration = 10.0,
        unsigned int seed = 0
    );

private:
    std::unique_ptr<ParticleSystem> system_;   // Particle system
    std::unique_ptr<Integrator> integrator_;   // Integration method
    scalar_t dt_;                              // Time step size
    scalar_t duration_;                        // Total simulation time
    scalar_t current_time_;                    // Current simulation time
    index_t current_step_;                     // Current step
    index_t total_steps_;                      // Total number of steps

    // Performance tracking
    std::chrono::steady_clock::time_point start_time_;  // Simulation start time
    std::chrono::steady_clock::time_point end_time_;    // Simulation end time
    scalar_t initial_energy_;                           // Initial total energy
    scalar_t final_energy_;                             // Final total energy
};

} // namespace nbody_sim