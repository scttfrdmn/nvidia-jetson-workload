// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "molecular_dynamics/molecular_system.hpp"
#include "molecular_dynamics/integrator.hpp"
#include <memory>
#include <functional>
#include <chrono>
#include <string>
#include <map>
#include <vector>

namespace molecular_dynamics {

/**
 * @brief Class for running molecular dynamics simulations.
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
     * @param system Molecular system
     * @param integrator Integration method
     * @param dt Time step in picoseconds
     * @param duration Total simulation time in picoseconds
     */
    Simulation(
        std::unique_ptr<MolecularSystem> system,
        std::unique_ptr<Integrator> integrator = nullptr,
        std::unique_ptr<Thermostat> thermostat = nullptr,
        scalar_t dt = DEFAULT_TIMESTEP,
        scalar_t duration = 10.0
    );

    /**
     * @brief Get the current simulation time.
     * 
     * @return scalar_t Current time in picoseconds
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
     * @return scalar_t Time step in picoseconds
     */
    scalar_t dt() const { return dt_; }

    /**
     * @brief Get the total simulation duration.
     * 
     * @return scalar_t Duration in picoseconds
     */
    scalar_t duration() const { return duration_; }

    /**
     * @brief Get the target temperature.
     * 
     * @return scalar_t Temperature in Kelvin
     */
    scalar_t temperature() const { return temperature_; }

    /**
     * @brief Set the target temperature.
     * 
     * @param temperature Temperature in Kelvin
     */
    void set_temperature(scalar_t temperature) { temperature_ = temperature; }

    /**
     * @brief Get the molecular system.
     * 
     * @return const MolecularSystem& Molecular system
     */
    const MolecularSystem& system() const { return *system_; }

    /**
     * @brief Get the molecular system (mutable).
     * 
     * @return MolecularSystem& Mutable molecular system
     */
    MolecularSystem& system() { return *system_; }

    /**
     * @brief Get the integrator.
     * 
     * @return const Integrator& Integrator
     */
    const Integrator& integrator() const { return *integrator_; }

    /**
     * @brief Get the thermostat.
     * 
     * @return const Thermostat& Thermostat (may be null)
     */
    const Thermostat* thermostat() const { return thermostat_.get(); }

    /**
     * @brief Set the thermostat.
     * 
     * @param thermostat New thermostat
     */
    void set_thermostat(std::unique_ptr<Thermostat> thermostat);

    /**
     * @brief Get the current device capabilities.
     * 
     * @return const DeviceCapabilities& Device capabilities
     */
    const DeviceCapabilities& device_capabilities() const { return device_capabilities_; }

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
        index_t callback_interval = 100
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
     * @brief Save trajectory to a DCD file.
     * 
     * @param filename Path to the output DCD file
     */
    void save_trajectory(const std::string& filename) const;

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
     * @param include_forces Whether to include force data
     * @return std::map<std::string, std::vector<double>> Map of data arrays
     */
    std::map<std::string, std::vector<double>> create_visualization_data(
        bool include_velocities = true,
        bool include_forces = false
    ) const;

    /**
     * @brief Create a water box simulation.
     * 
     * @param box_size Size of the cubic box in Angstroms
     * @param integration_type Integration method
     * @param thermostat_type Thermostat type
     * @param temperature Temperature in Kelvin
     * @param dt Time step in picoseconds
     * @param duration Total simulation time in picoseconds
     * @return std::unique_ptr<Simulation> New simulation
     */
    static std::unique_ptr<Simulation> create_water_box_simulation(
        scalar_t box_size,
        IntegrationType integration_type = IntegrationType::VelocityVerlet,
        ThermostatType thermostat_type = ThermostatType::Berendsen,
        scalar_t temperature = DEFAULT_TEMPERATURE,
        scalar_t dt = DEFAULT_TIMESTEP,
        scalar_t duration = 10.0
    );

    /**
     * @brief Create a Lennard-Jones fluid simulation.
     * 
     * @param num_particles Number of particles
     * @param box_size Size of the cubic box in Angstroms
     * @param integration_type Integration method
     * @param thermostat_type Thermostat type
     * @param temperature Temperature in Kelvin
     * @param dt Time step in picoseconds
     * @param duration Total simulation time in picoseconds
     * @param seed Random seed
     * @return std::unique_ptr<Simulation> New simulation
     */
    static std::unique_ptr<Simulation> create_lj_fluid_simulation(
        size_t num_particles,
        scalar_t box_size,
        IntegrationType integration_type = IntegrationType::VelocityVerlet,
        ThermostatType thermostat_type = ThermostatType::Berendsen,
        scalar_t temperature = DEFAULT_TEMPERATURE,
        scalar_t dt = DEFAULT_TIMESTEP,
        scalar_t duration = 10.0,
        unsigned int seed = 0
    );

    /**
     * @brief Create a simulation from a PDB file.
     * 
     * @param pdb_filename Path to PDB file
     * @param integration_type Integration method
     * @param thermostat_type Thermostat type
     * @param temperature Temperature in Kelvin
     * @param dt Time step in picoseconds
     * @param duration Total simulation time in picoseconds
     * @return std::unique_ptr<Simulation> New simulation
     */
    static std::unique_ptr<Simulation> create_from_pdb(
        const std::string& pdb_filename,
        IntegrationType integration_type = IntegrationType::VelocityVerlet,
        ThermostatType thermostat_type = ThermostatType::Berendsen,
        scalar_t temperature = DEFAULT_TEMPERATURE,
        scalar_t dt = DEFAULT_TIMESTEP,
        scalar_t duration = 10.0
    );

private:
    std::unique_ptr<MolecularSystem> system_;     // Molecular system
    std::unique_ptr<Integrator> integrator_;      // Integration method
    std::unique_ptr<Thermostat> thermostat_;      // Thermostat (optional)
    scalar_t dt_;                                 // Time step size in picoseconds
    scalar_t duration_;                           // Total simulation time in picoseconds
    scalar_t current_time_;                       // Current simulation time in picoseconds
    index_t current_step_;                        // Current step
    index_t total_steps_;                         // Total number of steps
    scalar_t temperature_;                        // Target temperature in Kelvin
    DeviceCapabilities device_capabilities_;      // Device capabilities for scaling

    // Trajectory storage
    bool store_trajectory_;
    std::vector<std::vector<Vec3>> trajectory_positions_;
    std::vector<std::vector<Vec3>> trajectory_velocities_;
    
    // Performance tracking
    std::chrono::steady_clock::time_point start_time_;  // Simulation start time
    std::chrono::steady_clock::time_point end_time_;    // Simulation end time
    scalar_t initial_energy_;                           // Initial total energy
    scalar_t final_energy_;                             // Final total energy
    
    // Initialize the default integrator if none is provided
    void initialize_default_integrator();
    
    // Initialize CUDA device if available
    void initialize_cuda_device();
    
    // Write a frame to the trajectory
    void write_trajectory_frame();
};

} // namespace molecular_dynamics