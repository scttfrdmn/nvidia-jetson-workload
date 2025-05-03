// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "molecular_dynamics/simulation.hpp"
#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <memory>
#include <fstream>

using namespace molecular_dynamics;

// Helper function to print performance information
void print_performance(const std::map<std::string, double>& metrics) {
    std::cout << "========== Performance Metrics ==========" << std::endl;
    std::cout << "Simulation time: " << metrics.at("total_time_ms") << " ms" << std::endl;
    std::cout << "Steps per second: " << metrics.at("steps_per_second") << std::endl;
    std::cout << "Particle count: " << metrics.at("particle_count") << std::endl;
    std::cout << "Initial energy: " << metrics.at("initial_energy") << " kJ/mol" << std::endl;
    std::cout << "Final energy: " << metrics.at("final_energy") << " kJ/mol" << std::endl;
    std::cout << "Energy conservation error: " << metrics.at("energy_conservation_error") * 100.0 << "%" << std::endl;
    std::cout << "=======================================" << std::endl;
}

// Simple progress reporter
void progress_callback(const Simulation& sim) {
    static int last_percent = -1;
    
    double percent = 100.0 * sim.current_step() / sim.total_steps();
    int int_percent = static_cast<int>(percent);
    
    if (int_percent > last_percent) {
        std::cout << "\rProgress: " << int_percent << "% (" 
                  << sim.current_step() << "/" << sim.total_steps() << " steps)" << std::flush;
        last_percent = int_percent;
    }
}

int main(int argc, char** argv) {
    std::cout << "Molecular Dynamics Simulation" << std::endl;
    std::cout << "Copyright 2025 Scott Friedman and Project Contributors" << std::endl;
    std::cout << std::endl;
    
    try {
        // Detect device capabilities
        DeviceCapabilities caps = detect_device_capabilities();
        
        // Create a Lennard-Jones fluid simulation
        size_t num_particles = 10000; // Default value
        scalar_t box_size = 50.0;     // Default value
        scalar_t temperature = DEFAULT_TEMPERATURE; // Default value
        
        // Parse command line arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            if (arg == "--particles" && i + 1 < argc) {
                num_particles = std::stoi(argv[++i]);
            } else if (arg == "--box-size" && i + 1 < argc) {
                box_size = std::stod(argv[++i]);
            } else if (arg == "--temperature" && i + 1 < argc) {
                temperature = std::stod(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --particles N       Number of particles (default: 10000)" << std::endl;
                std::cout << "  --box-size S        Size of the simulation box in Angstroms (default: 50.0)" << std::endl;
                std::cout << "  --temperature T     Temperature in Kelvin (default: 300.0)" << std::endl;
                std::cout << "  --help, -h          Show this help message" << std::endl;
                return 0;
            }
        }
        
        std::cout << "Setting up simulation with:" << std::endl;
        std::cout << "  Particles: " << num_particles << std::endl;
        std::cout << "  Box size: " << box_size << " Angstroms" << std::endl;
        std::cout << "  Temperature: " << temperature << " K" << std::endl;
        std::cout << std::endl;
        
        // Create the simulation
        auto sim = Simulation::create_lj_fluid_simulation(
            num_particles,
            box_size,
            IntegrationType::VelocityVerlet,
            ThermostatType::Berendsen,
            temperature,
            0.001, // 1 fs timestep
            10.0   // 10 ps simulation time
        );
        
        // Run the simulation with progress reporting
        std::cout << "Running simulation..." << std::endl;
        sim->run(progress_callback, 100); // Report progress every 100 steps
        std::cout << std::endl << "Simulation complete!" << std::endl;
        
        // Get and print performance metrics
        auto metrics = sim->get_performance_metrics();
        print_performance(metrics);
        
        // Save final state
        std::cout << "Saving trajectory to trajectory.dcd..." << std::endl;
        sim->save_trajectory("trajectory.dcd");
        
        std::cout << "Done!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}