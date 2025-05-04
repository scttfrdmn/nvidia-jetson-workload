// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "nbody_sim/particle.hpp"
#include "nbody_sim/integrator.hpp"
#include "nbody_sim/simulation.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <memory>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <map>

using namespace nbody_sim;

// Print usage information
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --system TYPE         System type: random, solar, galaxy (default: galaxy)\n";
    std::cout << "  --particles N         Number of particles (default: 5000)\n";
    std::cout << "  --integrator TYPE     Integrator: euler, leapfrog, verlet, rk4 (default: leapfrog)\n";
    std::cout << "  --dt STEP             Time step size (default: 0.01)\n";
    std::cout << "  --duration TIME       Total simulation time (default: 5.0)\n";
    std::cout << "  --seed SEED           Random seed for reproducibility\n";
    std::cout << "  --output-dir DIR      Output directory (default: ./output)\n";
    std::cout << "  --save-visualization  Save visualization data\n";
    std::cout << "  --quiet               Disable progress display\n";
    std::cout << "  --help                Display this help message\n";
}

// Parse command-line arguments
std::map<std::string, std::string> parse_args(int argc, char* argv[]) {
    std::map<std::string, std::string> args;
    
    // Set default values
    args["system"] = "galaxy";
    args["particles"] = "5000";
    args["integrator"] = "leapfrog";
    args["dt"] = "0.01";
    args["duration"] = "5.0";
    args["output-dir"] = "./output";
    args["save-visualization"] = "false";
    args["quiet"] = "false";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--save-visualization") {
            args["save-visualization"] = "true";
        } else if (arg == "--quiet") {
            args["quiet"] = "true";
        } else if (i + 1 < argc) {
            if (arg == "--system") {
                args["system"] = argv[++i];
            } else if (arg == "--particles") {
                args["particles"] = argv[++i];
            } else if (arg == "--integrator") {
                args["integrator"] = argv[++i];
            } else if (arg == "--dt") {
                args["dt"] = argv[++i];
            } else if (arg == "--duration") {
                args["duration"] = argv[++i];
            } else if (arg == "--seed") {
                args["seed"] = argv[++i];
            } else if (arg == "--output-dir") {
                args["output-dir"] = argv[++i];
            } else {
                std::cerr << "Unknown option: " << arg << "\n";
                print_usage(argv[0]);
                std::exit(1);
            }
        } else {
            std::cerr << "Missing argument for option: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    
    return args;
}

// Progress callback function
void display_progress(const Simulation& simulation) {
    const scalar_t progress = 100.0 * simulation.current_step() / simulation.total_steps();
    
    std::cout << "\rProgress: " << std::fixed << std::setprecision(1) << progress << "% ";
    std::cout << "(Step " << simulation.current_step() << "/" << simulation.total_steps() << ")";
    std::cout << std::flush;
}

// Save simulation results to files
void save_results(const Simulation& simulation, const std::string& output_dir, bool save_visualization) {
    // Create output directory if it doesn't exist
    std::string command = "mkdir -p " + output_dir;
    std::system(command.c_str());
    
    // Save final state
    std::string state_file = output_dir + "/final_state.bin";
    simulation.save_state(state_file);
    std::cout << "Final state saved to " << state_file << std::endl;
    
    // Save performance metrics
    std::string metrics_file = output_dir + "/metrics.json";
    std::ofstream metrics_out(metrics_file);
    if (metrics_out) {
        const auto metrics = simulation.get_performance_metrics();
        
        metrics_out << "{\n";
        for (auto it = metrics.begin(); it != metrics.end(); ++it) {
            metrics_out << "  \"" << it->first << "\": " << it->second;
            if (std::next(it) != metrics.end()) {
                metrics_out << ",";
            }
            metrics_out << "\n";
        }
        metrics_out << "}\n";
        
        std::cout << "Performance metrics saved to " << metrics_file << std::endl;
    }
    
    // Save visualization data if requested
    if (save_visualization) {
        std::string vis_file = output_dir + "/visualization.json";
        std::ofstream vis_out(vis_file);
        if (vis_out) {
            const auto vis_data = simulation.create_visualization_data();
            
            vis_out << "{\n";
            for (auto it = vis_data.begin(); it != vis_data.end(); ++it) {
                vis_out << "  \"" << it->first << "\": [";
                
                const auto& values = it->second;
                for (size_t i = 0; i < values.size(); ++i) {
                    if (i > 0) {
                        vis_out << ", ";
                    }
                    vis_out << values[i];
                }
                
                vis_out << "]";
                if (std::next(it) != vis_data.end()) {
                    vis_out << ",";
                }
                vis_out << "\n";
            }
            vis_out << "}\n";
            
            std::cout << "Visualization data saved to " << vis_file << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    auto args = parse_args(argc, argv);
    
    // Get parameter values
    const std::string system_type = args["system"];
    const size_t num_particles = std::stoul(args["particles"]);
    const std::string integrator_type = args["integrator"];
    const scalar_t dt = std::stod(args["dt"]);
    const scalar_t duration = std::stod(args["duration"]);
    const bool quiet = args["quiet"] == "true";
    const bool save_vis = args["save-visualization"] == "true";
    const std::string output_dir = args["output-dir"];
    
    // Convert integrator type string to enum
    IntegrationType integration_type;
    if (integrator_type == "euler") {
        integration_type = IntegrationType::Euler;
    } else if (integrator_type == "leapfrog") {
        integration_type = IntegrationType::Leapfrog;
    } else if (integrator_type == "verlet") {
        integration_type = IntegrationType::Verlet;
    } else if (integrator_type == "rk4") {
        integration_type = IntegrationType::RungeKutta4;
    } else {
        std::cerr << "Unknown integrator type: " << integrator_type << "\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Get random seed
    unsigned int seed = 0;
    if (args.find("seed") != args.end()) {
        seed = std::stoul(args["seed"]);
    } else {
        // Use current time as seed if not specified
        seed = static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    
    // Create simulation based on system type
    std::unique_ptr<Simulation> simulation;
    
    if (system_type == "random") {
        simulation = Simulation::create_random_simulation(
            num_particles,
            integration_type,
            dt,
            duration,
            seed
        );
    } else if (system_type == "solar") {
        simulation = Simulation::create_solar_system_simulation(
            integration_type,
            dt,
            duration
        );
    } else if (system_type == "galaxy") {
        simulation = Simulation::create_galaxy_simulation(
            num_particles,
            integration_type,
            dt,
            duration,
            seed
        );
    } else {
        std::cerr << "Unknown system type: " << system_type << "\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Display simulation configuration
    std::cout << "N-body Simulation Configuration:\n";
    std::cout << "  System Type: " << system_type << "\n";
    std::cout << "  Particles: " << num_particles << "\n";
    std::cout << "  Integrator: " << integrator_type << "\n";
    std::cout << "  Time Step: " << dt << "\n";
    std::cout << "  Duration: " << duration << "\n";
    std::cout << "  Total Steps: " << simulation->total_steps() << "\n";
    std::cout << "  Random Seed: " << seed << "\n";
    std::cout << std::endl;
    
    // Run simulation
    std::cout << "Starting simulation..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    
    if (quiet) {
        simulation->run();
    } else {
        simulation->run(display_progress, simulation->total_steps() / 100);
        std::cout << std::endl;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Simulation completed in " << elapsed.count() / 1000.0 << " seconds" << std::endl;
    
    // Save results
    save_results(*simulation, output_dir, save_vis);
    
    // Print performance summary
    const auto metrics = simulation->get_performance_metrics();
    
    std::cout << "\nPerformance Summary:" << std::endl;
    std::cout << "  Execution Time: " << metrics.at("total_time_ms") / 1000.0 << " seconds" << std::endl;
    std::cout << "  Steps per Second: " << metrics.at("steps_per_second") << std::endl;
    std::cout << "  Energy Conservation Error: " << std::scientific << metrics.at("energy_conservation_error") << std::endl;
    
    return 0;
}