/**
 * @file weather_sim_example.cpp
 * @brief Example usage of the Weather Simulation workload.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <weather_sim/weather_sim.hpp>
#include <weather_sim/initial_conditions.hpp>
#include <weather_sim/gpu_adaptability.hpp>
#include <weather_sim/output_manager.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>

// Simple class for handling command line arguments
class CommandLineArgs {
public:
    CommandLineArgs(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            args_.push_back(std::string(argv[i]));
        }
    }
    
    bool hasOption(const std::string& option) const {
        return std::find(args_.begin(), args_.end(), option) != args_.end();
    }
    
    std::string getOption(const std::string& option, const std::string& defaultValue = "") const {
        auto it = std::find(args_.begin(), args_.end(), option);
        if (it != args_.end() && ++it != args_.end()) {
            return *it;
        }
        return defaultValue;
    }
    
private:
    std::vector<std::string> args_;
};

void printAvailableModels() {
    std::cout << "Available simulation models:" << std::endl;
    std::cout << "  shallow_water     - Shallow water equations" << std::endl;
    std::cout << "  barotropic        - Barotropic vorticity equation" << std::endl;
    std::cout << "  primitive         - Primitive equations" << std::endl;
    std::cout << "  general           - General circulation model" << std::endl;
}

void printAvailableInitialConditions() {
    auto& factory = weather_sim::InitialConditionFactory::getInstance();
    auto available = factory.getAvailableInitialConditions();
    
    std::cout << "Available initial conditions:" << std::endl;
    for (const auto& name : available) {
        std::cout << "  " << name << std::endl;
    }
}

void printAvailableIntegrationMethods() {
    std::cout << "Available integration methods:" << std::endl;
    std::cout << "  euler             - Explicit Euler method" << std::endl;
    std::cout << "  rk2               - 2nd order Runge-Kutta method" << std::endl;
    std::cout << "  rk4               - 4th order Runge-Kutta method" << std::endl;
    std::cout << "  adams_bashforth   - Adams-Bashforth method" << std::endl;
    std::cout << "  semi_implicit     - Semi-implicit method" << std::endl;
}

void printAvailableBackends() {
    std::cout << "Available compute backends:" << std::endl;
    std::cout << "  cuda              - CUDA GPU backend" << std::endl;
    std::cout << "  cpu               - CPU backend" << std::endl;
    std::cout << "  hybrid            - Hybrid CPU-GPU backend" << std::endl;
    std::cout << "  adaptive          - Adaptive hybrid backend" << std::endl;
}

void printBenchmarkResults(
    const std::string& name,
    const weather_sim::SimulationConfig& config,
    const weather_sim::PerformanceMetrics& metrics
) {
    std::cout << "===============================================" << std::endl;
    std::cout << "Benchmark Results: " << name << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "Grid Size: " << config.grid_width << "x" << config.grid_height << std::endl;
    std::cout << "Time Steps: " << metrics.num_steps << std::endl;
    std::cout << "Total Time: " << metrics.total_time_ms << " ms" << std::endl;
    std::cout << "Compute Time: " << metrics.compute_time_ms << " ms";
    if (metrics.total_time_ms > 0) {
        std::cout << " (" << (metrics.compute_time_ms / metrics.total_time_ms * 100.0) << "%)";
    }
    std::cout << std::endl;
    
    std::cout << "Memory Transfer Time: " << metrics.memory_transfer_time_ms << " ms";
    if (metrics.total_time_ms > 0) {
        std::cout << " (" << (metrics.memory_transfer_time_ms / metrics.total_time_ms * 100.0) << "%)";
    }
    std::cout << std::endl;
    
    std::cout << "I/O Time: " << metrics.io_time_ms << " ms";
    if (metrics.total_time_ms > 0) {
        std::cout << " (" << (metrics.io_time_ms / metrics.total_time_ms * 100.0) << "%)";
    }
    std::cout << std::endl;
    
    if (metrics.num_steps > 0) {
        std::cout << "Time Per Step: " << (metrics.total_time_ms / metrics.num_steps) << " ms" << std::endl;
        double cell_updates_per_step = config.grid_width * config.grid_height;
        double total_cell_updates = cell_updates_per_step * metrics.num_steps;
        double mcups = total_cell_updates / (metrics.compute_time_ms * 1000.0); // Million Cell Updates Per Second
        std::cout << "Performance: " << std::fixed << std::setprecision(2) << mcups << " MCUPS" << std::endl;
    }
    std::cout << "===============================================" << std::endl;
}

weather_sim::SimulationModel getModelFromString(const std::string& model_str) {
    if (model_str == "shallow_water") {
        return weather_sim::SimulationModel::ShallowWater;
    } else if (model_str == "barotropic") {
        return weather_sim::SimulationModel::Barotropic;
    } else if (model_str == "primitive") {
        return weather_sim::SimulationModel::PrimitiveEquations;
    } else if (model_str == "general") {
        return weather_sim::SimulationModel::General;
    } else {
        // Default to shallow water
        std::cout << "Warning: Unknown model '" << model_str << "', using shallow water model" << std::endl;
        return weather_sim::SimulationModel::ShallowWater;
    }
}

weather_sim::IntegrationMethod getIntegrationMethodFromString(const std::string& method_str) {
    if (method_str == "euler") {
        return weather_sim::IntegrationMethod::ExplicitEuler;
    } else if (method_str == "rk2") {
        return weather_sim::IntegrationMethod::RungeKutta2;
    } else if (method_str == "rk4") {
        return weather_sim::IntegrationMethod::RungeKutta4;
    } else if (method_str == "adams_bashforth") {
        return weather_sim::IntegrationMethod::AdamsBashforth;
    } else if (method_str == "semi_implicit") {
        return weather_sim::IntegrationMethod::SemiImplicit;
    } else {
        // Default to Euler
        std::cout << "Warning: Unknown integration method '" << method_str << "', using explicit Euler" << std::endl;
        return weather_sim::IntegrationMethod::ExplicitEuler;
    }
}

weather_sim::ComputeBackend getBackendFromString(const std::string& backend_str) {
    if (backend_str == "cuda") {
        return weather_sim::ComputeBackend::CUDA;
    } else if (backend_str == "cpu") {
        return weather_sim::ComputeBackend::CPU;
    } else if (backend_str == "hybrid") {
        return weather_sim::ComputeBackend::Hybrid;
    } else if (backend_str == "adaptive") {
        return weather_sim::ComputeBackend::AdaptiveHybrid;
    } else {
        // Check if CUDA is available, otherwise default to CPU
        if (weather_sim::AdaptiveKernelManager::getInstance().isCudaAvailable()) {
            std::cout << "Warning: Unknown backend '" << backend_str << "', using CUDA" << std::endl;
            return weather_sim::ComputeBackend::CUDA;
        } else {
            std::cout << "Warning: Unknown backend '" << backend_str << "', using CPU" << std::endl;
            return weather_sim::ComputeBackend::CPU;
        }
    }
}

weather_sim::OutputFormat getOutputFormatFromString(const std::string& format_str) {
    if (format_str == "csv") {
        return weather_sim::OutputFormat::CSV;
    } else if (format_str == "netcdf") {
        return weather_sim::OutputFormat::NetCDF;
    } else if (format_str == "vtk") {
        return weather_sim::OutputFormat::VTK;
    } else if (format_str == "png") {
        return weather_sim::OutputFormat::PNG;
    } else {
        // Default to CSV
        std::cout << "Warning: Unknown output format '" << format_str << "', using CSV" << std::endl;
        return weather_sim::OutputFormat::CSV;
    }
}

void runBenchmark(
    const weather_sim::SimulationConfig& config,
    const std::string& initial_condition_name,
    int num_steps,
    bool output_enabled
) {
    // Create initial condition
    auto& factory = weather_sim::InitialConditionFactory::getInstance();
    auto initial_condition = factory.createInitialCondition(initial_condition_name);
    
    if (!initial_condition) {
        std::cout << "Error: Invalid initial condition '" << initial_condition_name << "'" << std::endl;
        printAvailableInitialConditions();
        return;
    }
    
    // Create simulation
    weather_sim::WeatherSimulation simulation(config);
    simulation.setInitialCondition(initial_condition);
    
    // Set up output manager if enabled
    std::shared_ptr<weather_sim::OutputManager> output_manager;
    if (output_enabled) {
        weather_sim::OutputConfig output_config;
        output_config.output_dir = "./output";
        output_config.prefix = "weather_sim";
        output_config.format = weather_sim::OutputFormat::CSV;
        output_config.output_interval = std::max(1, num_steps / 10); // Output ~10 frames
        
        // Create output manager
        auto& output_factory = weather_sim::OutputManagerFactory::getInstance();
        output_manager = output_factory.createOutputManager(output_config.format, output_config);
        
        if (output_manager) {
            simulation.setOutputManager(output_manager);
        } else {
            std::cout << "Warning: Failed to create output manager, output disabled" << std::endl;
        }
    }
    
    // Initialize simulation
    simulation.initialize();
    
    // Print simulation info
    std::cout << "Running simulation with:" << std::endl;
    std::cout << "  Model: ";
    switch (config.model) {
        case weather_sim::SimulationModel::ShallowWater:
            std::cout << "Shallow Water Equations";
            break;
        case weather_sim::SimulationModel::Barotropic:
            std::cout << "Barotropic Vorticity Equation";
            break;
        case weather_sim::SimulationModel::PrimitiveEquations:
            std::cout << "Primitive Equations";
            break;
        case weather_sim::SimulationModel::General:
            std::cout << "General Circulation Model";
            break;
    }
    std::cout << std::endl;
    
    std::cout << "  Grid size: " << config.grid_width << " x " << config.grid_height << std::endl;
    std::cout << "  Time step: " << config.dt << std::endl;
    std::cout << "  Steps: " << num_steps << std::endl;
    std::cout << "  Backend: ";
    switch (config.compute_backend) {
        case weather_sim::ComputeBackend::CUDA:
            std::cout << "CUDA GPU";
            break;
        case weather_sim::ComputeBackend::CPU:
            std::cout << "CPU";
            break;
        case weather_sim::ComputeBackend::Hybrid:
            std::cout << "Hybrid CPU-GPU";
            break;
        case weather_sim::ComputeBackend::AdaptiveHybrid:
            std::cout << "Adaptive Hybrid";
            break;
    }
    std::cout << std::endl;
    
    std::cout << "  Initial condition: " << initial_condition->getName() << std::endl;
    
    // Run simulation
    auto start_time = std::chrono::high_resolution_clock::now();
    
    simulation.run(num_steps);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Print benchmark results
    printBenchmarkResults("Weather Simulation", config, simulation.getPerformanceMetrics());
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help                   Show this help message" << std::endl;
    std::cout << "  --list-models            List available simulation models" << std::endl;
    std::cout << "  --list-initial           List available initial conditions" << std::endl;
    std::cout << "  --list-integration       List available integration methods" << std::endl;
    std::cout << "  --list-backends          List available compute backends" << std::endl;
    std::cout << "  --model MODEL            Set simulation model (default: shallow_water)" << std::endl;
    std::cout << "  --width WIDTH            Set grid width (default: 512)" << std::endl;
    std::cout << "  --height HEIGHT          Set grid height (default: 512)" << std::endl;
    std::cout << "  --dt DT                  Set time step (default: 0.01)" << std::endl;
    std::cout << "  --steps STEPS            Set number of time steps (default: 100)" << std::endl;
    std::cout << "  --initial INITIAL        Set initial condition (default: jet_stream)" << std::endl;
    std::cout << "  --integration METHOD     Set integration method (default: rk4)" << std::endl;
    std::cout << "  --backend BACKEND        Set compute backend (default: adaptive)" << std::endl;
    std::cout << "  --device DEVICE          Set GPU device ID (default: 0)" << std::endl;
    std::cout << "  --threads THREADS        Set number of CPU threads (default: auto)" << std::endl;
    std::cout << "  --output-format FORMAT   Set output format (default: csv)" << std::endl;
    std::cout << "  --output-dir DIR         Set output directory (default: ./output)" << std::endl;
    std::cout << "  --disable-output         Disable output generation" << std::endl;
}

int main(int argc, char** argv) {
    CommandLineArgs args(argc, argv);
    
    // Register all initial conditions
    weather_sim::registerAllInitialConditions();
    
    // Register all output managers
    // weather_sim::registerAllOutputManagers();
    
    // Initialize CUDA
    weather_sim::AdaptiveKernelManager::getInstance().initialize();
    
    // Check for help and list options
    if (args.hasOption("--help")) {
        printUsage(argv[0]);
        return 0;
    }
    
    if (args.hasOption("--list-models")) {
        printAvailableModels();
        return 0;
    }
    
    if (args.hasOption("--list-initial")) {
        printAvailableInitialConditions();
        return 0;
    }
    
    if (args.hasOption("--list-integration")) {
        printAvailableIntegrationMethods();
        return 0;
    }
    
    if (args.hasOption("--list-backends")) {
        printAvailableBackends();
        return 0;
    }
    
    // Parse command line arguments
    std::string model_str = args.getOption("--model", "shallow_water");
    int width = std::stoi(args.getOption("--width", "512"));
    int height = std::stoi(args.getOption("--height", "512"));
    float dt = std::stof(args.getOption("--dt", "0.01"));
    int steps = std::stoi(args.getOption("--steps", "100"));
    std::string initial = args.getOption("--initial", "jet_stream");
    std::string integration = args.getOption("--integration", "rk4");
    std::string backend = args.getOption("--backend", "adaptive");
    int device = std::stoi(args.getOption("--device", "0"));
    int threads = std::stoi(args.getOption("--threads", "0"));
    bool output_enabled = !args.hasOption("--disable-output");
    
    // Set up simulation configuration
    weather_sim::SimulationConfig config;
    config.model = getModelFromString(model_str);
    config.grid_width = width;
    config.grid_height = height;
    config.dt = dt;
    config.integration_method = getIntegrationMethodFromString(integration);
    config.compute_backend = getBackendFromString(backend);
    config.device_id = device;
    config.num_threads = threads;
    
    // Run the benchmark
    runBenchmark(config, initial, steps, output_enabled);
    
    return 0;
}