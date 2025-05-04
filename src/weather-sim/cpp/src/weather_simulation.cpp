/**
 * @file weather_simulation.cpp
 * @brief Implementation of the WeatherSimulation class.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include "../include/weather_sim/weather_sim.hpp"
#include "../include/weather_sim/gpu_adaptability.hpp"
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace weather_sim {

WeatherSimulation::WeatherSimulation(const SimulationConfig& config)
    : config_(config),
      current_time_(0.0),
      current_step_(0),
      dt_(config.dt)
{
    // Create grids
    current_grid_ = std::make_shared<WeatherGrid>(config);
    next_grid_ = std::make_shared<WeatherGrid>(config);
    tendency_grid_ = std::make_shared<WeatherGrid>(config);
    
    // Create temporary grid for multi-step methods
    if (config.integration_method == IntegrationMethod::RungeKutta4 ||
        config.integration_method == IntegrationMethod::RungeKutta2) {
        temp_grid_ = std::make_shared<WeatherGrid>(config);
    }
    
    // Select optimal backend
    selectOptimalBackend();
}

void WeatherSimulation::setInitialCondition(std::shared_ptr<InitialCondition> initial_condition) {
    initial_condition_ = initial_condition;
}

void WeatherSimulation::setOutputManager(std::shared_ptr<OutputManager> output_manager) {
    output_manager_ = output_manager;
}

void WeatherSimulation::initialize() {
    // Reset time and step counter
    current_time_ = 0.0;
    current_step_ = 0;
    
    // Reset performance metrics
    metrics_.reset();
    
    // Initialize current grid
    current_grid_->reset();
    
    // Apply initial condition
    if (initial_condition_) {
        initial_condition_->initialize(*current_grid_);
    }
    
    // Initialize output manager
    if (output_manager_) {
        output_manager_->initialize(*this);
    }
}

void WeatherSimulation::run(int num_steps) {
    if (num_steps <= 0) {
        return;
    }
    
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run for specified number of steps
    for (int i = 0; i < num_steps; ++i) {
        step();
        
        // Write output at specified intervals
        if (output_manager_ && config_.output_interval > 0 && 
            current_step_ % config_.output_interval == 0) {
            output_manager_->writeOutput(*this);
        }
        
        // Check if we've reached the maximum time
        if (current_time_ >= config_.max_time) {
            break;
        }
    }
    
    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Update performance metrics
    metrics_.total_time_ms += duration;
    
    // Print performance info
    std::cout << "Completed " << num_steps << " steps in " 
              << duration << " ms (" << (duration / (double)num_steps) 
              << " ms/step)" << std::endl;
}

void WeatherSimulation::runUntil(scalar_t max_time) {
    if (max_time <= current_time_) {
        return;
    }
    
    // Calculate number of steps needed
    int num_steps = static_cast<int>((max_time - current_time_) / dt_) + 1;
    
    // Run for calculated number of steps
    run(num_steps);
}

void WeatherSimulation::step() {
    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Choose the appropriate time integration method
    switch (config_.integration_method) {
        case IntegrationMethod::ExplicitEuler:
            stepExplicitEuler();
            break;
        case IntegrationMethod::RungeKutta2:
            stepRungeKutta2();
            break;
        case IntegrationMethod::RungeKutta4:
            stepRungeKutta4();
            break;
        case IntegrationMethod::AdamsBashforth:
            stepAdamsBashforth();
            break;
        case IntegrationMethod::SemiImplicit:
            stepSemiImplicit();
            break;
        default:
            // Default to explicit Euler
            stepExplicitEuler();
            break;
    }
    
    // Increment time and step counter
    current_time_ += dt_;
    current_step_++;
    
    // Calculate diagnostics
    current_grid_->calculateDiagnostics();
    
    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Update performance metrics
    metrics_.compute_time_ms += duration;
    metrics_.num_steps++;
}

void WeatherSimulation::stepExplicitEuler() {
    // Compute tendencies
    switch (config_.model) {
        case SimulationModel::ShallowWater:
            computeShallowWaterTendencies(*current_grid_, *tendency_grid_);
            break;
        case SimulationModel::Barotropic:
            computeBarotropicTendencies(*current_grid_, *tendency_grid_);
            break;
        case SimulationModel::PrimitiveEquations:
            computePrimitiveEquationsTendencies(*current_grid_, *tendency_grid_);
            break;
        default:
            // Default to shallow water
            computeShallowWaterTendencies(*current_grid_, *tendency_grid_);
            break;
    }
    
    // Update using explicit Euler method
    // next = current + dt * tendency
    
    // For velocity field
    auto& current_velocity = current_grid_->getVelocityField();
    auto& tendency_velocity = tendency_grid_->getVelocityField();
    auto& next_velocity = next_grid_->getVelocityField();
    
    for (index_t i = 0; i < current_velocity.u.size(); ++i) {
        next_velocity.u[i] = current_velocity.u[i] + dt_ * tendency_velocity.u[i];
        next_velocity.v[i] = current_velocity.v[i] + dt_ * tendency_velocity.v[i];
    }
    
    // For height field
    auto& current_height = current_grid_->getHeightField();
    auto& tendency_height = tendency_grid_->getHeightField();
    auto& next_height = next_grid_->getHeightField();
    
    for (index_t i = 0; i < current_height.data.size(); ++i) {
        next_height.data[i] = current_height.data[i] + dt_ * tendency_height.data[i];
    }
    
    // For other fields (if needed)
    if (config_.model == SimulationModel::PrimitiveEquations) {
        auto& current_temp = current_grid_->getTemperatureField();
        auto& tendency_temp = tendency_grid_->getTemperatureField();
        auto& next_temp = next_grid_->getTemperatureField();
        
        auto& current_pressure = current_grid_->getPressureField();
        auto& tendency_pressure = tendency_grid_->getPressureField();
        auto& next_pressure = next_grid_->getPressureField();
        
        for (index_t i = 0; i < current_temp.data.size(); ++i) {
            next_temp.data[i] = current_temp.data[i] + dt_ * tendency_temp.data[i];
            next_pressure.data[i] = current_pressure.data[i] + dt_ * tendency_pressure.data[i];
        }
    }
    
    // Swap grids
    current_grid_.swap(next_grid_);
}

void WeatherSimulation::stepRungeKutta2() {
    // RK2 method (midpoint method):
    // k1 = f(y_n)
    // k2 = f(y_n + dt/2 * k1)
    // y_{n+1} = y_n + dt * k2
    
    // Step 1: Compute k1 = f(y_n)
    switch (config_.model) {
        case SimulationModel::ShallowWater:
            computeShallowWaterTendencies(*current_grid_, *tendency_grid_);
            break;
        case SimulationModel::Barotropic:
            computeBarotropicTendencies(*current_grid_, *tendency_grid_);
            break;
        case SimulationModel::PrimitiveEquations:
            computePrimitiveEquationsTendencies(*current_grid_, *tendency_grid_);
            break;
        default:
            computeShallowWaterTendencies(*current_grid_, *tendency_grid_);
            break;
    }
    
    // Step 2: Compute y_n + dt/2 * k1
    // For velocity field
    auto& current_velocity = current_grid_->getVelocityField();
    auto& tendency_velocity = tendency_grid_->getVelocityField();
    auto& temp_velocity = temp_grid_->getVelocityField();
    
    for (index_t i = 0; i < current_velocity.u.size(); ++i) {
        temp_velocity.u[i] = current_velocity.u[i] + 0.5f * dt_ * tendency_velocity.u[i];
        temp_velocity.v[i] = current_velocity.v[i] + 0.5f * dt_ * tendency_velocity.v[i];
    }
    
    // For height field
    auto& current_height = current_grid_->getHeightField();
    auto& tendency_height = tendency_grid_->getHeightField();
    auto& temp_height = temp_grid_->getHeightField();
    
    for (index_t i = 0; i < current_height.data.size(); ++i) {
        temp_height.data[i] = current_height.data[i] + 0.5f * dt_ * tendency_height.data[i];
    }
    
    // For other fields (if needed)
    if (config_.model == SimulationModel::PrimitiveEquations) {
        auto& current_temp = current_grid_->getTemperatureField();
        auto& tendency_temp = tendency_grid_->getTemperatureField();
        auto& temp_temp = temp_grid_->getTemperatureField();
        
        auto& current_pressure = current_grid_->getPressureField();
        auto& tendency_pressure = tendency_grid_->getPressureField();
        auto& temp_pressure = temp_grid_->getPressureField();
        
        for (index_t i = 0; i < current_temp.data.size(); ++i) {
            temp_temp.data[i] = current_temp.data[i] + 0.5f * dt_ * tendency_temp.data[i];
            temp_pressure.data[i] = current_pressure.data[i] + 0.5f * dt_ * tendency_pressure.data[i];
        }
    }
    
    // Step 3: Compute k2 = f(y_n + dt/2 * k1)
    switch (config_.model) {
        case SimulationModel::ShallowWater:
            computeShallowWaterTendencies(*temp_grid_, *tendency_grid_);
            break;
        case SimulationModel::Barotropic:
            computeBarotropicTendencies(*temp_grid_, *tendency_grid_);
            break;
        case SimulationModel::PrimitiveEquations:
            computePrimitiveEquationsTendencies(*temp_grid_, *tendency_grid_);
            break;
        default:
            computeShallowWaterTendencies(*temp_grid_, *tendency_grid_);
            break;
    }
    
    // Step 4: Compute y_{n+1} = y_n + dt * k2
    // For velocity field
    auto& next_velocity = next_grid_->getVelocityField();
    
    for (index_t i = 0; i < current_velocity.u.size(); ++i) {
        next_velocity.u[i] = current_velocity.u[i] + dt_ * tendency_velocity.u[i];
        next_velocity.v[i] = current_velocity.v[i] + dt_ * tendency_velocity.v[i];
    }
    
    // For height field
    auto& next_height = next_grid_->getHeightField();
    
    for (index_t i = 0; i < current_height.data.size(); ++i) {
        next_height.data[i] = current_height.data[i] + dt_ * tendency_height.data[i];
    }
    
    // For other fields (if needed)
    if (config_.model == SimulationModel::PrimitiveEquations) {
        auto& next_temp = next_grid_->getTemperatureField();
        auto& next_pressure = next_grid_->getPressureField();
        
        for (index_t i = 0; i < current_temp.data.size(); ++i) {
            next_temp.data[i] = current_temp.data[i] + dt_ * tendency_temp.data[i];
            next_pressure.data[i] = current_pressure.data[i] + dt_ * tendency_pressure.data[i];
        }
    }
    
    // Swap grids
    current_grid_.swap(next_grid_);
}

void WeatherSimulation::stepRungeKutta4() {
    // RK4 method:
    // k1 = f(y_n)
    // k2 = f(y_n + dt/2 * k1)
    // k3 = f(y_n + dt/2 * k2)
    // k4 = f(y_n + dt * k3)
    // y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    // For simplicity, we'll just implement this for the shallow water model
    if (config_.model != SimulationModel::ShallowWater) {
        // Fall back to RK2 for other models
        stepRungeKutta2();
        return;
    }
    
    // Get references to the fields
    auto& current_velocity = current_grid_->getVelocityField();
    auto& current_height = current_grid_->getHeightField();
    
    auto& next_velocity = next_grid_->getVelocityField();
    auto& next_height = next_grid_->getHeightField();
    
    auto& temp_velocity = temp_grid_->getVelocityField();
    auto& temp_height = temp_grid_->getHeightField();
    
    auto& k1_velocity = tendency_grid_->getVelocityField();
    auto& k1_height = tendency_grid_->getHeightField();
    
    // Create vectors to store k2, k3, k4
    std::vector<scalar_t> k2_u(current_velocity.u.size());
    std::vector<scalar_t> k2_v(current_velocity.v.size());
    std::vector<scalar_t> k2_h(current_height.data.size());
    
    std::vector<scalar_t> k3_u(current_velocity.u.size());
    std::vector<scalar_t> k3_v(current_velocity.v.size());
    std::vector<scalar_t> k3_h(current_height.data.size());
    
    std::vector<scalar_t> k4_u(current_velocity.u.size());
    std::vector<scalar_t> k4_v(current_velocity.v.size());
    std::vector<scalar_t> k4_h(current_height.data.size());
    
    // Step 1: Compute k1 = f(y_n)
    computeShallowWaterTendencies(*current_grid_, *tendency_grid_);
    
    // Copy k1 values
    for (size_t i = 0; i < k1_velocity.u.size(); ++i) {
        k2_u[i] = k1_velocity.u[i];
        k2_v[i] = k1_velocity.v[i];
    }
    
    for (size_t i = 0; i < k1_height.data.size(); ++i) {
        k2_h[i] = k1_height.data[i];
    }
    
    // Step 2: Compute y_n + dt/2 * k1
    for (size_t i = 0; i < current_velocity.u.size(); ++i) {
        temp_velocity.u[i] = current_velocity.u[i] + 0.5f * dt_ * k1_velocity.u[i];
        temp_velocity.v[i] = current_velocity.v[i] + 0.5f * dt_ * k1_velocity.v[i];
    }
    
    for (size_t i = 0; i < current_height.data.size(); ++i) {
        temp_height.data[i] = current_height.data[i] + 0.5f * dt_ * k1_height.data[i];
    }
    
    // Step 3: Compute k2 = f(y_n + dt/2 * k1)
    computeShallowWaterTendencies(*temp_grid_, *tendency_grid_);
    
    // Copy k2 values and compute y_n + dt/2 * k2
    for (size_t i = 0; i < tendency_grid_->getVelocityField().u.size(); ++i) {
        k2_u[i] = tendency_grid_->getVelocityField().u[i];
        k2_v[i] = tendency_grid_->getVelocityField().v[i];
        
        temp_velocity.u[i] = current_velocity.u[i] + 0.5f * dt_ * k2_u[i];
        temp_velocity.v[i] = current_velocity.v[i] + 0.5f * dt_ * k2_v[i];
    }
    
    for (size_t i = 0; i < tendency_grid_->getHeightField().data.size(); ++i) {
        k2_h[i] = tendency_grid_->getHeightField().data[i];
        temp_height.data[i] = current_height.data[i] + 0.5f * dt_ * k2_h[i];
    }
    
    // Step 4: Compute k3 = f(y_n + dt/2 * k2)
    computeShallowWaterTendencies(*temp_grid_, *tendency_grid_);
    
    // Copy k3 values and compute y_n + dt * k3
    for (size_t i = 0; i < tendency_grid_->getVelocityField().u.size(); ++i) {
        k3_u[i] = tendency_grid_->getVelocityField().u[i];
        k3_v[i] = tendency_grid_->getVelocityField().v[i];
        
        temp_velocity.u[i] = current_velocity.u[i] + dt_ * k3_u[i];
        temp_velocity.v[i] = current_velocity.v[i] + dt_ * k3_v[i];
    }
    
    for (size_t i = 0; i < tendency_grid_->getHeightField().data.size(); ++i) {
        k3_h[i] = tendency_grid_->getHeightField().data[i];
        temp_height.data[i] = current_height.data[i] + dt_ * k3_h[i];
    }
    
    // Step 5: Compute k4 = f(y_n + dt * k3)
    computeShallowWaterTendencies(*temp_grid_, *tendency_grid_);
    
    // Copy k4 values
    for (size_t i = 0; i < tendency_grid_->getVelocityField().u.size(); ++i) {
        k4_u[i] = tendency_grid_->getVelocityField().u[i];
        k4_v[i] = tendency_grid_->getVelocityField().v[i];
    }
    
    for (size_t i = 0; i < tendency_grid_->getHeightField().data.size(); ++i) {
        k4_h[i] = tendency_grid_->getHeightField().data[i];
    }
    
    // Step 6: Compute y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    for (size_t i = 0; i < current_velocity.u.size(); ++i) {
        next_velocity.u[i] = current_velocity.u[i] + dt_ / 6.0f * (
            k1_velocity.u[i] + 2.0f * k2_u[i] + 2.0f * k3_u[i] + k4_u[i]
        );
        
        next_velocity.v[i] = current_velocity.v[i] + dt_ / 6.0f * (
            k1_velocity.v[i] + 2.0f * k2_v[i] + 2.0f * k3_v[i] + k4_v[i]
        );
    }
    
    for (size_t i = 0; i < current_height.data.size(); ++i) {
        next_height.data[i] = current_height.data[i] + dt_ / 6.0f * (
            k1_height.data[i] + 2.0f * k2_h[i] + 2.0f * k3_h[i] + k4_h[i]
        );
    }
    
    // Swap grids
    current_grid_.swap(next_grid_);
}

void WeatherSimulation::stepAdamsBashforth() {
    // This is a multi-step method that requires previous steps
    // For simplicity, we'll just use explicit Euler for the first step
    
    // For now, fall back to explicit Euler
    stepExplicitEuler();
}

void WeatherSimulation::stepSemiImplicit() {
    // Semi-implicit methods are more complex and require solving linear systems
    // For simplicity, we'll just use explicit Euler
    
    // For now, fall back to explicit Euler
    stepExplicitEuler();
}

void WeatherSimulation::computeShallowWaterTendencies(
    const WeatherGrid& in_grid, WeatherGrid& tendencies
) {
    // Get references to the fields
    const auto& velocity = in_grid.getVelocityField();
    const auto& height = in_grid.getHeightField();
    
    auto& vel_tendencies = tendencies.getVelocityField();
    auto& h_tendencies = tendencies.getHeightField();
    
    // Grid dimensions
    index_t width = in_grid.getWidth();
    index_t height_dim = in_grid.getHeight();
    scalar_t dx = in_grid.getDx();
    scalar_t dy = in_grid.getDy();
    scalar_t gravity = config_.gravity;
    scalar_t coriolis_f = config_.coriolis_f;
    
    // Check if we can use GPU
    if (config_.compute_backend == ComputeBackend::CUDA && isCudaAvailable()) {
        // This would be implemented using the GPU kernels
        // Placeholder for now
        
        // Launch GPU kernel
        // The actual GPU implementation would be here
        
        return;
    }
    
    // CPU implementation of shallow water equations
    #pragma omp parallel for collapse(2)
    for (index_t y = 0; y < height_dim; ++y) {
        for (index_t x = 0; x < width; ++x) {
            // Current cell index
            index_t idx = y * width + x;
            
            // Get neighboring cell indices with boundary handling
            index_t idx_left = (x > 0) ? idx - 1 : idx;
            index_t idx_right = (x < width - 1) ? idx + 1 : idx;
            index_t idx_top = (y > 0) ? idx - width : idx;
            index_t idx_bottom = (y < height_dim - 1) ? idx + width : idx;
            
            // Current cell values
            scalar_t u = velocity.u[idx];
            scalar_t v = velocity.v[idx];
            scalar_t h = height.data[idx];
            
            // Compute spatial derivatives using central differences
            scalar_t u_x = (velocity.u[idx_right] - velocity.u[idx_left]) / (2.0f * dx);
            scalar_t u_y = (velocity.u[idx_bottom] - velocity.u[idx_top]) / (2.0f * dy);
            
            scalar_t v_x = (velocity.v[idx_right] - velocity.v[idx_left]) / (2.0f * dx);
            scalar_t v_y = (velocity.v[idx_bottom] - velocity.v[idx_top]) / (2.0f * dy);
            
            scalar_t h_x = (height.data[idx_right] - height.data[idx_left]) / (2.0f * dx);
            scalar_t h_y = (height.data[idx_bottom] - height.data[idx_top]) / (2.0f * dy);
            
            // Compute tendencies (time derivatives)
            // du/dt = -u * du/dx - v * du/dy - g * dh/dx + f * v
            // dv/dt = -u * dv/dx - v * dv/dy - g * dh/dy - f * u
            // dh/dt = -h * (du/dx + dv/dy) - u * dh/dx - v * dh/dy
            
            vel_tendencies.u[idx] = -u * u_x - v * u_y - gravity * h_x + coriolis_f * v;
            vel_tendencies.v[idx] = -u * v_x - v * v_y - gravity * h_y - coriolis_f * u;
            h_tendencies.data[idx] = -h * (u_x + v_y) - u * h_x - v * h_y;
        }
    }
}

void WeatherSimulation::computeBarotropicTendencies(
    const WeatherGrid& in_grid, WeatherGrid& tendencies
) {
    // Barotropic vorticity equation
    // This is a simplification for now
    
    // Fall back to shallow water tendencies
    computeShallowWaterTendencies(in_grid, tendencies);
}

void WeatherSimulation::computePrimitiveEquationsTendencies(
    const WeatherGrid& in_grid, WeatherGrid& tendencies
) {
    // Primitive equations
    // This is a simplification for now
    
    // Fall back to shallow water tendencies
    computeShallowWaterTendencies(in_grid, tendencies);
}

void WeatherSimulation::selectOptimalBackend() {
    // Check if CUDA is available
    if (config_.compute_backend == ComputeBackend::CUDA && !isCudaAvailable()) {
        std::cerr << "CUDA requested but not available, falling back to CPU" << std::endl;
        config_.compute_backend = ComputeBackend::CPU;
    }
    
    // For Hybrid and AdaptiveHybrid, we need to initialize the HybridExecutionManager
    if (config_.compute_backend == ComputeBackend::Hybrid || 
        config_.compute_backend == ComputeBackend::AdaptiveHybrid) {
        HybridExecutionManager::getInstance().initialize(config_.device_id, config_.num_threads);
    }
    
    // Log the chosen backend
    std::cout << "Using compute backend: ";
    switch (config_.compute_backend) {
        case ComputeBackend::CUDA:
            std::cout << "CUDA GPU" << std::endl;
            break;
        case ComputeBackend::CPU:
            std::cout << "CPU" << std::endl;
            break;
        case ComputeBackend::Hybrid:
            std::cout << "Hybrid CPU-GPU" << std::endl;
            break;
        case ComputeBackend::AdaptiveHybrid:
            std::cout << "Adaptive Hybrid CPU-GPU" << std::endl;
            break;
    }
}

bool WeatherSimulation::isCudaAvailable() {
    return AdaptiveKernelManager::getInstance().isCudaAvailable();
}

} // namespace weather_sim