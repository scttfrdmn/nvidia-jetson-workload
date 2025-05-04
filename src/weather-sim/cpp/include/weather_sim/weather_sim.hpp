/**
 * @file weather_sim.hpp
 * @brief Weather simulation using GPU-accelerated numerical methods.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <random>
#include <chrono>
#include <iostream>

namespace weather_sim {

/**
 * @brief Common types and utilities for the weather simulation.
 */
using scalar_t = float; // Use float for better GPU performance
using index_t = int32_t; // Use 32-bit integers for indexing

/**
 * @brief Enumeration for different weather simulation models.
 */
enum class SimulationModel {
    ShallowWater,  // Shallow water equations
    Barotropic,    // Barotropic vorticity equation
    PrimitiveEquations, // Primitive equations
    General        // General circulation model (GCM)
};

/**
 * @brief Enumeration for boundary conditions.
 */
enum class BoundaryCondition {
    Periodic,  // Periodic (wrap-around) boundary
    Reflective, // Reflective boundary (no-flow)
    Outflow,   // Outflow boundary (gradient=0)
    Custom     // Custom boundary condition
};

/**
 * @brief Enumeration for time integration methods.
 */
enum class IntegrationMethod {
    ExplicitEuler,  // First-order explicit Euler
    RungeKutta2,    // Second-order Runge-Kutta
    RungeKutta4,    // Fourth-order Runge-Kutta
    AdamsBashforth, // Adams-Bashforth multi-step method
    SemiImplicit    // Semi-implicit method
};

/**
 * @brief Enumeration for grid types.
 */
enum class GridType {
    Cartesian,      // Regular Cartesian grid
    Staggered,      // Staggered grid (Arakawa C-grid)
    Icosahedral,    // Icosahedral grid
    SphericalHarmonic // Spectral grid
};

/**
 * @brief Enumeration for computational backends.
 */
enum class ComputeBackend {
    CUDA,          // CUDA GPU backend
    CPU,           // CPU backend
    Hybrid,        // Hybrid CPU-GPU
    AdaptiveHybrid // Adaptive hybrid (dynamically balances workload)
};

/**
 * @brief Structure to hold grid point data.
 */
struct GridPoint {
    scalar_t u;        // Wind velocity (x or lon)
    scalar_t v;        // Wind velocity (y or lat)
    scalar_t w;        // Vertical velocity (if applicable)
    scalar_t p;        // Pressure
    scalar_t t;        // Temperature
    scalar_t q;        // Humidity (water vapor mixing ratio)
    scalar_t h;        // Height field (for shallow water)
    scalar_t vorticity; // Vorticity
    scalar_t divergence; // Divergence
    
    // Default constructor
    GridPoint() : u(0), v(0), w(0), p(0), t(0), q(0), h(0), vorticity(0), divergence(0) {}
};

/**
 * @brief Structure to hold a 2D vector field.
 */
struct VectorField2D {
    std::vector<scalar_t> u; // x component
    std::vector<scalar_t> v; // y component
    index_t width;
    index_t height;
    
    VectorField2D(index_t w, index_t h) : 
        u(w * h, 0), v(w * h, 0), width(w), height(h) {}
    
    inline index_t index(index_t x, index_t y) const {
        return y * width + x;
    }
    
    inline void set(index_t x, index_t y, scalar_t u_val, scalar_t v_val) {
        index_t idx = index(x, y);
        u[idx] = u_val;
        v[idx] = v_val;
    }
    
    inline void get(index_t x, index_t y, scalar_t& u_val, scalar_t& v_val) const {
        index_t idx = index(x, y);
        u_val = u[idx];
        v_val = v[idx];
    }
};

/**
 * @brief Structure to hold a 2D scalar field.
 */
struct ScalarField2D {
    std::vector<scalar_t> data;
    index_t width;
    index_t height;
    
    ScalarField2D(index_t w, index_t h) : data(w * h, 0), width(w), height(h) {}
    
    inline index_t index(index_t x, index_t y) const {
        return y * width + x;
    }
    
    inline scalar_t& operator()(index_t x, index_t y) {
        return data[index(x, y)];
    }
    
    inline const scalar_t& operator()(index_t x, index_t y) const {
        return data[index(x, y)];
    }
    
    inline void fill(scalar_t value) {
        std::fill(data.begin(), data.end(), value);
    }
};

/**
 * @brief Configuration for a weather simulation.
 */
struct SimulationConfig {
    // Model parameters
    SimulationModel model = SimulationModel::ShallowWater;
    GridType grid_type = GridType::Staggered;
    IntegrationMethod integration_method = IntegrationMethod::RungeKutta4;
    BoundaryCondition boundary_condition = BoundaryCondition::Periodic;
    
    // Grid parameters
    index_t grid_width = 256;    // Number of grid cells in x direction
    index_t grid_height = 256;   // Number of grid cells in y direction
    index_t num_levels = 1;      // Number of vertical levels (1 for 2D)
    scalar_t dx = 1.0;           // Grid spacing in x direction
    scalar_t dy = 1.0;           // Grid spacing in y direction
    scalar_t dt = 0.01;          // Time step
    
    // Physical parameters
    scalar_t gravity = 9.81;     // Gravity (m/s^2)
    scalar_t coriolis_f = 0.0;   // Coriolis parameter
    scalar_t beta = 0.0;         // Beta-plane parameter
    scalar_t viscosity = 0.0;    // Viscosity coefficient
    scalar_t diffusivity = 0.0;  // Diffusivity coefficient
    
    // Computational parameters
    ComputeBackend compute_backend = ComputeBackend::CUDA;
    bool double_precision = false; // Use double precision (affects accuracy vs. performance)
    int device_id = 0;           // GPU device ID
    int num_threads = 0;         // Number of CPU threads (0 = auto)
    
    // Time control
    scalar_t max_time = 10.0;    // Maximum simulation time
    int max_steps = 1000;        // Maximum number of time steps
    int output_interval = 10;    // Interval between outputs
    std::string output_path = "./output"; // Path for output files
    
    // Random seed for initial conditions
    unsigned int random_seed = std::random_device{}();
};

/**
 * @brief Performance metrics for the simulation.
 */
struct PerformanceMetrics {
    double total_time_ms = 0.0;
    double compute_time_ms = 0.0;
    double memory_transfer_time_ms = 0.0;
    double io_time_ms = 0.0;
    int num_steps = 0;
    
    void reset() {
        total_time_ms = 0.0;
        compute_time_ms = 0.0;
        memory_transfer_time_ms = 0.0;
        io_time_ms = 0.0;
        num_steps = 0;
    }
    
    void print() const {
        std::cout << "Performance Metrics:" << std::endl;
        std::cout << "  Total time: " << total_time_ms << " ms" << std::endl;
        std::cout << "  Compute time: " << compute_time_ms << " ms (" 
                  << (compute_time_ms / total_time_ms * 100.0) << "%)" << std::endl;
        std::cout << "  Memory transfer time: " << memory_transfer_time_ms << " ms ("
                  << (memory_transfer_time_ms / total_time_ms * 100.0) << "%)" << std::endl;
        std::cout << "  I/O time: " << io_time_ms << " ms ("
                  << (io_time_ms / total_time_ms * 100.0) << "%)" << std::endl;
        std::cout << "  Steps: " << num_steps << std::endl;
        std::cout << "  Time per step: " << (total_time_ms / num_steps) << " ms" << std::endl;
    }
};

// Forward declarations
class WeatherGrid;
class WeatherSimulation;
class InitialCondition;
class OutputManager;

/**
 * @brief Abstract base class for initial conditions.
 */
class InitialCondition {
public:
    virtual ~InitialCondition() = default;
    
    /**
     * @brief Initialize the weather grid with specified conditions.
     * @param grid The grid to initialize
     */
    virtual void initialize(WeatherGrid& grid) const = 0;
    
    /**
     * @brief Get the name of the initial condition.
     * @return Name of the initial condition
     */
    virtual std::string getName() const = 0;
};

/**
 * @brief Class representing a weather simulation grid.
 */
class WeatherGrid {
public:
    /**
     * @brief Construct a new Weather Grid.
     * @param width Width of the grid (in cells)
     * @param height Height of the grid (in cells)
     * @param num_levels Number of vertical levels
     */
    WeatherGrid(index_t width, index_t height, index_t num_levels = 1);
    
    /**
     * @brief Construct a grid from a configuration.
     * @param config The simulation configuration
     */
    explicit WeatherGrid(const SimulationConfig& config);
    
    /**
     * @brief Reset the grid to default values.
     */
    void reset();
    
    /**
     * @brief Get width of the grid.
     * @return Grid width
     */
    index_t getWidth() const { return width_; }
    
    /**
     * @brief Get height of the grid.
     * @return Grid height
     */
    index_t getHeight() const { return height_; }
    
    /**
     * @brief Get number of levels in the grid.
     * @return Number of levels
     */
    index_t getNumLevels() const { return num_levels_; }
    
    /**
     * @brief Get grid spacing in x direction.
     * @return Grid spacing dx
     */
    scalar_t getDx() const { return dx_; }
    
    /**
     * @brief Get grid spacing in y direction.
     * @return Grid spacing dy
     */
    scalar_t getDy() const { return dy_; }
    
    /**
     * @brief Set grid spacing.
     * @param dx Grid spacing in x direction
     * @param dy Grid spacing in y direction
     */
    void setSpacing(scalar_t dx, scalar_t dy);
    
    /**
     * @brief Get velocity field.
     * @return Reference to velocity field
     */
    VectorField2D& getVelocityField() { return velocity_; }
    
    /**
     * @brief Get velocity field (const version).
     * @return Const reference to velocity field
     */
    const VectorField2D& getVelocityField() const { return velocity_; }
    
    /**
     * @brief Get height field.
     * @return Reference to height field
     */
    ScalarField2D& getHeightField() { return height_; }
    
    /**
     * @brief Get height field (const version).
     * @return Const reference to height field
     */
    const ScalarField2D& getHeightField() const { return height_; }
    
    /**
     * @brief Get pressure field.
     * @return Reference to pressure field
     */
    ScalarField2D& getPressureField() { return pressure_; }
    
    /**
     * @brief Get pressure field (const version).
     * @return Const reference to pressure field
     */
    const ScalarField2D& getPressureField() const { return pressure_; }
    
    /**
     * @brief Get temperature field.
     * @return Reference to temperature field
     */
    ScalarField2D& getTemperatureField() { return temperature_; }
    
    /**
     * @brief Get temperature field (const version).
     * @return Const reference to temperature field
     */
    const ScalarField2D& getTemperatureField() const { return temperature_; }
    
    /**
     * @brief Get humidity field.
     * @return Reference to humidity field
     */
    ScalarField2D& getHumidityField() { return humidity_; }
    
    /**
     * @brief Get humidity field (const version).
     * @return Const reference to humidity field
     */
    const ScalarField2D& getHumidityField() const { return humidity_; }
    
    /**
     * @brief Get vorticity field.
     * @return Reference to vorticity field
     */
    ScalarField2D& getVorticityField() { return vorticity_; }
    
    /**
     * @brief Get vorticity field (const version).
     * @return Const reference to vorticity field
     */
    const ScalarField2D& getVorticityField() const { return vorticity_; }
    
    /**
     * @brief Calculate diagnostics like vorticity, divergence, etc.
     */
    void calculateDiagnostics();
    
    /**
     * @brief Swap grid data with another grid.
     * @param other The grid to swap with
     */
    void swap(WeatherGrid& other);
    
private:
    index_t width_;       // Grid width
    index_t height_;      // Grid height
    index_t num_levels_;  // Number of vertical levels
    scalar_t dx_;         // Grid spacing in x direction
    scalar_t dy_;         // Grid spacing in y direction
    
    // Primary fields
    VectorField2D velocity_;      // Velocity field (u, v components)
    ScalarField2D height_;        // Height field (for shallow water)
    ScalarField2D pressure_;      // Pressure field
    ScalarField2D temperature_;   // Temperature field
    ScalarField2D humidity_;      // Humidity field
    
    // Diagnostic fields
    ScalarField2D vorticity_;     // Vorticity field
    ScalarField2D divergence_;    // Divergence field
};

/**
 * @brief Class for managing the weather simulation.
 */
class WeatherSimulation {
public:
    /**
     * @brief Construct a new Weather Simulation.
     * @param config The simulation configuration
     */
    explicit WeatherSimulation(const SimulationConfig& config);
    
    /**
     * @brief Set the initial condition.
     * @param initial_condition The initial condition to use
     */
    void setInitialCondition(std::shared_ptr<InitialCondition> initial_condition);
    
    /**
     * @brief Set the output manager.
     * @param output_manager The output manager to use
     */
    void setOutputManager(std::shared_ptr<OutputManager> output_manager);
    
    /**
     * @brief Initialize the simulation.
     */
    void initialize();
    
    /**
     * @brief Run the simulation for the specified number of steps.
     * @param num_steps Number of steps to run
     */
    void run(int num_steps);
    
    /**
     * @brief Run the simulation until the specified time.
     * @param max_time Maximum simulation time
     */
    void runUntil(scalar_t max_time);
    
    /**
     * @brief Perform a single time step.
     */
    void step();
    
    /**
     * @brief Get the current time.
     * @return Current simulation time
     */
    scalar_t getCurrentTime() const { return current_time_; }
    
    /**
     * @brief Get the current step count.
     * @return Current step count
     */
    int getCurrentStep() const { return current_step_; }
    
    /**
     * @brief Get the time step.
     * @return Time step size
     */
    scalar_t getDt() const { return dt_; }
    
    /**
     * @brief Set the time step.
     * @param dt New time step size
     */
    void setDt(scalar_t dt) { dt_ = dt; }
    
    /**
     * @brief Get the simulation configuration.
     * @return Current configuration
     */
    const SimulationConfig& getConfig() const { return config_; }
    
    /**
     * @brief Get the current grid.
     * @return Reference to the current grid
     */
    WeatherGrid& getCurrentGrid() { return *current_grid_; }
    
    /**
     * @brief Get the current grid (const version).
     * @return Const reference to the current grid
     */
    const WeatherGrid& getCurrentGrid() const { return *current_grid_; }
    
    /**
     * @brief Get performance metrics.
     * @return Reference to performance metrics
     */
    const PerformanceMetrics& getPerformanceMetrics() const { return metrics_; }
    
    /**
     * @brief Reset performance metrics.
     */
    void resetPerformanceMetrics() { metrics_.reset(); }
    
private:
    // Helper methods for different time integration schemes
    void stepExplicitEuler();
    void stepRungeKutta2();
    void stepRungeKutta4();
    void stepAdamsBashforth();
    void stepSemiImplicit();
    
    // Helper methods for different models
    void computeShallowWaterTendencies(const WeatherGrid& in_grid, WeatherGrid& tendencies);
    void computeBarotropicTendencies(const WeatherGrid& in_grid, WeatherGrid& tendencies);
    void computePrimitiveEquationsTendencies(const WeatherGrid& in_grid, WeatherGrid& tendencies);
    
    // Member variables
    SimulationConfig config_;                       // Simulation configuration
    scalar_t current_time_ = 0.0;                   // Current simulation time
    int current_step_ = 0;                          // Current step count
    scalar_t dt_;                                   // Time step
    
    std::shared_ptr<WeatherGrid> current_grid_;     // Current state
    std::shared_ptr<WeatherGrid> next_grid_;        // Next state
    std::shared_ptr<WeatherGrid> temp_grid_;        // Temporary grid for multi-step methods
    std::shared_ptr<WeatherGrid> tendency_grid_;    // Grid for storing tendencies
    
    std::shared_ptr<InitialCondition> initial_condition_;  // Initial condition
    std::shared_ptr<OutputManager> output_manager_;        // Output manager
    
    PerformanceMetrics metrics_;                    // Performance metrics
    
    // Helper for device adaptivity
    void selectOptimalBackend();
    bool isCudaAvailable();
};

/**
 * @brief Abstract base class for output managers.
 */
class OutputManager {
public:
    virtual ~OutputManager() = default;
    
    /**
     * @brief Initialize the output manager.
     * @param simulation Reference to the simulation
     */
    virtual void initialize(const WeatherSimulation& simulation) = 0;
    
    /**
     * @brief Write output for the current state.
     * @param simulation Reference to the simulation
     */
    virtual void writeOutput(const WeatherSimulation& simulation) = 0;
    
    /**
     * @brief Finalize the output manager.
     * @param simulation Reference to the simulation
     */
    virtual void finalize(const WeatherSimulation& simulation) = 0;
};

} // namespace weather_sim