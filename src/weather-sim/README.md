# Weather Simulation Workload

A GPU-accelerated weather simulation workload for scientific computing on Nvidia Jetson Orin NX and AWS Graviton g5g instances.

## Overview

This workload implements several weather models with GPU acceleration, including:

- Shallow Water Equations
- Barotropic Vorticity Equation
- Primitive Equations

The implementation features:

- GPU-CPU hybrid computing with dynamic load balancing
- Architecture-specific optimizations for Jetson Orin NX and T4 GPUs
- Multiple time integration methods (Explicit Euler, RK2, RK4, etc.)
- Various initial conditions (vortex, jet stream, front, etc.)
- Python bindings with visualization utilities

## Building

### Prerequisites

- CMake 3.12 or higher
- CUDA Toolkit 11.0 or higher
- C++17 compatible compiler
- Python 3.8 or higher (for Python bindings)
- pybind11 (for Python bindings)
- matplotlib (for visualization)

### Build Instructions

To build the C++ library and Python bindings:

```bash
mkdir build
cd build
cmake .. -DBUILD_WEATHER_SIM=ON
make -j
```

## Usage

### C++ Example

```cpp
#include <weather_sim/weather_sim.hpp>
#include <weather_sim/initial_conditions.hpp>

int main() {
    // Create configuration
    weather_sim::SimulationConfig config;
    config.grid_width = 512;
    config.grid_height = 512;
    config.model = weather_sim::SimulationModel::ShallowWater;
    config.integration_method = weather_sim::IntegrationMethod::RungeKutta4;
    
    // Create simulation
    weather_sim::WeatherSimulation sim(config);
    
    // Set initial condition
    auto initial = std::make_shared<weather_sim::VortexInitialCondition>(0.5, 0.5, 0.1, 10.0);
    sim.setInitialCondition(initial);
    
    // Initialize and run
    sim.initialize();
    sim.run(100);
    
    // Get results
    const auto& grid = sim.getCurrentGrid();
    const auto& metrics = sim.getPerformanceMetrics();
    
    return 0;
}
```

### Python Example

```python
from weather_sim import WeatherSimulation, SimulationConfig, IntegrationMethod, SimulationModel
from weather_sim.visualization import visualize_height, visualize_velocity

# Create simulation
sim = WeatherSimulationWrapper(
    width=512,
    height=512,
    model="shallow_water",
    integration_method="rk4"
)

# Set initial condition
sim.set_initial_condition("vortex", x_center=0.5, y_center=0.5, radius=0.1, strength=10.0)

# Initialize and run
sim.initialize()
sim.run(100)

# Visualize results
grid = sim.get_grid()
u, v = grid.get_velocity_field()
height = grid.get_height_field()

visualize_height(height, u, v, show_velocity=True)
visualize_velocity(u, v, streamlines=True)
```

## GPU Adaptability

The workload automatically detects and adapts to different GPU architectures:

- Jetson Orin NX (SM 8.7)
- AWS Graviton g5g with T4 (SM 7.5)
- Generic high-end GPUs (SM ≥ 8.0)

Optimizations include:

- Tiled algorithms with shared memory
- Architecture-specific kernel implementations
- Dynamic workload distribution between CPU and GPU
- Optimal kernel launch parameters for each architecture

## Performance Metrics

The workload collects performance metrics including:

- Total computation time
- Memory transfer time
- I/O time
- Steps per second
- Million cell updates per second (MCUPS)

## License

Copyright 2025 Scott Friedman. All rights reserved.