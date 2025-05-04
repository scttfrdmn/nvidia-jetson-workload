"""
Weather Simulation Python Module.

This module provides Python bindings for the Weather Simulation workload.
It includes classes for creating and running weather simulations, as well
as utilities for visualization and analysis.

Author: Scott Friedman
Copyright 2025 Scott Friedman. All rights reserved.
"""

from .weather_simulation import (
    WeatherSimulation,
    SimulationConfig,
    IntegrationMethod,
    SimulationModel,
    ComputeBackend,
    GridType,
    BoundaryCondition,
    create_initial_condition,
    get_available_initial_conditions,
    is_cuda_available,
    get_device_info
)

from .visualization import (
    visualize_field,
    visualize_velocity,
    visualize_vorticity,
    visualize_height,
    animate_simulation,
    plot_performance
)

__version__ = '0.1.0'
__author__ = 'Scott Friedman'