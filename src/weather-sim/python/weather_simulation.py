"""
Weather Simulation Python Module.

This module provides a high-level Python API for the Weather Simulation workload.

Author: Scott Friedman
Copyright 2025 Scott Friedman. All rights reserved.
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import os
import time
from pathlib import Path

try:
    from .pyweather_sim import (
        WeatherGrid, WeatherSimulation, SimulationConfig, InitialConditionFactory,
        AdaptiveKernelManager, PerformanceMetrics, OutputConfig,
        SimulationModel, IntegrationMethod, GridType, BoundaryCondition, 
        ComputeBackend, DeviceType, OutputFormat,
        UniformInitialCondition, RandomInitialCondition, ZonalFlowInitialCondition,
        VortexInitialCondition, JetStreamInitialCondition, BreakingWaveInitialCondition,
        FrontInitialCondition, MountainInitialCondition, AtmosphericProfileInitialCondition,
        register_all_initial_conditions
    )
    
    # Register initial conditions
    register_all_initial_conditions()
except ImportError:
    print("Warning: Failed to import C++ bindings. Using mock implementation.")
    
    # Mock enumerations for documentation/testing without C++ library
    class SimulationModel:
        ShallowWater = 0
        Barotropic = 1
        PrimitiveEquations = 2
        General = 3
    
    class IntegrationMethod:
        ExplicitEuler = 0
        RungeKutta2 = 1
        RungeKutta4 = 2
        AdamsBashforth = 3
        SemiImplicit = 4
    
    class GridType:
        Cartesian = 0
        Staggered = 1
        Icosahedral = 2
        SphericalHarmonic = 3
    
    class BoundaryCondition:
        Periodic = 0
        Reflective = 1
        Outflow = 2
        Custom = 3
    
    class ComputeBackend:
        CUDA = 0
        CPU = 1
        Hybrid = 2
        AdaptiveHybrid = 3
    
    class DeviceType:
        Unknown = 0
        CPU = 1
        JetsonOrinNX = 2
        T4 = 3
        HighEndGPU = 4
        OtherGPU = 5
    
    class OutputFormat:
        CSV = 0
        NetCDF = 1
        VTK = 2
        PNG = 3
        Custom = 4
    
    # Mock classes (minimal implementation)
    class WeatherGrid:
        def __init__(self, width, height, num_levels=1):
            self.width = width
            self.height = height
            self.num_levels = num_levels
            
            # Create numpy arrays for fields
            self.velocity_u = np.zeros((height, width), dtype=np.float32)
            self.velocity_v = np.zeros((height, width), dtype=np.float32)
            self.height_field = np.ones((height, width), dtype=np.float32) * 10.0
            self.pressure_field = np.ones((height, width), dtype=np.float32) * 1013.25
            self.temperature_field = np.ones((height, width), dtype=np.float32) * 288.15
            self.humidity_field = np.zeros((height, width), dtype=np.float32)
            self.vorticity_field = np.zeros((height, width), dtype=np.float32)
        
        def get_velocity_field(self):
            return self.velocity_u, self.velocity_v
        
        def get_height_field(self):
            return self.height_field
        
        def get_pressure_field(self):
            return self.pressure_field
        
        def get_temperature_field(self):
            return self.temperature_field
        
        def get_humidity_field(self):
            return self.humidity_field
        
        def get_vorticity_field(self):
            return self.vorticity_field
        
        def get_width(self):
            return self.width
        
        def get_height(self):
            return self.height
        
        def calculate_diagnostics(self):
            # Mock implementation
            pass
    
    class SimulationConfig:
        def __init__(self):
            self.model = SimulationModel.ShallowWater
            self.grid_type = GridType.Staggered
            self.integration_method = IntegrationMethod.RungeKutta4
            self.boundary_condition = BoundaryCondition.Periodic
            self.grid_width = 256
            self.grid_height = 256
            self.num_levels = 1
            self.dx = 1.0
            self.dy = 1.0
            self.dt = 0.01
            self.gravity = 9.81
            self.coriolis_f = 0.0
            self.beta = 0.0
            self.viscosity = 0.0
            self.diffusivity = 0.0
            self.compute_backend = ComputeBackend.CPU
            self.double_precision = False
            self.device_id = 0
            self.num_threads = 0
            self.max_time = 10.0
            self.max_steps = 1000
            self.output_interval = 10
            self.output_path = "./output"
            self.random_seed = 42
    
    class PerformanceMetrics:
        def __init__(self):
            self.total_time_ms = 0.0
            self.compute_time_ms = 0.0
            self.memory_transfer_time_ms = 0.0
            self.io_time_ms = 0.0
            self.num_steps = 0
    
    class WeatherSimulation:
        def __init__(self, config):
            self.config = config
            self.current_grid = WeatherGrid(config.grid_width, config.grid_height)
            self.current_time = 0.0
            self.current_step = 0
            self.performance_metrics = PerformanceMetrics()
        
        def initialize(self):
            pass
        
        def step(self):
            # Mock implementation
            self.current_time += self.config.dt
            self.current_step += 1
        
        def run(self, steps):
            for _ in range(steps):
                self.step()
        
        def get_current_grid(self):
            return self.current_grid
        
        def get_current_time(self):
            return self.current_time
        
        def get_current_step(self):
            return self.current_step
        
        def get_performance_metrics(self):
            return self.performance_metrics


# High-level wrapper classes

class WeatherSimulationWrapper:
    """High-level wrapper for the Weather Simulation."""
    
    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        model: Union[str, int] = "shallow_water",
        dt: float = 0.01,
        integration_method: Union[str, int] = "rk4",
        backend: Union[str, int] = "adaptive",
        device_id: int = 0,
        threads: int = 0,
        output_interval: int = 10,
        output_path: str = "./output"
    ):
        """
        Initialize a new Weather Simulation.
        
        Args:
            width: Grid width in cells
            height: Grid height in cells
            model: Simulation model (shallow_water, barotropic, primitive, general)
            dt: Time step size
            integration_method: Integration method (euler, rk2, rk4, adams_bashforth, semi_implicit)
            backend: Compute backend (cuda, cpu, hybrid, adaptive)
            device_id: GPU device ID (if using GPU)
            threads: Number of CPU threads (0 = auto)
            output_interval: Interval between outputs
            output_path: Path for output files
        """
        self.config = SimulationConfig()
        self.config.grid_width = width
        self.config.grid_height = height
        self.config.dt = dt
        self.config.output_interval = output_interval
        self.config.output_path = output_path
        self.config.device_id = device_id
        self.config.num_threads = threads
        
        # Set model
        if isinstance(model, str):
            model_map = {
                "shallow_water": SimulationModel.ShallowWater,
                "barotropic": SimulationModel.Barotropic,
                "primitive": SimulationModel.PrimitiveEquations,
                "general": SimulationModel.General
            }
            self.config.model = model_map.get(model.lower(), SimulationModel.ShallowWater)
        else:
            self.config.model = model
        
        # Set integration method
        if isinstance(integration_method, str):
            method_map = {
                "euler": IntegrationMethod.ExplicitEuler,
                "rk2": IntegrationMethod.RungeKutta2,
                "rk4": IntegrationMethod.RungeKutta4,
                "adams_bashforth": IntegrationMethod.AdamsBashforth,
                "semi_implicit": IntegrationMethod.SemiImplicit
            }
            self.config.integration_method = method_map.get(integration_method.lower(), IntegrationMethod.RungeKutta4)
        else:
            self.config.integration_method = integration_method
        
        # Set backend
        if isinstance(backend, str):
            backend_map = {
                "cuda": ComputeBackend.CUDA,
                "cpu": ComputeBackend.CPU,
                "hybrid": ComputeBackend.Hybrid,
                "adaptive": ComputeBackend.AdaptiveHybrid
            }
            self.config.compute_backend = backend_map.get(backend.lower(), ComputeBackend.AdaptiveHybrid)
        else:
            self.config.compute_backend = backend
        
        # Create simulation
        self.simulation = WeatherSimulation(self.config)
        self.initialized = False
        
        # Output data
        self.output_data = []
    
    def set_initial_condition(self, condition_name: str, **kwargs):
        """
        Set the initial condition for the simulation.
        
        Args:
            condition_name: Name of the initial condition
            **kwargs: Parameters for the initial condition
        """
        initial_condition = create_initial_condition(condition_name, **kwargs)
        if initial_condition:
            self.simulation.set_initial_condition(initial_condition)
    
    def initialize(self):
        """Initialize the simulation."""
        self.simulation.initialize()
        self.initialized = True
    
    def step(self):
        """Perform a single time step."""
        if not self.initialized:
            self.initialize()
        
        self.simulation.step()
        
        # Store output if necessary
        if self.config.output_interval > 0 and self.simulation.get_current_step() % self.config.output_interval == 0:
            self._store_output()
    
    def run(self, steps: int):
        """
        Run the simulation for a specified number of steps.
        
        Args:
            steps: Number of steps to run
        """
        if not self.initialized:
            self.initialize()
        
        start_time = time.time()
        self.simulation.run(steps)
        end_time = time.time()
        
        # Print performance info
        elapsed = (end_time - start_time) * 1000  # ms
        print(f"Completed {steps} steps in {elapsed:.2f} ms ({elapsed / steps:.2f} ms/step)")
    
    def run_until(self, max_time: float):
        """
        Run the simulation until a specified time.
        
        Args:
            max_time: Maximum simulation time
        """
        if not self.initialized:
            self.initialize()
        
        start_time = time.time()
        self.simulation.run_until(max_time)
        end_time = time.time()
        
        # Print performance info
        steps = self.simulation.get_current_step()
        elapsed = (end_time - start_time) * 1000  # ms
        print(f"Reached time {max_time} in {elapsed:.2f} ms ({elapsed / steps:.2f} ms/step)")
    
    def get_grid(self):
        """Get the current grid."""
        return self.simulation.get_current_grid()
    
    def get_metrics(self):
        """Get performance metrics."""
        return self.simulation.get_performance_metrics()
    
    def get_output_data(self):
        """Get stored output data."""
        return self.output_data
    
    def _store_output(self):
        """Store current state for output."""
        grid = self.simulation.get_current_grid()
        time = self.simulation.get_current_time()
        step = self.simulation.get_current_step()
        
        # Create a snapshot of current state
        snapshot = {
            'time': time,
            'step': step,
            'u': grid.get_velocity_field()[0].copy(),
            'v': grid.get_velocity_field()[1].copy(),
            'height': grid.get_height_field().copy(),
            'vorticity': grid.get_vorticity_field().copy()
        }
        
        self.output_data.append(snapshot)


# Helper functions

def create_initial_condition(name: str, **kwargs) -> Optional[object]:
    """
    Create an initial condition by name with parameters.
    
    Args:
        name: Name of the initial condition
        **kwargs: Parameters for the initial condition
    
    Returns:
        Initial condition object, or None if not found
    """
    try:
        if name == "uniform":
            return UniformInitialCondition(
                kwargs.get("u", 0.0),
                kwargs.get("v", 0.0),
                kwargs.get("h", 10.0),
                kwargs.get("p", 1000.0),
                kwargs.get("t", 300.0),
                kwargs.get("q", 0.0)
            )
        elif name == "random":
            return RandomInitialCondition(
                kwargs.get("seed", 0),
                kwargs.get("amplitude", 1.0)
            )
        elif name == "zonal_flow":
            return ZonalFlowInitialCondition(
                kwargs.get("u_max", 10.0),
                kwargs.get("h_mean", 10.0),
                kwargs.get("beta", 0.1)
            )
        elif name == "vortex":
            return VortexInitialCondition(
                kwargs.get("x_center", 0.5),
                kwargs.get("y_center", 0.5),
                kwargs.get("radius", 0.1),
                kwargs.get("strength", 10.0),
                kwargs.get("h_mean", 10.0)
            )
        elif name == "jet_stream":
            return JetStreamInitialCondition(
                kwargs.get("y_center", 0.5),
                kwargs.get("width", 0.1),
                kwargs.get("strength", 10.0),
                kwargs.get("h_mean", 10.0)
            )
        elif name == "breaking_wave":
            return BreakingWaveInitialCondition(
                kwargs.get("amplitude", 1.0),
                kwargs.get("wavelength", 0.2),
                kwargs.get("h_mean", 10.0)
            )
        elif name == "front":
            return FrontInitialCondition(
                kwargs.get("y_position", 0.5),
                kwargs.get("width", 0.05),
                kwargs.get("temp_difference", 10.0),
                kwargs.get("wind_shear", 5.0)
            )
        elif name == "mountain":
            return MountainInitialCondition(
                kwargs.get("x_center", 0.3),
                kwargs.get("y_center", 0.5),
                kwargs.get("radius", 0.1),
                kwargs.get("height", 1.0),
                kwargs.get("u_base", 5.0)
            )
        elif name == "atmospheric_profile":
            return AtmosphericProfileInitialCondition(
                kwargs.get("profile_name", "standard")
            )
        else:
            # Try to use factory
            return InitialConditionFactory.get_instance().create_initial_condition(name)
    except Exception as e:
        print(f"Error creating initial condition '{name}': {e}")
        return None

def get_available_initial_conditions() -> List[str]:
    """
    Get a list of available initial conditions.
    
    Returns:
        List of initial condition names
    """
    try:
        return InitialConditionFactory.get_instance().get_available_initial_conditions()
    except Exception:
        # Fallback list if C++ library not available
        return [
            "uniform", "random", "zonal_flow", "vortex", "jet_stream",
            "breaking_wave", "front", "mountain", "atmospheric_profile"
        ]

def is_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        return AdaptiveKernelManager.get_instance().is_cuda_available()
    except Exception:
        return False

def get_device_info() -> Dict:
    """
    Get information about the compute device.
    
    Returns:
        Dictionary with device information
    """
    try:
        manager = AdaptiveKernelManager.get_instance()
        manager.initialize()
        capabilities = manager.get_device_capabilities()
        
        device_type_map = {
            DeviceType.Unknown: "Unknown",
            DeviceType.CPU: "CPU",
            DeviceType.JetsonOrinNX: "Jetson Orin NX",
            DeviceType.T4: "NVIDIA T4",
            DeviceType.HighEndGPU: "High-End GPU",
            DeviceType.OtherGPU: "Other GPU"
        }
        
        return {
            "device_type": device_type_map.get(capabilities.device_type, "Unknown"),
            "device_name": capabilities.device_name,
            "compute_capability": f"{capabilities.compute_capability_major}.{capabilities.compute_capability_minor}",
            "cuda_cores": capabilities.cuda_cores,
            "multiprocessors": capabilities.multiprocessors,
            "global_memory_mb": capabilities.global_memory / (1024 * 1024),
            "compute_power_ratio": capabilities.compute_power_ratio,
            "cuda_available": manager.is_cuda_available()
        }
    except Exception as e:
        return {
            "device_type": "Unknown",
            "device_name": "Unknown",
            "error": str(e),
            "cuda_available": False
        }