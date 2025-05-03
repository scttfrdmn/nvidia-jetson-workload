#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Integrated benchmarking suite for all GPU-accelerated scientific workloads.
Provides consistent performance metrics across different hardware configurations.
"""

import argparse
import json
import os
import time
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Union, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import visualization utilities
from benchmark.visualization import (
    plot_execution_time_comparison,
    plot_memory_usage,
    plot_gpu_utilization,
    plot_energy_consumption,
    generate_summary_report
)

class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, 
                 workload_name: str,
                 device_name: str,
                 device_capabilities: Dict[str, Any],
                 execution_time: float,
                 memory_usage: Dict[str, float],
                 gpu_utilization: Optional[float] = None,
                 energy_consumption: Optional[float] = None,
                 throughput: Optional[float] = None,
                 additional_metrics: Optional[Dict[str, Any]] = None,
                 cost_metrics: Optional[Dict[str, Any]] = None):
        """
        Initialize benchmark result.
        
        Args:
            workload_name: Name of the workload
            device_name: Name of the device
            device_capabilities: Dictionary of device capabilities
            execution_time: Execution time in seconds
            memory_usage: Dictionary with keys 'host' and 'device' (in MB)
            gpu_utilization: GPU utilization percentage (0-100)
            energy_consumption: Energy consumption in joules
            throughput: Workload-specific throughput metric
            additional_metrics: Additional workload-specific metrics
            cost_metrics: Cost metrics from various compute environments
        """
        self.workload_name = workload_name
        self.device_name = device_name
        self.device_capabilities = device_capabilities
        self.execution_time = execution_time
        self.memory_usage = memory_usage
        self.gpu_utilization = gpu_utilization
        self.energy_consumption = energy_consumption
        self.throughput = throughput
        self.additional_metrics = additional_metrics or {}
        self.cost_metrics = cost_metrics or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "workload_name": self.workload_name,
            "device_name": self.device_name,
            "device_capabilities": self.device_capabilities,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "gpu_utilization": self.gpu_utilization,
            "energy_consumption": self.energy_consumption,
            "throughput": self.throughput,
            "additional_metrics": self.additional_metrics,
            "cost_metrics": self.cost_metrics,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create result from dictionary."""
        return cls(
            workload_name=data["workload_name"],
            device_name=data["device_name"],
            device_capabilities=data["device_capabilities"],
            execution_time=data["execution_time"],
            memory_usage=data["memory_usage"],
            gpu_utilization=data.get("gpu_utilization"),
            energy_consumption=data.get("energy_consumption"),
            throughput=data.get("throughput"),
            additional_metrics=data.get("additional_metrics", {}),
            cost_metrics=data.get("cost_metrics", {})
        )
        
    def calculate_cost_metrics(self, 
                              compare_with_cloud: bool = True,
                              aws_instance_type: str = "g4dn.xlarge",
                              azure_instance_type: str = "Standard_NC4as_T4_v3",
                              gcp_instance_type: str = "n1-standard-4-t4",
                              include_dgx_spark: bool = True,
                              dgx_system_type: str = "dgx_a100",
                              dgx_quantity: int = 1,
                              dgx_config_file: Optional[str] = None,
                              include_slurm_cluster: bool = True,
                              slurm_node_type: str = "basic_gpu",
                              slurm_nodes: int = 4,
                              slurm_config_file: Optional[str] = None) -> None:
        """
        Calculate cost metrics for various compute environments.
        
        Args:
            compare_with_cloud: Whether to compare with cloud providers
            aws_instance_type: AWS instance type for comparison
            azure_instance_type: Azure instance type for comparison
            gcp_instance_type: GCP instance type for comparison
            include_dgx_spark: Whether to include DGX Spark system in comparison
            dgx_system_type: DGX system type ("dgx_a100", "dgx_h100", "dgx_station_a100", "dgx_station_h100", "dgx_superpod")
            dgx_quantity: Number of DGX systems (for non-SuperPOD systems)
            dgx_config_file: Path to DGX configuration file (overrides other DGX parameters if provided)
            include_slurm_cluster: Whether to include Slurm cluster in comparison
            slurm_node_type: Slurm node type ("basic_cpu", "basic_gpu", "highend_gpu", "jetson_cluster", "custom")
            slurm_nodes: Number of nodes in Slurm cluster
            slurm_config_file: Path to Slurm cluster configuration file (overrides other Slurm parameters if provided)
        """
        from benchmark.cost_modeling import (
            ComputeEnvironment,
            CostModelFactory,
            calculate_cost_comparison
        )
        
        # Create Jetson cost model
        jetson_model = CostModelFactory.create_model(ComputeEnvironment.LOCAL_JETSON)
        
        # Estimate Jetson cost
        jetson_cost = jetson_model.estimate_cost(
            self.execution_time,
            self.memory_usage,
            self.gpu_utilization,
            self.energy_consumption,
            self.additional_metrics
        )
        
        # Store Jetson cost
        self.cost_metrics["jetson"] = jetson_cost
        
        # Compare with cloud providers if requested
        if compare_with_cloud:
            cloud_costs = {}
            
            # AWS
            aws_model = CostModelFactory.create_model(
                ComputeEnvironment.AWS_GPU, 
                instance_type=aws_instance_type
            )
            cloud_costs["aws"] = aws_model.estimate_cost(
                self.execution_time,
                self.memory_usage,
                self.gpu_utilization,
                self.energy_consumption,
                self.additional_metrics
            )
            
            # Azure
            azure_model = CostModelFactory.create_model(
                ComputeEnvironment.AZURE_GPU, 
                instance_type=azure_instance_type
            )
            cloud_costs["azure"] = azure_model.estimate_cost(
                self.execution_time,
                self.memory_usage,
                self.gpu_utilization,
                self.energy_consumption,
                self.additional_metrics
            )
            
            # GCP
            gcp_model = CostModelFactory.create_model(
                ComputeEnvironment.GCP_GPU, 
                instance_type=gcp_instance_type
            )
            cloud_costs["gcp"] = gcp_model.estimate_cost(
                self.execution_time,
                self.memory_usage,
                self.gpu_utilization,
                self.energy_consumption,
                self.additional_metrics
            )
            
            # DGX Spark
            if include_dgx_spark:
                dgx_model = CostModelFactory.create_model(
                    ComputeEnvironment.DGX_SPARK,
                    system_type=dgx_system_type,
                    quantity=dgx_quantity,
                    config_file=dgx_config_file
                )
                cloud_costs["dgx_spark"] = dgx_model.estimate_cost(
                    self.execution_time,
                    self.memory_usage,
                    self.gpu_utilization,
                    self.energy_consumption,
                    self.additional_metrics
                )
            
            # Slurm cluster
            if include_slurm_cluster:
                slurm_model = CostModelFactory.create_model(
                    ComputeEnvironment.SLURM_CLUSTER,
                    nodes=slurm_nodes,
                    node_type=slurm_node_type,
                    config_file=slurm_config_file
                )
                cloud_costs["slurm_cluster"] = slurm_model.estimate_cost(
                    self.execution_time,
                    self.memory_usage,
                    self.gpu_utilization,
                    self.energy_consumption,
                    self.additional_metrics
                )
            
            # Store cloud costs
            self.cost_metrics["cloud"] = cloud_costs
            
            # Calculate cost comparison metrics
            self.cost_metrics["comparison"] = calculate_cost_comparison(
                jetson_cost,
                cloud_costs,
                self.workload_name,
                self.execution_time,
                self.throughput
            )

class WorkloadBenchmark:
    """Base class for workload benchmarks."""
    
    def __init__(self, name: str, device_id: int = 0):
        """
        Initialize workload benchmark.
        
        Args:
            name: Name of the workload
            device_id: GPU device ID to use
        """
        self.name = name
        self.device_id = device_id
        self.device_capabilities = self._get_device_capabilities()
        self.device_name = self._get_device_name()
    
    def _get_device_capabilities(self) -> Dict[str, Any]:
        """Get device capabilities."""
        try:
            import pycuda.driver as cuda
            cuda.init()
            device = cuda.Device(self.device_id)
            return {
                "name": device.name(),
                "compute_capability": f"{device.compute_capability()[0]}.{device.compute_capability()[1]}",
                "total_memory": device.total_memory() / (1024 ** 2),  # Convert to MB
                "clock_rate": device.clock_rate() / 1000,  # Convert to MHz
                "num_multiprocessors": device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
            }
        except (ImportError, Exception) as e:
            print(f"Warning: Could not get GPU capabilities: {e}")
            return {"name": "CPU", "compute_capability": "0.0"}
    
    def _get_device_name(self) -> str:
        """Get device name."""
        if "name" in self.device_capabilities:
            return self.device_capabilities["name"]
        
        import platform
        if platform.system() == "Darwin":
            return "Apple M1/M2"
        elif platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return line.split(":")[1].strip()
            except:
                pass
        return platform.processor() or "Unknown CPU"
    
    def _measure_memory_usage(self) -> Dict[str, float]:
        """Measure memory usage of host and device."""
        import psutil
        host_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        
        device_memory = 0.0
        try:
            import pycuda.driver as cuda
            cuda.init()
            device = cuda.Device(self.device_id)
            free_memory, total_memory = cuda.mem_get_info()
            device_memory = (total_memory - free_memory) / (1024 * 1024)  # MB
        except (ImportError, Exception):
            pass
        
        return {"host": host_memory, "device": device_memory}
    
    def _measure_gpu_utilization(self) -> Optional[float]:
        """Measure GPU utilization."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            pynvml.nvmlShutdown()
            return float(util.gpu)
        except (ImportError, Exception):
            return None
    
    def _measure_energy_consumption(self) -> Optional[float]:
        """Measure energy consumption."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            power_start = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
            yield
            power_end = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            duration = time.time() - start_time
            pynvml.nvmlShutdown()
            return (power_start + power_end) / 2.0 * duration  # Average power * duration = energy in joules
        except (ImportError, Exception):
            return None
    
    def run(self, **kwargs) -> BenchmarkResult:
        """
        Run the benchmark.
        
        Args:
            **kwargs: Additional parameters for the benchmark
        
        Returns:
            BenchmarkResult: Benchmark results
        """
        raise NotImplementedError("Subclasses must implement run()")

class NBodySimbenchmark(WorkloadBenchmark):
    """Benchmark for N-body simulation workload."""
    
    def __init__(self, device_id: int = 0):
        super().__init__("nbody_sim", device_id)
    
    def run(self, 
            num_particles: int = 10000, 
            num_steps: int = 1000, 
            dt: float = 0.01,
            system_type: str = "random",
            integrator: str = "leapfrog",
            **kwargs) -> BenchmarkResult:
        """
        Run N-body simulation benchmark.
        
        Args:
            num_particles: Number of particles
            num_steps: Number of simulation steps
            dt: Time step size
            system_type: Type of system (random, solar_system, galaxy)
            integrator: Integrator type (euler, leapfrog, verlet, rk4)
        
        Returns:
            BenchmarkResult: Benchmark results
        """
        try:
            # Import N-body simulation module
            from nbody_sim.simulation import Simulation
            from nbody_sim.integrator import IntegratorType
            
            # Create simulation
            sim = Simulation()
            sim.initialize(
                num_particles=num_particles,
                system_type=system_type,
                integrator_type=getattr(IntegratorType, integrator.upper()),
                device_id=self.device_id
            )
            
            # Measure memory before running
            memory_before = self._measure_memory_usage()
            
            # Run simulation and measure time
            start_time = time.time()
            for _ in range(num_steps):
                sim.step(dt)
            execution_time = time.time() - start_time
            
            # Measure memory after running
            memory_after = self._measure_memory_usage()
            memory_usage = {
                "host": memory_after["host"] - memory_before["host"],
                "device": memory_after["device"] - memory_before["device"]
            }
            
            # Calculate throughput (steps per second)
            throughput = num_steps / execution_time
            
            # Additional metrics
            additional_metrics = {
                "num_particles": num_particles,
                "num_steps": num_steps,
                "interactions_per_second": (num_particles ** 2) * throughput
            }
            
            # Get GPU utilization
            gpu_utilization = self._measure_gpu_utilization()
            
            # Create benchmark result
            return BenchmarkResult(
                workload_name=self.name,
                device_name=self.device_name,
                device_capabilities=self.device_capabilities,
                execution_time=execution_time,
                memory_usage=memory_usage,
                gpu_utilization=gpu_utilization,
                throughput=throughput,
                additional_metrics=additional_metrics
            )
        
        except Exception as e:
            print(f"Error running N-body simulation benchmark: {e}")
            raise

class MolecularDynamicsBenchmark(WorkloadBenchmark):
    """Benchmark for Molecular Dynamics simulation workload."""
    
    def __init__(self, device_id: int = 0):
        super().__init__("molecular_dynamics", device_id)
    
    def run(self, 
            num_atoms: int = 5000, 
            num_steps: int = 1000, 
            dt: float = 0.001,
            force_field: str = "lennard_jones",
            thermostat: str = "none",
            **kwargs) -> BenchmarkResult:
        """
        Run Molecular Dynamics simulation benchmark.
        
        Args:
            num_atoms: Number of atoms
            num_steps: Number of simulation steps
            dt: Time step size
            force_field: Type of force field (lennard_jones, coulomb)
            thermostat: Type of thermostat (none, berendsen, nose_hoover)
        
        Returns:
            BenchmarkResult: Benchmark results
        """
        try:
            # Import Molecular Dynamics module
            sys.path.append(str(project_root / "src" / "molecular-dynamics" / "python"))
            import molecular_dynamics as md
            
            # Create simulation
            sim = md.Simulation()
            sim.initialize(
                num_atoms=num_atoms,
                force_field=force_field,
                thermostat=thermostat,
                device_id=self.device_id
            )
            
            # Measure memory before running
            memory_before = self._measure_memory_usage()
            
            # Run simulation and measure time
            start_time = time.time()
            for _ in range(num_steps):
                sim.step(dt)
            execution_time = time.time() - start_time
            
            # Measure memory after running
            memory_after = self._measure_memory_usage()
            memory_usage = {
                "host": memory_after["host"] - memory_before["host"],
                "device": memory_after["device"] - memory_before["device"]
            }
            
            # Calculate throughput (steps per second)
            throughput = num_steps / execution_time
            
            # Additional metrics
            additional_metrics = {
                "num_atoms": num_atoms,
                "num_steps": num_steps,
                "interactions_per_second": (num_atoms ** 2) * throughput
            }
            
            # Get GPU utilization
            gpu_utilization = self._measure_gpu_utilization()
            
            # Create benchmark result
            return BenchmarkResult(
                workload_name=self.name,
                device_name=self.device_name,
                device_capabilities=self.device_capabilities,
                execution_time=execution_time,
                memory_usage=memory_usage,
                gpu_utilization=gpu_utilization,
                throughput=throughput,
                additional_metrics=additional_metrics
            )
        
        except Exception as e:
            print(f"Error running Molecular Dynamics benchmark: {e}")
            raise

class WeatherSimulationBenchmark(WorkloadBenchmark):
    """Benchmark for Weather Simulation workload."""
    
    def __init__(self, device_id: int = 0):
        super().__init__("weather_sim", device_id)
    
    def run(self, 
            grid_size: int = 512, 
            num_steps: int = 1000, 
            dt: float = 0.01,
            model: str = "shallow_water",
            **kwargs) -> BenchmarkResult:
        """
        Run Weather Simulation benchmark.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            num_steps: Number of simulation steps
            dt: Time step size
            model: Model type (shallow_water, barotropic, primitive)
        
        Returns:
            BenchmarkResult: Benchmark results
        """
        try:
            # Import Weather Simulation module
            sys.path.append(str(project_root / "src" / "weather-sim" / "python"))
            import weather_sim
            
            # Create simulation
            sim = weather_sim.WeatherSimulation()
            sim.initialize(
                grid_size=grid_size,
                model=model,
                device_id=self.device_id
            )
            
            # Measure memory before running
            memory_before = self._measure_memory_usage()
            
            # Run simulation and measure time
            start_time = time.time()
            for _ in range(num_steps):
                sim.step(dt)
            execution_time = time.time() - start_time
            
            # Measure memory after running
            memory_after = self._measure_memory_usage()
            memory_usage = {
                "host": memory_after["host"] - memory_before["host"],
                "device": memory_after["device"] - memory_before["device"]
            }
            
            # Calculate throughput (steps per second)
            throughput = num_steps / execution_time
            
            # Additional metrics
            additional_metrics = {
                "grid_size": grid_size,
                "num_steps": num_steps,
                "grid_points_per_second": (grid_size ** 2) * throughput
            }
            
            # Get GPU utilization
            gpu_utilization = self._measure_gpu_utilization()
            
            # Create benchmark result
            return BenchmarkResult(
                workload_name=self.name,
                device_name=self.device_name,
                device_capabilities=self.device_capabilities,
                execution_time=execution_time,
                memory_usage=memory_usage,
                gpu_utilization=gpu_utilization,
                throughput=throughput,
                additional_metrics=additional_metrics
            )
        
        except Exception as e:
            print(f"Error running Weather Simulation benchmark: {e}")
            raise

class MedicalImagingBenchmark(WorkloadBenchmark):
    """Benchmark for Medical Imaging workload."""
    
    def __init__(self, device_id: int = 0):
        super().__init__("medical_imaging", device_id)
    
    def run(self, 
            image_size: int = 512, 
            task: str = "ct_reconstruction",
            num_iterations: int = 10,
            **kwargs) -> BenchmarkResult:
        """
        Run Medical Imaging benchmark.
        
        Args:
            image_size: Size of the image (image_size x image_size)
            task: Task type (ct_reconstruction, segmentation, registration)
            num_iterations: Number of iterations
        
        Returns:
            BenchmarkResult: Benchmark results
        """
        try:
            # Import Medical Imaging module
            sys.path.append(str(project_root / "src" / "medical-imaging" / "python"))
            import medical_imaging as mi
            
            # Measure memory before running
            memory_before = self._measure_memory_usage()
            
            # Initialize adaptative kernel manager
            akm = mi.AdaptiveKernelManager.get_instance()
            akm.initialize(device_id=self.device_id)
            
            # Create test data based on task
            if task == "ct_reconstruction":
                # Create phantom
                phantom = np.zeros((image_size, image_size), dtype=np.float32)
                # Create a simple phantom with circles
                center_x, center_y = image_size // 2, image_size // 2
                for i in range(5):
                    radius = image_size // 10 * (i + 1)
                    value = 1.0 - i * 0.2
                    for y in range(image_size):
                        for x in range(image_size):
                            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                            if dist <= radius and (i == 0 or dist > image_size // 10 * i):
                                phantom[y, x] = value
                
                # Convert to MedicalImage
                phantom_img = mi.from_numpy(phantom)
                
                # Create CT reconstructor
                reconstructor = mi.CTReconstructor()
                reconstructor.set_image_dimensions(image_size, image_size)
                reconstructor.set_angles(np.linspace(0, np.pi, 180, dtype=np.float32))
                
                # Forward projection to create sinogram
                sinogram = reconstructor.forward_project(phantom_img)
                
                # Run benchmark
                start_time = time.time()
                for _ in range(num_iterations):
                    reconstructor.set_method(mi.ReconstructionMethod.FilteredBackProjection)
                    result = reconstructor.process(sinogram)
                execution_time = time.time() - start_time
                
                # Additional metrics
                additional_metrics = {
                    "image_size": image_size,
                    "num_iterations": num_iterations,
                    "num_angles": 180,
                    "method": "FilteredBackProjection"
                }
                
            elif task == "segmentation":
                # Create test image
                image = np.zeros((image_size, image_size), dtype=np.float32)
                # Create a simple image with shapes
                center_x, center_y = image_size // 2, image_size // 2
                for i in range(5):
                    radius = image_size // 10 * (i + 1)
                    value = 1.0 - i * 0.2
                    for y in range(image_size):
                        for x in range(image_size):
                            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                            if dist <= radius and (i == 0 or dist > image_size // 10 * i):
                                image[y, x] = value
                
                # Convert to MedicalImage
                image_obj = mi.from_numpy(image)
                
                # Create segmenter
                segmenter = mi.Segmenter()
                
                # Run benchmark
                start_time = time.time()
                for _ in range(num_iterations):
                    # Try different segmentation methods
                    segmenter.set_method(mi.SegmentationMethod.Thresholding)
                    segmenter.set_threshold(0.5)
                    result1 = segmenter.process(image_obj)
                    
                    segmenter.set_method(mi.SegmentationMethod.Watershed)
                    result2 = segmenter.process(image_obj)
                execution_time = time.time() - start_time
                
                # Additional metrics
                additional_metrics = {
                    "image_size": image_size,
                    "num_iterations": num_iterations,
                    "methods": ["Thresholding", "Watershed"]
                }
                
            elif task == "registration":
                # Create fixed image
                fixed_image = np.zeros((image_size, image_size), dtype=np.float32)
                # Create a simple image with circles
                center_x, center_y = image_size // 2, image_size // 2
                for i in range(5):
                    radius = image_size // 10 * (i + 1)
                    value = 1.0 - i * 0.2
                    for y in range(image_size):
                        for x in range(image_size):
                            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                            if dist <= radius and (i == 0 or dist > image_size // 10 * i):
                                fixed_image[y, x] = value
                
                # Create moving image (shifted and rotated)
                moving_image = np.zeros((image_size, image_size), dtype=np.float32)
                shift_x, shift_y = 10, 5
                for y in range(image_size):
                    for x in range(image_size):
                        # Apply a simple transformation (shift)
                        src_x = x - shift_x
                        src_y = y - shift_y
                        if 0 <= src_x < image_size and 0 <= src_y < image_size:
                            moving_image[y, x] = fixed_image[src_y, src_x]
                
                # Convert to MedicalImage
                fixed_img = mi.from_numpy(fixed_image)
                moving_img = mi.from_numpy(moving_image)
                
                # Create registrator
                registrator = mi.Registrator()
                registrator.set_method(mi.RegistrationMethod.Rigid)
                
                # Run benchmark
                start_time = time.time()
                for _ in range(num_iterations):
                    result = registrator.register(fixed_img, moving_img)
                execution_time = time.time() - start_time
                
                # Additional metrics
                additional_metrics = {
                    "image_size": image_size,
                    "num_iterations": num_iterations,
                    "method": "Rigid"
                }
            
            else:
                raise ValueError(f"Unknown task: {task}")
            
            # Measure memory after running
            memory_after = self._measure_memory_usage()
            memory_usage = {
                "host": memory_after["host"] - memory_before["host"],
                "device": memory_after["device"] - memory_before["device"]
            }
            
            # Calculate throughput (iterations per second)
            throughput = num_iterations / execution_time
            
            # Get GPU utilization
            gpu_utilization = self._measure_gpu_utilization()
            
            # Create benchmark result
            return BenchmarkResult(
                workload_name=f"{self.name}_{task}",
                device_name=self.device_name,
                device_capabilities=self.device_capabilities,
                execution_time=execution_time,
                memory_usage=memory_usage,
                gpu_utilization=gpu_utilization,
                throughput=throughput,
                additional_metrics=additional_metrics
            )
        
        except Exception as e:
            print(f"Error running Medical Imaging benchmark: {e}")
            raise

class GeospatialBenchmark(WorkloadBenchmark):
    """Benchmark for Geospatial Analysis workload."""
    
    def __init__(self, device_id: int = 0):
        super().__init__("geospatial", device_id)
    
    def run(self, 
            dataset_type: str = "dem", 
            operation: str = "viewshed",
            data_size: int = 1024,
            **kwargs) -> BenchmarkResult:
        """
        Run Geospatial Analysis benchmark.
        
        Args:
            dataset_type: Type of dataset (dem, point_cloud, raster, vector)
            operation: Operation to perform (viewshed, terrain_derivatives, etc.)
            data_size: Size of the dataset (pixels or points)
            **kwargs: Additional parameters for the benchmark
            
        Returns:
            BenchmarkResult: Benchmark results
        """
        try:
            # Import Geospatial module
            sys.path.append(str(project_root / "src" / "geospatial" / "python"))
            import geospatial
            
            # Measure memory before running
            memory_before = self._measure_memory_usage()
            
            if dataset_type == "dem" and operation == "viewshed":
                # Create a synthetic DEM for benchmarking
                dem_file = kwargs.get("dem_file")
                
                # If no DEM file is provided, create a synthetic one
                if not dem_file:
                    dem_file = self._create_synthetic_dem(data_size)
                
                # Create DEM processor
                dem_processor = geospatial.DEMProcessor(dem_file, device_id=self.device_id)
                
                # Observer position at center of DEM
                width, height = dem_processor.get_dimensions()
                observer_point = (width // 2, height // 2)
                observer_height = kwargs.get("observer_height", 1.8)
                radius = kwargs.get("radius", 0.0)
                
                # Run benchmark
                start_time = time.time()
                viewshed = dem_processor.compute_viewshed(observer_point, observer_height, radius)
                execution_time = time.time() - start_time
                
                # Additional metrics
                additional_metrics = {
                    "dem_size": f"{width}x{height}",
                    "observer_height": observer_height,
                    "radius": radius
                }
                
                # Calculate throughput (pixels per second)
                throughput = (width * height) / execution_time
                
            elif dataset_type == "dem" and operation == "terrain_derivatives":
                # Create a synthetic DEM for benchmarking
                dem_file = kwargs.get("dem_file")
                
                # If no DEM file is provided, create a synthetic one
                if not dem_file:
                    dem_file = self._create_synthetic_dem(data_size)
                
                # Create DEM processor
                dem_processor = geospatial.DEMProcessor(dem_file, device_id=self.device_id)
                
                # Run benchmark
                start_time = time.time()
                terrain = dem_processor.compute_terrain_derivatives(z_factor=kwargs.get("z_factor", 1.0))
                execution_time = time.time() - start_time
                
                # Additional metrics
                width, height = dem_processor.get_dimensions()
                additional_metrics = {
                    "dem_size": f"{width}x{height}",
                    "z_factor": kwargs.get("z_factor", 1.0)
                }
                
                # Calculate throughput (pixels per second)
                throughput = (width * height) / execution_time
                
            elif dataset_type == "point_cloud" and operation == "classification":
                # Point cloud benchmarking
                # This requires a point cloud file, create synthetic one if not provided
                point_cloud_file = kwargs.get("point_cloud_file")
                
                if not point_cloud_file:
                    # For now we just report a simulated result
                    # In a real implementation, create a synthetic point cloud file
                    execution_time = data_size / 1e6  # Simulate processing time
                    throughput = data_size / execution_time
                    additional_metrics = {
                        "num_points": data_size,
                        "simulated": True
                    }
                else:
                    # Create point cloud processor
                    point_cloud = geospatial.PointCloud(point_cloud_file, device_id=self.device_id)
                    
                    # Run benchmark
                    start_time = time.time()
                    classified = point_cloud.classify_points()
                    execution_time = time.time() - start_time
                    
                    # Additional metrics
                    num_points = point_cloud.get_num_points()
                    additional_metrics = {
                        "num_points": num_points,
                        "simulated": False
                    }
                    
                    # Calculate throughput (points per second)
                    throughput = num_points / execution_time
            
            else:
                raise ValueError(f"Unsupported benchmark: {dataset_type} - {operation}")
            
            # Measure memory after running
            memory_after = self._measure_memory_usage()
            memory_usage = {
                "host": memory_after["host"] - memory_before["host"],
                "device": memory_after["device"] - memory_before["device"]
            }
            
            # Get GPU utilization
            gpu_utilization = self._measure_gpu_utilization()
            
            # Create benchmark result
            return BenchmarkResult(
                workload_name=f"{self.name}_{dataset_type}_{operation}",
                device_name=self.device_name,
                device_capabilities=self.device_capabilities,
                execution_time=execution_time,
                memory_usage=memory_usage,
                gpu_utilization=gpu_utilization,
                throughput=throughput,
                additional_metrics=additional_metrics
            )
        
        except Exception as e:
            print(f"Error running Geospatial benchmark: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_synthetic_dem(self, size: int) -> str:
        """Create a synthetic DEM for benchmarking."""
        import tempfile
        import numpy as np
        from pathlib import Path
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            temp_file = f.name
        
        # Create a simple DEM (sinusoidal terrain)
        x = np.linspace(0, 10, size)
        y = np.linspace(0, 10, size)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y) * 100 + 1000  # Elevation in meters
        
        # Save the DEM to the temporary file
        np.save(temp_file, Z)
        
        return temp_file


class BenchmarkSuite:
    """Integrated benchmark suite for all workloads."""
    
    def __init__(self, 
                device_id: int = 0, 
                output_dir: str = "results",
                enable_cost_modeling: bool = False,
                aws_instance_type: str = "g4dn.xlarge",
                azure_instance_type: str = "Standard_NC4as_T4_v3",
                gcp_instance_type: str = "n1-standard-4-t4",
                # DGX Spark configuration
                include_dgx_spark: bool = True,
                dgx_system_type: str = "dgx_a100",
                dgx_quantity: int = 1,
                dgx_config_file: Optional[str] = None,
                # Slurm cluster configuration
                include_slurm_cluster: bool = True,
                slurm_node_type: str = "basic_gpu",
                slurm_nodes: int = 4,
                slurm_config_file: Optional[str] = None):
        """
        Initialize benchmark suite.
        
        Args:
            device_id: GPU device ID to use
            output_dir: Directory to store results
            enable_cost_modeling: Whether to enable cost modeling
            
            # Cloud provider configuration
            aws_instance_type: AWS instance type for comparison
            azure_instance_type: Azure instance type for comparison
            gcp_instance_type: GCP instance type for comparison
            
            # DGX Spark configuration
            include_dgx_spark: Whether to include DGX Spark system in comparison
            dgx_system_type: DGX system type ("dgx_a100", "dgx_h100", "dgx_station_a100", "dgx_station_h100", "dgx_superpod")
            dgx_quantity: Number of DGX systems (for non-SuperPOD systems)
            dgx_config_file: Path to DGX configuration file (overrides other DGX parameters if provided)
            
            # Slurm cluster configuration
            include_slurm_cluster: Whether to include Slurm cluster in comparison
            slurm_node_type: Slurm node type ("basic_cpu", "basic_gpu", "highend_gpu", "jetson_cluster", "custom")
            slurm_nodes: Number of nodes in Slurm cluster
            slurm_config_file: Path to Slurm cluster configuration file (overrides other Slurm parameters if provided)
        """
        self.device_id = device_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cost modeling configuration
        self.enable_cost_modeling = enable_cost_modeling
        
        # Cloud provider configuration
        self.aws_instance_type = aws_instance_type
        self.azure_instance_type = azure_instance_type
        self.gcp_instance_type = gcp_instance_type
        
        # DGX Spark configuration
        self.include_dgx_spark = include_dgx_spark
        self.dgx_system_type = dgx_system_type
        self.dgx_quantity = dgx_quantity
        self.dgx_config_file = dgx_config_file
        
        # Slurm cluster configuration
        self.include_slurm_cluster = include_slurm_cluster
        self.slurm_node_type = slurm_node_type
        self.slurm_nodes = slurm_nodes
        self.slurm_config_file = slurm_config_file
        
        # Create workload benchmarks
        self.benchmarks = {
            "nbody_sim": NBodySimbenchmark(device_id),
            "molecular_dynamics": MolecularDynamicsBenchmark(device_id),
            "weather_sim": WeatherSimulationBenchmark(device_id),
            "medical_imaging": MedicalImagingBenchmark(device_id),
            "geospatial": GeospatialBenchmark(device_id)
        }
        
        # Results storage
        self.results = {}
    
    def run_benchmark(self, benchmark_name: str, **kwargs) -> BenchmarkResult:
        """
        Run a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark to run
            **kwargs: Additional parameters for the benchmark
        
        Returns:
            BenchmarkResult: Benchmark results
        """
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        print(f"Running {benchmark_name} benchmark...")
        result = self.benchmarks[benchmark_name].run(**kwargs)
        
        # Calculate cost metrics if enabled
        if self.enable_cost_modeling:
            print(f"Calculating cost metrics for {benchmark_name}...")
            result.calculate_cost_metrics(
                compare_with_cloud=True,
                aws_instance_type=self.aws_instance_type,
                azure_instance_type=self.azure_instance_type,
                gcp_instance_type=self.gcp_instance_type,
                # DGX Spark configuration
                include_dgx_spark=self.include_dgx_spark,
                dgx_system_type=self.dgx_system_type,
                dgx_quantity=self.dgx_quantity,
                dgx_config_file=self.dgx_config_file,
                # Slurm cluster configuration
                include_slurm_cluster=self.include_slurm_cluster,
                slurm_node_type=self.slurm_node_type,
                slurm_nodes=self.slurm_nodes,
                slurm_config_file=self.slurm_config_file
            )
        
        self.results[benchmark_name] = result
        
        # Save result to file
        self._save_result(result)
        
        return result
    
    def run_all(self, 
                nbody_params: Optional[Dict[str, Any]] = None,
                md_params: Optional[Dict[str, Any]] = None,
                weather_params: Optional[Dict[str, Any]] = None,
                medical_params: Optional[Dict[str, Any]] = None,
                geospatial_params: Optional[Dict[str, Any]] = None) -> Dict[str, BenchmarkResult]:
        """
        Run all benchmarks.
        
        Args:
            nbody_params: Parameters for N-body simulation benchmark
            md_params: Parameters for Molecular Dynamics benchmark
            weather_params: Parameters for Weather Simulation benchmark
            medical_params: Parameters for Medical Imaging benchmark
            geospatial_params: Parameters for Geospatial Analysis benchmark
        
        Returns:
            Dict[str, BenchmarkResult]: Dictionary of benchmark results
        """
        nbody_params = nbody_params or {}
        md_params = md_params or {}
        weather_params = weather_params or {}
        medical_params = medical_params or {}
        geospatial_params = geospatial_params or {}
        
        print("Running all benchmarks...")
        
        try:
            self.run_benchmark("nbody_sim", **nbody_params)
        except Exception as e:
            print(f"Error running N-body simulation benchmark: {e}")
        
        try:
            self.run_benchmark("molecular_dynamics", **md_params)
        except Exception as e:
            print(f"Error running Molecular Dynamics benchmark: {e}")
        
        try:
            self.run_benchmark("weather_sim", **weather_params)
        except Exception as e:
            print(f"Error running Weather Simulation benchmark: {e}")
        
        try:
            # Run Medical Imaging benchmarks for different tasks
            for task in ["ct_reconstruction", "segmentation", "registration"]:
                params = medical_params.copy()
                params["task"] = task
                self.run_benchmark("medical_imaging", **params)
        except Exception as e:
            print(f"Error running Medical Imaging benchmark: {e}")
            
        try:
            # Run Geospatial benchmarks for different operations
            dataset_type = geospatial_params.get("dataset_type", "dem")
            
            if dataset_type == "dem":
                for operation in ["viewshed", "terrain_derivatives"]:
                    params = geospatial_params.copy()
                    params["operation"] = operation
                    self.run_benchmark("geospatial", **params)
            elif dataset_type == "point_cloud":
                params = geospatial_params.copy()
                params["operation"] = "classification"
                self.run_benchmark("geospatial", **params)
        except Exception as e:
            print(f"Error running Geospatial benchmark: {e}")
        
        return self.results
    
    def _save_result(self, result: BenchmarkResult) -> None:
        """
        Save benchmark result to file.
        
        Args:
            result: Benchmark result to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.workload_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"Saved result to {filepath}")
    
    def load_results(self, directory: Optional[str] = None) -> Dict[str, List[BenchmarkResult]]:
        """
        Load benchmark results from files.
        
        Args:
            directory: Directory to load results from (default: self.output_dir)
        
        Returns:
            Dict[str, List[BenchmarkResult]]: Dictionary of benchmark results
        """
        directory = directory or self.output_dir
        results = {}
        
        for file in Path(directory).glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    result = BenchmarkResult.from_dict(data)
                    
                    if result.workload_name not in results:
                        results[result.workload_name] = []
                    
                    results[result.workload_name].append(result)
            except Exception as e:
                print(f"Error loading result from {file}: {e}")
        
        return results
    
    def generate_reports(self, 
                        directory: Optional[str] = None,
                        output_file: Optional[str] = None) -> None:
        """
        Generate reports from benchmark results.
        
        Args:
            directory: Directory to load results from (default: self.output_dir)
            output_file: Output file for the report (default: benchmark_report.html)
        """
        # Load results
        results = self.load_results(directory)
        
        if not results:
            print("No benchmark results found.")
            return
        
        # Set output file
        output_file = output_file or (self.output_dir / "benchmark_report.html")
        
        # Generate report
        generate_summary_report(results, output_file)
        
        print(f"Generated report: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Integrated benchmarking suite for GPU-accelerated scientific workloads")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--output", type=str, default="results", help="Directory to store results")
    parser.add_argument("--report", action="store_true", help="Generate report from existing results")
    
    # Workload selection
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--nbody", action="store_true", help="Run N-body simulation benchmark")
    parser.add_argument("--md", action="store_true", help="Run Molecular Dynamics benchmark")
    parser.add_argument("--weather", action="store_true", help="Run Weather Simulation benchmark")
    parser.add_argument("--medical", action="store_true", help="Run Medical Imaging benchmark")
    parser.add_argument("--geospatial", action="store_true", help="Run Geospatial Analysis benchmark")
    
    # N-body parameters
    parser.add_argument("--nbody-particles", type=int, default=10000, help="Number of particles for N-body simulation")
    parser.add_argument("--nbody-steps", type=int, default=1000, help="Number of steps for N-body simulation")
    parser.add_argument("--nbody-system", type=str, default="random", choices=["random", "solar_system", "galaxy"], help="System type for N-body simulation")
    parser.add_argument("--nbody-integrator", type=str, default="leapfrog", choices=["euler", "leapfrog", "verlet", "rk4"], help="Integrator for N-body simulation")
    
    # Molecular Dynamics parameters
    parser.add_argument("--md-atoms", type=int, default=5000, help="Number of atoms for Molecular Dynamics")
    parser.add_argument("--md-steps", type=int, default=1000, help="Number of steps for Molecular Dynamics")
    parser.add_argument("--md-forcefield", type=str, default="lennard_jones", choices=["lennard_jones", "coulomb"], help="Force field for Molecular Dynamics")
    
    # Weather Simulation parameters
    parser.add_argument("--weather-grid", type=int, default=512, help="Grid size for Weather Simulation")
    parser.add_argument("--weather-steps", type=int, default=1000, help="Number of steps for Weather Simulation")
    parser.add_argument("--weather-model", type=str, default="shallow_water", choices=["shallow_water", "barotropic", "primitive"], help="Model for Weather Simulation")
    
    # Medical Imaging parameters
    parser.add_argument("--medical-size", type=int, default=512, help="Image size for Medical Imaging")
    parser.add_argument("--medical-task", type=str, default="ct_reconstruction", choices=["ct_reconstruction", "segmentation", "registration"], help="Task for Medical Imaging")
    parser.add_argument("--medical-iterations", type=int, default=10, help="Number of iterations for Medical Imaging")
    
    # Geospatial parameters
    parser.add_argument("--geo-dataset", type=str, default="dem", choices=["dem", "point_cloud", "raster", "vector"], help="Dataset type for Geospatial Analysis")
    parser.add_argument("--geo-operation", type=str, default="viewshed", choices=["viewshed", "terrain_derivatives", "classification"], help="Operation for Geospatial Analysis")
    parser.add_argument("--geo-size", type=int, default=1024, help="Size of dataset (pixels or points) for Geospatial Analysis")
    parser.add_argument("--geo-file", type=str, default="", help="Path to dataset file (optional, will create synthetic if not provided)")
    parser.add_argument("--geo-height", type=float, default=1.8, help="Observer height for viewshed analysis (meters)")
    
    # Cost modeling parameters
    cost_group = parser.add_argument_group('Cost Modeling', 'Settings for cost comparison analysis')
    cost_group.add_argument("--cost-analysis", action="store_true", help="Enable cost modeling and comparison")
    
    # Cloud provider options
    cost_group.add_argument("--aws-instance", type=str, default="g4dn.xlarge", 
                          choices=["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge", "p3.2xlarge", "g3s.xlarge"],
                          help="AWS instance type for cost comparison")
    cost_group.add_argument("--azure-instance", type=str, default="Standard_NC4as_T4_v3",
                          choices=["Standard_NC4as_T4_v3", "Standard_NC6s_v3", "Standard_ND96asr_A100_v4"],
                          help="Azure instance type for cost comparison")
    cost_group.add_argument("--gcp-instance", type=str, default="n1-standard-4-t4",
                          choices=["n1-standard-4-t4", "n1-standard-8-v100", "a2-highgpu-1g"],
                          help="GCP instance type for cost comparison")
                          
    # DGX Spark options
    cost_group.add_argument("--no-dgx-spark", action="store_true", help="Exclude DGX Spark from cost comparison")
    cost_group.add_argument("--dgx-system-type", type=str, default="dgx_a100",
                          choices=["dgx_a100", "dgx_h100", "dgx_station_a100", "dgx_station_h100", "dgx_superpod"],
                          help="DGX system type for cost comparison")
    cost_group.add_argument("--dgx-quantity", type=int, default=1, 
                          help="Number of DGX systems (for non-SuperPOD systems)")
    cost_group.add_argument("--dgx-config", type=str, default=None,
                          help="Path to DGX configuration file (overrides other DGX parameters if provided)")
                          
    # Slurm cluster options
    cost_group.add_argument("--no-slurm-cluster", action="store_true", help="Exclude Slurm cluster from cost comparison")
    cost_group.add_argument("--slurm-node-type", type=str, default="basic_gpu",
                          choices=["basic_cpu", "basic_gpu", "highend_gpu", "jetson_cluster", "custom"],
                          help="Slurm node type for cost comparison")
    cost_group.add_argument("--slurm-nodes", type=int, default=4, 
                          help="Number of nodes in Slurm cluster for cost comparison")
    cost_group.add_argument("--slurm-config", type=str, default=None,
                          help="Path to Slurm cluster configuration file (overrides other Slurm parameters if provided)")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    suite = BenchmarkSuite(
        device_id=args.device, 
        output_dir=args.output,
        enable_cost_modeling=args.cost_analysis,
        # Cloud provider configuration
        aws_instance_type=args.aws_instance,
        azure_instance_type=args.azure_instance,
        gcp_instance_type=args.gcp_instance,
        # DGX Spark configuration
        include_dgx_spark=not args.no_dgx_spark,
        dgx_system_type=args.dgx_system_type,
        dgx_quantity=args.dgx_quantity,
        dgx_config_file=args.dgx_config,
        # Slurm cluster configuration
        include_slurm_cluster=not args.no_slurm_cluster,
        slurm_node_type=args.slurm_node_type,
        slurm_nodes=args.slurm_nodes,
        slurm_config_file=args.slurm_config
    )
    
    # Generate report if requested
    if args.report:
        suite.generate_reports()
        return
    
    # Run benchmarks
    if args.all or not (args.nbody or args.md or args.weather or args.medical or args.geospatial):
        # Prepare parameters
        nbody_params = {
            "num_particles": args.nbody_particles,
            "num_steps": args.nbody_steps,
            "system_type": args.nbody_system,
            "integrator": args.nbody_integrator
        }
        
        md_params = {
            "num_atoms": args.md_atoms,
            "num_steps": args.md_steps,
            "force_field": args.md_forcefield
        }
        
        weather_params = {
            "grid_size": args.weather_grid,
            "num_steps": args.weather_steps,
            "model": args.weather_model
        }
        
        medical_params = {
            "image_size": args.medical_size,
            "task": args.medical_task,
            "num_iterations": args.medical_iterations
        }
        
        geospatial_params = {
            "dataset_type": args.geo_dataset,
            "operation": args.geo_operation,
            "data_size": args.geo_size,
            "dem_file": args.geo_file if args.geo_dataset == "dem" else None,
            "point_cloud_file": args.geo_file if args.geo_dataset == "point_cloud" else None,
            "observer_height": args.geo_height
        }
        
        # Run all benchmarks
        suite.run_all(
            nbody_params=nbody_params,
            md_params=md_params,
            weather_params=weather_params,
            medical_params=medical_params,
            geospatial_params=geospatial_params
        )
    else:
        # Run individual benchmarks
        if args.nbody:
            suite.run_benchmark("nbody_sim", 
                               num_particles=args.nbody_particles,
                               num_steps=args.nbody_steps,
                               system_type=args.nbody_system,
                               integrator=args.nbody_integrator)
        
        if args.md:
            suite.run_benchmark("molecular_dynamics",
                               num_atoms=args.md_atoms,
                               num_steps=args.md_steps,
                               force_field=args.md_forcefield)
        
        if args.weather:
            suite.run_benchmark("weather_sim",
                               grid_size=args.weather_grid,
                               num_steps=args.weather_steps,
                               model=args.weather_model)
        
        if args.medical:
            suite.run_benchmark("medical_imaging",
                               image_size=args.medical_size,
                               task=args.medical_task,
                               num_iterations=args.medical_iterations)
        
        if args.geospatial:
            # Check if advanced geospatial benchmarks are available
            advanced_geospatial = False
            try:
                # Try to import advanced geospatial benchmark modules
                sys.path.append(str(project_root / "src" / "geospatial"))
                from benchmark.geospatial_benchmark import GeospatialBenchmarkSuite
                from benchmark.datasets import TerrainType
                advanced_geospatial = True
            except ImportError:
                # Fall back to basic benchmark
                print("Advanced geospatial benchmarks not available, using basic benchmark...")
                advanced_geospatial = False
            
            if advanced_geospatial:
                # Use advanced geospatial benchmarks via adapter
                from benchmark.scripts.run_geospatial_benchmark import GeospatialBenchmarkAdapter
                
                print("Running advanced geospatial benchmarks...")
                adapter = GeospatialBenchmarkAdapter(
                    device_id=args.device,
                    output_dir=args.output,
                    enable_cost_modeling=args.cost_analysis,
                    aws_instance_type=args.aws_instance,
                    azure_instance_type=args.azure_instance,
                    gcp_instance_type=args.gcp_instance
                )
                adapter.set_main_suite(suite)
                
                # Map dataset type to terrain type
                terrain_map = {
                    "dem": "rolling_hills",
                    "point_cloud": "uniform",
                    "raster": "flat",
                    "vector": "random"
                }
                
                # Map operation to DEM size
                size_map = {
                    "viewshed": "medium",
                    "terrain_derivatives": "medium",
                    "classification": "medium"
                }
                
                # Run geospatial benchmarks
                geospatial_results = adapter.run_benchmarks(
                    dem_size=size_map.get(args.geo_operation, "medium"),
                    dem_type=terrain_map.get(args.geo_dataset, "rolling_hills"),
                    pc_size="medium",
                    pc_density="uniform"
                )
                
                # Add results to main suite
                for name, result in geospatial_results.items():
                    suite.results[name] = result
            else:
                # Use basic benchmark
                suite.run_benchmark("geospatial",
                                  dataset_type=args.geo_dataset,
                                  operation=args.geo_operation,
                                  data_size=args.geo_size,
                                  dem_file=args.geo_file if args.geo_dataset == "dem" else None,
                                  point_cloud_file=args.geo_file if args.geo_dataset == "point_cloud" else None,
                                  observer_height=args.geo_height)
    
    # Generate report
    suite.generate_reports()

if __name__ == "__main__":
    main()