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
                 additional_metrics: Optional[Dict[str, Any]] = None):
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
            additional_metrics=data.get("additional_metrics", {})
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

class BenchmarkSuite:
    """Integrated benchmark suite for all workloads."""
    
    def __init__(self, device_id: int = 0, output_dir: str = "results"):
        """
        Initialize benchmark suite.
        
        Args:
            device_id: GPU device ID to use
            output_dir: Directory to store results
        """
        self.device_id = device_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create workload benchmarks
        self.benchmarks = {
            "nbody_sim": NBodySimbenchmark(device_id),
            "molecular_dynamics": MolecularDynamicsBenchmark(device_id),
            "weather_sim": WeatherSimulationBenchmark(device_id),
            "medical_imaging": MedicalImagingBenchmark(device_id)
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
        self.results[benchmark_name] = result
        
        # Save result to file
        self._save_result(result)
        
        return result
    
    def run_all(self, 
                nbody_params: Optional[Dict[str, Any]] = None,
                md_params: Optional[Dict[str, Any]] = None,
                weather_params: Optional[Dict[str, Any]] = None,
                medical_params: Optional[Dict[str, Any]] = None) -> Dict[str, BenchmarkResult]:
        """
        Run all benchmarks.
        
        Args:
            nbody_params: Parameters for N-body simulation benchmark
            md_params: Parameters for Molecular Dynamics benchmark
            weather_params: Parameters for Weather Simulation benchmark
            medical_params: Parameters for Medical Imaging benchmark
        
        Returns:
            Dict[str, BenchmarkResult]: Dictionary of benchmark results
        """
        nbody_params = nbody_params or {}
        md_params = md_params or {}
        weather_params = weather_params or {}
        medical_params = medical_params or {}
        
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
    
    args = parser.parse_args()
    
    # Create benchmark suite
    suite = BenchmarkSuite(device_id=args.device, output_dir=args.output)
    
    # Generate report if requested
    if args.report:
        suite.generate_reports()
        return
    
    # Run benchmarks
    if args.all or not (args.nbody or args.md or args.weather or args.medical):
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
        
        # Run all benchmarks
        suite.run_all(
            nbody_params=nbody_params,
            md_params=md_params,
            weather_params=weather_params,
            medical_params=medical_params
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
    
    # Generate report
    suite.generate_reports()

if __name__ == "__main__":
    main()