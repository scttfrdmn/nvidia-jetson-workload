#!/usr/bin/env python3
"""
Comprehensive benchmark script for the Geospatial Analysis workload.

This script benchmarks the performance of DEM processing, point cloud processing,
and other geospatial operations on different hardware configurations and collects
detailed performance metrics for analysis.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import tempfile
import shutil
import platform
import subprocess
import multiprocessing
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from contextlib import contextmanager

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

# Import geospatial modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))
import geospatial
from geospatial.dem import DEMProcessor
from geospatial.point_cloud import PointCloud, PointClass

# Import dataset utilities
from datasets import (
    create_synthetic_dem,
    create_synthetic_point_cloud,
    create_standard_benchmark_datasets,
    TerrainType,
    PointCloudDensity,
    clear_datasets
)

# Import benchmarking utilities from main benchmark suite
benchmark_dir = os.path.join(project_root, "benchmark")
sys.path.append(benchmark_dir)

try:
    from benchmark.cost_modeling import (
        ComputeEnvironment,
        CostModelFactory,
        calculate_cost_comparison
    )
    COST_MODELING_AVAILABLE = True
except ImportError:
    COST_MODELING_AVAILABLE = False
    print("Warning: Cost modeling module not available. Cost comparisons will be disabled.")

# Context manager for timing code execution
@contextmanager
def timer(operation=None, silent=False):
    """
    Context manager to time code execution.
    
    Args:
        operation: Name of the operation being timed
        silent: Whether to suppress output
        
    Yields:
        Start time of the operation
        
    Returns:
        Execution time in seconds
    """
    start = time.time()
    yield start
    end = time.time()
    execution_time = end - start
    
    if operation and not silent:
        print(f"{operation}: {execution_time:.4f} seconds")
    
    return execution_time

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary with memory usage in MB
    """
    import psutil
    process = psutil.Process(os.getpid())
    host_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Try to get GPU memory usage
    device_memory = 0.0
    try:
        import pycuda.driver as cuda
        cuda.init()
        device_count = cuda.Device.count()
        
        if device_count > 0:
            device = cuda.Device(0)
            context = device.make_context()
            free_memory, total_memory = cuda.mem_get_info()
            device_memory = (total_memory - free_memory) / (1024 * 1024)  # MB
            context.pop()
    except (ImportError, Exception):
        pass
    
    return {"host": host_memory, "device": device_memory}

def get_gpu_utilization(device_id: int = 0) -> Optional[float]:
    """
    Get GPU utilization percentage.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        GPU utilization as a percentage (0-100) or None if not available
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return float(util.gpu)
    except (ImportError, Exception):
        return None

def get_system_info() -> Dict[str, Any]:
    """
    Get detailed system information.
    
    Returns:
        Dictionary with system information
    """
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": multiprocessing.cpu_count(),
        "memory": {}
    }
    
    # Get memory information
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["memory"] = {
            "total": vm.total / (1024 * 1024),  # MB
            "available": vm.available / (1024 * 1024),  # MB
            "used": vm.used / (1024 * 1024),  # MB
            "percent": vm.percent
        }
    except ImportError:
        pass
    
    # Get GPU information
    info["gpu"] = {}
    try:
        import pycuda.driver as cuda
        cuda.init()
        device_count = cuda.Device.count()
        
        if device_count > 0:
            info["gpu"]["device_count"] = device_count
            info["gpu"]["devices"] = []
            
            for i in range(device_count):
                device = cuda.Device(i)
                device_info = {
                    "name": device.name(),
                    "compute_capability": f"{device.compute_capability()[0]}.{device.compute_capability()[1]}",
                    "total_memory": device.total_memory() / (1024 * 1024),  # MB
                    "clock_rate": device.clock_rate() / 1000  # MHz
                }
                info["gpu"]["devices"].append(device_info)
        else:
            info["gpu"]["device_count"] = 0
    except (ImportError, Exception):
        info["gpu"]["device_count"] = 0
    
    return info

class Benchmark:
    """Base class for benchmarks."""
    
    def __init__(self, name: str, device_id: int = 0):
        """
        Initialize benchmark.
        
        Args:
            name: Name of the benchmark
            device_id: GPU device ID to use
        """
        self.name = name
        self.device_id = device_id
        self.results = {}
        
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the benchmark.
        
        Args:
            **kwargs: Additional parameters for the benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        raise NotImplementedError("Subclasses must implement run()")

class DEMBenchmark(Benchmark):
    """Benchmark for DEM processing operations."""
    
    def __init__(self, device_id: int = 0):
        super().__init__("dem_processing", device_id)
        
    def run(self, dem_path: str, **kwargs) -> Dict[str, Any]:
        """
        Run DEM processing benchmark.
        
        Args:
            dem_path: Path to DEM file
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "name": self.name,
            "params": {
                "dem_path": dem_path,
                **kwargs
            },
            "operations": {},
            "memory": {},
            "throughput": {}
        }
        
        # Get initial memory usage
        memory_before = get_memory_usage()
        
        # Load DEM
        print(f"Loading DEM from {dem_path}...")
        with timer("DEM loading") as elapsed:
            dem_proc = DEMProcessor(dem_path, self.device_id)
        results["operations"]["loading"] = time.time() - elapsed
        
        # Get DEM dimensions
        width, height = dem_proc.get_dimensions()
        print(f"DEM dimensions: {width}x{height}")
        results["params"]["dimensions"] = [width, height]
        
        # Run viewshed analysis
        print("Running viewshed analysis...")
        observer_point = (width/2, height/2)  # Center point
        observer_height = 10.0
        radius = width / 4
        
        with timer("Viewshed analysis") as elapsed:
            viewshed = dem_proc.compute_viewshed(observer_point, observer_height, radius)
        results["operations"]["viewshed"] = time.time() - elapsed
        results["throughput"]["viewshed"] = (width * height) / results["operations"]["viewshed"]
        
        # Run terrain derivatives computation
        print("Computing terrain derivatives...")
        with timer("Terrain derivatives") as elapsed:
            terrain = dem_proc.compute_terrain_derivatives(z_factor=1.0)
        results["operations"]["terrain_derivatives"] = time.time() - elapsed
        results["throughput"]["terrain_derivatives"] = (width * height) / results["operations"]["terrain_derivatives"]
        
        # Run hydrological features computation
        print("Computing hydrological features...")
        with timer("Hydrological features") as elapsed:
            hydro = dem_proc.compute_hydrological_features()
        results["operations"]["hydrological_features"] = time.time() - elapsed
        results["throughput"]["hydrological_features"] = (width * height) / results["operations"]["hydrological_features"]
        
        # Run least cost path computation
        print("Computing least cost path...")
        start_point = (width/4, height/4)
        end_point = (3*width/4, 3*height/4)
        
        with timer("Least cost path") as elapsed:
            path = dem_proc.compute_least_cost_path(start_point, end_point)
        results["operations"]["least_cost_path"] = time.time() - elapsed
        results["throughput"]["least_cost_path"] = (width * height) / results["operations"]["least_cost_path"]
        
        # Run statistics computation
        print("Computing statistics...")
        with timer("Statistics") as elapsed:
            stats = dem_proc.compute_statistics()
        results["operations"]["statistics"] = time.time() - elapsed
        results["throughput"]["statistics"] = (width * height) / results["operations"]["statistics"]
        
        # Run resampling
        print("Resampling DEM...")
        with timer("Resampling") as elapsed:
            resampled = dem_proc.resample(0.5)
        results["operations"]["resampling"] = time.time() - elapsed
        results["throughput"]["resampling"] = (width * height) / results["operations"]["resampling"]
        
        # Calculate total execution time
        total_execution_time = sum(results["operations"].values())
        results["total_execution_time"] = total_execution_time
        
        # Calculate total throughput (operations per second)
        total_operations = len(results["operations"])
        results["total_throughput"] = total_operations / total_execution_time
        
        # Get memory usage
        memory_after = get_memory_usage()
        results["memory"] = {
            "host": memory_after["host"] - memory_before["host"],
            "device": memory_after["device"] - memory_before["device"]
        }
        
        # Get GPU utilization
        results["gpu_utilization"] = get_gpu_utilization(self.device_id)
        
        # Clean up
        del dem_proc
        del viewshed
        del terrain
        del hydro
        del path
        del resampled
        gc.collect()
        
        return results

class PointCloudBenchmark(Benchmark):
    """Benchmark for point cloud processing operations."""
    
    def __init__(self, device_id: int = 0):
        super().__init__("point_cloud_processing", device_id)
        
    def run(self, point_cloud_path: str, **kwargs) -> Dict[str, Any]:
        """
        Run point cloud processing benchmark.
        
        Args:
            point_cloud_path: Path to point cloud file
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "name": self.name,
            "params": {
                "point_cloud_path": point_cloud_path,
                **kwargs
            },
            "operations": {},
            "memory": {},
            "throughput": {}
        }
        
        # Get initial memory usage
        memory_before = get_memory_usage()
        
        # Load point cloud
        print(f"Loading point cloud from {point_cloud_path}...")
        with timer("Point cloud loading") as elapsed:
            try:
                pc = PointCloud(point_cloud_path, self.device_id)
                num_points = pc.get_num_points()
                print(f"Point cloud loaded: {num_points} points")
                results["params"]["num_points"] = num_points
                results["operations"]["loading"] = time.time() - elapsed
                results["throughput"]["loading"] = num_points / results["operations"]["loading"]
            except Exception as e:
                print(f"Error loading point cloud: {e}")
                return results
        
        try:
            # Run point classification
            print("Classifying points...")
            with timer("Point classification") as elapsed:
                classified = pc.classify_points()
            results["operations"]["classification"] = time.time() - elapsed
            results["throughput"]["classification"] = num_points / results["operations"]["classification"]
            
            # Run ground filtering
            print("Filtering ground points...")
            with timer("Ground filtering") as elapsed:
                ground = classified.filter_by_class(PointClass.Ground)
            results["operations"]["ground_filtering"] = time.time() - elapsed
            results["throughput"]["ground_filtering"] = num_points / results["operations"]["ground_filtering"]
            
            # Run DEM creation
            print("Creating DEM...")
            with timer("DEM creation") as elapsed:
                dem = ground.create_dem(resolution=1.0)
            results["operations"]["dem_creation"] = time.time() - elapsed
            results["throughput"]["dem_creation"] = num_points / results["operations"]["dem_creation"]
            
            # Run DSM creation
            print("Creating DSM...")
            with timer("DSM creation") as elapsed:
                dsm = pc.create_dsm(resolution=1.0)
            results["operations"]["dsm_creation"] = time.time() - elapsed
            results["throughput"]["dsm_creation"] = num_points / results["operations"]["dsm_creation"]
            
            # Run building extraction
            print("Extracting buildings...")
            with timer("Building extraction") as elapsed:
                buildings = pc.extract_buildings(min_height=2.0, min_area=10.0)
            results["operations"]["building_extraction"] = time.time() - elapsed
            results["throughput"]["building_extraction"] = num_points / results["operations"]["building_extraction"]
            
            # Run vegetation extraction
            print("Extracting vegetation...")
            with timer("Vegetation extraction") as elapsed:
                vegetation = pc.extract_vegetation(height_classes=[0.5, 2.0, 5.0, 15.0], resolution=1.0)
            results["operations"]["vegetation_extraction"] = time.time() - elapsed
            results["throughput"]["vegetation_extraction"] = num_points / results["operations"]["vegetation_extraction"]
            
            # Run segmentation
            print("Segmenting points...")
            with timer("Point segmentation") as elapsed:
                segments = pc.segment_points(max_distance=1.0, min_points=10)
            results["operations"]["segmentation"] = time.time() - elapsed
            results["throughput"]["segmentation"] = num_points / results["operations"]["segmentation"]
            
            # Calculate total execution time
            total_execution_time = sum(results["operations"].values())
            results["total_execution_time"] = total_execution_time
            
            # Calculate total throughput (points per second)
            results["total_throughput"] = num_points / total_execution_time
            
        except Exception as e:
            print(f"Error during point cloud benchmarking: {e}")
            import traceback
            traceback.print_exc()
        
        # Get memory usage
        memory_after = get_memory_usage()
        results["memory"] = {
            "host": memory_after["host"] - memory_before["host"],
            "device": memory_after["device"] - memory_before["device"]
        }
        
        # Get GPU utilization
        results["gpu_utilization"] = get_gpu_utilization(self.device_id)
        
        # Clean up
        del pc
        gc.collect()
        
        return results

class GeospatialBenchmarkSuite:
    """Comprehensive benchmark suite for geospatial workloads."""
    
    def __init__(self, 
                device_id: int = 0, 
                output_dir: str = "results",
                enable_cost_modeling: bool = False,
                aws_instance_type: str = "g4dn.xlarge",
                azure_instance_type: str = "Standard_NC4as_T4_v3",
                gcp_instance_type: str = "n1-standard-4-t4"):
        """
        Initialize benchmark suite.
        
        Args:
            device_id: GPU device ID to use
            output_dir: Directory to store results
            enable_cost_modeling: Whether to enable cost modeling
            aws_instance_type: AWS instance type for cost comparison
            azure_instance_type: Azure instance type for cost comparison
            gcp_instance_type: GCP instance type for cost comparison
        """
        self.device_id = device_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cost modeling configuration
        self.enable_cost_modeling = enable_cost_modeling and COST_MODELING_AVAILABLE
        self.aws_instance_type = aws_instance_type
        self.azure_instance_type = azure_instance_type
        self.gcp_instance_type = gcp_instance_type
        
        # Create benchmarks
        self.dem_benchmark = DEMBenchmark(device_id)
        self.point_cloud_benchmark = PointCloudBenchmark(device_id)
        
        # Results storage
        self.results = {}
        
        # System information
        self.system_info = get_system_info()
    
    def run_dem_benchmark(self, 
                         dem_size: str = "medium", 
                         terrain_type: str = TerrainType.ROLLING_HILLS, 
                         dem_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run DEM processing benchmark.
        
        Args:
            dem_size: Size of the DEM (small, medium, large)
            terrain_type: Type of terrain (flat, rolling_hills, mountains, canyon)
            dem_path: Optional path to custom DEM file
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n===== Running DEM benchmark ({dem_size}, {terrain_type}) =====")
        
        # Use provided DEM or create synthetic one
        if dem_path and os.path.exists(dem_path):
            print(f"Using existing DEM: {dem_path}")
        else:
            # Map size string to actual size
            size_map = {
                "small": 512,
                "medium": 1024,
                "large": 2048,
                "xlarge": 4096
            }
            
            if dem_size not in size_map:
                print(f"Unknown DEM size: {dem_size}, defaulting to medium")
                dem_size = "medium"
            
            pixel_size = size_map[dem_size]
            
            print(f"Creating synthetic DEM ({pixel_size}x{pixel_size}, {terrain_type})...")
            dem_path = create_synthetic_dem(
                size=pixel_size,
                terrain_type=terrain_type,
                z_scale=100.0
            )
        
        # Run benchmark
        result = self.dem_benchmark.run(dem_path=dem_path)
        
        # Add to results
        dem_key = f"dem_{dem_size}_{terrain_type}"
        self.results[dem_key] = result
        
        # Calculate cost metrics if enabled
        if self.enable_cost_modeling:
            print("Calculating cost metrics...")
            self._calculate_cost_metrics(dem_key, result)
        
        return result
    
    def run_point_cloud_benchmark(self, 
                                 point_cloud_size: str = "medium", 
                                 density_pattern: str = "uniform",
                                 point_cloud_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run point cloud processing benchmark.
        
        Args:
            point_cloud_size: Size of the point cloud (small, medium, large)
            density_pattern: Point density pattern (uniform, clustered)
            point_cloud_path: Optional path to custom point cloud file
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\n===== Running point cloud benchmark ({point_cloud_size}, {density_pattern}) =====")
        
        # Use provided point cloud or create synthetic one
        if point_cloud_path and os.path.exists(point_cloud_path):
            print(f"Using existing point cloud: {point_cloud_path}")
        else:
            # Map size string to actual size
            size_map = {
                "small": 100000,
                "medium": 1000000,
                "large": 10000000,
                "xlarge": 50000000
            }
            
            if point_cloud_size not in size_map:
                print(f"Unknown point cloud size: {point_cloud_size}, defaulting to medium")
                point_cloud_size = "medium"
            
            num_points = size_map[point_cloud_size]
            
            print(f"Creating synthetic point cloud ({num_points} points, {density_pattern})...")
            point_cloud_path = create_synthetic_point_cloud(
                num_points=num_points,
                density_pattern=density_pattern
            )
        
        # Run benchmark
        result = self.point_cloud_benchmark.run(point_cloud_path=point_cloud_path)
        
        # Add to results
        pc_key = f"point_cloud_{point_cloud_size}_{density_pattern}"
        self.results[pc_key] = result
        
        # Calculate cost metrics if enabled
        if self.enable_cost_modeling:
            print("Calculating cost metrics...")
            self._calculate_cost_metrics(pc_key, result)
        
        return result
    
    def _calculate_cost_metrics(self, benchmark_key: str, result: Dict[str, Any]) -> None:
        """
        Calculate cost metrics for benchmark result.
        
        Args:
            benchmark_key: Key for the benchmark result
            result: Benchmark result dictionary
        """
        if not self.enable_cost_modeling or not COST_MODELING_AVAILABLE:
            return
        
        # Create a BenchmarkResult-like object that has the necessary attributes
        class CostBenchmarkResult:
            def __init__(self, result_dict):
                self.workload_name = result_dict["name"]
                self.device_name = "Jetson"
                self.device_capabilities = {"name": "NVIDIA Jetson", "compute_capability": "8.7"}
                self.execution_time = result_dict["total_execution_time"]
                self.memory_usage = result_dict["memory"]
                self.gpu_utilization = result_dict.get("gpu_utilization")
                self.energy_consumption = None
                self.throughput = result_dict["total_throughput"]
                self.additional_metrics = result_dict["params"]
                self.cost_metrics = {}
        
        # Create benchmark result object
        benchmark_result = CostBenchmarkResult(result)
        
        # Calculate cost metrics
        jetson_model = CostModelFactory.create_model(ComputeEnvironment.LOCAL_JETSON)
        jetson_cost = jetson_model.estimate_cost(
            benchmark_result.execution_time,
            benchmark_result.memory_usage,
            benchmark_result.gpu_utilization,
            None,  # No energy consumption data
            benchmark_result.additional_metrics
        )
        
        # Calculate cloud costs
        cloud_costs = {}
        
        # AWS
        aws_model = CostModelFactory.create_model(
            ComputeEnvironment.AWS_GPU, 
            instance_type=self.aws_instance_type
        )
        cloud_costs["aws"] = aws_model.estimate_cost(
            benchmark_result.execution_time,
            benchmark_result.memory_usage,
            benchmark_result.gpu_utilization,
            None,  # No energy consumption data
            benchmark_result.additional_metrics
        )
        
        # Azure
        azure_model = CostModelFactory.create_model(
            ComputeEnvironment.AZURE_GPU, 
            instance_type=self.azure_instance_type
        )
        cloud_costs["azure"] = azure_model.estimate_cost(
            benchmark_result.execution_time,
            benchmark_result.memory_usage,
            benchmark_result.gpu_utilization,
            None,  # No energy consumption data
            benchmark_result.additional_metrics
        )
        
        # GCP
        gcp_model = CostModelFactory.create_model(
            ComputeEnvironment.GCP_GPU, 
            instance_type=self.gcp_instance_type
        )
        cloud_costs["gcp"] = gcp_model.estimate_cost(
            benchmark_result.execution_time,
            benchmark_result.memory_usage,
            benchmark_result.gpu_utilization,
            None,  # No energy consumption data
            benchmark_result.additional_metrics
        )
        
        # Calculate cost comparison
        comparison = calculate_cost_comparison(
            jetson_cost,
            cloud_costs,
            benchmark_result.workload_name,
            benchmark_result.execution_time,
            benchmark_result.throughput
        )
        
        # Store cost metrics
        cost_metrics = {
            "jetson": jetson_cost,
            "cloud": cloud_costs,
            "comparison": comparison
        }
        
        # Add cost metrics to result
        result["cost_metrics"] = cost_metrics
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all benchmarks.
        
        Returns:
            Dictionary with benchmark results
        """
        print("\n===== Running all benchmarks =====")
        
        # DEM benchmarks
        self.run_dem_benchmark(dem_size="small", terrain_type=TerrainType.FLAT)
        self.run_dem_benchmark(dem_size="medium", terrain_type=TerrainType.ROLLING_HILLS)
        self.run_dem_benchmark(dem_size="medium", terrain_type=TerrainType.MOUNTAINS)
        
        # Point cloud benchmarks
        self.run_point_cloud_benchmark(point_cloud_size="small", density_pattern="uniform")
        self.run_point_cloud_benchmark(point_cloud_size="medium", density_pattern="clustered")
        
        return self.results
    
    def save_results(self) -> str:
        """
        Save benchmark results to file.
        
        Returns:
            Path to the results file
        """
        # Add system information to results
        results_with_metadata = {
            "system_info": self.system_info,
            "timestamp": datetime.now().isoformat(),
            "benchmarks": self.results
        }
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"geospatial_benchmark_{timestamp}.json"
        
        # Save results
        with open(results_file, "w") as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        return str(results_file)
    
    def generate_report(self, results_file: Optional[str] = None) -> str:
        """
        Generate benchmark report.
        
        Args:
            results_file: Path to results file (default: use current results)
            
        Returns:
            Path to the report file
        """
        # Allow visualizing previously saved results
        if results_file:
            with open(results_file, "r") as f:
                results_with_metadata = json.load(f)
            benchmarks = results_with_metadata["benchmarks"]
            system_info = results_with_metadata["system_info"]
        else:
            benchmarks = self.results
            system_info = self.system_info
        
        # Import visualization utilities
        sys.path.append(os.path.join(project_root, "benchmark"))
        try:
            from benchmark.visualization import generate_summary_report
            
            # Convert our results format to the expected format
            adapted_results = {}
            for key, benchmark in benchmarks.items():
                adapted_results[key] = [{
                    "workload_name": benchmark["name"],
                    "device_name": system_info.get("gpu", {}).get("devices", [{}])[0].get("name", "Unknown"),
                    "device_capabilities": system_info.get("gpu", {}).get("devices", [{}])[0],
                    "execution_time": benchmark["total_execution_time"],
                    "memory_usage": benchmark["memory"],
                    "gpu_utilization": benchmark.get("gpu_utilization"),
                    "throughput": benchmark["total_throughput"],
                    "additional_metrics": benchmark["params"],
                    "cost_metrics": benchmark.get("cost_metrics", {})
                }]
            
            # Generate report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"geospatial_report_{timestamp}.html"
            
            # Generate report
            generate_summary_report(adapted_results, str(report_file))
            
            print(f"\nReport generated: {report_file}")
            return str(report_file)
            
        except ImportError:
            print("Warning: Visualization utilities not available. Cannot generate report.")
            return ""

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Geospatial Analysis workload")
    
    # Device selection
    parser.add_argument("--device", type=int, default=0, 
                       help="GPU device ID to use")
    
    # Benchmark selection
    parser.add_argument("--all", action="store_true",
                       help="Run all benchmarks")
    parser.add_argument("--dem", action="store_true",
                       help="Run DEM benchmarks")
    parser.add_argument("--point-cloud", action="store_true",
                       help="Run point cloud benchmarks")
    
    # DEM parameters
    parser.add_argument("--dem-size", type=str, default="medium",
                       choices=["small", "medium", "large", "xlarge"],
                       help="Size of synthetic DEM")
    parser.add_argument("--dem-type", type=str, default="rolling_hills",
                       choices=["flat", "rolling_hills", "mountains", "canyon", "coastal", "urban", "random"],
                       help="Type of terrain for synthetic DEM")
    parser.add_argument("--dem-file", type=str, default=None,
                       help="Path to custom DEM file")
    
    # Point cloud parameters
    parser.add_argument("--pc-size", type=str, default="medium",
                       choices=["small", "medium", "large", "xlarge"],
                       help="Size of synthetic point cloud")
    parser.add_argument("--pc-density", type=str, default="uniform",
                       choices=["uniform", "clustered", "grid"],
                       help="Density pattern for synthetic point cloud")
    parser.add_argument("--pc-file", type=str, default=None,
                       help="Path to custom point cloud file")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--report", action="store_true",
                       help="Generate report from results")
    parser.add_argument("--results-file", type=str, default=None,
                       help="Path to existing results file for report generation")
    
    # Cost modeling
    parser.add_argument("--cost-analysis", action="store_true",
                       help="Enable cost modeling and comparison")
    parser.add_argument("--aws-instance", type=str, default="g4dn.xlarge",
                       help="AWS instance type for cost comparison")
    parser.add_argument("--azure-instance", type=str, default="Standard_NC4as_T4_v3",
                       help="Azure instance type for cost comparison")
    parser.add_argument("--gcp-instance", type=str, default="n1-standard-4-t4",
                       help="GCP instance type for cost comparison")
    
    # Dataset management
    parser.add_argument("--clear-datasets", action="store_true",
                       help="Clear generated datasets before starting")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Clear datasets if requested
    if args.clear_datasets:
        clear_datasets()
    
    # Create benchmark suite
    suite = GeospatialBenchmarkSuite(
        device_id=args.device,
        output_dir=args.output_dir,
        enable_cost_modeling=args.cost_analysis and COST_MODELING_AVAILABLE,
        aws_instance_type=args.aws_instance,
        azure_instance_type=args.azure_instance,
        gcp_instance_type=args.gcp_instance
    )
    
    # Generate report from existing results
    if args.report and args.results_file:
        suite.generate_report(args.results_file)
        return
    
    # Run benchmarks
    if args.all:
        suite.run_all_benchmarks()
    else:
        if args.dem:
            suite.run_dem_benchmark(
                dem_size=args.dem_size,
                terrain_type=args.dem_type,
                dem_path=args.dem_file
            )
        
        if args.point_cloud:
            suite.run_point_cloud_benchmark(
                point_cloud_size=args.pc_size,
                density_pattern=args.pc_density,
                point_cloud_path=args.pc_file
            )
    
    # Save results
    results_file = suite.save_results()
    
    # Generate report
    if args.report:
        suite.generate_report()

if __name__ == "__main__":
    main()