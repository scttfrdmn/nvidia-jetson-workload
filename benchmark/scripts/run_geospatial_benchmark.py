#!/usr/bin/env python3
"""
Integration script for running Geospatial Analysis benchmarks with the main benchmark suite.

This script serves as a bridge between the main benchmark suite and the specialized
Geospatial Analysis benchmark module, allowing for consistent cost comparison and reporting.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import main benchmark suite modules
from benchmark.benchmark_suite import BenchmarkSuite, BenchmarkResult

# Import cost modeling utilities
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

# Import geospatial benchmark module
geospatial_dir = project_root / "src" / "geospatial"
sys.path.append(str(geospatial_dir))
from benchmark.geospatial_benchmark import GeospatialBenchmarkSuite

class GeospatialBenchmarkAdapter:
    """
    Adapter class to integrate Geospatial benchmarks with the main benchmark suite.
    
    This class adapts the Geospatial benchmark results to the format expected
    by the main benchmark suite and vice versa.
    """
    
    def __init__(self, 
                device_id: int = 0, 
                output_dir: str = "results",
                enable_cost_modeling: bool = False,
                aws_instance_type: str = "g4dn.xlarge",
                azure_instance_type: str = "Standard_NC4as_T4_v3",
                gcp_instance_type: str = "n1-standard-4-t4"):
        """
        Initialize the adapter.
        
        Args:
            device_id: GPU device ID to use
            output_dir: Directory to store results
            enable_cost_modeling: Whether to enable cost modeling
            aws_instance_type: AWS instance type for comparison
            azure_instance_type: Azure instance type for comparison
            gcp_instance_type: GCP instance type for comparison
        """
        self.device_id = device_id
        self.output_dir = Path(output_dir)
        self.enable_cost_modeling = enable_cost_modeling and COST_MODELING_AVAILABLE
        self.aws_instance_type = aws_instance_type
        self.azure_instance_type = azure_instance_type
        self.gcp_instance_type = gcp_instance_type
        
        # Create geospatial benchmark suite
        self.geospatial_suite = GeospatialBenchmarkSuite(
            device_id=device_id,
            output_dir=str(output_dir),
            enable_cost_modeling=enable_cost_modeling,
            aws_instance_type=aws_instance_type,
            azure_instance_type=azure_instance_type,
            gcp_instance_type=gcp_instance_type
        )
        
        # Reference to main benchmark suite
        self.main_suite = None
    
    def set_main_suite(self, main_suite: BenchmarkSuite):
        """
        Set the main benchmark suite reference.
        
        Args:
            main_suite: Main benchmark suite
        """
        self.main_suite = main_suite
    
    def run_benchmarks(self, 
                      dem_size: str = "medium", 
                      dem_type: str = "rolling_hills",
                      pc_size: str = "medium",
                      pc_density: str = "uniform") -> Dict[str, BenchmarkResult]:
        """
        Run geospatial benchmarks and convert results to main suite format.
        
        Args:
            dem_size: Size of DEM (small, medium, large)
            dem_type: Type of terrain (flat, rolling_hills, mountains, canyon)
            pc_size: Size of point cloud (small, medium, large)
            pc_density: Density pattern (uniform, clustered)
            
        Returns:
            Dictionary of benchmark results in main suite format
        """
        # Run DEM benchmark
        dem_result = self.geospatial_suite.run_dem_benchmark(
            dem_size=dem_size,
            terrain_type=dem_type
        )
        
        # Run point cloud benchmark
        pc_result = self.geospatial_suite.run_point_cloud_benchmark(
            point_cloud_size=pc_size,
            density_pattern=pc_density
        )
        
        # Save native geospatial results
        self.geospatial_suite.save_results()
        
        # Convert to main suite format
        main_suite_results = self._convert_to_main_suite_format({
            f"geospatial_dem_{dem_size}_{dem_type}": dem_result,
            f"geospatial_pc_{pc_size}_{pc_density}": pc_result
        })
        
        return main_suite_results
    
    def _convert_to_main_suite_format(self, 
                                     geospatial_results: Dict[str, Dict[str, Any]]) -> Dict[str, BenchmarkResult]:
        """
        Convert geospatial benchmark results to main suite format.
        
        Args:
            geospatial_results: Dictionary of geospatial benchmark results
            
        Returns:
            Dictionary of benchmark results in main suite format
        """
        main_suite_results = {}
        
        for benchmark_name, result in geospatial_results.items():
            # Get system information from the first result
            if not hasattr(self, 'system_info'):
                self.system_info = self.geospatial_suite.system_info
            
            # Extract device capabilities
            device_capabilities = {}
            if self.system_info.get("gpu", {}).get("device_count", 0) > 0:
                device_capabilities = self.system_info.get("gpu", {}).get("devices", [{}])[0]
            
            # Create BenchmarkResult
            benchmark_result = BenchmarkResult(
                workload_name=benchmark_name,
                device_name=device_capabilities.get("name", "Unknown"),
                device_capabilities=device_capabilities,
                execution_time=result.get("total_execution_time", 0),
                memory_usage=result.get("memory", {"host": 0, "device": 0}),
                gpu_utilization=result.get("gpu_utilization"),
                energy_consumption=None,
                throughput=result.get("total_throughput", 0),
                additional_metrics=result.get("params", {}),
                cost_metrics=result.get("cost_metrics", {})
            )
            
            # Calculate cost metrics if not already calculated
            if self.enable_cost_modeling and not result.get("cost_metrics") and COST_MODELING_AVAILABLE:
                benchmark_result.calculate_cost_metrics(
                    compare_with_cloud=True,
                    aws_instance_type=self.aws_instance_type,
                    azure_instance_type=self.azure_instance_type,
                    gcp_instance_type=self.gcp_instance_type,
                    include_dgx_spark=True,
                    include_slurm_cluster=True
                )
            
            main_suite_results[benchmark_name] = benchmark_result
        
        return main_suite_results

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run Geospatial Analysis benchmarks with the main benchmark suite")
    
    # Device selection
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID to use")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--report", action="store_true",
                       help="Generate report from results")
    
    # DEM parameters
    parser.add_argument("--dem-size", type=str, default="medium",
                       choices=["small", "medium", "large"],
                       help="Size of synthetic DEM")
    parser.add_argument("--dem-type", type=str, default="rolling_hills",
                       choices=["flat", "rolling_hills", "mountains", "canyon"],
                       help="Type of terrain for synthetic DEM")
    
    # Point cloud parameters
    parser.add_argument("--pc-size", type=str, default="medium",
                       choices=["small", "medium", "large"],
                       help="Size of synthetic point cloud")
    parser.add_argument("--pc-density", type=str, default="uniform",
                       choices=["uniform", "clustered"],
                       help="Density pattern for synthetic point cloud")
    
    # Cost modeling
    parser.add_argument("--cost-analysis", action="store_true",
                       help="Enable cost modeling and comparison")
    parser.add_argument("--aws-instance", type=str, default="g4dn.xlarge",
                       help="AWS instance type for cost comparison")
    parser.add_argument("--azure-instance", type=str, default="Standard_NC4as_T4_v3",
                       help="Azure instance type for cost comparison")
    parser.add_argument("--gcp-instance", type=str, default="n1-standard-4-t4",
                       help="GCP instance type for cost comparison")
    
    args = parser.parse_args()
    
    # Create benchmark suite
    main_suite = BenchmarkSuite(
        device_id=args.device,
        output_dir=args.output_dir,
        enable_cost_modeling=args.cost_analysis,
        aws_instance_type=args.aws_instance,
        azure_instance_type=args.azure_instance,
        gcp_instance_type=args.gcp_instance
    )
    
    # Create geospatial adapter
    adapter = GeospatialBenchmarkAdapter(
        device_id=args.device,
        output_dir=args.output_dir,
        enable_cost_modeling=args.cost_analysis,
        aws_instance_type=args.aws_instance,
        azure_instance_type=args.azure_instance,
        gcp_instance_type=args.gcp_instance
    )
    
    # Set main suite reference
    adapter.set_main_suite(main_suite)
    
    # Run benchmarks
    geospatial_results = adapter.run_benchmarks(
        dem_size=args.dem_size,
        dem_type=args.dem_type,
        pc_size=args.pc_size,
        pc_density=args.pc_density
    )
    
    # Add results to main suite
    for name, result in geospatial_results.items():
        main_suite.results[name] = result
    
    # Generate report
    if args.report:
        main_suite.generate_reports()
    
    print("\nGeospatial benchmarks completed successfully!")

if __name__ == "__main__":
    main()