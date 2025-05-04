#!/usr/bin/env python3
"""
Performance metrics for Geospatial Analysis benchmarking.

This module provides standardized metrics for evaluating geospatial
processing performance across different hardware configurations.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path

class GeospatialMetrics:
    """Class for tracking and calculating geospatial performance metrics."""
    
    # Standard metric names
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    THROUGHPUT = "throughput"
    COST_EFFICIENCY = "cost_efficiency"
    ENERGY_EFFICIENCY = "energy_efficiency"
    
    # Geospatial-specific metrics
    CELLS_PER_SECOND = "cells_per_second"  # For raster operations
    POINTS_PER_SECOND = "points_per_second"  # For point cloud operations
    VIEWSHED_SPEED = "viewshed_speed"  # Cells processed per second for viewshed
    DEM_PROCESSING_SPEED = "dem_processing_speed"  # Cells processed per second for terrain derivatives
    HYDRO_PROCESSING_SPEED = "hydro_processing_speed"  # Cells processed per second for hydrological features
    POINT_CLASSIFICATION_SPEED = "point_classification_speed"  # Points classified per second
    SURFACE_RECONSTRUCTION_SPEED = "surface_reconstruction_speed"  # Points processed per second for surface reconstruction
    FEATURE_EXTRACTION_SPEED = "feature_extraction_speed"  # Points processed per second for feature extraction
    
    def __init__(self):
        """Initialize metrics tracking."""
        self.metrics = {}
        self.start_time = datetime.now()
        
    def record_metric(self, name: str, value: Any, timestamp: Optional[datetime] = None):
        """
        Record a metric value.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "timestamp": timestamp.isoformat()
        })
    
    def get_metric(self, name: str) -> List[Dict[str, Any]]:
        """
        Get all values for a metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            List of metric values with timestamps
        """
        return self.metrics.get(name, [])
    
    def get_latest_metric(self, name: str) -> Optional[Any]:
        """
        Get the latest value for a metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Latest value or None if not found
        """
        values = self.get_metric(name)
        if values:
            return values[-1]["value"]
        return None
    
    def get_average_metric(self, name: str) -> Optional[float]:
        """
        Get the average value for a metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Average value or None if not found
        """
        values = self.get_metric(name)
        if values:
            try:
                # Extract numeric values
                numeric_values = [float(v["value"]) for v in values]
                return sum(numeric_values) / len(numeric_values)
            except (ValueError, TypeError):
                return None
        return None
    
    def calculate_raster_throughput(self, operation_name: str, width: int, height: int, execution_time: float) -> float:
        """
        Calculate throughput for raster operations.
        
        Args:
            operation_name: Name of the operation
            width: Width of the raster in cells
            height: Height of the raster in cells
            execution_time: Execution time in seconds
            
        Returns:
            Throughput in cells per second
        """
        # Calculate total number of cells processed
        num_cells = width * height
        
        # Calculate throughput
        throughput = num_cells / execution_time
        
        # Record metrics
        self.record_metric(f"{operation_name}_{self.CELLS_PER_SECOND}", throughput)
        
        return throughput
    
    def calculate_point_cloud_throughput(self, operation_name: str, num_points: int, execution_time: float) -> float:
        """
        Calculate throughput for point cloud operations.
        
        Args:
            operation_name: Name of the operation
            num_points: Number of points processed
            execution_time: Execution time in seconds
            
        Returns:
            Throughput in points per second
        """
        # Calculate throughput
        throughput = num_points / execution_time
        
        # Record metrics
        self.record_metric(f"{operation_name}_{self.POINTS_PER_SECOND}", throughput)
        
        return throughput
    
    def record_viewshed_performance(self, width: int, height: int, execution_time: float):
        """
        Record viewshed analysis performance.
        
        Args:
            width: Width of the DEM in cells
            height: Height of the DEM in cells
            execution_time: Execution time in seconds
        """
        throughput = self.calculate_raster_throughput("viewshed", width, height, execution_time)
        self.record_metric(self.VIEWSHED_SPEED, throughput)
    
    def record_dem_derivatives_performance(self, width: int, height: int, execution_time: float):
        """
        Record DEM derivatives performance.
        
        Args:
            width: Width of the DEM in cells
            height: Height of the DEM in cells
            execution_time: Execution time in seconds
        """
        throughput = self.calculate_raster_throughput("dem_derivatives", width, height, execution_time)
        self.record_metric(self.DEM_PROCESSING_SPEED, throughput)
    
    def record_hydro_features_performance(self, width: int, height: int, execution_time: float):
        """
        Record hydrological features performance.
        
        Args:
            width: Width of the DEM in cells
            height: Height of the DEM in cells
            execution_time: Execution time in seconds
        """
        throughput = self.calculate_raster_throughput("hydro_features", width, height, execution_time)
        self.record_metric(self.HYDRO_PROCESSING_SPEED, throughput)
    
    def record_point_classification_performance(self, num_points: int, execution_time: float):
        """
        Record point classification performance.
        
        Args:
            num_points: Number of points classified
            execution_time: Execution time in seconds
        """
        throughput = self.calculate_point_cloud_throughput("classification", num_points, execution_time)
        self.record_metric(self.POINT_CLASSIFICATION_SPEED, throughput)
    
    def record_surface_reconstruction_performance(self, num_points: int, execution_time: float):
        """
        Record surface reconstruction performance.
        
        Args:
            num_points: Number of points processed
            execution_time: Execution time in seconds
        """
        throughput = self.calculate_point_cloud_throughput("surface_reconstruction", num_points, execution_time)
        self.record_metric(self.SURFACE_RECONSTRUCTION_SPEED, throughput)
    
    def record_feature_extraction_performance(self, num_points: int, execution_time: float):
        """
        Record feature extraction performance.
        
        Args:
            num_points: Number of points processed
            execution_time: Execution time in seconds
        """
        throughput = self.calculate_point_cloud_throughput("feature_extraction", num_points, execution_time)
        self.record_metric(self.FEATURE_EXTRACTION_SPEED, throughput)
    
    def record_cost_efficiency(self, operation_name: str, cost: float, throughput: float):
        """
        Record cost efficiency metrics.
        
        Args:
            operation_name: Name of the operation
            cost: Cost of the operation
            throughput: Throughput of the operation
        """
        # Calculate cost efficiency (operations per dollar)
        cost_efficiency = throughput / cost if cost > 0 else float('inf')
        
        # Record metrics
        self.record_metric(f"{operation_name}_{self.COST_EFFICIENCY}", cost_efficiency)
    
    def record_energy_efficiency(self, operation_name: str, energy_consumption: float, throughput: float):
        """
        Record energy efficiency metrics.
        
        Args:
            operation_name: Name of the operation
            energy_consumption: Energy consumption in joules
            throughput: Throughput of the operation
        """
        # Calculate energy efficiency (operations per joule)
        energy_efficiency = throughput / energy_consumption if energy_consumption > 0 else float('inf')
        
        # Record metrics
        self.record_metric(f"{operation_name}_{self.ENERGY_EFFICIENCY}", energy_efficiency)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.
        
        Returns:
            Dictionary with metric summaries
        """
        summary = {
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration": (datetime.now() - self.start_time).total_seconds()
        }
        
        # Calculate summary statistics for each metric
        for name, values in self.metrics.items():
            try:
                numeric_values = [float(v["value"]) for v in values]
                summary[name] = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": sum(numeric_values) / len(numeric_values),
                    "median": sorted(numeric_values)[len(numeric_values) // 2],
                    "count": len(numeric_values)
                }
                
                # Include raw values for time series analysis
                summary[f"{name}_raw"] = values
            except (ValueError, TypeError):
                # Handle non-numeric metrics
                summary[name] = {
                    "values": [v["value"] for v in values],
                    "count": len(values)
                }
        
        return summary
    
    def save_to_file(self, filename: str) -> str:
        """
        Save metrics to a file.
        
        Args:
            filename: Path to save metrics
            
        Returns:
            Path to the saved file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Get summary
        summary = self.get_summary()
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return filename

class PerformanceProfiler:
    """Utility class for profiling geospatial operations."""
    
    def __init__(self, metrics: Optional[GeospatialMetrics] = None):
        """
        Initialize performance profiler.
        
        Args:
            metrics: Optional GeospatialMetrics instance to record to
        """
        self.metrics = metrics or GeospatialMetrics()
        self.operation_timers = {}
    
    def start_operation(self, operation_name: str):
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation
        """
        self.operation_timers[operation_name] = time.time()
    
    def end_operation(self, operation_name: str) -> float:
        """
        End timing an operation and record metrics.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Execution time in seconds
        """
        if operation_name not in self.operation_timers:
            raise ValueError(f"Operation {operation_name} was not started")
        
        start_time = self.operation_timers.pop(operation_name)
        execution_time = time.time() - start_time
        
        # Record execution time metric
        self.metrics.record_metric(f"{operation_name}_{GeospatialMetrics.EXECUTION_TIME}", execution_time)
        
        return execution_time
    
    def record_memory_usage(self, operation_name: str, host_memory: float, device_memory: float):
        """
        Record memory usage.
        
        Args:
            operation_name: Name of the operation
            host_memory: Host memory usage in MB
            device_memory: Device memory usage in MB
        """
        self.metrics.record_metric(
            f"{operation_name}_{GeospatialMetrics.MEMORY_USAGE}",
            {"host": host_memory, "device": device_memory}
        )
    
    def record_gpu_utilization(self, operation_name: str, utilization: float):
        """
        Record GPU utilization.
        
        Args:
            operation_name: Name of the operation
            utilization: GPU utilization percentage (0-100)
        """
        self.metrics.record_metric(
            f"{operation_name}_{GeospatialMetrics.GPU_UTILIZATION}",
            utilization
        )
    
    def get_metrics(self) -> GeospatialMetrics:
        """
        Get the metrics object.
        
        Returns:
            GeospatialMetrics object
        """
        return self.metrics