"""
Geospatial Analysis benchmarking package.

This package provides benchmarking tools for the Geospatial Analysis workload,
including performance measurements, dataset generation, and cost comparisons.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

from .geospatial_benchmark import GeospatialBenchmarkSuite
from .metrics import GeospatialMetrics, PerformanceProfiler
from .datasets import (
    TerrainType,
    PointCloudDensity,
    create_synthetic_dem,
    create_synthetic_point_cloud,
    create_standard_benchmark_datasets
)