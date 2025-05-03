# Geospatial Analysis Workload

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors

## Overview

The Geospatial Analysis workload provides GPU-accelerated algorithms for processing and analyzing geographic data, optimized for both NVIDIA Jetson Orin NX and AWS Graviton g5g instances with T4 GPUs.

## Key Features

- **Digital Elevation Model (DEM) Processing**
  - Viewshed Analysis
  - Terrain Derivatives (slope, aspect, curvature)
  - Hydrological Features
  - Least-cost Path Analysis
  - Sink/Depression Filling

- **LiDAR Point Cloud Processing**
  - Point Cloud Classification
  - Surface Reconstruction
  - Feature Extraction
  - Building Footprint Extraction
  - Vegetation Analysis

- **Raster/Vector Operations**
  - Spatial Indexing
  - Geometric Operations
  - Spatial Statistics
  - Image Processing

## GPU Optimizations

This workload implements the GPU adaptability pattern to automatically select optimized code paths based on the available GPU:

- **Jetson Orin NX (SM 8.7)**
  - Leverages Tensor Cores for matrix operations
  - Optimized for unified memory architecture
  - Balanced compute-memory operations

- **AWS T4 GPUs (SM 7.5)**
  - Maximizes global memory bandwidth
  - Optimized for higher clock speeds
  - Enhanced floating-point performance

- **CPU Fallback**
  - Implemented for systems without CUDA capability
  - Multi-threaded CPU implementation

## Dependencies

- CUDA Toolkit 11.0+
- GDAL 3.0+
- [Optional] PDAL for Point Cloud processing
- [Optional] Boost 1.65+ for advanced spatial algorithms
- Python 3.8+ (for Python bindings)

## Installation

### Building from Source

1. Install dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install libgdal-dev libpdal-dev libboost-all-dev

# CentOS/RHEL
sudo yum install gdal-devel pdal-devel boost-devel
```

2. Build with CMake:

```bash
cd src/geospatial/cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

3. Install Python bindings:

```bash
cd src/geospatial/python
pip install -e .
```

### Docker Container

A prebuilt Docker container is available:

```bash
docker pull ghcr.io/scttfrdmn/nvidia-jetson-workload/geospatial:latest
```

## Usage Examples

### Python API

```python
from geospatial import DEMProcessor, PointCloud

# Process a DEM
dem = DEMProcessor("elevation.tif", device_id=0)
viewshed = dem.compute_viewshed(
    observer_point=(356420, 4842624),
    observer_height=1.8,
    radius=5000
)
dem.save_result(viewshed, "viewshed_result.tif")

# Process a point cloud
point_cloud = PointCloud("lidar_scan.las", device_id=0)
classified_points = point_cloud.classify_points()
ground_points = classified_points.filter_by_class("ground")
dem = ground_points.create_dem(resolution=1.0)
dem.save("ground_surface.tif")
```

### C++ API

```cpp
#include <geospatial/dem_processing.hpp>
#include <geospatial/point_cloud.hpp>

// Process DEM
auto dem = geospatial::DEMProcessor("elevation.tif");
auto viewshed = dem.computeViewshed(
    {356420, 4842624},  // observer point
    1.8,                // observer height
    5000                // radius
);
dem.saveResult(viewshed, "viewshed_result.tif");

// Process point cloud
auto pointCloud = geospatial::PointCloud("lidar_scan.las");
auto classifiedPoints = pointCloud.classifyPoints();
auto groundPoints = classifiedPoints.filterByClass(geospatial::PointClass::Ground);
auto dem = groundPoints.createDEM(1.0);  // 1.0m resolution
dem.save("ground_surface.tif");
```

## Benchmarking

The workload includes benchmarking capabilities that integrate with the project's benchmarking suite. To run benchmarks:

```bash
cd benchmark
./run_benchmarks.sh --workload geospatial
```

Benchmark results include:
- Execution time for key operations
- Memory usage (host and device)
- GPU utilization
- Cross-device performance comparisons

## Documentation

See the `docs/user-guide/geospatial.md` for detailed documentation on all available functionality.

## License

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors