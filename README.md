# NVIDIA Jetson & AWS Graviton Workloads

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

GPU-accelerated scientific workloads for NVIDIA Jetson Orin NX and AWS Graviton g5g instances.

## Overview

This project provides GPU-accelerated scientific workloads designed for both NVIDIA Jetson Orin NX devices and AWS Graviton g5g instances with NVIDIA T4 GPUs. It includes implementations of complex scientific computations that showcase exceptional multi-core and GPU programming with high arithmetic throughput and coordination, ensuring high system utilization and good load balance across disparate system components and capabilities.

## Key Features

- **GPU Adaptability Pattern**: Automatically detects GPU capabilities and selects optimized kernels
- **Cross-Platform Performance**: Efficiently scales between Jetson Orin NX and AWS Graviton g5g instances
- **ARM CPU Optimization**: Leverages ARM-specific optimizations for both CPU and hybrid CPU/GPU computing
- **Comprehensive Benchmarking**: Includes tools for performance comparison across different hardware
- **Python Bindings**: All workloads provide Python bindings for high-level control and visualization

## Workloads

The project includes the following workloads:

1. **N-body Gravitational Simulation**
   - Physics-based simulation of gravitational interactions between particles
   - Multiple integrators: Euler, Leapfrog, Verlet, RK4
   - Several system types: random, solar system, galaxy
   - C++/CUDA implementation with Python bindings
   - Comprehensive benchmark and visualization tools

2. **Molecular Dynamics Simulation**
   - Simulation of molecular systems with multiple force fields
   - Supports multiple integrators and thermostats
   - Implements Lennard-Jones potentials and Coulomb interactions
   - GPU-accelerated with architecture-specific optimizations
   - Energy conservation and performance tracking

3. **Weather Simulation**
   - Implementation of multiple numerical weather models
   - Shallow Water Equations, Barotropic Model, Primitive Equations
   - CUDA kernels with architecture-specific optimizations
   - GPU adaptability pattern for optimal cross-device performance
   - Python bindings with NumPy integration and visualization tools

4. **Medical Imaging**
   - Comprehensive medical image processing workload
   - CT reconstruction (filtered backprojection, iterative methods)
   - Image processing (convolution, filtering, transformation)
   - Segmentation (thresholding, watershed, level set, graph cut)
   - Registration (image warping, mutual information)
   - CUDA kernels with architecture-specific optimizations
   - Python bindings with NumPy integration and visualization utilities

## GPU Optimization Techniques

- Automatic detection of GPU compute capability
- Dynamically selects optimal kernel parameters based on device type
- Tiled algorithm implementation with configurable tile sizes
- Shared memory optimization scaled to hardware capabilities
- Texture memory usage for bandwidth-limited applications
- Cooperative groups for better thread coordination in CUDA
- Fallback to CPU implementation when GPU is not available
- Support for multiple CUDA architectures in a single binary
- Hybrid CPU-GPU computing with dynamic workload balancing

## Getting Started

### Requirements

- NVIDIA Jetson Orin NX (16GB RAM) or AWS Graviton g5g instance with NVIDIA T4 GPU
- JetPack 6.0+ (for Jetson) or AWS AMI with CUDA support (for Graviton)
- CUDA 12.0+
- For Python workloads: Python 3.10+, NumPy, Matplotlib
- For C++ workloads: GCC 11.0+, CMake 3.22+

### Building from Source

```bash
# Clone the repository
git clone https://github.com/username/nvidia-jetson-workload.git
cd nvidia-jetson-workload

# Build all workloads
./build.sh

# Build a specific workload
cd src/nbody_sim/cpp
./build_and_test.sh
```

### Running Workloads

Each workload can be run through either C++ executables or Python:

```bash
# C++ examples
./build/bin/nbody_sim
./build/bin/weather_sim

# Python examples
python -m nbody_sim.examples.solar_system
python -m medical_imaging.examples.ct_reconstruction_example
```

## Documentation

For more information on the GPU adaptability pattern and performance optimizations, see:

- [GPU Adaptability Pattern](GPU_ADAPTABILITY.md)
- [Completed Tasks](COMPLETED.md)

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.