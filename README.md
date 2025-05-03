# NVIDIA Jetson Workload

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2024 nvidia-jetson-workload contributors -->

Scientific demo workloads for NVIDIA Jetson Orin NX systems.

## Overview

This project provides GPU-accelerated scientific workloads designed for NVIDIA Jetson Orin NX devices. It includes implementations of complex scientific computations that showcase the computational capabilities of Jetson systems, with a focus on hybrid CPU/GPU processing. The workloads are designed to run for 2-5 minutes each, making them ideal for demonstrations.

## Workloads

The project includes the following workloads:

1. **Weather/Climate Simulation**
   - Fluid dynamics simulation showcasing atmospheric modeling
   - Parallelized across GPU cores for maximum performance
   - Available in C++/CUDA and Python implementations

2. **Medical Image Processing**
   - AI-driven segmentation and analysis of medical imaging data
   - GPU-accelerated neural network inference
   - Available in C++/CUDA and Python implementations

3. **N-body Gravitational Simulation**
   - Physics-based simulation of gravitational interactions
   - Optimized for Jetson's CUDA capabilities
   - Available in C++/CUDA and Python implementations

## Visualization

The project includes a browser-based dashboard for visualizing results from workloads running on headless Jetson nodes, with:

- Real-time visualization of simulation data
- Performance metrics and comparisons
- Slurm job management integration

## Requirements

- NVIDIA Jetson Orin NX (16GB RAM)
- JetPack 6.0+
- CUDA 12.0+
- For Python workloads: Python 3.10+
- For C++ workloads: GCC 11.0+, CMake 3.22+
- For visualization: Modern web browser

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.