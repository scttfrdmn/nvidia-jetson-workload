# User Guide for NVIDIA Jetson & AWS Graviton Workloads

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This user guide provides comprehensive documentation for using the GPU-accelerated scientific workloads with NVIDIA Jetson Orin NX devices and AWS Graviton g5g instances.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Available Workloads](#available-workloads)
4. [Running Workloads](#running-workloads)
5. [Benchmarking](#benchmarking)
6. [Cost Analysis](#cost-analysis)
7. [Deployment](#deployment)
8. [Visualization Dashboard](#visualization-dashboard)
9. [Advanced Configuration](#advanced-configuration)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)

## Introduction

This project provides GPU-accelerated scientific workloads designed for both NVIDIA Jetson Orin NX devices and AWS Graviton g5g instances with NVIDIA T4 GPUs. It includes implementations of complex scientific computations that showcase exceptional multi-core and GPU programming with high arithmetic throughput and coordination, ensuring high system utilization and good load balance across disparate system components and capabilities.

### Key Features

- **GPU Adaptability Pattern**: Automatically detects GPU capabilities and selects optimized kernels
- **Cross-Platform Performance**: Efficiently scales between Jetson Orin NX and AWS Graviton g5g instances
- **ARM CPU Optimization**: Leverages ARM-specific optimizations for both CPU and hybrid CPU/GPU computing
- **Comprehensive Benchmarking**: Includes tools for performance comparison across different hardware
- **Python Bindings**: All workloads provide Python bindings for high-level control and visualization

## Installation

### Requirements

- NVIDIA Jetson Orin NX (16GB RAM) or AWS Graviton g5g instance with NVIDIA T4 GPU
- JetPack 6.0+ (for Jetson) or AWS AMI with CUDA support (for Graviton)
- CUDA 12.0+
- For Python workloads: Python 3.10+, NumPy, Matplotlib
- For C++ workloads: GCC 11.0+, CMake 3.22+

### Clone the Repository

```bash
git clone https://github.com/username/nvidia-jetson-workload.git
cd nvidia-jetson-workload
```

### Build from Source

Build all workloads:

```bash
./build.sh
```

Or build a specific workload:

```bash
cd src/nbody_sim/cpp
./build_and_test.sh
```

### Using Docker

For containerized execution, use the provided Docker images:

```bash
docker pull ghcr.io/username/nvidia-jetson-workload/benchmark:latest
docker run --gpus all -v $(pwd)/results:/app/benchmark/results ghcr.io/username/nvidia-jetson-workload/benchmark --all
```

### Using Singularity

For high-performance computing environments, use the Singularity container:

```bash
singularity build benchmark.sif containers/benchmark.def
singularity run --nv benchmark.sif --all
```

## Available Workloads

### 1. N-body Gravitational Simulation

Physics-based simulation of gravitational interactions between particles.

- Multiple integrators: Euler, Leapfrog, Verlet, RK4
- Several system types: random, solar system, galaxy
- C++/CUDA implementation with Python bindings
- Comprehensive benchmark and visualization tools

[Learn more about N-body Simulation](workloads/nbody-sim.md)

### 2. Molecular Dynamics Simulation

Simulation of molecular systems with multiple force fields.

- Supports multiple integrators and thermostats
- Implements Lennard-Jones potentials and Coulomb interactions
- GPU-accelerated with architecture-specific optimizations
- Energy conservation and performance tracking

[Learn more about Molecular Dynamics Simulation](workloads/molecular-dynamics.md)

### 3. Weather Simulation

Implementation of multiple numerical weather models.

- Shallow Water Equations, Barotropic Model, Primitive Equations
- CUDA kernels with architecture-specific optimizations
- GPU adaptability pattern for optimal cross-device performance
- Python bindings with NumPy integration and visualization tools

[Learn more about Weather Simulation](workloads/weather-sim.md)

### 4. Medical Imaging

Comprehensive medical image processing workload.

- CT reconstruction (filtered backprojection, iterative methods)
- Image processing (convolution, filtering, transformation)
- Segmentation (thresholding, watershed, level set, graph cut)
- Registration (image warping, mutual information)
- CUDA kernels with architecture-specific optimizations
- Python bindings with NumPy integration and visualization utilities

[Learn more about Medical Imaging](workloads/medical-imaging.md)

## Running Workloads

### C++ Executables

Run workloads using their C++ executables:

```bash
# N-body Simulation
./build/bin/nbody_sim --num-particles 10000 --system-type random --integrator leapfrog

# Weather Simulation
./build/bin/weather_sim --grid-size 512 --model shallow_water
```

### Python Interface

All workloads can be run through their Python interfaces:

```python
# N-body Simulation
from nbody_sim.simulation import Simulation
from nbody_sim.integrator import IntegratorType

sim = Simulation()
sim.initialize(
    num_particles=10000,
    system_type="random",
    integrator_type=IntegratorType.LEAPFROG
)

for _ in range(1000):
    sim.step(0.01)

# Visualize results
from nbody_sim import visualization
visualization.plot_system(sim)
```

### Example Scripts

Ready-to-use example scripts are provided for each workload:

```bash
# N-body Simulation
python -m nbody_sim.examples.solar_system

# Medical Imaging
python -m medical_imaging.examples.ct_reconstruction_example
```

## Benchmarking

The benchmark suite provides a unified framework for evaluating the performance of all implemented scientific workloads across different hardware configurations.

### Running Benchmarks

Run benchmarks for all workloads:

```bash
./benchmark/scripts/run_benchmarks.sh --all
```

Run benchmarks for specific workloads:

```bash
./benchmark/scripts/run_benchmarks.sh --nbody --weather
```

Specify GPU device:

```bash
./benchmark/scripts/run_benchmarks.sh --device 1 --all
```

Generate report from existing results:

```bash
./benchmark/scripts/run_benchmarks.sh --report
```

### Benchmark Results

Benchmark results include:

- Execution time comparison
- GPU utilization
- Memory usage
- Energy consumption
- Throughput metrics
- Hardware comparison
- Performance radar charts

Results are saved in HTML format in the `benchmark/results` directory. This report includes interactive charts and tables for analyzing performance.

### Automated Benchmarking

You can configure GitHub Actions to run benchmarks automatically:

- On a schedule (e.g., weekly)
- On demand through workflow_dispatch
- For specific workloads or all workloads
- On different AWS instance types

Results can be automatically uploaded to an S3 bucket for tracking performance over time.

## Deployment

### Deploying Workloads to Target Systems

The unified deployment script allows you to deploy workloads to one or more target systems:

```bash
# Deploy all workloads to all nodes
./scripts/deploy-all.sh

# Deploy specific workloads to specific nodes
./scripts/deploy-all.sh --workloads nbody_sim,weather_sim --nodes orin1,orin2

# Specify SSH user and key
./scripts/deploy-all.sh --user jetson --key ~/.ssh/id_rsa

# Deploy to AWS instances
./scripts/deploy-all.sh --aws --instances i-123456,i-789012
```

### Continuous Integration/Continuous Deployment

CI/CD workflows are provided for:

- Building and testing code
- Linting and code quality checks
- Building Docker images
- Deploying to staging and production environments

For more information, see [CI/CD Documentation](ci-cd.md).

## Visualization Dashboard

The project includes a web-based dashboard for visualizing workload results and benchmark data.

### Starting the Dashboard

```bash
cd src/visualization
npm install
npm start
```

Then navigate to http://localhost:3000 in your browser.

### Dashboard Features

- Real-time performance monitoring
- Workload management
- Benchmark result visualization
- Hardware comparison charts
- Cluster status overview

For more information, see [Visualization Dashboard Documentation](visualization-dashboard.md).

## Cost Analysis

The cost comparison framework allows you to compare execution costs between NVIDIA Jetson devices, cloud providers, DGX systems, and Slurm clusters.

### Running Cost Analysis

Enable cost analysis by adding the `--cost-analysis` flag to benchmark commands:

```bash
# Run all benchmarks with cost analysis
./benchmark/scripts/run_benchmarks.sh --all --cost-analysis

# Run with DGX and Slurm comparisons
./benchmark/scripts/run_benchmarks.sh --nbody --cost-analysis \
  --dgx-system-type dgx_a100 --slurm-node-type highend_gpu
  
# Use custom configuration files
./benchmark/scripts/run_benchmarks.sh --all --cost-analysis \
  --dgx-config benchmark/configs/dgx_custom.yaml \
  --slurm-config benchmark/configs/slurm_custom.yaml
```

### Cost Metrics and Reports

The cost analysis includes:

- **Total Cost**: Hardware amortization, operational costs, and cloud instance costs
- **Cost Efficiency**: Operations per dollar spent
- **Cost Ratio**: Comparison between different computing environments
- **Break-Even Analysis**: When local computing becomes more economical than cloud

Results are visualized in the benchmark report with comparative charts and tables.

### Sample Reports and Resources

- [Cost Comparison Guide](cost-comparison.md) - Full documentation of cost modeling features
- [Sample Cost Comparison Report](sample-reports/cost-comparison-sample.md) - Example reports with interpretation
- [Detailed Cost Analysis Example](sample-reports/detailed-cost-analysis-example.md) - Step-by-step walkthrough of cost calculations
- [Interpreting Cost Results](sample-reports/interpreting-cost-results.md) - Guide to understanding metrics and visualizations

For more information, see [Cost Comparison Guide](cost-comparison.md).

## Advanced Configuration

### GPU Adaptability Pattern

The GPU adaptability pattern automatically tunes workload parameters based on the available hardware. Understanding this pattern is key to optimizing performance across different devices.

For more information, see [GPU Adaptability Pattern Documentation](../gpu_adaptability_pattern.md).

### Performance Tuning

Each workload can be further tuned for specific hardware configurations:

- Tiled algorithm parameters
- Shared memory usage
- Thread block sizes
- Memory access patterns

For more information, see [Performance Tuning Guide](performance-tuning.md).

## Troubleshooting

### Common Issues

- **CUDA not available**: Ensure CUDA drivers are installed and `nvcc` is in your PATH
- **Build fails**: Check that you have the required dependencies installed
- **Benchmark errors**: Verify that all workloads are built correctly
- **Deployment fails**: Check SSH connectivity and permissions on target systems

### Getting Help

- Check the [FAQ](faq.md) for common questions
- Open an issue on GitHub
- Consult the [troubleshooting guide](troubleshooting.md) for detailed steps

## Contributing

Contributions to this project are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

For more information on development, see the [Developer Guide](../developer-guide.md).