# Benchmarking Guide

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This guide provides detailed instructions for using the benchmarking suite to evaluate and compare the performance of GPU-accelerated scientific workloads across different hardware configurations.

## Table of Contents

1. [Overview](#overview)
2. [Benchmark Suite Architecture](#benchmark-suite-architecture)
3. [Running Benchmarks](#running-benchmarks)
4. [Understanding Benchmark Results](#understanding-benchmark-results)
5. [Comparing Hardware Configurations](#comparing-hardware-configurations)
6. [Integration with CI/CD](#integration-with-cicd)
7. [Custom Benchmark Configurations](#custom-benchmark-configurations)
8. [Adding New Benchmarks](#adding-new-benchmarks)

## Overview

The benchmarking suite provides a unified framework for evaluating the performance of all implemented scientific workloads across different hardware configurations. It measures key metrics such as execution time, memory usage, GPU utilization, energy consumption, and workload-specific throughput metrics.

## Benchmark Suite Architecture

The benchmark suite consists of the following key components:

- **Benchmark Runner** (`benchmark_suite.py`): Core benchmarking tool that executes workloads and collects metrics
- **Visualization Module** (`visualization.py`): Generates plots and HTML reports from benchmark results
- **Command-line Interface** (`run_benchmarks.sh`): Script for running benchmarks with various options
- **Docker/Singularity Support**: Containers for reproducible benchmarking across environments
- **CI/CD Integration**: GitHub Actions workflows for automated benchmarking

### Key Features

- **Unified Metrics**: Consistent collection of metrics across all workloads
- **Hardware Adaptability**: Automatic detection and adaptation to different GPU architectures
- **Cross-Workload Comparison**: Tools for comparing performance across different workloads
- **Cross-Hardware Comparison**: Tools for comparing performance across different hardware configurations
- **Comprehensive Reporting**: Detailed HTML reports with interactive charts and tables

## Running Benchmarks

### Basic Usage

Run benchmarks for all workloads:

```bash
./benchmark/scripts/run_benchmarks.sh --all
```

Run benchmarks for specific workloads:

```bash
./benchmark/scripts/run_benchmarks.sh --nbody --medical
```

Specify GPU device:

```bash
./benchmark/scripts/run_benchmarks.sh --device 1 --all
```

Generate report from existing results:

```bash
./benchmark/scripts/run_benchmarks.sh --report
```

### Command-line Options

The `run_benchmarks.sh` script accepts the following options:

| Option | Description |
|--------|-------------|
| `--all` | Run all benchmarks |
| `--nbody` | Run N-body simulation benchmark |
| `--md` | Run Molecular Dynamics benchmark |
| `--weather` | Run Weather Simulation benchmark |
| `--medical` | Run Medical Imaging benchmark |
| `--device N` | Use GPU device N |
| `--output DIR` | Specify output directory for results |
| `--report` | Generate report from existing results |

### N-body Simulation Benchmark Options

| Option | Description | Default |
|--------|-------------|---------|
| `--nbody-particles N` | Number of particles | 10000 |
| `--nbody-steps N` | Number of simulation steps | 1000 |
| `--nbody-system TYPE` | System type (random, solar_system, galaxy) | random |
| `--nbody-integrator TYPE` | Integrator (euler, leapfrog, verlet, rk4) | leapfrog |

### Molecular Dynamics Benchmark Options

| Option | Description | Default |
|--------|-------------|---------|
| `--md-atoms N` | Number of atoms | 5000 |
| `--md-steps N` | Number of simulation steps | 1000 |
| `--md-forcefield TYPE` | Force field (lennard_jones, coulomb) | lennard_jones |

### Weather Simulation Benchmark Options

| Option | Description | Default |
|--------|-------------|---------|
| `--weather-grid N` | Grid size | 512 |
| `--weather-steps N` | Number of simulation steps | 1000 |
| `--weather-model TYPE` | Model (shallow_water, barotropic, primitive) | shallow_water |

### Medical Imaging Benchmark Options

| Option | Description | Default |
|--------|-------------|---------|
| `--medical-size N` | Image size | 512 |
| `--medical-task TYPE` | Task (ct_reconstruction, segmentation, registration) | ct_reconstruction |
| `--medical-iterations N` | Number of iterations | 10 |

## Understanding Benchmark Results

Benchmark results are stored in the specified output directory (default: `benchmark/results`). The main output is an HTML report (`benchmark_report.html`) that includes various charts and tables.

### Execution Time

The execution time chart shows how long each workload takes to run on different hardware configurations. Lower execution time is better.

### GPU Utilization

The GPU utilization chart shows how effectively each workload uses the GPU. Higher utilization generally indicates better GPU usage. However, very high utilization (>95%) might also indicate a bottleneck.

### Memory Usage

The memory usage charts show both host (CPU) and device (GPU) memory usage for each workload. This helps identify memory-bound workloads and optimize memory usage.

### Energy Consumption

The energy consumption chart shows how much energy each workload consumes, providing insights into power efficiency. Lower energy consumption is generally better, especially for power-constrained environments like Jetson devices.

### Throughput

The throughput chart shows workload-specific throughput metrics:
- N-body Simulation: Interactions per second
- Molecular Dynamics: Interactions per second
- Weather Simulation: Grid points per second
- Medical Imaging: Iterations per second

Higher throughput is better.

### Performance Radar

The performance radar chart provides a visual representation of how a workload performs across different metrics and hardware configurations. This multi-dimensional view helps identify overall performance characteristics.

## Comparing Hardware Configurations

The benchmark suite is designed to facilitate direct comparisons between different hardware configurations. To compare:

1. Run the same benchmarks on different hardware
2. Copy the result files to a common location
3. Generate a comparative report

```bash
# Run benchmarks on Jetson Orin
./benchmark/scripts/run_benchmarks.sh --all --output benchmark/results/jetson

# Run benchmarks on AWS Graviton g5g
./benchmark/scripts/run_benchmarks.sh --all --output benchmark/results/graviton

# Generate comparative report
python benchmark/benchmark_suite.py --report --directory benchmark/results
```

### Speedup Analysis

The hardware comparison section of the report includes a speedup analysis table that shows:
- Execution time speedup
- Throughput ratio
- Energy efficiency ratio

This helps quantify the relative performance advantage of one hardware configuration over another.

## Integration with CI/CD

The benchmark suite integrates with GitHub Actions for automated benchmarking.

### GitHub Actions Workflow

The `.github/workflows/benchmark.yml` workflow allows you to:
- Run benchmarks on demand through workflow_dispatch
- Run scheduled benchmarks (e.g., weekly)
- Specify workloads to benchmark
- Specify AWS instance type
- Upload results to S3

### Tracking Performance Over Time

By running benchmarks regularly and storing results, you can track performance changes over time, such as:
- Performance regressions
- Improvements from optimizations
- Effects of hardware or driver updates

### Setting Up S3 Result Storage

To store benchmark results in S3:

1. Create an S3 bucket for benchmark results
2. Set up AWS credentials as GitHub secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
   - `BENCHMARK_BUCKET`
3. Enable result upload in the workflow

## Custom Benchmark Configurations

You can create custom benchmark configurations for specific testing needs.

### Python API

Use the Python API for fine-grained control:

```python
from benchmark.benchmark_suite import BenchmarkSuite

# Create benchmark suite
suite = BenchmarkSuite(device_id=0, output_dir="custom_results")

# Run N-body simulation benchmark with custom parameters
suite.run_benchmark("nbody_sim", 
                   num_particles=50000,
                   num_steps=500,
                   system_type="galaxy",
                   integrator="rk4")

# Generate reports
suite.generate_reports()
```

### Configuration Files

You can also use configuration files for reproducible benchmarking:

```json
{
  "nbody_sim": {
    "num_particles": 50000,
    "num_steps": 500,
    "system_type": "galaxy",
    "integrator": "rk4"
  },
  "weather_sim": {
    "grid_size": 1024,
    "num_steps": 500,
    "model": "primitive"
  }
}
```

```bash
./benchmark/scripts/run_benchmarks.sh --config path/to/config.json
```

## Adding New Benchmarks

You can extend the benchmark suite with new workloads by following these steps:

1. Create a new benchmark class in `benchmark_suite.py`:

```python
class NewWorkloadBenchmark(WorkloadBenchmark):
    """Benchmark for new workload."""
    
    def __init__(self, device_id: int = 0):
        super().__init__("new_workload", device_id)
    
    def run(self, **kwargs) -> BenchmarkResult:
        """Run the benchmark."""
        # Implementation details
        pass
```

2. Add the new benchmark to the `BenchmarkSuite` class:

```python
self.benchmarks = {
    "nbody_sim": NBodySimBenchmark(device_id),
    "molecular_dynamics": MolecularDynamicsBenchmark(device_id),
    "weather_sim": WeatherSimulationBenchmark(device_id),
    "medical_imaging": MedicalImagingBenchmark(device_id),
    "new_workload": NewWorkloadBenchmark(device_id)
}
```

3. Update the command-line interface in `run_benchmarks.sh`

4. Add visualization support in `visualization.py`

For more details, see the [Developer Guide](../developer-guide.md).