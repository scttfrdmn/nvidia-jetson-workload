# Benchmark Suite for GPU-Accelerated Scientific Workloads

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This benchmark suite provides a unified framework for evaluating the performance of all implemented scientific workloads across different hardware configurations.

## Features

- **Integrated Benchmarking**: Run benchmarks for all workloads with a single command
- **Comprehensive Metrics**: Measure execution time, memory usage, GPU utilization, and throughput
- **Visualization Tools**: Generate plots and reports for performance comparison
- **Hardware Adaptability**: Automatically detects and adapts to different GPU architectures
- **Docker/Singularity Support**: Run benchmarks in containers for reproducibility

## Workloads

The benchmark suite includes the following workloads:

1. **N-body Gravitational Simulation**
   - Configurable particle count, system types, and integrators
   - Measures interactions per second and scaling efficiency

2. **Molecular Dynamics Simulation**
   - Configurable atom count, force fields, and thermostats
   - Measures energy conservation and force calculation performance

3. **Weather Simulation**
   - Multiple numerical models (Shallow Water, Barotropic, Primitive Equations)
   - Measures grid points processed per second

4. **Medical Imaging**
   - Multiple tasks (CT reconstruction, segmentation, registration)
   - Measures processing throughput for different image sizes

## Usage

### Command Line

```bash
# Run all benchmarks
./scripts/run_benchmarks.sh --all

# Run specific workload benchmarks
./scripts/run_benchmarks.sh --nbody --medical

# Specify GPU device
./scripts/run_benchmarks.sh --device 1 --all

# Generate report from existing results
./scripts/run_benchmarks.sh --report
```

### Python API

```python
from benchmark.benchmark_suite import BenchmarkSuite

# Create benchmark suite
suite = BenchmarkSuite(device_id=0, output_dir="results")

# Run all benchmarks
suite.run_all(
    nbody_params={"num_particles": 10000, "num_steps": 1000},
    md_params={"num_atoms": 5000, "num_steps": 1000},
    weather_params={"grid_size": 512, "num_steps": 1000},
    medical_params={"image_size": 512, "num_iterations": 10}
)

# Generate reports
suite.generate_reports()
```

## Docker

Build and run the Docker container:

```bash
# Build container
docker build -t gpu-workloads-benchmark -f containers/benchmark.Dockerfile .

# Run benchmarks
docker run --gpus all -v $(pwd)/results:/app/benchmark/results gpu-workloads-benchmark --all
```

## Singularity

Build and run the Singularity container:

```bash
# Build container
singularity build benchmark.sif containers/benchmark.def

# Run benchmarks
singularity run --nv benchmark.sif --all
```

## Output

The benchmark suite generates a comprehensive HTML report with:

- **Performance Metrics Table**: Execution time, memory usage, GPU utilization, throughput
- **Execution Time Comparison**: Bar chart comparing execution time across workloads and devices
- **Memory Usage**: Charts for host and device memory usage
- **GPU Utilization**: Utilization percentage for each workload
- **Throughput Comparison**: Workload-specific throughput metrics
- **System Information**: Details about the hardware used for benchmarking

## Customizing Benchmarks

Each workload supports various parameters for customization:

### N-body Simulation

- `--nbody-particles`: Number of particles (default: 10000)
- `--nbody-steps`: Number of simulation steps (default: 1000)
- `--nbody-system`: System type (random, solar_system, galaxy)
- `--nbody-integrator`: Integrator (euler, leapfrog, verlet, rk4)

### Molecular Dynamics

- `--md-atoms`: Number of atoms (default: 5000)
- `--md-steps`: Number of simulation steps (default: 1000)
- `--md-forcefield`: Force field (lennard_jones, coulomb)

### Weather Simulation

- `--weather-grid`: Grid size (default: 512)
- `--weather-steps`: Number of simulation steps (default: 1000)
- `--weather-model`: Model (shallow_water, barotropic, primitive)

### Medical Imaging

- `--medical-size`: Image size (default: 512)
- `--medical-task`: Task (ct_reconstruction, segmentation, registration)
- `--medical-iterations`: Number of iterations (default: 10)