# N-Body Simulation Workload

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2024 nvidia-jetson-workload contributors -->

This directory contains implementations of an N-body gravitational simulation workload for NVIDIA Jetson devices.

## Overview

The N-body simulation models the gravitational interactions between particles in a system. It demonstrates GPU acceleration by offloading the computationally intensive force calculations to the CUDA cores on the Jetson's GPU.

The workload includes:
- A pure Python implementation using NumPy
- A C++/CUDA implementation for maximum performance
- Integration with Slurm for job submission and scheduling

## Python Implementation

The Python implementation is located in the `python/` directory and consists of the following components:

- `particle.py`: Defines the `Particle` and `ParticleSystem` classes
- `integrator.py`: Implements various numerical integration methods (Euler, Leapfrog, Verlet, RK4)
- `simulation.py`: Main simulation class that combines a particle system with an integrator
- `cli.py`: Command-line interface for running the simulation
- `run_test.py`: Simple test script to verify functionality

### Running the Python Version

To run the Python implementation on a Jetson device:

```bash
# Basic run with default settings
python -m src.nbody_sim.python.cli

# Advanced run with specific parameters
python -m src.nbody_sim.python.cli \
    --system-type galaxy \
    --num-particles 10000 \
    --integrator leapfrog \
    --duration 5.0 \
    --dt 0.01 \
    --output-dir ./nbody-output
```

## C++/CUDA Implementation

The C++/CUDA implementation is located in the `cpp/` directory (coming soon).

## Slurm Integration

The Slurm job scripts are located in the `slurm/nbody-sim/` directory:

- `nbody_job.sbatch`: Slurm job script for running the simulation
- `submit_job.py`: Python script for submitting jobs to Slurm

### Submitting a Job to Slurm

```bash
# Submit a job to Slurm with default parameters
python slurm/nbody-sim/submit_job.py

# Submit a job with specific parameters
python slurm/nbody-sim/submit_job.py \
    --system-type galaxy \
    --num-particles 50000 \
    --duration 10.0 \
    --time-step 0.005 \
    --integrator verlet \
    --nodes orin1 orin2 \
    --wait
```

## Performance Optimization

The N-body simulation is optimized for the Jetson Orin NX by:

1. Using GPU acceleration for the O(n²) force calculation
2. Implementing efficient memory access patterns
3. Utilizing shared memory for frequently accessed data
4. Balancing work between CPU and GPU