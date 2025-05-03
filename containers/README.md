# Container Definitions

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This directory contains container definitions for the workloads.

## Container Types

### Docker Containers

Docker containers are used primarily for development and testing:

```bash
# Build the Docker container
docker build -t nvidia-jetson-workload/nbody-sim -f containers/nbody-sim.Dockerfile .

# Run the Docker container
docker run --gpus all nvidia-jetson-workload/nbody-sim --system-type galaxy --num-particles 10000
```

### Singularity/Apptainer Containers

Singularity containers are used for deployment to Slurm clusters:

```bash
# Build the Singularity container
singularity build nbody-sim.sif containers/nbody-sim.def

# Run the Singularity container
singularity run --nv nbody-sim.sif --system-type galaxy --num-particles 10000
```

## Container Compatibility

The containers are designed to work on both:
- NVIDIA Jetson Orin NX (ARM64 with CUDA)
- AWS Graviton g5g instances (ARM64 with NVIDIA GPUs)

This is achieved by using NVIDIA's L4T (Linux for Tegra) container images as the base.

## Container Structure

Each container includes:
1. All dependencies pre-installed
2. The complete codebase at `/opt/nvidia-jetson-workload`
3. Default output directory at `/output`
4. Entry point configured to run the workload

## Building Cross-Platform Containers

For cross-platform development, you can use Docker's BuildX:

```bash
# Create a builder instance
docker buildx create --name armbuilder --use

# Build for multiple platforms
docker buildx build --platform linux/arm64 -t nvidia-jetson-workload/nbody-sim:arm64 -f containers/nbody-sim.Dockerfile .
```