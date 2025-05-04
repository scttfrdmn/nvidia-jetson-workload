# Cost Modeling Configuration Files

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This directory contains example configuration files for the cost modeling system. These files can be used to define custom hardware configurations for cost comparison analysis.

## Slurm Cluster Configurations

Slurm cluster configuration files define the specifications and costs associated with running workloads on a Slurm-managed cluster.

- `slurm_cluster_basic.yaml`: A basic Slurm cluster with 4 GPU-equipped nodes
- `slurm_cluster_highend.yaml`: A high-end research cluster with 16 powerful GPU nodes
- `slurm_cluster_jetson.yaml`: A cluster built with 32 Jetson Orin NX nodes
- `slurm_cluster_custom.yaml`: A custom cluster configuration with 8 nodes using RTX A6000 GPUs

## DGX System Configurations

DGX system configuration files define the specifications and costs associated with running workloads on NVIDIA DGX systems.

- `dgx_a100.yaml`: A single DGX A100 system
- `dgx_h100.yaml`: A pair of DGX H100 systems
- `dgx_station_a100.yaml`: A DGX Station A100 workstation
- `dgx_superpod.yaml`: A DGX SuperPOD with 20 DGX H100 nodes

## Usage

To use these configuration files with the benchmark suite, use the following command-line arguments:

### Using a Slurm cluster configuration

```bash
./benchmark/scripts/run_benchmarks.sh --all --cost-analysis --slurm-config benchmark/configs/slurm_cluster_highend.yaml
```

### Using a DGX system configuration

```bash
./benchmark/scripts/run_benchmarks.sh --all --cost-analysis --dgx-config benchmark/configs/dgx_superpod.yaml
```

### Using both configurations

```bash
./benchmark/scripts/run_benchmarks.sh --all --cost-analysis \
  --slurm-config benchmark/configs/slurm_cluster_highend.yaml \
  --dgx-config benchmark/configs/dgx_superpod.yaml
```

## Custom Configuration Format

You can create your own configuration files by copying and modifying the provided examples. The configuration files use YAML format and must contain the required parameters for the corresponding system type.