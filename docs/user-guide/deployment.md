# Deployment Guide

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This guide provides detailed instructions for deploying GPU-accelerated scientific workloads to target systems, including Jetson Orin NX devices and AWS Graviton g5g instances.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Unified Deployment Script](#unified-deployment-script)
4. [Deployment to Jetson Devices](#deployment-to-jetson-devices)
5. [Deployment to AWS Instances](#deployment-to-aws-instances)
6. [CI/CD Integration](#cicd-integration)
7. [Containerized Deployment](#containerized-deployment)
8. [Slurm Integration](#slurm-integration)
9. [Troubleshooting](#troubleshooting)

## Overview

The project includes a unified deployment system that allows you to build and deploy one or more workloads to one or more target systems. It supports both Jetson Orin NX devices and AWS Graviton g5g instances.

## Prerequisites

### For All Deployments

- SSH access to target systems
- Python 3.10+ installed on target systems
- CUDA 12.0+ installed on target systems

### For Jetson Deployments

- JetPack 6.0+ installed on Jetson devices
- Properly configured `/etc/hosts` file or DNS resolution for Jetson devices

### For AWS Deployments

- AWS CLI configured with appropriate credentials
- EC2 instances with proper security groups allowing SSH access
- Graviton g5g instances with NVIDIA T4 GPUs

## Unified Deployment Script

The `deploy-all.sh` script provides a unified interface for deploying workloads to target systems.

### Basic Usage

```bash
# Deploy all workloads to all configured nodes
./scripts/deploy-all.sh

# Deploy specific workloads to specific nodes
./scripts/deploy-all.sh --workloads nbody_sim,weather_sim --nodes orin1,orin2

# Build and deploy
./scripts/deploy-all.sh --build --workloads all
```

### Available Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-w, --workloads WORKLOADS` | Comma-separated list of workloads to deploy (default: all) |
| `-n, --nodes NODES` | Comma-separated list of target nodes (default: orin1,orin2,orin3,orin4) |
| `-u, --user USER` | SSH user (default: ubuntu) |
| `-k, --key SSH_KEY` | Path to SSH private key |
| `-d, --dir DEPLOY_DIR` | Directory on target to deploy to (default: /opt/nvidia-jetson-workload) |
| `-b, --build` | Build workloads before deploying |
| `-a, --aws` | AWS deployment mode (uses instance IDs instead of hostnames) |
| `-i, --instances INSTANCES` | Comma-separated list of AWS instance IDs |

### Examples

```bash
# Deploy all workloads to Jetson devices
./scripts/deploy-all.sh --workloads all --nodes orin1,orin2,orin3,orin4 --user jetson

# Deploy specific workloads to specific Jetson devices
./scripts/deploy-all.sh --workloads nbody_sim,weather_sim --nodes orin1,orin2 --user jetson

# Deploy all workloads to AWS instances
./scripts/deploy-all.sh --aws --instances i-1234567890abcdef0,i-0987654321fedcba0

# Build and deploy specific workloads to AWS instances
./scripts/deploy-all.sh --aws --instances i-1234567890abcdef0 --workloads medical_imaging --build
```

## Deployment to Jetson Devices

### Setting Up Jetson Devices

Use the provided setup script to prepare Jetson devices:

```bash
./scripts/setup-jetson.sh orin1 jetson
```

This script:
1. Installs required packages
2. Configures SSH access
3. Sets up environment variables
4. Creates necessary directories

### Deployment Flow

When deploying to Jetson devices, the script:

1. Builds the specified workloads (if `--build` is specified)
2. Creates a deployment package with binaries and Python modules
3. Establishes SSH connections to target devices
4. Copies files to target devices
5. Sets up Python environment and dependencies
6. Configures environment variables
7. Verifies deployment

### Test Deployment

After deploying, verify the installation:

```bash
ssh jetson@orin1 "cd /opt/nvidia-jetson-workload && python -c 'import nbody_sim; print(\"N-body simulation module loaded successfully\")'"
```

## Deployment to AWS Instances

### Setting Up AWS Instances

1. Launch AWS Graviton g5g instances with NVIDIA T4 GPUs
2. Install CUDA 12.0+ and other dependencies
3. Configure security groups to allow SSH access

### AWS Deployment Mode

The `--aws` flag switches the script to AWS deployment mode, which:
1. Takes instance IDs instead of hostnames
2. Uses AWS CLI to get instance public IPs
3. Handles AWS-specific environment setup

### AWS Deployment Flow

When deploying to AWS instances, the script:

1. Builds the specified workloads (if `--build` is specified)
2. Creates a deployment package with binaries and Python modules
3. Gets public IPs for the specified instance IDs
4. Establishes SSH connections to instances
5. Copies files to instances
6. Sets up Python environment and dependencies
7. Configures environment variables
8. Verifies deployment

### Example: Deploying to a g5g.2xlarge Instance

```bash
# Launch instance (one-time setup)
aws ec2 run-instances --image-id ami-12345678 --instance-type g5g.2xlarge --key-name your-key --security-group-ids sg-12345678

# Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=instance-type,Values=g5g.2xlarge" "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].InstanceId" --output text)

# Deploy workloads
./scripts/deploy-all.sh --aws --instances $INSTANCE_ID --workloads all --build
```

## CI/CD Integration

The project includes GitHub Actions workflows for automated deployment.

### Deployment Workflow

The `.github/workflows/deploy.yml` workflow allows you to:
- Deploy to staging or production environments
- Deploy specific workloads
- Deploy to multiple instances in parallel

### Setting Up CI/CD Deployment

1. Configure GitHub secrets:
   - `AWS_ACCESS_KEY_ID`: AWS access key ID
   - `AWS_SECRET_ACCESS_KEY`: AWS secret access key
   - `AWS_REGION`: AWS region
   - `STAGING_INSTANCE_ID`: Comma-separated list of staging instance IDs
   - `PRODUCTION_INSTANCE_ID`: Comma-separated list of production instance IDs
   - `SSH_PRIVATE_KEY`: SSH private key for accessing instances

2. Trigger deployment:
   - Manually through workflow_dispatch
   - Automatically on specific branch pushes

### Sample Workflow Dispatch Configuration

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
      workloads:
        description: 'Workloads to deploy (comma-separated)'
        required: false
        default: 'all'
        type: string
```

## Containerized Deployment

For simplified deployment, you can use Docker or Singularity containers.

### Docker Deployment

```bash
# Build Docker image
docker build -t workloads -f containers/benchmark.Dockerfile .

# Run on a single node
docker run --gpus all -v /path/to/results:/app/benchmark/results workloads --nbody --medical

# Run on multiple nodes
for node in orin1 orin2 orin3 orin4; do
  ssh $node "docker pull user/workloads:latest && docker run --gpus all workloads"
done
```

### Singularity Deployment

Singularity is particularly useful for HPC environments:

```bash
# Build Singularity image
singularity build workloads.sif containers/benchmark.def

# Copy to node and run
scp workloads.sif user@orin1:/tmp/
ssh user@orin1 "singularity run --nv /tmp/workloads.sif"
```

## Slurm Integration

For HPC clusters with Slurm, the project includes Slurm job scripts.

### Slurm Job Templates

Templates are provided in the `slurm` directory:

```bash
# Submit a job to run N-body simulation
sbatch slurm/nbody-sim/nbody_job.sbatch

# Submit with specific parameters
sbatch --export=NUM_PARTICLES=100000,STEPS=10000 slurm/nbody-sim/nbody_job.sbatch
```

### Automatic Job Submission

The `submit_job.py` script allows programmatic job submission:

```bash
python slurm/nbody-sim/submit_job.py --nodes 4 --time 2:00:00 --particles 100000
```

## Troubleshooting

### Common Deployment Issues

#### SSH Connection Failures

```
Error: Cannot connect to [host] as [user]
```

Solutions:
- Verify that the host is reachable with `ping`
- Check that SSH keys are properly set up with `ssh-copy-id`
- Verify that the SSH port (22) is open in firewalls and security groups

#### Permission Denied

```
Permission denied when creating directory on target
```

Solutions:
- Verify that the target user has write permissions to the target directory
- Use a different target directory with `--dir`
- Use sudo on the target system: `--user root` (if you have sudo access)

#### Missing Dependencies

```
ImportError: No module named 'numpy'
```

Solutions:
- Verify that required Python packages are installed on the target system
- Install manually with `pip install -r requirements.txt`
- Check for architecture compatibility (arm64 vs. x86_64)

#### CUDA Not Found

```
Failed to load CUDA library
```

Solutions:
- Verify that CUDA is installed on the target system
- Check CUDA version compatibility (12.0+ required)
- Ensure LD_LIBRARY_PATH includes CUDA libraries
- Check for appropriate drivers for the GPU

### Logs and Debugging

The deployment script outputs detailed logs to help diagnose issues. You can increase verbosity by setting the `DEBUG` environment variable:

```bash
DEBUG=1 ./scripts/deploy-all.sh
```

For persistent logs, redirect output to a file:

```bash
./scripts/deploy-all.sh --workloads all --nodes orin1 | tee deployment.log
```