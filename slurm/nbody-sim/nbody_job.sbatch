#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 nvidia-jetson-workload contributors

#SBATCH --job-name=nbody-sim        # Job name
#SBATCH --output=nbody-%j.log       # Output file (%j expands to jobId)
#SBATCH --error=nbody-%j.err        # Error file
#SBATCH --nodes=1                   # Request one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=4           # Use 4 CPU cores
#SBATCH --gres=gpu:1                # Request one GPU
#SBATCH --time=00:10:00             # Time limit (10 minutes)
#SBATCH --mem=4G                    # Memory limit

# Print node information
echo "Job running on $(hostname)"
echo "Node information:"
lscpu | grep "Model name"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# Set up environment variables
export PYTHONPATH=/home/ubuntu/nvidia-jetson-workload:$PYTHONPATH

# Create output directory for this job
JOB_OUTPUT_DIR="/home/ubuntu/nbody-outputs/job-$SLURM_JOB_ID"
mkdir -p $JOB_OUTPUT_DIR

# Parse command-line arguments, with defaults
NBODY_TYPE=${1:-"galaxy"}
NBODY_PARTICLES=${2:-5000}
NBODY_DURATION=${3:-5.0}
NBODY_TIME_STEP=${4:-0.01}
NBODY_INTEGRATOR=${5:-"leapfrog"}

echo "=== N-body Simulation Configuration ==="
echo "Type: $NBODY_TYPE"
echo "Particles: $NBODY_PARTICLES"
echo "Duration: $NBODY_DURATION"
echo "Time Step: $NBODY_TIME_STEP"
echo "Integrator: $NBODY_INTEGRATOR"
echo "======================================="

# Change to source directory
cd /home/ubuntu/nvidia-jetson-workload

# Print current working directory
echo "Working directory: $(pwd)"

# Run the simulation
echo "Starting N-body simulation at $(date)"
START_TIME=$(date +%s.%N)

python -m src.nbody_sim.python.cli \
    --system-type $NBODY_TYPE \
    --num-particles $NBODY_PARTICLES \
    --duration $NBODY_DURATION \
    --dt $NBODY_TIME_STEP \
    --integrator $NBODY_INTEGRATOR \
    --output-dir $JOB_OUTPUT_DIR \
    --save-visualization

END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
echo "Simulation completed in $ELAPSED seconds"

# Print summary of results
echo "Results saved to $JOB_OUTPUT_DIR"
echo "Performance metrics:"
cat $JOB_OUTPUT_DIR/metrics.json

# Compress output files for easier downloading
cd $(dirname $JOB_OUTPUT_DIR)
tar -czvf "nbody-job-$SLURM_JOB_ID.tar.gz" "job-$SLURM_JOB_ID"
echo "Outputs compressed to nbody-job-$SLURM_JOB_ID.tar.gz"