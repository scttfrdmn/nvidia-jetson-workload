#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Unified deployment script for all workloads
# This script deploys all or selected workloads to target nodes

set -e

# Default values
WORKLOADS="all"
NODES=("orin1" "orin2" "orin3" "orin4")
TARGET_USER="ubuntu"
SSH_KEY=""
DEPLOY_DIR="/opt/nvidia-jetson-workload"
BUILD=false
AWS_MODE=false
AWS_INSTANCES=()

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# Parse command line arguments
function show_help {
  echo "Usage: $0 [OPTIONS]"
  echo "Deploy workloads to target nodes"
  echo ""
  echo "Options:"
  echo "  -h, --help                   Show this help message"
  echo "  -w, --workloads WORKLOADS    Comma-separated list of workloads to deploy (default: all)"
  echo "                               Available: nbody_sim,molecular_dynamics,weather_sim,medical_imaging,all"
  echo "  -n, --nodes NODES            Comma-separated list of target nodes (default: orin1,orin2,orin3,orin4)"
  echo "  -u, --user USER              SSH user (default: ubuntu)"
  echo "  -k, --key SSH_KEY            Path to SSH private key (optional)"
  echo "  -d, --dir DEPLOY_DIR         Directory on target to deploy to (default: /opt/nvidia-jetson-workload)"
  echo "  -b, --build                  Build workloads before deploying"
  echo "  -a, --aws                    AWS deployment mode (uses instance IDs instead of hostnames)"
  echo "  -i, --instances INSTANCES    Comma-separated list of AWS instance IDs (for AWS mode)"
  echo ""
  echo "Example:"
  echo "  $0 --workloads nbody_sim,weather_sim --nodes orin1,orin2 --user jetson --key ~/.ssh/id_rsa"
  echo "  $0 --aws --instances i-123456,i-789012 --workloads all"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -w|--workloads)
      WORKLOADS="$2"
      shift 2
      ;;
    -n|--nodes)
      IFS=',' read -ra NODES <<< "$2"
      shift 2
      ;;
    -u|--user)
      TARGET_USER="$2"
      shift 2
      ;;
    -k|--key)
      SSH_KEY="$2"
      shift 2
      ;;
    -d|--dir)
      DEPLOY_DIR="$2"
      shift 2
      ;;
    -b|--build)
      BUILD=true
      shift 1
      ;;
    -a|--aws)
      AWS_MODE=true
      shift 1
      ;;
    -i|--instances)
      IFS=',' read -ra AWS_INSTANCES <<< "$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check for AWS mode requirements
if [ "$AWS_MODE" = true ] && [ ${#AWS_INSTANCES[@]} -eq 0 ]; then
  echo "Error: AWS mode requires instance IDs (--instances)"
  show_help
  exit 1
fi

# Use AWS instances if in AWS mode
if [ "$AWS_MODE" = true ]; then
  NODES=("${AWS_INSTANCES[@]}")
fi

# Move to project root
cd "${PROJECT_ROOT}"

# SSH command construction
SSH_CMD="ssh"
if [ -n "$SSH_KEY" ]; then
  SSH_CMD="$SSH_CMD -i $SSH_KEY"
fi
SSH_CMD="$SSH_CMD -o StrictHostKeyChecking=no -o BatchMode=yes"

# SCP command construction
SCP_CMD="scp"
if [ -n "$SSH_KEY" ]; then
  SCP_CMD="$SCP_CMD -i $SSH_KEY"
fi
SCP_CMD="$SCP_CMD -o StrictHostKeyChecking=no"

# Build workloads if requested
function build_workloads {
  local workloads="$1"
  
  echo "Building workloads: $workloads"
  
  # Create build directory if it doesn't exist
  mkdir -p "${BUILD_DIR}"
  
  if [[ "$workloads" == "all" || "$workloads" == *"nbody_sim"* ]]; then
    echo "Building N-body simulation workload..."
    (cd src/nbody_sim/cpp && ./build_and_test.sh)
  fi
  
  if [[ "$workloads" == "all" || "$workloads" == *"molecular_dynamics"* ]]; then
    echo "Building Molecular Dynamics workload..."
    (cd src/molecular-dynamics/cpp && ./build.sh)
  fi
  
  if [[ "$workloads" == "all" || "$workloads" == *"weather_sim"* ]]; then
    echo "Building Weather Simulation workload..."
    (cd src/weather-sim/cpp && cmake -B build -S . && cmake --build build --parallel)
  fi
  
  if [[ "$workloads" == "all" || "$workloads" == *"medical_imaging"* ]]; then
    echo "Building Medical Imaging workload..."
    (cd src/medical-imaging/cpp && cmake -B build -S . && cmake --build build --parallel)
  fi
}

# Prepare deployment package
function prepare_deployment_package {
  local workloads="$1"
  local temp_dir="$2"
  
  echo "Preparing deployment package for workloads: $workloads"
  
  # Create directory for common files
  mkdir -p "${temp_dir}/common"
  cp requirements.txt pyproject.toml "${temp_dir}/common/"
  
  # Prepare specific workloads
  if [[ "$workloads" == "all" || "$workloads" == *"nbody_sim"* ]]; then
    echo "Preparing N-body simulation workload..."
    
    # Create directory structure
    mkdir -p "${temp_dir}/nbody_sim/bin"
    mkdir -p "${temp_dir}/nbody_sim/python"
    
    # Copy binaries
    if [ -d "src/nbody_sim/cpp/build/bin" ]; then
      cp -r src/nbody_sim/cpp/build/bin/* "${temp_dir}/nbody_sim/bin/"
    fi
    
    # Copy Python modules
    cp -r src/nbody_sim/python/* "${temp_dir}/nbody_sim/python/"
  fi
  
  if [[ "$workloads" == "all" || "$workloads" == *"molecular_dynamics"* ]]; then
    echo "Preparing Molecular Dynamics workload..."
    
    # Create directory structure
    mkdir -p "${temp_dir}/molecular-dynamics/lib"
    mkdir -p "${temp_dir}/molecular-dynamics/python"
    
    # Copy binaries
    if [ -d "src/molecular-dynamics/cpp/build/lib" ]; then
      cp -r src/molecular-dynamics/cpp/build/lib/* "${temp_dir}/molecular-dynamics/lib/"
    fi
    
    # Copy Python modules
    cp -r src/molecular-dynamics/python/* "${temp_dir}/molecular-dynamics/python/"
  fi
  
  if [[ "$workloads" == "all" || "$workloads" == *"weather_sim"* ]]; then
    echo "Preparing Weather Simulation workload..."
    
    # Create directory structure
    mkdir -p "${temp_dir}/weather-sim/lib"
    mkdir -p "${temp_dir}/weather-sim/python"
    
    # Copy binaries
    if [ -d "src/weather-sim/cpp/build/lib" ]; then
      cp -r src/weather-sim/cpp/build/lib/* "${temp_dir}/weather-sim/lib/"
    fi
    
    # Copy Python modules
    cp -r src/weather-sim/python/* "${temp_dir}/weather-sim/python/"
  fi
  
  if [[ "$workloads" == "all" || "$workloads" == *"medical_imaging"* ]]; then
    echo "Preparing Medical Imaging workload..."
    
    # Create directory structure
    mkdir -p "${temp_dir}/medical-imaging/lib"
    mkdir -p "${temp_dir}/medical-imaging/python"
    
    # Copy binaries
    if [ -d "src/medical-imaging/cpp/build/lib" ]; then
      cp -r src/medical-imaging/cpp/build/lib/* "${temp_dir}/medical-imaging/lib/"
    fi
    
    # Copy Python modules
    cp -r src/medical-imaging/python/* "${temp_dir}/medical-imaging/python/"
  fi
  
  # Include benchmark
  if [[ "$workloads" == "all" ]]; then
    echo "Preparing Benchmark suite..."
    
    # Create directory structure
    mkdir -p "${temp_dir}/benchmark"
    
    # Copy benchmark files
    cp -r benchmark/* "${temp_dir}/benchmark/"
  fi
}

# Deploy to a target node
function deploy_to_node {
  local node="$1"
  local target_user="$2"
  local deploy_dir="$3"
  local temp_dir="$4"
  
  local target_host="$node"
  if [ "$AWS_MODE" = true ]; then
    echo "Getting public IP for instance $node..."
    target_host=$(aws ec2 describe-instances --instance-ids "$node" --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
    if [ -z "$target_host" ]; then
      echo "Error: Could not get public IP for instance $node"
      return 1
    fi
    echo "Instance $node has IP $target_host"
  fi
  
  echo "Deploying to ${target_user}@${target_host}:${deploy_dir}..."
  
  # Check SSH connection
  if ! $SSH_CMD ${target_user}@${target_host} "exit"; then
    echo "Error: Cannot connect to ${target_host} as ${target_user}"
    return 1
  fi
  
  # Create target directory
  $SSH_CMD ${target_user}@${target_host} "mkdir -p ${deploy_dir}"
  
  # Copy common files
  $SCP_CMD -r "${temp_dir}/common/"* ${target_user}@${target_host}:${deploy_dir}/
  
  # Copy workload directories
  for workload_dir in "${temp_dir}"/*; do
    if [ -d "$workload_dir" ] && [ "$(basename "$workload_dir")" != "common" ]; then
      workload_name=$(basename "$workload_dir")
      echo "Copying $workload_name to ${target_host}..."
      $SCP_CMD -r "$workload_dir" ${target_user}@${target_host}:${deploy_dir}/
    fi
  done
  
  # Install Python dependencies
  $SSH_CMD ${target_user}@${target_host} "cd ${deploy_dir} && python3 -m pip install --user -r requirements.txt"
  
  # Set up environment
  $SSH_CMD ${target_user}@${target_host} "mkdir -p ~/.bashrc.d && echo 'export PYTHONPATH=\${PYTHONPATH}:${deploy_dir}' > ~/.bashrc.d/nvidia-jetson-workload.sh && chmod +x ~/.bashrc.d/nvidia-jetson-workload.sh"
  
  echo "Deployment to ${target_host} completed successfully"
  return 0
}

# Build if requested
if [ "$BUILD" = true ]; then
  build_workloads "$WORKLOADS"
fi

# Create temporary directory for deployment
temp_dir=$(mktemp -d)
echo "Using temporary directory: $temp_dir"

# Prepare deployment package
prepare_deployment_package "$WORKLOADS" "$temp_dir"

# Deploy to all nodes in parallel
echo "Deploying to ${#NODES[@]} nodes in parallel..."
pids=()

for node in "${NODES[@]}"; do
  echo "Starting deployment to ${node}..."
  deploy_to_node "$node" "$TARGET_USER" "$DEPLOY_DIR" "$temp_dir" &
  pids+=($!)
done

# Wait for all deployments to finish
success=true
for pid in "${pids[@]}"; do
  if ! wait $pid; then
    success=false
  fi
done

# Clean up temporary directory
rm -rf "${temp_dir}"

if $success; then
  echo "All deployments completed successfully!"
else
  echo "One or more deployments failed. Check output for details."
  exit 1
fi