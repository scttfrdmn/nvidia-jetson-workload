#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 nvidia-jetson-workload contributors

# This script deploys code to all Jetson nodes in parallel

set -e

# Default values
JETSON_USER=${1:-"ubuntu"}
REMOTE_DIR=${2:-"/home/${JETSON_USER}/nvidia-jetson-workload"}
NODES=("orin1" "orin2" "orin3" "orin4")
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Display usage
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [username] [remote-dir]"
    echo "Deploys code to all Jetson nodes in parallel"
    echo ""
    echo "  [username]   Username for SSH login (default: ubuntu)"
    echo "  [remote-dir] Remote directory (default: /home/<username>/nvidia-jetson-workload)"
    exit 0
fi

# Check if we have SSH access to all nodes
for node in "${NODES[@]}"; do
    if ! ssh -q -o BatchMode=yes -o ConnectTimeout=5 ${JETSON_USER}@${node} exit; then
        echo "Error: Cannot connect to ${node} without password."
        echo "Please run scripts/setup-ssh-keys.sh first to set up SSH keys."
        exit 1
    fi
done

# Deploy to all nodes in parallel
echo "Deploying to all Jetson nodes in parallel..."
pids=()

for node in "${NODES[@]}"; do
    echo "Starting deployment to ${node}..."
    ${SCRIPT_DIR}/deploy.sh ${node} ${JETSON_USER} ${REMOTE_DIR} &
    pids+=($!)
done

# Wait for all deployments to finish
success=true
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        success=false
    fi
done

if $success; then
    echo "All deployments completed successfully!"
else
    echo "One or more deployments failed. Check output for details."
    exit 1
fi