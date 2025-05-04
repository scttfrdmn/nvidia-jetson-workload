#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# This script deploys code to the Jetson devices for development and testing

set -e

# Default values
JETSON_HOST=${1:-"orin1"}
JETSON_USER=${2:-"ubuntu"}
REMOTE_DIR=${3:-"/home/${JETSON_USER}/nvidia-jetson-workload"}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <jetson-hostname> [user] [remote-dir]"
    echo "Example: $0 orin1 ubuntu /home/ubuntu/nvidia-jetson-workload"
    echo "Available nodes: orin1, orin2, orin3, orin4"
    exit 1
fi

echo "Deploying to ${JETSON_USER}@${JETSON_HOST}:${REMOTE_DIR}"

# Ensure remote directory exists
ssh ${JETSON_USER}@${JETSON_HOST} "mkdir -p ${REMOTE_DIR}"

# Sync code to remote machine, excluding unnecessary files
rsync -avz --progress \
    --exclude '.git/' \
    --exclude '*.o' \
    --exclude '*.so' \
    --exclude '*.a' \
    --exclude '*.pyc' \
    --exclude '__pycache__' \
    --exclude 'node_modules' \
    --exclude 'build/' \
    --exclude 'dist/' \
    ${PROJECT_ROOT}/ ${JETSON_USER}@${JETSON_HOST}:${REMOTE_DIR}/

# Run remote setup if needed
if [[ "$4" == "--setup" ]]; then
    echo "Running setup script on remote machine..."
    ssh ${JETSON_USER}@${JETSON_HOST} "cd ${REMOTE_DIR} && bash scripts/setup-jetson.sh"
fi

echo "Deployment complete!"