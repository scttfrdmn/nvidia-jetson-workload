#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 nvidia-jetson-workload contributors

# This script sets up SSH keys for passwordless login to the Jetson cluster

set -e

# Default values
NODES=("linux-0" "orin1" "orin2" "orin3" "orin4")
SSH_USER=${1:-"ubuntu"}
SSH_KEY_TYPE="ed25519"
SSH_KEY_FILE="$HOME/.ssh/id_${SSH_KEY_TYPE}_jetson_cluster"

# Display usage
function show_usage {
    echo "Usage: $0 [username]"
    echo "Sets up SSH keys for passwordless login to the Jetson cluster"
    echo ""
    echo "  [username]  Username for SSH login (default: ubuntu)"
    echo ""
    echo "The script will:"
    echo "  1. Generate a new SSH key if it doesn't exist"
    echo "  2. Copy the public key to all cluster nodes"
    echo "  3. Update SSH config for easier access"
}

# Parse arguments
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Check if SSH key exists, generate if not
if [[ ! -f "$SSH_KEY_FILE" ]]; then
    echo "Generating new SSH key at $SSH_KEY_FILE"
    ssh-keygen -t $SSH_KEY_TYPE -f "$SSH_KEY_FILE" -N "" -C "jetson-cluster-access"
else
    echo "Using existing SSH key at $SSH_KEY_FILE"
fi

# Add to SSH config if not already there
SSH_CONFIG="$HOME/.ssh/config"
if ! grep -q "Host orin[1-4]" "$SSH_CONFIG" 2>/dev/null; then
    echo "Updating SSH config file"
    cat >> "$SSH_CONFIG" << EOF

# Jetson Cluster Configuration
Host linux-0 orin1 orin2 orin3 orin4
    User ${SSH_USER}
    IdentityFile $SSH_KEY_FILE
    StrictHostKeyChecking no

EOF
    echo "Updated SSH config at $SSH_CONFIG"
else
    echo "SSH config already contains entries for the cluster"
fi

# Copy SSH key to each node
for node in "${NODES[@]}"; do
    echo "Copying SSH key to ${node}..."
    ssh-copy-id -i "$SSH_KEY_FILE.pub" -f -o StrictHostKeyChecking=no ${SSH_USER}@${node} || {
        echo "Warning: Failed to copy key to ${node}. You may need to manually enter your password for this node."
    }
done

echo ""
echo "SSH key setup complete. You should now be able to SSH to cluster nodes without a password."
echo "Try: ssh orin1"