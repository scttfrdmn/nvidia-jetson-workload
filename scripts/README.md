# NVIDIA Jetson Workload Scripts

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors

This directory contains utility scripts for the NVIDIA Jetson & AWS Graviton Workloads project.

## Available Scripts

### `create_release.sh`

This script automates the process of creating a release package for the project.

**Usage:**
```
./create_release.sh VERSION [OPTIONS]
```

**Arguments:**
- `VERSION`: Version number (e.g., 1.2.0)

**Options:**
- `-h, --help`: Show help message
- `-t, --type TYPE`: Release type: major, minor, patch, hotfix (default: minor)
- `-d, --dir DIR`: Directory to store release files (default: release)
- `-p, --publish`: Publish the release (tag git, push to GitHub, etc.)

For detailed information, see the [Release Process Guide](/docs/user-guide/release-process.md).

### `deploy-all.sh`

Unified deployment script for all workloads to both Jetson devices and AWS instances.

**Usage:**
```
./deploy-all.sh [OPTIONS] TARGET_LIST
```

See the [Deployment Guide](/docs/user-guide/deployment.md) for details.

### `deploy.sh`

Single-node deployment script for individual workloads.

### `setup-jetson.sh`

Script to set up a Jetson device with required dependencies.

### `setup-ssh-keys.sh`

Utility script to set up SSH keys for deployment to multiple nodes.

### `update_copyright.sh`

Script to update copyright notices in all source files.