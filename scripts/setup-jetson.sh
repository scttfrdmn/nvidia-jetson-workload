#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# This script sets up the Jetson environment for the workloads

set -e

echo "Setting up NVIDIA Jetson Workload environment..."

# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    protobuf-compiler \
    libprotobuf-dev

# Install python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Set up protobuf
cd src/proto
protoc --cpp_out=../weather-sim/cpp --python_out=../weather-sim/python weather.proto
protoc --cpp_out=../medical-imaging/cpp --python_out=../medical-imaging/python medical.proto
protoc --cpp_out=../nbody-sim/cpp --python_out=../nbody-sim/python nbody.proto

# Build C++ components
for workload in weather-sim medical-imaging nbody-sim; do
    echo "Building C++ workload: ${workload}"
    cd ../$(workload)/cpp
    mkdir -p build && cd build
    cmake ..
    make -j$(nproc)
    cd ../../../proto
done

echo "Setup complete!"