#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Script to build the molecular dynamics simulation workload

set -e  # Exit on error

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build completed successfully!"
echo "Executable located at: $(pwd)/molecular_dynamics_app"

# Run tests if requested
if [[ "$1" == "--test" ]]; then
    echo "Running tests..."
    ctest --output-on-failure
    echo "Tests completed successfully!"
fi