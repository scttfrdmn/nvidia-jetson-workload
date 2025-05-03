#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Script to build and run the C++ tests for the N-body simulation

set -e  # Exit on error

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Debug -DNBODY_BUILD_TESTS=ON

# Build
echo "Building..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run tests
echo "Running tests..."
ctest --output-on-failure

# Also run the test executable directly for more verbose output
echo "Running test executable directly..."
./nbody_sim_test

echo "Done!"