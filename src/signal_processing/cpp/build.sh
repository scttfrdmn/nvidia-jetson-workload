#!/bin/bash
# Build script for Signal Processing module
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
cmake .. -DWITH_CUDA=ON

# Build
make -j$(nproc)

# Run tests
ctest -V

# Return to original directory
cd ..