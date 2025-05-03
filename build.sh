#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Script to build all workloads in the project

set -e  # Exit on error

# Parse command line arguments
BUILD_TYPE="Release"
RUN_TESTS=0

for arg in "$@"; do
  case $arg in
    --debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    --test)
      RUN_TESTS=1
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --debug    Build in debug mode"
      echo "  --test     Run tests after building"
      echo "  --help     Show this help message"
      exit 0
      ;;
  esac
done

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE

# Build
echo "Building..."
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Build completed successfully!"

# Run tests if requested
if [ $RUN_TESTS -eq 1 ]; then
    echo "Running tests..."
    ctest --output-on-failure
    echo "Tests completed successfully!"
fi