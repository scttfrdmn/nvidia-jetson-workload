#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Benchmark script for time-frequency analysis across different architectures

# Set environment variables
export SIGNAL_PROC_TEST_SIZE=100000  # Use larger signal for benchmarks
export SIGNAL_PROC_VERBOSE=1        # Enable detailed output

# Build directory
BUILD_DIR="../../build"

# Ensure executable exists
if [ ! -f "${BUILD_DIR}/src/signal_processing/cpp/test/test_time_frequency" ]; then
    echo "Error: test_time_frequency executable not found!"
    echo "Please build the project first using CMake."
    exit 1
fi

# Run benchmark
echo "==================================================================="
echo "Running Time-Frequency Analysis Benchmark"
echo "==================================================================="
echo ""

# Get system info
echo "System Information:"
echo "-------------------"
if [ -f /proc/cpuinfo ]; then
    echo "CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d ":" -f2 | xargs)"
else
    echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")"
fi

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    echo "GPU: None detected"
fi
echo ""

# Run benchmark tests
echo "Running Time-Frequency Analysis Tests..."
echo "-------------------"
cd "${BUILD_DIR}" || exit 1
./src/signal_processing/cpp/test/test_time_frequency

echo ""
echo "Benchmark completed"
echo "==================================================================="