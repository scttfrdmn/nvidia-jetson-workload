#!/bin/bash
# Benchmark script for Signal Processing filters
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Check that executables exist
if [ ! -f ../build/test/test_fir_filter_cuda ] || \
   [ ! -f ../build/test/test_multirate_filter_cuda ] || \
   [ ! -f ../build/test/test_adaptive_filter_cuda ]; then
    echo "Error: Test executables not found. Run build.sh first."
    exit 1
fi

# Create results directory
RESULTS_DIR="benchmark_results"
mkdir -p ${RESULTS_DIR}

# Current timestamp for results
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${RESULTS_DIR}/benchmark_results_${TIMESTAMP}.txt"

# Function to run a benchmark and save results
run_benchmark() {
    local test_name=$1
    local executable=$2
    
    echo "Running benchmark: ${test_name}"
    echo "-----------------------------------------" >> ${RESULTS_FILE}
    echo "Benchmark: ${test_name}" >> ${RESULTS_FILE}
    echo "Timestamp: $(date)" >> ${RESULTS_FILE}
    echo "-----------------------------------------" >> ${RESULTS_FILE}
    
    # Run the benchmark and capture output
    ../build/test/${executable} >> ${RESULTS_FILE} 2>&1
    
    echo "Done. Results saved to ${RESULTS_FILE}"
    echo ""
}

# Header info
echo "Signal Processing Workload Benchmarks" > ${RESULTS_FILE}
echo "===================================" >> ${RESULTS_FILE}
echo "System Information:" >> ${RESULTS_FILE}
echo "Date: $(date)" >> ${RESULTS_FILE}
echo "Hostname: $(hostname)" >> ${RESULTS_FILE}
echo "OS: $(uname -a)" >> ${RESULTS_FILE}
echo "" >> ${RESULTS_FILE}

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:" >> ${RESULTS_FILE}
    nvidia-smi --query-gpu=name,driver_version,compute_capability --format=csv >> ${RESULTS_FILE}
else
    echo "No NVIDIA GPU detected. Running CPU benchmarks only." >> ${RESULTS_FILE}
fi
echo "" >> ${RESULTS_FILE}

# Run benchmarks
echo "Running benchmarks..."
run_benchmark "FIR Filter" "test_fir_filter_cuda"
run_benchmark "Multirate Filter" "test_multirate_filter_cuda"
run_benchmark "Adaptive Filter" "test_adaptive_filter_cuda"

# Summarize results
echo ""
echo "Benchmark Summary"
echo "================="
echo "Results saved to: ${RESULTS_FILE}"
echo ""
echo "To view results: cat ${RESULTS_FILE}"