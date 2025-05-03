#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Script to run benchmarks for GPU-accelerated scientific workloads
# This script provides an easy way to run benchmarks with various configurations

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Default values
DEVICE=0
OUTPUT_DIR="${PROJECT_ROOT}/benchmark/results"
RUN_ALL=false
RUN_NBODY=false
RUN_MD=false
RUN_WEATHER=false
RUN_MEDICAL=false
GENERATE_REPORT=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --nbody)
            RUN_NBODY=true
            shift
            ;;
        --md)
            RUN_MD=true
            shift
            ;;
        --weather)
            RUN_WEATHER=true
            shift
            ;;
        --medical)
            RUN_MEDICAL=true
            shift
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--device ID] [--output DIR] [--all] [--nbody] [--md] [--weather] [--medical] [--report]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Generate command
COMMAND="python ${PROJECT_ROOT}/benchmark/benchmark_suite.py --device ${DEVICE} --output ${OUTPUT_DIR}"

if [ "$GENERATE_REPORT" = true ]; then
    COMMAND="${COMMAND} --report"
elif [ "$RUN_ALL" = true ]; then
    COMMAND="${COMMAND} --all"
else
    if [ "$RUN_NBODY" = true ]; then
        COMMAND="${COMMAND} --nbody"
    fi
    
    if [ "$RUN_MD" = true ]; then
        COMMAND="${COMMAND} --md"
    fi
    
    if [ "$RUN_WEATHER" = true ]; then
        COMMAND="${COMMAND} --weather"
    fi
    
    if [ "$RUN_MEDICAL" = true ]; then
        COMMAND="${COMMAND} --medical"
    fi
    
    # If no workload specified, run all
    if [ "$RUN_NBODY" = false ] && [ "$RUN_MD" = false ] && [ "$RUN_WEATHER" = false ] && [ "$RUN_MEDICAL" = false ]; then
        COMMAND="${COMMAND} --all"
    fi
fi

# Print command
echo "Running command: ${COMMAND}"

# Run command
eval "${COMMAND}"

# Print report path
if [ "$GENERATE_REPORT" = true ] || [ "$RUN_ALL" = true ] || [ "$RUN_NBODY" = true ] || [ "$RUN_MD" = true ] || [ "$RUN_WEATHER" = true ] || [ "$RUN_MEDICAL" = true ]; then
    echo "Report generated at: ${OUTPUT_DIR}/benchmark_report.html"
fi