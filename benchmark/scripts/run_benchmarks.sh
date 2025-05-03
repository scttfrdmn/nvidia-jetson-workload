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
RUN_GEOSPATIAL=false
GENERATE_REPORT=false

# Cost modeling options
ENABLE_COST_MODELING=false
AWS_INSTANCE="g4dn.xlarge"
AZURE_INSTANCE="Standard_NC4as_T4_v3"
GCP_INSTANCE="n1-standard-4-t4"

# DGX Spark options
INCLUDE_DGX_SPARK=true
DGX_SYSTEM_TYPE="dgx_a100"
DGX_QUANTITY=1
DGX_CONFIG_FILE=""

# Slurm cluster options
INCLUDE_SLURM_CLUSTER=true
SLURM_NODE_TYPE="basic_gpu"
SLURM_NODES=4
SLURM_CONFIG_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        # Basic options
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
        --geospatial)
            RUN_GEOSPATIAL=true
            shift
            ;;
        --report)
            GENERATE_REPORT=true
            shift
            ;;
            
        # Cost modeling options
        --cost-analysis)
            ENABLE_COST_MODELING=true
            shift
            ;;
        --aws-instance)
            AWS_INSTANCE="$2"
            shift 2
            ;;
        --azure-instance)
            AZURE_INSTANCE="$2"
            shift 2
            ;;
        --gcp-instance)
            GCP_INSTANCE="$2"
            shift 2
            ;;
            
        # DGX Spark options
        --no-dgx-spark)
            INCLUDE_DGX_SPARK=false
            shift
            ;;
        --dgx-system-type)
            DGX_SYSTEM_TYPE="$2"
            shift 2
            ;;
        --dgx-quantity)
            DGX_QUANTITY="$2"
            shift 2
            ;;
        --dgx-config)
            DGX_CONFIG_FILE="$2"
            shift 2
            ;;
            
        # Slurm cluster options
        --no-slurm-cluster)
            INCLUDE_SLURM_CLUSTER=false
            shift
            ;;
        --slurm-node-type)
            SLURM_NODE_TYPE="$2"
            shift 2
            ;;
        --slurm-nodes)
            SLURM_NODES="$2"
            shift 2
            ;;
        --slurm-config)
            SLURM_CONFIG_FILE="$2"
            shift 2
            ;;
            
        # Unknown options
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [options]"
            echo "Basic options:"
            echo "  --device ID              GPU device ID to use"
            echo "  --output DIR             Directory to store results"
            echo "  --all                    Run all benchmarks"
            echo "  --nbody                  Run N-body simulation benchmark"
            echo "  --md                     Run Molecular Dynamics benchmark"
            echo "  --weather                Run Weather Simulation benchmark"
            echo "  --medical                Run Medical Imaging benchmark"
            echo "  --geospatial             Run Geospatial Analysis benchmark"
            echo "  --report                 Generate report from existing results"
            echo ""
            echo "Cost modeling options:"
            echo "  --cost-analysis          Enable cost modeling and comparison"
            echo "  --aws-instance TYPE      AWS instance type (default: g4dn.xlarge)"
            echo "  --azure-instance TYPE    Azure instance type (default: Standard_NC4as_T4_v3)"
            echo "  --gcp-instance TYPE      GCP instance type (default: n1-standard-4-t4)"
            echo ""
            echo "DGX Spark options:"
            echo "  --no-dgx-spark           Exclude DGX Spark from cost comparison"
            echo "  --dgx-system-type TYPE   DGX system type (default: dgx_a100)"
            echo "  --dgx-quantity NUM       Number of DGX systems (default: 1)"
            echo "  --dgx-config FILE        Path to DGX configuration file"
            echo ""
            echo "Slurm cluster options:"
            echo "  --no-slurm-cluster       Exclude Slurm cluster from cost comparison"
            echo "  --slurm-node-type TYPE   Slurm node type (default: basic_gpu)"
            echo "  --slurm-nodes NUM        Number of nodes in Slurm cluster (default: 4)"
            echo "  --slurm-config FILE      Path to Slurm cluster configuration file"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Generate command
COMMAND="python ${PROJECT_ROOT}/benchmark/benchmark_suite.py --device ${DEVICE} --output ${OUTPUT_DIR}"

# Workload options
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
    
    if [ "$RUN_GEOSPATIAL" = true ]; then
        COMMAND="${COMMAND} --geospatial"
    fi
    
    # If no workload specified, run all
    if [ "$RUN_NBODY" = false ] && [ "$RUN_MD" = false ] && [ "$RUN_WEATHER" = false ] && [ "$RUN_MEDICAL" = false ] && [ "$RUN_GEOSPATIAL" = false ]; then
        COMMAND="${COMMAND} --all"
    fi
fi

# Cost modeling options
if [ "$ENABLE_COST_MODELING" = true ]; then
    COMMAND="${COMMAND} --cost-analysis"
    COMMAND="${COMMAND} --aws-instance ${AWS_INSTANCE}"
    COMMAND="${COMMAND} --azure-instance ${AZURE_INSTANCE}"
    COMMAND="${COMMAND} --gcp-instance ${GCP_INSTANCE}"
    
    # DGX Spark options
    if [ "$INCLUDE_DGX_SPARK" = false ]; then
        COMMAND="${COMMAND} --no-dgx-spark"
    else
        # Add DGX Spark parameters only if included
        COMMAND="${COMMAND} --dgx-system-type ${DGX_SYSTEM_TYPE}"
        COMMAND="${COMMAND} --dgx-quantity ${DGX_QUANTITY}"
        if [ -n "$DGX_CONFIG_FILE" ]; then
            COMMAND="${COMMAND} --dgx-config ${DGX_CONFIG_FILE}"
        fi
    fi
    
    # Slurm cluster options
    if [ "$INCLUDE_SLURM_CLUSTER" = false ]; then
        COMMAND="${COMMAND} --no-slurm-cluster"
    else
        # Add Slurm cluster parameters only if included
        COMMAND="${COMMAND} --slurm-node-type ${SLURM_NODE_TYPE}"
        COMMAND="${COMMAND} --slurm-nodes ${SLURM_NODES}"
        if [ -n "$SLURM_CONFIG_FILE" ]; then
            COMMAND="${COMMAND} --slurm-config ${SLURM_CONFIG_FILE}"
        fi
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