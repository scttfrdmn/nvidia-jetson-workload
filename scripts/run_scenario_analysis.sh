#!/bin/bash
# Run scenario analysis for geo-financial integration
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

set -e

# Default values
OUTPUT_DIR="./output"
GENERATE_DATA=false
NUM_ASSETS=200
GPU_DEVICE=-1
PLOT_FORMAT="png"
HEADLESS=false
ALL_ANALYSES=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --generate-data)
      GENERATE_DATA=true
      shift
      ;;
    --num-assets)
      NUM_ASSETS="$2"
      shift 2
      ;;
    --device)
      GPU_DEVICE="$2"
      shift 2
      ;;
    --plot-format)
      PLOT_FORMAT="$2"
      shift 2
      ;;
    --headless)
      HEADLESS=true
      shift
      ;;
    --all-analyses)
      ALL_ANALYSES=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo "Run scenario analysis for geo-financial integration"
      echo ""
      echo "Options:"
      echo "  --output-dir DIR       Directory to store output files (default: ./output)"
      echo "  --generate-data         Generate synthetic test data (default: false)"
      echo "  --num-assets N         Number of assets to generate (default: 200)"
      echo "  --device N             GPU device ID to use (-1 for CPU, default: -1)"
      echo "  --plot-format FORMAT   Format for plots (png, pdf, svg, default: png)"
      echo "  --headless             Run in headless mode without interactive display"
      echo "  --all-analyses         Run all analyses (basic, sensitivity, stress test)"
      echo "  --verbose              Enable verbose output"
      echo "  --help                 Display this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information."
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set Python path to include project root
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH}"

# Log execution parameters
log_file="${OUTPUT_DIR}/scenario_analysis_log.txt"
echo "========================================" > "${log_file}"
echo "Scenario Analysis - $(date)" >> "${log_file}"
echo "----------------------------------------" >> "${log_file}"
echo "Output directory: ${OUTPUT_DIR}" >> "${log_file}"
echo "Generate data: ${GENERATE_DATA}" >> "${log_file}"
echo "Number of assets: ${NUM_ASSETS}" >> "${log_file}"
echo "GPU device: ${GPU_DEVICE}" >> "${log_file}"
echo "Plot format: ${PLOT_FORMAT}" >> "${log_file}"
echo "Headless mode: ${HEADLESS}" >> "${log_file}"
echo "All analyses: ${ALL_ANALYSES}" >> "${log_file}"
echo "----------------------------------------" >> "${log_file}"

# Function to run command with logging
run_command() {
  local cmd="$1"
  local description="$2"
  
  echo "Running: ${description}" | tee -a "${log_file}"
  if [ "${VERBOSE}" = true ]; then
    echo "Command: ${cmd}" | tee -a "${log_file}"
  fi
  
  echo "----------------------------------------" >> "${log_file}"
  if eval "${cmd}" >> "${log_file}" 2>&1; then
    echo "✓ ${description} completed successfully" | tee -a "${log_file}"
  else
    echo "✗ ${description} failed (exit code: $?)" | tee -a "${log_file}"
    echo "See log file for details: ${log_file}"
    return 1
  fi
  echo "----------------------------------------" >> "${log_file}"
}

# Generate test data if requested
if [ "${GENERATE_DATA}" = true ]; then
  data_dir="${OUTPUT_DIR}/data"
  mkdir -p "${data_dir}"
  
  cmd="python -m src.integrations.geo_financial.generate_test_data --output-dir \"${data_dir}\" --num-assets ${NUM_ASSETS} --seed 42"
  run_command "${cmd}" "Generating test data"
else
  data_dir="./data/geo_financial"
fi

# Build the common parameters
common_params="--output-dir \"${OUTPUT_DIR}\" --device-id ${GPU_DEVICE} --plot-format ${PLOT_FORMAT}"
if [ "${HEADLESS}" = true ]; then
  common_params="${common_params} --headless"
fi

# Run scenario analysis
cmd="python -m src.integrations.geo_financial.examples.scenario_analysis ${common_params}"
run_command "${cmd}" "Running scenario analysis"

# Generate final report if all analyses were run
if [ "${ALL_ANALYSES}" = true ]; then
  report_file="${OUTPUT_DIR}/scenario_analysis_report.md"
  echo "# Scenario Analysis Report" > "${report_file}"
  echo "" >> "${report_file}"
  echo "Generated on: $(date)" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "## Overview" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "This report contains results from comprehensive scenario analysis of the geospatial financial portfolio." >> "${report_file}"
  echo "" >> "${report_file}"
  echo "## Basic Scenario Comparison" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "![Scenario Comparison]($(basename "${OUTPUT_DIR}")/scenario_comparison.${PLOT_FORMAT})" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "## Economic Impact Analysis" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "![Economic Impact]($(basename "${OUTPUT_DIR}")/economic_impact.${PLOT_FORMAT})" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "## Sensitivity Analysis" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "![Sensitivity Analysis]($(basename "${OUTPUT_DIR}")/sensitivity_analysis.${PLOT_FORMAT})" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "## Stress Test Results" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "![Stress Test]($(basename "${OUTPUT_DIR}")/stress_test.${PLOT_FORMAT})" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "## Comprehensive Dashboard" >> "${report_file}"
  echo "" >> "${report_file}"
  echo "![Scenario Dashboard]($(basename "${OUTPUT_DIR}")/scenario_dashboard.${PLOT_FORMAT})" >> "${report_file}"
  
  echo "Generated summary report: ${report_file}" | tee -a "${log_file}"
fi

echo "Scenario analysis completed. Results are available in ${OUTPUT_DIR}" | tee -a "${log_file}"