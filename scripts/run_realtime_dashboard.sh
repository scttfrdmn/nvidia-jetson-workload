#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Script to run the Real-time Geospatial Financial Dashboard

# Default values
DATA_DIR="data/geo_financial"
OUTPUT_DIR="results/geo_financial/realtime_dashboard"
DEVICE_ID=0
GENERATE_DATA=false
NUM_ASSETS=20
DEM_SIZE=500
PORT=8050
DEBUG=false
HEADLESS=false
HEADLESS_DURATION=3600
HEADLESS_OUTPUT_INTERVAL=300
UPDATE_INTERVAL=5
SEED=42

# Function to display usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --data-dir DIR               Directory for input data (default: $DATA_DIR)"
    echo "  --output-dir DIR             Directory for output files (default: $OUTPUT_DIR)"
    echo "  --device-id ID               GPU device ID (-1 for CPU) (default: $DEVICE_ID)"
    echo "  --generate-data              Generate synthetic data (default: false)"
    echo "  --num-assets N               Number of assets to generate (default: $NUM_ASSETS)"
    echo "  --dem-size N                 Size of DEM to generate (default: $DEM_SIZE)"
    echo "  --port N                     Port to run the dashboard on (default: $PORT)"
    echo "  --debug                      Run in debug mode (default: false)"
    echo "  --headless                   Run in headless mode (default: false)"
    echo "  --headless-duration N        Duration to run headless mode in seconds (default: $HEADLESS_DURATION)"
    echo "  --headless-output-interval N Interval to save outputs in headless mode (default: $HEADLESS_OUTPUT_INTERVAL)"
    echo "  --update-interval N          Dashboard update interval in seconds (default: $UPDATE_INTERVAL)"
    echo "  --seed N                     Random seed for reproducibility (default: $SEED)"
    echo "  --help                       Display this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device-id)
            DEVICE_ID="$2"
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
        --dem-size)
            DEM_SIZE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --headless-duration)
            HEADLESS_DURATION="$2"
            shift 2
            ;;
        --headless-output-interval)
            HEADLESS_OUTPUT_INTERVAL="$2"
            shift 2
            ;;
        --update-interval)
            UPDATE_INTERVAL="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd )"

# Create directories if they don't exist
mkdir -p "$PROJECT_ROOT/$DATA_DIR"
mkdir -p "$PROJECT_ROOT/$OUTPUT_DIR"

# Check if required packages are installed
if ! python -c "import dash" &>/dev/null; then
    echo "Dash package not found. Installing required packages..."
    pip install dash dash-bootstrap-components plotly
fi

# Build command
CMD="python -m src.integrations.geo_financial.realtime_dashboard \
--data-dir $DATA_DIR \
--output-dir $OUTPUT_DIR \
--device-id $DEVICE_ID \
--update-interval $UPDATE_INTERVAL \
--port $PORT \
--seed $SEED"

# Add optional flags
if $GENERATE_DATA; then
    CMD="$CMD --generate-data --num-assets $NUM_ASSETS --dem-size $DEM_SIZE"
fi

if $DEBUG; then
    CMD="$CMD --debug"
fi

if $HEADLESS; then
    CMD="$CMD --headless --headless-duration $HEADLESS_DURATION --headless-output-interval $HEADLESS_OUTPUT_INTERVAL"
fi

# Print the command
echo "Running command: $CMD"

# Execute the command
cd "$PROJECT_ROOT"
eval "$CMD"