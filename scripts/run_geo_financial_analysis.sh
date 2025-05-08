#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Script to run the Geospatial Financial Risk Analysis notebook

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NOTEBOOK_DIR="${PROJECT_ROOT}/src/integrations/geo_financial"
VENV_PATH="${PROJECT_ROOT}/.venv"

# Parse command line arguments
DOCKER_MODE=false
CREATE_VENV=false

function print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Run the Geospatial Financial Risk Analysis notebook"
    echo ""
    echo "Options:"
    echo "  -h, --help         Show this help message"
    echo "  -d, --docker       Run using Docker container"
    echo "  -v, --create-venv  Create a virtual environment if it doesn't exist"
    echo ""
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_usage
            exit 0
            ;;
        -d|--docker)
            DOCKER_MODE=true
            shift
            ;;
        -v|--create-venv)
            CREATE_VENV=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [ "$DOCKER_MODE" = true ]; then
    echo "Running Geospatial Financial Risk Analysis notebook in Docker..."
    
    # Check if both containers exist
    if docker image inspect financial-modeling:latest &>/dev/null && docker image inspect geospatial:latest &>/dev/null; then
        echo "Using existing Docker images"
    else
        echo "Building Docker images..."
        docker build -t financial-modeling -f "${PROJECT_ROOT}/containers/financial-modeling.Dockerfile" "${PROJECT_ROOT}"
        docker build -t geospatial -f "${PROJECT_ROOT}/containers/geospatial.Dockerfile" "${PROJECT_ROOT}"
    fi
    
    # Create custom Dockerfile for integration
    TEMP_DOCKERFILE=$(mktemp)
    cat > ${TEMP_DOCKERFILE} << EOF
FROM financial-modeling:latest
USER root
COPY --from=geospatial:latest /app/src/geospatial /app/src/geospatial
COPY src/integrations/geo_financial /app/src/integrations/geo_financial
RUN pip install jupyter jupyterlab matplotlib seaborn
WORKDIR /app
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]
EOF
    
    echo "Building integrated Docker image..."
    docker build -t geo-financial -f ${TEMP_DOCKERFILE} "${PROJECT_ROOT}"
    rm ${TEMP_DOCKERFILE}
    
    echo "Running Docker container..."
    docker run -it --rm -p 8888:8888 --gpus all -v "${NOTEBOOK_DIR}:/app/src/integrations/geo_financial" geo-financial
else
    # Setup Python environment
    PYTHON_CMD="python3"
    if ! command -v python3 &> /dev/null; then
        PYTHON_CMD="python"
    fi
    
    # Create virtual environment if requested
    if [ "$CREATE_VENV" = true ]; then
        if [ ! -d "$VENV_PATH" ]; then
            echo "Creating virtual environment..."
            $PYTHON_CMD -m venv "$VENV_PATH"
        fi
    fi
    
    # Activate virtual environment if it exists
    if [ -d "$VENV_PATH" ]; then
        echo "Activating virtual environment..."
        source "$VENV_PATH/bin/activate"
    fi
    
    # Install dependencies
    echo "Installing dependencies..."
    pip install -e "${PROJECT_ROOT}/src/financial_modeling/python"
    pip install -e "${PROJECT_ROOT}/src/geospatial/python"
    pip install numpy pandas matplotlib seaborn jupyter jupyterlab
    
    # Launch Jupyter Lab
    echo "Launching Jupyter Lab with the Geospatial Financial Risk Analysis notebook..."
    jupyter lab "${NOTEBOOK_DIR}/geospatial_financial_analysis.ipynb"
fi