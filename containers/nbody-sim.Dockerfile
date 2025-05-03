# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 nvidia-jetson-workload contributors

# Use NVIDIA's L4T PyTorch image as base for ARM compatibility
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

LABEL maintainer="NVIDIA Jetson Workload Contributors"
LABEL description="N-body simulation workload for NVIDIA Jetson"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONPATH=/opt/nvidia-jetson-workload:$PYTHONPATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    python3-venv \
    libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create output directory
RUN mkdir -p /output && chmod 777 /output

# Create app directory
WORKDIR /opt/nvidia-jetson-workload

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir matplotlib seaborn

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x /opt/nvidia-jetson-workload/src/nbody_sim/python/run_test.py && \
    chmod +x /opt/nvidia-jetson-workload/src/nbody_sim/python/cli.py

# Default command
ENTRYPOINT ["python3", "-m", "src.nbody_sim.python.cli", "--output-dir", "/output"]

# Default arguments (can be overridden)
CMD ["--system-type", "galaxy", "--num-particles", "10000"]