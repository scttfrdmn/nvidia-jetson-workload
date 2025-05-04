# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Multi-stage build for GPU-accelerated scientific workloads benchmark suite
# Base stage with CUDA
FROM nvidia/cuda:12.0.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    libboost-all-dev \
    libfftw3-dev \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build stage for C++/CUDA workloads
FROM base as build

# Copy source code
COPY . /app
WORKDIR /app

# Build all workloads
RUN ./build.sh

# Final stage
FROM base

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Install additional dependencies for benchmarking
RUN pip3 install --no-cache-dir \
    matplotlib \
    numpy \
    pandas \
    pycuda \
    psutil \
    pynvml

# Copy built libraries and Python code
COPY --from=build /app/build /app/build
COPY src/ /app/src/
COPY benchmark/ /app/benchmark/
COPY scripts/ /app/scripts/
COPY requirements.txt LICENSE README.md CONTRIBUTING.md /app/

# Set working directory
WORKDIR /app

# Create volume for results
VOLUME /app/benchmark/results

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH

# Entrypoint
ENTRYPOINT ["/app/benchmark/scripts/run_benchmarks.sh"]

# Default command (run all benchmarks)
CMD ["--all"]