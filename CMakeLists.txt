# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

cmake_minimum_required(VERSION 3.18)
project(nvidia-jetson-workload LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect CUDA architectures if possible
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCH_LIST)
if(CUDA_ARCH_LIST)
    string(REGEX MATCHALL "sm_([0-9]+)" CUDA_ARCH_LIST "${CUDA_ARCH_LIST}")
    list(SORT CUDA_ARCH_LIST)
    message(STATUS "Automatically detected CUDA architectures: ${CUDA_ARCH_LIST}")
else()
    # Fallback to known architectures for Jetson Orin NX and AWS Graviton g5g
    set(CUDA_ARCH_LIST 72 75 87)
    message(STATUS "Using default CUDA architectures: ${CUDA_ARCH_LIST}")
endif()

# Convert SM list to CMake format
set(CMAKE_CUDA_ARCHITECTURES)
foreach(ARCH ${CUDA_ARCH_LIST})
    string(REGEX MATCH "([0-9]+)" ARCH_NUM "${ARCH}")
    list(APPEND CMAKE_CUDA_ARCHITECTURES ${ARCH_NUM})
endforeach()

# Remove duplicates and sort
list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES)
list(SORT CMAKE_CUDA_ARCHITECTURES)
message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Options
option(BUILD_NBODY_SIM "Build N-body simulation workload" ON)
option(BUILD_WEATHER_SIM "Build weather simulation workload" ON)
option(BUILD_MEDICAL_IMAGING "Build medical imaging workload" OFF)
option(BUILD_MOLECULAR_DYNAMICS "Build molecular dynamics workload" ON)

# Add subdirectories for each workload
if(BUILD_NBODY_SIM)
    add_subdirectory(src/nbody_sim/cpp)
endif()

if(BUILD_WEATHER_SIM)
    add_subdirectory(src/weather-sim/cpp)
    add_subdirectory(src/weather-sim/cpp/pybind)
endif()

if(BUILD_MEDICAL_IMAGING)
    add_subdirectory(src/medical-imaging/cpp)
endif()

if(BUILD_MOLECULAR_DYNAMICS)
    add_subdirectory(src/molecular-dynamics/cpp)
endif()

# Print configuration
message(STATUS "Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Building N-body Simulation: ${BUILD_NBODY_SIM}")
message(STATUS "  Building Weather Simulation: ${BUILD_WEATHER_SIM}")
message(STATUS "  Building Medical Imaging: ${BUILD_MEDICAL_IMAGING}")
message(STATUS "  Building Molecular Dynamics: ${BUILD_MOLECULAR_DYNAMICS}")