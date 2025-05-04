# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

cmake_minimum_required(VERSION 3.15)
project(geospatial_workload LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Project version
set(PROJECT_VERSION "1.0.0")

# Add CUDA flags for both Jetson Orin NX (SM 8.7) and AWS T4 (SM 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75 87)

# Option to build with or without CUDA
option(WITH_CUDA "Build with CUDA support" ON)

# Find dependencies
find_package(GDAL REQUIRED)

# Enable CUDA if requested and available
if(WITH_CUDA)
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND)
        message(WARNING "CUDA toolkit not found, building CPU-only version")
        set(WITH_CUDA OFF)
    endif()
endif()

# Define source files
set(SOURCES
    src/gpu_adaptability.cpp
)

# Add CUDA kernels if CUDA is enabled
if(WITH_CUDA)
    set(CUDA_SOURCES
        src/kernels/dem_kernels.cu
        src/kernels/point_cloud_kernels.cu
    )
    add_definitions(-DWITH_CUDA)
endif()

# Define include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GDAL_INCLUDE_DIRS}
)

# Create library
if(WITH_CUDA)
    add_library(geospatial SHARED ${SOURCES} ${CUDA_SOURCES})
    target_link_libraries(geospatial ${GDAL_LIBRARIES} CUDA::cudart)
else()
    add_library(geospatial SHARED ${SOURCES})
    target_link_libraries(geospatial ${GDAL_LIBRARIES})
endif()

# Set library properties
set_target_properties(geospatial PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

# Add Python bindings
add_subdirectory(pybind)

# Install targets
install(TARGETS geospatial
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install headers
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

# Add tests
enable_testing()
add_subdirectory(test)