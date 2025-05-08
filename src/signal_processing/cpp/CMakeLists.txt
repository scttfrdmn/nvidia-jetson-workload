# CMakeLists.txt for Signal Processing module
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

cmake_minimum_required(VERSION 3.18)

# Set project name
project(signal_processing LANGUAGES CXX)

# Check for CUDA
option(WITH_CUDA "Build with CUDA support" ON)
if(WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DWITH_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS})
  message(STATUS "Building with CUDA support")
  message(STATUS "CUDA Version: ${CUDA_VERSION}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Define source files
set(SIGNAL_PROCESSING_SOURCES
  src/digital_filtering.cpp
  src/spectral_analysis.cpp
  src/time_frequency.cpp
  src/device_detection.cpp
  src/wavelet_transform.cpp
)

if(WITH_CUDA)
  set(SIGNAL_PROCESSING_CUDA_SOURCES
    src/kernels/filter_kernels.cu
    src/kernels/fft_kernels.cu
    src/kernels/time_frequency_kernels.cu
    src/kernels/high_perf_gpu_support.cu
    src/kernels/wavelet_kernels.cu
  )
endif()

# Create the library
if(WITH_CUDA)
  add_library(signal_processing STATIC ${SIGNAL_PROCESSING_SOURCES} ${SIGNAL_PROCESSING_CUDA_SOURCES})
  set_target_properties(signal_processing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  
  # Set CUDA architectures for high-performance GPUs and edge devices
  # SM 7.0 (V100), SM 7.5 (T4), SM 8.0 (A100), SM 8.7 (Orin), SM 9.0 (H100)
  set_target_properties(signal_processing PROPERTIES CUDA_ARCHITECTURES "70;75;80;87;90")
else()
  add_library(signal_processing STATIC ${SIGNAL_PROCESSING_SOURCES})
endif()

# Link libraries
target_link_libraries(signal_processing 
  PUBLIC
    ${CUDA_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
)

# Add test subdirectory
add_subdirectory(test)

# Add pybind subdirectory if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pybind/CMakeLists.txt")
  add_subdirectory(pybind)
endif()

# Installation rules
install(TARGETS signal_processing
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h*"
)