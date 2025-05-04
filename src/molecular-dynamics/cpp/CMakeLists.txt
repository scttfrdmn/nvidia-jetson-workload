# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

cmake_minimum_required(VERSION 3.18)
project(molecular_dynamics LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard and architectures
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

# Build options
option(MOLECULAR_DYNAMICS_BUILD_TESTS "Build tests" ON)
option(MOLECULAR_DYNAMICS_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(MOLECULAR_DYNAMICS_ENABLE_LOGGING "Enable logging" ON)
option(MOLECULAR_DYNAMICS_ENABLE_PROFILING "Enable performance profiling" ON)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Find packages
find_package(CUDAToolkit REQUIRED)

# Set compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Define sources
set(MOLECULAR_DYNAMICS_SOURCES
    src/atom.cpp
    src/molecular_system.cpp
    src/integrator.cpp
    src/simulation.cpp
    src/cuda_kernels.cu
)

# Create main library
add_library(molecular_dynamics ${MOLECULAR_DYNAMICS_SOURCES})
target_link_libraries(molecular_dynamics PRIVATE CUDA::cudart)

# Create executable
add_executable(molecular_dynamics_app src/main.cpp)
target_link_libraries(molecular_dynamics_app PRIVATE molecular_dynamics)

# Install targets
install(TARGETS molecular_dynamics molecular_dynamics_app
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/
    DESTINATION include/molecular_dynamics
    FILES_MATCHING PATTERN "*.h*"
)

# Testing
if(MOLECULAR_DYNAMICS_BUILD_TESTS)
    enable_testing()
    
    # Find Google Test package
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.12.1
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    
    # Test executable
    add_executable(molecular_dynamics_test
        test/test_main.cpp
        test/test_atom.cpp
        test/test_molecular_system.cpp
        test/test_integrator.cpp
        test/test_simulation.cpp
    )
    
    target_link_libraries(molecular_dynamics_test
        PRIVATE
        molecular_dynamics
        GTest::gtest
        GTest::gtest_main
    )
    
    # Add tests
    add_test(NAME molecular_dynamics_test COMMAND molecular_dynamics_test)
endif()

# Python bindings
if(MOLECULAR_DYNAMICS_BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG QUIET)
    
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    pybind11_add_module(pymolecular_dynamics src/python_bindings.cpp)
    target_link_libraries(pymolecular_dynamics PRIVATE molecular_dynamics)
endif()

# Print configuration
message(STATUS "Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Tests: ${MOLECULAR_DYNAMICS_BUILD_TESTS}")
message(STATUS "  Build Python Bindings: ${MOLECULAR_DYNAMICS_BUILD_PYTHON_BINDINGS}")
message(STATUS "  Enable Logging: ${MOLECULAR_DYNAMICS_ENABLE_LOGGING}")
message(STATUS "  Enable Profiling: ${MOLECULAR_DYNAMICS_ENABLE_PROFILING}")