# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

cmake_minimum_required(VERSION 3.18)
project(nbody_sim LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard and architectures
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 72 75 87) # For Jetson Orin NX and AWS Graviton g5g

# Build options
option(NBODY_BUILD_TESTS "Build tests" ON)
option(NBODY_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(NBODY_ENABLE_LOGGING "Enable logging" ON)
option(NBODY_ENABLE_PROFILING "Enable performance profiling" ON)

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
set(NBODY_SOURCES
    src/particle.cpp
    src/integrator.cpp
    src/simulation.cpp
    src/cuda_kernels.cu
    src/adaptive_kernels.cu
    src/device_adaptor.cpp
)

# Create main library
add_library(nbody_sim ${NBODY_SOURCES})
target_link_libraries(nbody_sim PRIVATE CUDA::cudart)

# Create executable
add_executable(nbody_sim_app src/main.cpp)
target_link_libraries(nbody_sim_app PRIVATE nbody_sim)

# Install targets
install(TARGETS nbody_sim nbody_sim_app
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY include/
    DESTINATION include/nbody_sim
    FILES_MATCHING PATTERN "*.h*"
)

# Testing
if(NBODY_BUILD_TESTS)
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
    add_executable(nbody_sim_test
        test/test_main.cpp
        test/test_particle.cpp
        test/test_integrator.cpp
        test/test_simulation.cpp
    )
    
    target_link_libraries(nbody_sim_test
        PRIVATE
        nbody_sim
        GTest::gtest
        GTest::gtest_main
    )
    
    # Add tests
    add_test(NAME nbody_sim_test COMMAND nbody_sim_test)
endif()

# Python bindings
if(NBODY_BUILD_PYTHON_BINDINGS)
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
    
    pybind11_add_module(pynbody_sim src/python_bindings.cpp)
    target_link_libraries(pynbody_sim PRIVATE nbody_sim)
endif()

# Print configuration
message(STATUS "Configuration:")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Tests: ${NBODY_BUILD_TESTS}")
message(STATUS "  Build Python Bindings: ${NBODY_BUILD_PYTHON_BINDINGS}")
message(STATUS "  Enable Logging: ${NBODY_ENABLE_LOGGING}")
message(STATUS "  Enable Profiling: ${NBODY_ENABLE_PROFILING}")