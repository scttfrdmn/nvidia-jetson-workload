# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

cmake_minimum_required(VERSION 3.18)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 QUIET)
if (NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.10.4
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Add Python module
pybind11_add_module(_signal_processing
    bindings.cpp
)

# Link to signal_processing library
target_link_libraries(_signal_processing PRIVATE signal_processing)

# Installation rules
install(TARGETS _signal_processing DESTINATION ${Python_SITEARCH})