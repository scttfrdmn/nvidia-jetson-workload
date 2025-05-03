cmake_minimum_required(VERSION 3.18)
project(medical_imaging_python LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find Python
find_package(Python 3.7 COMPONENTS Interpreter Development NumPy REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# CUDA architecture detection
include(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/cmake/CudaArchitectureDetection.cmake)
detect_cuda_architectures(DETECTED_CUDA_ARCH)

if(DEFINED CUDA_ARCH)
    # Use user-specified CUDA architecture
    message(STATUS "Using user-specified CUDA architecture(s): ${CUDA_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
elseif(DETECTED_CUDA_ARCH)
    # Use detected CUDA architecture
    message(STATUS "Detected CUDA architecture: ${DETECTED_CUDA_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_CUDA_ARCH})
else()
    # Default to common architectures
    message(STATUS "Using default CUDA architectures: 70;75;80;86")
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

# Include directories
include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/include
)

# Add source files
set(MODULE_SOURCES
    medical_imaging_module.cpp
)

# Add CUDA kernel files
set(KERNEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src/kernels/ct_reconstruction_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src/kernels/image_processing_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src/kernels/segmentation_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src/kernels/registration_kernels.cu
)

# Add implementation files
set(IMPLEMENTATION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src/gpu_adaptability.cpp
)

# Create pybind11 module
pybind11_add_module(_medical_imaging
    ${MODULE_SOURCES}
    ${KERNEL_SOURCES}
    ${IMPLEMENTATION_SOURCES}
)

# Set target properties
set_target_properties(_medical_imaging PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(_medical_imaging PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_cuFFT_LIBRARY}
)

# Installation
install(TARGETS _medical_imaging DESTINATION medical_imaging)