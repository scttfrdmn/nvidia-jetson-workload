cmake_minimum_required(VERSION 3.14)
project(signal_processing_python LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Option to use CUDA
option(USE_CUDA "Build with CUDA support" ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.10.0
)
FetchContent_MakeAvailable(pybind11)

# Path to main C++ library
set(CPP_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cpp)
set(CPP_INCLUDE_DIR ${CPP_LIB_DIR}/include)
set(CPP_SRC_DIR ${CPP_LIB_DIR}/src)
set(CPP_KERNELS_DIR ${CPP_SRC_DIR}/kernels)

# Find source files
file(GLOB CPP_SOURCES "${CPP_SRC_DIR}/*.cpp")
file(GLOB MODULE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/signal_processing/_signal_processing.cpp")

# CUDA configuration
if(USE_CUDA)
    # Find CUDA
    find_package(CUDA REQUIRED)
    
    # Add CUDA directories
    include_directories(${CUDA_INCLUDE_DIRS})
    
    # Define CUDA architectures
    if(NOT DEFINED CUDA_ARCHS)
        # Default CUDA architectures for our targets
        set(CUDA_ARCHS "75;87")
    endif()
    
    # Build CUDA code for each architecture
    set(CUDA_NVCC_FLAGS "")
    foreach(arch ${CUDA_ARCHS})
        list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${arch},code=sm_${arch}")
    endforeach()
    
    # Add CUDA compilation flag
    add_definitions(-DUSE_CUDA)
    
    # Find CUDA kernel files
    file(GLOB CUDA_SOURCES "${CPP_KERNELS_DIR}/*.cu")
    
    # Set CUDA compile flags
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    list(APPEND CUDA_NVCC_FLAGS "--std=c++17" "-O3")
    
    # Compile CUDA sources
    cuda_add_library(signal_processing_cuda STATIC ${CUDA_SOURCES})
    target_include_directories(signal_processing_cuda PRIVATE ${CPP_INCLUDE_DIR})
endif()

# Create the Python module
pybind11_add_module(_signal_processing ${MODULE_SOURCES} ${CPP_SOURCES})

# Include directories
target_include_directories(_signal_processing PRIVATE ${CPP_INCLUDE_DIR})

# Link CUDA libraries if enabled
if(USE_CUDA)
    target_link_libraries(_signal_processing PRIVATE signal_processing_cuda ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY} ${CUDA_cublas_LIBRARY})
endif()

# Set output directory
set_target_properties(_signal_processing PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/signal_processing"
)

# Install the module
install(TARGETS _signal_processing DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/signal_processing")