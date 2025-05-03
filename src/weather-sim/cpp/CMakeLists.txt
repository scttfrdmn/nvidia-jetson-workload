cmake_minimum_required(VERSION 3.12)
project(weather_simulation LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Find CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    add_definitions(-DHAVE_CUDA)
    
    # Detect CUDA architectures (dynamically based on available hardware)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(CUDA_ARCHS)
    string(REPLACE " " ";" CUDA_ARCHS "${CUDA_ARCHS}")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})
    
    # If no GPU found or CUDA architecture detection failed, set default architectures
    if(NOT CUDA_ARCHS)
        # Support for Jetson Orin NX (SM 8.7), T4 (SM 7.5), and fallback (SM 6.0)
        set(CMAKE_CUDA_ARCHITECTURES "60;75;87")
    endif()
    
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA not found, building CPU-only version")
endif()

# Check for optional dependencies
find_package(NetCDF QUIET)
if(NetCDF_FOUND)
    add_definitions(-DHAVE_NETCDF)
endif()

find_package(VTK QUIET)
if(VTK_FOUND)
    add_definitions(-DHAVE_VTK)
endif()

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${NETCDF_INCLUDES}
    ${VTK_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/weather_grid.cpp
    src/weather_simulation.cpp
    src/initial_conditions.cpp
    src/gpu_adaptability.cpp
)

# CUDA kernel files
set(CUDA_SOURCES
    src/kernels/shallow_water_kernels.cu
)

# Create library
add_library(weather_sim ${SOURCES})

# Add CUDA library if CUDA is found
if(CUDA_FOUND)
    cuda_add_library(weather_sim_cuda ${CUDA_SOURCES})
    target_link_libraries(weather_sim weather_sim_cuda)
endif()

# Link with optional dependencies
if(OpenMP_CXX_FOUND)
    target_link_libraries(weather_sim OpenMP::OpenMP_CXX)
endif()

if(NetCDF_FOUND)
    target_link_libraries(weather_sim ${NETCDF_LIBRARIES})
endif()

if(VTK_FOUND)
    target_link_libraries(weather_sim ${VTK_LIBRARIES})
endif()

if(CUDA_FOUND)
    target_link_libraries(weather_sim ${CUDA_LIBRARIES})
endif()

# Install targets
install(TARGETS weather_sim DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# Add example executable
add_executable(weather_sim_example examples/weather_sim_example.cpp)
target_link_libraries(weather_sim_example weather_sim)

# Enable testing
enable_testing()
add_subdirectory(test)