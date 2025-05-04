cmake_minimum_required(VERSION 3.12)
project(pyweather_sim)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
if(NOT TARGET pybind11::pybind11)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.10.4
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Add pybind11 module
pybind11_add_module(pyweather_sim ../src/python_bindings.cpp)

# Include directories
target_include_directories(pyweather_sim PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CUDA_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(pyweather_sim PRIVATE weather_sim)

# Set output directory
set_target_properties(pyweather_sim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python
)

# Install module
install(TARGETS pyweather_sim DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python)