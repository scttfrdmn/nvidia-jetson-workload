cmake_minimum_required(VERSION 3.14)

# Include Catch2 for testing
find_package(Catch2 REQUIRED)
include(CTest)
include(Catch)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
)

# List of test source files
set(TEST_SOURCES
    test_dem_processing.cpp
    test_point_cloud.cpp
)

# Create test executables
add_executable(geospatial_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(geospatial_tests
    PRIVATE
    geospatial
    Catch2::Catch2WithMain
)

# Add compile definitions for test configuration
target_compile_definitions(geospatial_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_data"
)

# Create a directory for test data if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_data)

# Register tests with CTest
catch_discover_tests(geospatial_tests)

# If synthetic test data generation is needed
add_custom_command(
    TARGET geospatial_tests
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Preparing test data..."
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_test_data.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating synthetic test data for geospatial tests"
)