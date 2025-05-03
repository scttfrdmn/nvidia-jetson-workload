# Test configuration

# Find GTest
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Weather Grid tests
add_executable(weather_grid_test weather_grid_test.cpp)
target_link_libraries(weather_grid_test weather_sim gtest gtest_main)
add_test(NAME weather_grid_test COMMAND weather_grid_test)

# Initial Conditions tests
add_executable(initial_conditions_test initial_conditions_test.cpp)
target_link_libraries(initial_conditions_test weather_sim gtest gtest_main)
add_test(NAME initial_conditions_test COMMAND initial_conditions_test)

# Weather Simulation tests
add_executable(weather_simulation_test weather_simulation_test.cpp)
target_link_libraries(weather_simulation_test weather_sim gtest gtest_main)
add_test(NAME weather_simulation_test COMMAND weather_simulation_test)

# GPU Adaptability tests
if(CUDA_FOUND)
    add_executable(gpu_adaptability_test gpu_adaptability_test.cpp)
    target_link_libraries(gpu_adaptability_test weather_sim weather_sim_cuda gtest gtest_main)
    add_test(NAME gpu_adaptability_test COMMAND gpu_adaptability_test)
endif()