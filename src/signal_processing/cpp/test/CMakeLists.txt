# Test CMakeLists.txt for Signal Processing module
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Add test executables
add_executable(test_fir_filter_cuda test_fir_filter_cuda.cpp)
target_link_libraries(test_fir_filter_cuda PRIVATE signal_processing)

add_executable(test_multirate_filter_cuda test_multirate_filter_cuda.cpp)
target_link_libraries(test_multirate_filter_cuda PRIVATE signal_processing)

add_executable(test_adaptive_filter_cuda test_adaptive_filter_cuda.cpp)
target_link_libraries(test_adaptive_filter_cuda PRIVATE signal_processing)

add_executable(test_time_frequency test_time_frequency.cpp)
target_link_libraries(test_time_frequency PRIVATE signal_processing)

add_executable(test_wavelet_transform test_wavelet_transform.cpp)
target_link_libraries(test_wavelet_transform PRIVATE signal_processing)

# Add to CTest registry
add_test(NAME TestFIRFilterCUDA COMMAND test_fir_filter_cuda)
add_test(NAME TestMultirateFilterCUDA COMMAND test_multirate_filter_cuda)
add_test(NAME TestAdaptiveFilterCUDA COMMAND test_adaptive_filter_cuda)
add_test(NAME TestTimeFrequency COMMAND test_time_frequency)
add_test(NAME TestWaveletTransform COMMAND test_wavelet_transform)

# Set CUDA specific flags if CUDA is available
if(WITH_CUDA)
  set_target_properties(test_fir_filter_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(test_fir_filter_cuda PROPERTIES CUDA_ARCHITECTURES "75;87")
  
  set_target_properties(test_multirate_filter_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(test_multirate_filter_cuda PROPERTIES CUDA_ARCHITECTURES "75;87")
  
  set_target_properties(test_adaptive_filter_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(test_adaptive_filter_cuda PROPERTIES CUDA_ARCHITECTURES "75;87")
  
  set_target_properties(test_time_frequency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(test_time_frequency PROPERTIES CUDA_ARCHITECTURES "75;87")
  
  set_target_properties(test_wavelet_transform PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(test_wavelet_transform PROPERTIES CUDA_ARCHITECTURES "75;87")
endif()