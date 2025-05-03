// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "nbody_sim/device_adaptor.hpp"
#include <iostream>
#include <cuda_runtime.h>

namespace nbody_sim {

DeviceCapabilities detect_device_capabilities() {
    DeviceCapabilities caps;
    
    // Default to CPU fallback
    caps.device_type = DeviceType::CPU;
    caps.compute_capability_major = 0;
    caps.compute_capability_minor = 0;
    caps.global_memory_bytes = 0;
    caps.multiprocessor_count = 0;
    caps.max_threads_per_multiprocessor = 0;
    caps.max_threads_per_block = DEFAULT_BLOCK_SIZE;
    caps.max_shared_memory_per_block = 16 * 1024; // 16 KB default
    
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cout << "No CUDA-capable device found, using CPU fallback" << std::endl;
        return caps;
    }
    
    // Get properties of device 0 (we're not handling multi-GPU setups here)
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    
    // Set capabilities based on device properties
    caps.compute_capability_major = props.major;
    caps.compute_capability_minor = props.minor;
    caps.global_memory_bytes = props.totalGlobalMem;
    caps.multiprocessor_count = props.multiProcessorCount;
    caps.max_threads_per_multiprocessor = props.maxThreadsPerMultiProcessor;
    caps.max_threads_per_block = props.maxThreadsPerBlock;
    caps.max_shared_memory_per_block = props.sharedMemPerBlock;
    
    // Determine device type
    if (props.major == 8 && props.minor == 7) {
        // Jetson Orin (SM 8.7)
        caps.device_type = DeviceType::JetsonOrin;
        std::cout << "Detected NVIDIA Jetson Orin (SM 8.7)" << std::endl;
    } else if (props.major == 7 && props.minor == 5) {
        // T4 (SM 7.5)
        caps.device_type = DeviceType::T4;
        std::cout << "Detected NVIDIA T4 (SM 7.5)" << std::endl;
    } else if (props.major >= 8) {
        // Other high-end GPUs (SM >= 8.0)
        caps.device_type = DeviceType::HighEnd;
        std::cout << "Detected high-end NVIDIA GPU (SM " << props.major << "." << props.minor << ")" << std::endl;
    } else {
        // Other GPUs
        caps.device_type = DeviceType::OtherGPU;
        std::cout << "Detected NVIDIA GPU (SM " << props.major << "." << props.minor << ")" << std::endl;
    }
    
    std::cout << "GPU Memory: " << (caps.global_memory_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << caps.multiprocessor_count << std::endl;
    std::cout << "Max threads per block: " << caps.max_threads_per_block << std::endl;
    std::cout << "Shared memory per block: " << (caps.max_shared_memory_per_block / 1024) << " KB" << std::endl;
    
    return caps;
}

int get_optimal_block_size(const DeviceCapabilities& caps) {
    switch (caps.device_type) {
        case DeviceType::JetsonOrin:
            return 256;  // Optimal for SM 8.7
        case DeviceType::T4:
            return 128;  // Optimal for SM 7.5
        case DeviceType::HighEnd:
            return 256;  // General high-end GPU
        default:
            return DEFAULT_BLOCK_SIZE;
    }
}

int get_optimal_tile_size(const DeviceCapabilities& caps) {
    switch (caps.device_type) {
        case DeviceType::JetsonOrin:
            return 16;  // 16x16 tiles
        case DeviceType::T4:
            return 8;   // 8x8 tiles
        case DeviceType::HighEnd:
            return 32;  // 32x32 tiles
        default:
            return 8;
    }
}

size_t get_optimal_shared_memory(const DeviceCapabilities& caps) {
    switch (caps.device_type) {
        case DeviceType::JetsonOrin:
            return 48 * 1024;  // 48 KB
        case DeviceType::T4:
            return 32 * 1024;  // 32 KB
        case DeviceType::HighEnd:
            return 96 * 1024;  // 96 KB
        default:
            return 16 * 1024;  // 16 KB
    }
}

int get_optimal_grid_size(const DeviceCapabilities& caps, int total_elements) {
    int block_size = get_optimal_block_size(caps);
    return (total_elements + block_size - 1) / block_size;
}

} // namespace nbody_sim