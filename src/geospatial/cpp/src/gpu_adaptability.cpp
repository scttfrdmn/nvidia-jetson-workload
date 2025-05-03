/**
 * @file gpu_adaptability.cpp
 * @brief Implementation of GPU adaptability pattern for geospatial workloads
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include "geospatial/gpu_adaptability.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace geospatial {

#ifdef __CUDACC__
// Convert compute capability to DeviceType
DeviceType computeCapabilityToDeviceType(const std::string& compute_capability) {
    if (compute_capability == "7.5") {
        return DeviceType::NVIDIA_SM_7_5;
    } else if (compute_capability == "8.7") {
        return DeviceType::NVIDIA_SM_8_7;
    } else if (!compute_capability.empty()) {
        return DeviceType::NVIDIA_OTHER;
    }
    return DeviceType::CPU;
}
#endif

DeviceCapabilities DeviceAdaptor::getDeviceCapabilities(int device_id) {
    DeviceCapabilities capabilities;
    capabilities.name = "CPU";
    capabilities.compute_capability = "";
    capabilities.total_memory = 0;
    capabilities.clock_rate = 0;
    capabilities.num_multiprocessors = 0;
    capabilities.unified_memory_support = false;
    capabilities.device_type = DeviceType::CPU;

#ifdef __CUDACC__
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(error) << std::endl;
        return capabilities;
    }
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return capabilities;
    }
    
    if (device_id < 0 || device_id >= device_count) {
        std::cerr << "Invalid device ID: " << device_id << ". Using device 0 instead." << std::endl;
        device_id = 0;
    }
    
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, device_id);
    
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
        return capabilities;
    }
    
    // Convert compute capability to string
    std::string cc = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    
    capabilities.name = prop.name;
    capabilities.compute_capability = cc;
    capabilities.total_memory = prop.totalGlobalMem;
    capabilities.clock_rate = prop.clockRate / 1000; // Convert to MHz
    capabilities.num_multiprocessors = prop.multiProcessorCount;
    capabilities.unified_memory_support = prop.unifiedAddressing != 0;
    capabilities.device_type = computeCapabilityToDeviceType(cc);
#endif

    return capabilities;
}

std::tuple<int, int, int> DeviceAdaptor::getOptimalDEMKernelConfig(
    int width, int height, int device_id) {
    
    DeviceCapabilities caps = getDeviceCapabilities(device_id);
    
    // Default values for different device types
    int block_dim_x = 16;
    int block_dim_y = 16;
    int tile_size = 256;
    
    // Adjust based on device type
    switch (caps.device_type) {
        case DeviceType::NVIDIA_SM_8_7: // Jetson Orin NX
            // Jetson Orin NX has more shared memory per SM
            // and benefits from larger tiles
            block_dim_x = 32;
            block_dim_y = 8;
            tile_size = 512;
            break;
            
        case DeviceType::NVIDIA_SM_7_5: // T4 GPU
            // T4 has good memory bandwidth and benefits from
            // better thread distribution
            block_dim_x = 16;
            block_dim_y = 16;
            tile_size = 384;
            break;
            
        case DeviceType::NVIDIA_OTHER:
            // Other NVIDIA GPUs - intermediate configuration
            block_dim_x = 16;
            block_dim_y = 16;
            tile_size = 256;
            break;
            
        case DeviceType::CPU:
        default:
            // CPU-based execution with small tiles for cache efficiency
            block_dim_x = 1;
            block_dim_y = 1;
            tile_size = 64;
            break;
    }
    
    // Adjust tile size based on DEM dimensions
    int max_dimension = std::max(width, height);
    if (max_dimension < tile_size) {
        tile_size = std::max(64, (max_dimension / 64) * 64); // Multiple of 64
    }
    
    return std::make_tuple(block_dim_x, block_dim_y, tile_size);
}

std::tuple<int, int> DeviceAdaptor::getOptimalPointCloudKernelConfig(
    int num_points, int device_id) {
    
    DeviceCapabilities caps = getDeviceCapabilities(device_id);
    
    // Default values
    int block_size = 256;
    int points_per_thread = 1;
    
    // Adjust based on device type
    switch (caps.device_type) {
        case DeviceType::NVIDIA_SM_8_7: // Jetson Orin NX
            // Jetson has more compute-bound workloads
            block_size = 128;
            points_per_thread = 4;
            break;
            
        case DeviceType::NVIDIA_SM_7_5: // T4 GPU
            // T4 balances compute and memory bandwidth
            block_size = 256;
            points_per_thread = 2;
            break;
            
        case DeviceType::NVIDIA_OTHER:
            // Generic configuration
            block_size = 256;
            points_per_thread = 1;
            break;
            
        case DeviceType::CPU:
        default:
            // CPU-based execution
            block_size = 1;
            points_per_thread = 1000; // Process many points per thread on CPU
            break;
    }
    
    // Ensure we have enough blocks for the data
    int total_threads = std::ceil(static_cast<double>(num_points) / points_per_thread);
    if (total_threads < block_size && total_threads > 0) {
        // If we have fewer points than the block size, adjust the block size
        // to be a multiple of 32 (warp size) that can accommodate all points
        block_size = ((total_threads + 31) / 32) * 32;
        block_size = std::max(32, block_size); // Minimum block size of 32
    }
    
    return std::make_tuple(block_size, points_per_thread);
}

std::tuple<int, int, int> DeviceAdaptor::getOptimalImageKernelConfig(
    int width, int height, int num_bands, int device_id) {
    
    DeviceCapabilities caps = getDeviceCapabilities(device_id);
    
    // Default values
    int block_dim_x = 16;
    int block_dim_y = 16;
    int tile_size = 256;
    
    // Adjust based on device type and number of bands
    switch (caps.device_type) {
        case DeviceType::NVIDIA_SM_8_7: // Jetson Orin NX
            if (num_bands <= 3) { // RGB or fewer bands
                block_dim_x = 32;
                block_dim_y = 8;
                tile_size = 512;
            } else { // Multispectral/hyperspectral
                block_dim_x = 32;
                block_dim_y = 4;
                tile_size = 256;
            }
            break;
            
        case DeviceType::NVIDIA_SM_7_5: // T4 GPU
            if (num_bands <= 3) { // RGB or fewer bands
                block_dim_x = 16;
                block_dim_y = 16;
                tile_size = 512;
            } else { // Multispectral/hyperspectral
                block_dim_x = 16;
                block_dim_y = 8;
                tile_size = 256;
            }
            break;
            
        case DeviceType::NVIDIA_OTHER:
            // Generic configuration
            block_dim_x = 16;
            block_dim_y = 16;
            tile_size = 256;
            break;
            
        case DeviceType::CPU:
        default:
            // CPU-based execution
            block_dim_x = 1;
            block_dim_y = 1;
            tile_size = 64;
            break;
    }
    
    // Adjust tile size based on image dimensions
    int max_dimension = std::max(width, height);
    if (max_dimension < tile_size) {
        tile_size = std::max(64, (max_dimension / 64) * 64); // Multiple of 64
    }
    
    return std::make_tuple(block_dim_x, block_dim_y, tile_size);
}

std::tuple<int, int> DeviceAdaptor::getOptimalVectorKernelConfig(
    int num_features, int avg_vertices_per_feature, int device_id) {
    
    DeviceCapabilities caps = getDeviceCapabilities(device_id);
    
    // Default values
    int block_size = 256;
    int features_per_block = 1;
    
    // For vector operations, adjust based on complexity
    int complexity = avg_vertices_per_feature;
    
    // Adjust based on device type and feature complexity
    switch (caps.device_type) {
        case DeviceType::NVIDIA_SM_8_7: // Jetson Orin NX
            if (complexity < 100) { // Simple features
                block_size = 256;
                features_per_block = 4;
            } else if (complexity < 1000) { // Moderate complexity
                block_size = 256;
                features_per_block = 2;
            } else { // Complex features
                block_size = 128;
                features_per_block = 1;
            }
            break;
            
        case DeviceType::NVIDIA_SM_7_5: // T4 GPU
            if (complexity < 100) { // Simple features
                block_size = 256;
                features_per_block = 8;
            } else if (complexity < 1000) { // Moderate complexity
                block_size = 256;
                features_per_block = 4;
            } else { // Complex features
                block_size = 256;
                features_per_block = 1;
            }
            break;
            
        case DeviceType::NVIDIA_OTHER:
            // Generic configuration
            block_size = 256;
            features_per_block = 1;
            break;
            
        case DeviceType::CPU:
        default:
            // CPU-based execution
            block_size = 1;
            features_per_block = 100; // Process many features per thread on CPU
            break;
    }
    
    return std::make_tuple(block_size, features_per_block);
}

bool DeviceAdaptor::hasUnifiedMemory(int device_id) {
    DeviceCapabilities caps = getDeviceCapabilities(device_id);
    return caps.unified_memory_support;
}

std::tuple<int, int> DeviceAdaptor::getRecommendedTileSize(
    int total_width, int total_height, int bytes_per_pixel, int device_id) {
    
    DeviceCapabilities caps = getDeviceCapabilities(device_id);
    
    // Conservative estimate: use at most 25% of device memory for a single tile
    size_t max_tile_memory = caps.total_memory / 4;
    
    if (caps.device_type == DeviceType::CPU) {
        // For CPU, limit to a more conservative amount (128 MB)
        max_tile_memory = 128 * 1024 * 1024;
    }
    
    // Calculate tile dimensions based on bytes per pixel and border
    int border_size = 128; // Border size in pixels for operations that require neighbor access
    
    // Maximum number of pixels that fit in memory
    size_t max_pixels = max_tile_memory / bytes_per_pixel;
    
    // Calculate the largest square tile that will fit
    int max_tile_side = static_cast<int>(std::sqrt(max_pixels));
    
    // Adjust for border
    max_tile_side -= 2 * border_size;
    max_tile_side = std::max(256, max_tile_side); // Minimum tile size of 256
    
    // Ensure tile size is a multiple of 64 for good memory access patterns
    max_tile_side = (max_tile_side / 64) * 64;
    
    // Actual tile dimensions (capped by the total image size)
    int tile_width = std::min(max_tile_side, total_width);
    int tile_height = std::min(max_tile_side, total_height);
    
    return std::make_tuple(tile_width, tile_height);
}

} // namespace geospatial