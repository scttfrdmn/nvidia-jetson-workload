/**
 * @file device_detection.cpp
 * @brief Implementation of device detection and capability identification
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include "signal_processing/device_detection.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <sstream>
#include <iomanip>

namespace signal_processing {

// Get cores per SM for different compute capabilities
int DeviceDetector::getCoresPerSM(int major, int minor) {
    // Values from CUDA Programming Guide
    if (major == 9) { // Hopper: H100
        return 128;
    } else if (major == 8) {
        if (minor == 7) { // Ampere: Jetson Orin
            return 128;
        } else { // Ampere: A100
            return 64;
        }
    } else if (major == 7) {
        if (minor == 5) { // Turing: T4
            return 64;
        } else { // Volta: V100
            return 64;
        }
    } else if (major == 6) { // Pascal
        return 64;
    } else if (major == 5) { // Maxwell
        return 128;
    } else if (major == 3) { // Kepler
        return 192;
    } else {
        return 32; // Default for unknown architecture
    }
}

// Check if Tensor Cores are available
bool DeviceDetector::hasTensorCores(int major, int minor) {
    // Tensor Cores were introduced in Volta (SM 7.0)
    // They are also present in Ampere (SM 8.x) including Jetson Orin (SM 8.7)
    // And in Hopper (SM 9.0)
    return (major >= 7);
}

// Get available Tensor Core precision modes
std::map<std::string, bool> DeviceDetector::getTensorCorePrecisions(int major, int minor) {
    std::map<std::string, bool> precisions;
    
    // Default all to false
    precisions["FP16"] = false;
    precisions["BF16"] = false;
    precisions["TF32"] = false;
    precisions["FP8"] = false;
    precisions["INT8"] = false;
    precisions["INT4"] = false;
    
    // Volta (SM 7.0) - First generation Tensor Cores
    if (major == 7 && minor == 0) {
        precisions["FP16"] = true;
        precisions["INT8"] = true;
    }
    // Turing (SM 7.5)
    else if (major == 7 && minor == 5) {
        precisions["FP16"] = true;
        precisions["INT8"] = true;
        precisions["INT4"] = true;
    }
    // Ampere (SM 8.0, 8.6, 8.7) - Second generation Tensor Cores
    else if (major == 8) {
        precisions["FP16"] = true;
        precisions["BF16"] = true;
        precisions["TF32"] = true;
        precisions["INT8"] = true;
        precisions["INT4"] = true;
    }
    // Hopper (SM 9.0) - Third generation Tensor Cores
    else if (major == 9) {
        precisions["FP16"] = true;
        precisions["BF16"] = true;
        precisions["TF32"] = true;
        precisions["FP8"] = true;
        precisions["INT8"] = true;
        precisions["INT4"] = true;
    }
    
    return precisions;
}

// Get device type from compute capability
DeviceType getDeviceTypeFromComputeCapability(int major, int minor, const std::string& name) {
    if (major == 9 && minor == 0) {
        return DeviceType::NVIDIA_SM_9_0; // H100
    } else if (major == 8) {
        if (minor == 7) {
            // Differentiate between Orin NX and Nano based on name
            if (name.find("Orin Nano") != std::string::npos) {
                return DeviceType::NVIDIA_SM_8_7_NANO;
            } else {
                return DeviceType::NVIDIA_SM_8_7_NX; // Default to NX
            }
        } else if (minor == 0) {
            return DeviceType::NVIDIA_SM_8_0; // A100
        }
    } else if (major == 7) {
        if (minor == 5) {
            return DeviceType::NVIDIA_SM_7_5; // T4
        } else if (minor == 0) {
            return DeviceType::NVIDIA_SM_7_0; // V100
        }
    }
    
    return DeviceType::NVIDIA_OTHER;
}

// Determine memory tier based on device type and total memory
DeviceMemoryTier getMemoryTier(DeviceType device_type, size_t total_memory) {
    // Convert memory to GB for easier comparison
    double memory_gb = static_cast<double>(total_memory) / (1024.0 * 1024.0 * 1024.0);
    
    switch (device_type) {
        case DeviceType::NVIDIA_SM_9_0:  // H100
            if (memory_gb >= 70.0) {
                return DeviceMemoryTier::HIGH;     // 80GB H100
            } else {
                return DeviceMemoryTier::LOW;      // Lower memory variants
            }
            
        case DeviceType::NVIDIA_SM_8_0:  // A100
            if (memory_gb >= 70.0) {
                return DeviceMemoryTier::HIGH;     // 80GB A100
            } else if (memory_gb >= 50.0) {
                return DeviceMemoryTier::MEDIUM;   // 64GB A100
            } else {
                return DeviceMemoryTier::LOW;      // 40GB A100
            }
            
        case DeviceType::NVIDIA_SM_8_7_NX:  // Jetson Orin NX
            if (memory_gb >= 14.0) {
                return DeviceMemoryTier::HIGH;     // 16GB Orin NX
            } else if (memory_gb >= 7.0) {
                return DeviceMemoryTier::MEDIUM;   // 8GB Orin NX
            } else {
                return DeviceMemoryTier::LOW;      // 4GB Orin NX (if exists)
            }
            
        case DeviceType::NVIDIA_SM_8_7_NANO:  // Jetson Orin Nano
            if (memory_gb >= 7.0) {
                return DeviceMemoryTier::HIGH;     // 8GB Orin Nano
            } else {
                return DeviceMemoryTier::LOW;      // 4GB Orin Nano
            }
            
        case DeviceType::NVIDIA_SM_7_0:  // V100
            if (memory_gb >= 30.0) {
                return DeviceMemoryTier::MEDIUM;   // 32GB V100
            } else {
                return DeviceMemoryTier::LOW;      // 16GB V100
            }
            
        case DeviceType::NVIDIA_SM_7_5:  // T4
            return DeviceMemoryTier::LOW;          // T4 typically only has 16GB
            
        default:
            return DeviceMemoryTier::UNKNOWN;
    }
}

DeviceCapabilities DeviceDetector::getDeviceCapabilities(int device_id) {
    DeviceCapabilities caps;
    
    // Default to CPU
    caps.name = "CPU";
    caps.compute_capability = "";
    caps.total_memory = 0;
    caps.clock_rate = 0;
    caps.num_multiprocessors = 0;
    caps.cores_per_sm = 0;
    caps.unified_memory_support = false;
    caps.tensor_cores_support = false;
    caps.device_type = DeviceType::CPU;
    caps.memory_tier = DeviceMemoryTier::UNKNOWN;
    caps.tensor_core_precisions = {};
    
    // CPU detection logic
    if (device_id < 0) {
        // Get CPU name and characteristics
        caps.name = "CPU";
        caps.device_type = DeviceType::CPU;
        
        // Could extract more CPU details here if needed
        // For now, just set some default values
        caps.num_multiprocessors = 1;
        caps.cores_per_sm = 1;
        
        return caps;
    }
    
#ifdef __CUDACC__
    // Check for CUDA devices
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        // No CUDA devices available or error
        return caps;
    }
    
    if (device_id >= device_count) {
        // Invalid device ID
        return caps;
    }
    
    cudaDeviceProp props;
    error = cudaGetDeviceProperties(&props, device_id);
    
    if (error != cudaSuccess) {
        // Error getting device properties
        return caps;
    }
    
    // Set device capabilities based on CUDA properties
    caps.name = props.name;
    caps.compute_capability = std::to_string(props.major) + "." + std::to_string(props.minor);
    caps.total_memory = props.totalGlobalMem;
    caps.clock_rate = props.clockRate / 1000; // Convert to MHz
    caps.num_multiprocessors = props.multiProcessorCount;
    caps.cores_per_sm = getCoresPerSM(props.major, props.minor);
    caps.unified_memory_support = (props.unifiedAddressing != 0);
    caps.tensor_cores_support = hasTensorCores(props.major, props.minor);
    caps.tensor_core_precisions = getTensorCorePrecisions(props.major, props.minor);
    caps.device_type = getDeviceTypeFromComputeCapability(props.major, props.minor, props.name);
    caps.memory_tier = getMemoryTier(caps.device_type, caps.total_memory);
#endif
    
    return caps;
}

std::vector<DeviceCapabilities> DeviceDetector::getAllDeviceCapabilities() {
    std::vector<DeviceCapabilities> all_caps;
    
    // Add CPU capabilities
    all_caps.push_back(getDeviceCapabilities(-1));
    
#ifdef __CUDACC__
    // Check for CUDA devices
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error == cudaSuccess && device_count > 0) {
        for (int device_id = 0; device_id < device_count; ++device_id) {
            all_caps.push_back(getDeviceCapabilities(device_id));
        }
    }
#endif
    
    return all_caps;
}

int DeviceDetector::getDeviceCount() {
#ifdef __CUDACC__
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
#else
    return 0;
#endif
}

bool DeviceDetector::isCudaAvailable() {
#ifdef __CUDACC__
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

int DeviceDetector::getOptimalBlockSize(
    DeviceType device_type, 
    int compute_capability_major,
    int compute_capability_minor) {
    
    switch (device_type) {
        case DeviceType::NVIDIA_SM_9_0: // H100
            return 1024;
        case DeviceType::NVIDIA_SM_8_0: // A100
            return 512;
        case DeviceType::NVIDIA_SM_8_7_NX: // Jetson Orin NX
            return 128;
        case DeviceType::NVIDIA_SM_8_7_NANO: // Jetson Orin Nano
            return 64;
        case DeviceType::NVIDIA_SM_7_5: // T4
            return 256;
        case DeviceType::NVIDIA_SM_7_0: // V100
            return 512;
        case DeviceType::NVIDIA_OTHER:
            // For unknown GPUs, choose based on compute capability
            if (compute_capability_major >= 8) {
                return 256;
            } else if (compute_capability_major >= 6) {
                return 128;
            } else {
                return 64;
            }
        case DeviceType::CPU:
        default:
            return 1; // CPU doesn't use blocks
    }
}

int DeviceDetector::getOptimalSharedMemorySize(DeviceType device_type) {
    switch (device_type) {
        case DeviceType::NVIDIA_SM_9_0: // H100
            return 228 * 1024; // 228 KB per SM (maximum)
        case DeviceType::NVIDIA_SM_8_0: // A100
            return 164 * 1024; // 164 KB per SM (maximum)
        case DeviceType::NVIDIA_SM_8_7_NX: // Jetson Orin NX
            return 64 * 1024;  // 64 KB recommended
        case DeviceType::NVIDIA_SM_8_7_NANO: // Jetson Orin Nano
            return 32 * 1024;  // 32 KB recommended (more conservative)
        case DeviceType::NVIDIA_SM_7_5: // T4
            return 48 * 1024;  // 48 KB recommended
        case DeviceType::NVIDIA_SM_7_0: // V100
            return 96 * 1024;  // 96 KB per SM (maximum)
        case DeviceType::NVIDIA_OTHER:
            return 48 * 1024;  // Conservative default
        case DeviceType::CPU:
        default:
            return 0;         // Not applicable for CPU
    }
}

int DeviceDetector::getOptimalTileSize(DeviceType device_type) {
    switch (device_type) {
        case DeviceType::NVIDIA_SM_9_0: // H100
            return 32;  // 32x32 tiles
        case DeviceType::NVIDIA_SM_8_0: // A100
            return 32;  // 32x32 tiles
        case DeviceType::NVIDIA_SM_8_7_NX: // Jetson Orin NX
            return 16;  // 16x16 tiles
        case DeviceType::NVIDIA_SM_8_7_NANO: // Jetson Orin Nano
            return 8;   // 8x8 tiles (more conservative)
        case DeviceType::NVIDIA_SM_7_5: // T4
            return 16;  // 16x16 tiles
        case DeviceType::NVIDIA_SM_7_0: // V100
            return 32;  // 32x32 tiles
        case DeviceType::NVIDIA_OTHER:
            return 16;  // Conservative default
        case DeviceType::CPU:
        default:
            return 8;   // Small tiles for CPU cache efficiency
    }
}

std::pair<int, int> DeviceDetector::getOptimalFFTConfig(
    DeviceType device_type, 
    int fft_size) {
    
    int block_size = getOptimalBlockSize(device_type);
    int shared_mem_size = fft_size * 2 * sizeof(float); // Complex numbers (2 floats)
    
    // Adjust block size based on FFT size
    if (fft_size < block_size) {
        // For small FFTs, use smaller blocks
        block_size = std::max(32, (fft_size + 31) / 32 * 32);
    }
    
    // For large FFTs, limit shared memory usage based on device
    int max_shared_mem = getOptimalSharedMemorySize(device_type);
    if (shared_mem_size > max_shared_mem) {
        // Cannot fit entire FFT in shared memory
        // Adjust strategy based on device
        switch (device_type) {
            case DeviceType::NVIDIA_SM_9_0: // H100 - larger shared memory
            case DeviceType::NVIDIA_SM_8_0: // A100 - larger shared memory
                // These devices can handle larger FFTs
                shared_mem_size = std::min(shared_mem_size, max_shared_mem);
                break;
            default:
                // For other devices, limit shared memory usage more aggressively
                shared_mem_size = std::min(shared_mem_size, max_shared_mem / 2);
                break;
        }
    }
    
    return std::make_pair(block_size, shared_mem_size);
}

std::pair<int, int> DeviceDetector::getOptimalFilterConfig(
    DeviceType device_type, 
    int filter_length) {
    
    int block_size = getOptimalBlockSize(device_type);
    
    // Shared memory includes filter coefficients and input data
    // For sliding window, we need (block_size + filter_length - 1) elements
    int shared_mem_size = (block_size + filter_length - 1) * sizeof(float);
    
    // For large filters, adjust strategy
    if (filter_length > 1024) {
        // Use constant memory for filter coefficients on older GPUs
        if (device_type == DeviceType::NVIDIA_SM_7_5 || 
            device_type == DeviceType::NVIDIA_SM_7_0 ||
            device_type == DeviceType::NVIDIA_OTHER) {
            // Just allocate space for input data in shared memory
            shared_mem_size = block_size * sizeof(float);
        }
    }
    
    // Ensure we don't exceed device limits
    int max_shared_mem = getOptimalSharedMemorySize(device_type);
    if (shared_mem_size > max_shared_mem) {
        // Need to reduce block size or use a different algorithm
        // Calculate maximum possible block size
        int max_block_size = (max_shared_mem / sizeof(float)) - filter_length + 1;
        max_block_size = std::max(32, (max_block_size / 32) * 32); // Round to multiple of 32
        
        block_size = std::min(block_size, max_block_size);
        shared_mem_size = (block_size + filter_length - 1) * sizeof(float);
    }
    
    return std::make_pair(block_size, shared_mem_size);
}

const std::map<DeviceType, std::string>& DeviceDetector::getDeviceTypeNames() {
    static const std::map<DeviceType, std::string> device_type_names = {
        {DeviceType::CPU, "CPU"},
        {DeviceType::NVIDIA_SM_7_0, "NVIDIA V100 (SM 7.0)"},
        {DeviceType::NVIDIA_SM_7_5, "NVIDIA T4 (SM 7.5)"},
        {DeviceType::NVIDIA_SM_8_0, "NVIDIA A100 (SM 8.0)"},
        {DeviceType::NVIDIA_SM_8_7_NX, "NVIDIA Jetson Orin NX (SM 8.7)"},
        {DeviceType::NVIDIA_SM_8_7_NANO, "NVIDIA Jetson Orin Nano (SM 8.7)"},
        {DeviceType::NVIDIA_SM_9_0, "NVIDIA H100 (SM 9.0)"},
        {DeviceType::NVIDIA_OTHER, "NVIDIA GPU (Other)"}
    };
    
    return device_type_names;
}

void DeviceDetector::printDeviceCapabilities(const DeviceCapabilities& caps) {
    const auto& device_type_names = getDeviceTypeNames();
    
    std::cout << "Device: " << caps.name << std::endl;
    std::cout << "Type: " << device_type_names.at(caps.device_type) << std::endl;
    
    if (caps.device_type != DeviceType::CPU) {
        std::cout << "Compute Capability: " << caps.compute_capability << std::endl;
        
        // Show memory information with tier
        std::cout << "Global Memory: " << (caps.total_memory / (1024.0 * 1024.0 * 1024.0)) << " GB";
        
        std::string tier_str = "";
        switch (caps.memory_tier) {
            case DeviceMemoryTier::LOW:
                tier_str = "Low";
                break;
            case DeviceMemoryTier::MEDIUM:
                tier_str = "Medium";
                break;
            case DeviceMemoryTier::HIGH:
                tier_str = "High";
                break;
            default:
                tier_str = "Unknown";
        }
        std::cout << " (Memory Tier: " << tier_str << ")" << std::endl;
        
        std::cout << "Clock Rate: " << caps.clock_rate << " MHz" << std::endl;
        std::cout << "Multiprocessors: " << caps.num_multiprocessors << std::endl;
        std::cout << "Cores per SM: " << caps.cores_per_sm << std::endl;
        std::cout << "Total CUDA Cores: " << caps.getTotalCores() << std::endl;
        std::cout << "Unified Memory: " << (caps.unified_memory_support ? "Yes" : "No") << std::endl;
        
        // Print Tensor Core information
        if (caps.tensor_cores_support) {
            std::cout << "Tensor Cores: Yes" << std::endl;
            std::cout << "Supported Tensor Core Precision Formats: ";
            bool first = true;
            for (const auto& pair : caps.tensor_core_precisions) {
                if (pair.second) {
                    if (!first) {
                        std::cout << ", ";
                    }
                    std::cout << pair.first;
                    first = false;
                }
            }
            std::cout << std::endl;
        } else {
            std::cout << "Tensor Cores: No" << std::endl;
        }
        
        std::cout << "Theoretical FP32 Performance: " << std::fixed << std::setprecision(2) 
                  << caps.getTheoricalPeakTFLOPS() << " TFLOPS" << std::endl;
        std::cout << "Theoretical Memory Bandwidth: " << std::fixed << std::setprecision(1) 
                  << caps.getTheoreticalMemoryBandwidth() << " GB/s" << std::endl;
    }
    
    std::cout << std::endl;
}

} // namespace signal_processing