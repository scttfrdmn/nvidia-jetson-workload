/**
 * @file device_detection.h
 * @brief Device detection and capability identification for signal processing
 * 
 * This header provides utilities for detecting GPU capabilities and
 * selecting optimal implementation strategies based on device characteristics.
 * It supports a wide range of NVIDIA GPUs from Jetson Orin to data center GPUs.
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#ifndef SIGNAL_PROCESSING_DEVICE_DETECTION_H
#define SIGNAL_PROCESSING_DEVICE_DETECTION_H

#include <string>
#include <vector>
#include <map>
#include <utility>
#include <iostream>

namespace signal_processing {

/**
 * @enum DeviceType
 * @brief Enumeration of supported device types
 */
enum class DeviceType {
    CPU,                  ///< CPU-only execution
    NVIDIA_SM_7_0,        ///< NVIDIA GPU with SM 7.0 (e.g., V100)
    NVIDIA_SM_7_5,        ///< NVIDIA GPU with SM 7.5 (e.g., T4)
    NVIDIA_SM_8_0,        ///< NVIDIA GPU with SM 8.0 (e.g., A100)
    NVIDIA_SM_8_7_NX,     ///< NVIDIA GPU with SM 8.7 (Jetson Orin NX)
    NVIDIA_SM_8_7_NANO,   ///< NVIDIA GPU with SM 8.7 (Jetson Orin Nano)
    NVIDIA_SM_9_0,        ///< NVIDIA GPU with SM 9.0 (e.g., H100)
    NVIDIA_OTHER          ///< Other NVIDIA GPU
};

/**
 * @enum DeviceMemoryTier
 * @brief Enumeration of memory size tiers for devices that have multiple VRAM configurations
 */
enum class DeviceMemoryTier {
    LOW,        ///< Lower memory tier (e.g., 16GB V100, 40GB A100, 4GB Orin NX)
    MEDIUM,     ///< Medium memory tier (e.g., 32GB V100, 8GB Orin NX)
    HIGH,       ///< High memory tier (e.g., 80GB A100, 80GB H100, 16GB Orin NX)
    UNKNOWN     ///< Unknown memory tier
};

/**
 * @struct DeviceCapabilities
 * @brief Structure holding device capability information
 */
struct DeviceCapabilities {
    std::string name;               ///< Device name
    std::string compute_capability; ///< Compute capability (e.g., "8.7")
    size_t total_memory;            ///< Total device memory in bytes
    int clock_rate;                 ///< Clock rate in MHz
    int num_multiprocessors;        ///< Number of multiprocessors
    int cores_per_sm;               ///< CUDA cores per SM
    bool unified_memory_support;    ///< Whether the device supports unified memory
    bool tensor_cores_support;      ///< Whether the device has Tensor Cores
    DeviceType device_type;         ///< Device type classification
    DeviceMemoryTier memory_tier;   ///< Memory size tier classification
    std::map<std::string, bool> tensor_core_precisions; ///< Supported Tensor Core precision formats
    
    /**
     * @brief Get the total number of CUDA cores
     * @return Number of CUDA cores
     */
    int getTotalCores() const {
        return num_multiprocessors * cores_per_sm;
    }
    
    /**
     * @brief Get theoretical peak floating-point performance (FP32) in TFLOPS
     * @return FP32 TFLOPS
     */
    float getTheoricalPeakTFLOPS() const {
        // Clock rate is in MHz, convert to GHz and compute 2 ops per core per cycle
        return 2.0f * getTotalCores() * (clock_rate / 1000.0f) / 1000.0f;
    }
    
    /**
     * @brief Get theoretical memory bandwidth in GB/s
     * @return Memory bandwidth in GB/s
     * 
     * Note: This is an approximation based on device type, 
     * actual memory bandwidth depends on specific model
     */
    float getTheoreticalMemoryBandwidth() const {
        switch (device_type) {
            case DeviceType::NVIDIA_SM_7_0:
                return 900.0f;    // V100: ~900 GB/s
            case DeviceType::NVIDIA_SM_7_5:
                return 320.0f;    // T4: ~320 GB/s
            case DeviceType::NVIDIA_SM_8_0:
                return 2000.0f;   // A100: ~2 TB/s
            case DeviceType::NVIDIA_SM_8_7_NX:
                return 204.8f;    // Orin NX: ~204.8 GB/s
            case DeviceType::NVIDIA_SM_8_7_NANO:
                return 102.4f;    // Orin Nano: ~102.4 GB/s
            case DeviceType::NVIDIA_SM_9_0:
                return 3000.0f;   // H100: ~3 TB/s
            case DeviceType::NVIDIA_OTHER:
                return 500.0f;    // Conservative estimate
            case DeviceType::CPU:
            default:
                return 50.0f;     // Typical CPU memory bandwidth
        }
    }
};

/**
 * @class DeviceDetector
 * @brief Detects and provides information about available computing devices
 */
class DeviceDetector {
public:
    /**
     * @brief Get device capabilities for a specific device
     * @param device_id Device ID (-1 for CPU)
     * @return Device capabilities
     */
    static DeviceCapabilities getDeviceCapabilities(int device_id = 0);
    
    /**
     * @brief Get capabilities for all available devices
     * @return Vector of device capabilities
     */
    static std::vector<DeviceCapabilities> getAllDeviceCapabilities();
    
    /**
     * @brief Get the number of available CUDA devices
     * @return Number of CUDA devices (0 if CUDA is not available)
     */
    static int getDeviceCount();
    
    /**
     * @brief Check if CUDA is available
     * @return True if CUDA is available, false otherwise
     */
    static bool isCudaAvailable();
    
    /**
     * @brief Get optimal block size for a specific device type
     * @param device_type Device type
     * @param compute_capability_major Major compute capability
     * @param compute_capability_minor Minor compute capability
     * @return Recommended block size
     */
    static int getOptimalBlockSize(
        DeviceType device_type, 
        int compute_capability_major = 0,
        int compute_capability_minor = 0);
    
    /**
     * @brief Get optimal shared memory size for a specific device type
     * @param device_type Device type
     * @return Recommended shared memory size (in bytes per block)
     */
    static int getOptimalSharedMemorySize(DeviceType device_type);
    
    /**
     * @brief Get optimal tile size for a specific device type
     * @param device_type Device type
     * @return Recommended tile size
     */
    static int getOptimalTileSize(DeviceType device_type);
    
    /**
     * @brief Get optimal FFT configuration for a specific device type
     * @param device_type Device type
     * @param fft_size FFT size
     * @return Pair of (block_size, shared_memory_size)
     */
    static std::pair<int, int> getOptimalFFTConfig(
        DeviceType device_type, 
        int fft_size);
    
    /**
     * @brief Get optimal filter configuration for a specific device type
     * @param device_type Device type
     * @param filter_length Filter length
     * @return Pair of (block_size, shared_memory_size)
     */
    static std::pair<int, int> getOptimalFilterConfig(
        DeviceType device_type, 
        int filter_length);
    
    /**
     * @brief Get core count per SM for a specific compute capability
     * @param major Major compute capability
     * @param minor Minor compute capability
     * @return Cores per SM
     */
    static int getCoresPerSM(int major, int minor);
    
    /**
     * @brief Check if Tensor Cores are available for a specific compute capability
     * @param major Major compute capability
     * @param minor Minor compute capability
     * @return True if Tensor Cores are available, false otherwise
     */
    static bool hasTensorCores(int major, int minor);
    
    /**
     * @brief Get available Tensor Core precision modes
     * @param major Major compute capability
     * @param minor Minor compute capability
     * @return Map of supported precision formats (FP16, BF16, FP8, etc.) to boolean
     */
    static std::map<std::string, bool> getTensorCorePrecisions(int major, int minor);
    
    /**
     * @brief Get a map of device type to readable name
     * @return Map of device type to name
     */
    static const std::map<DeviceType, std::string>& getDeviceTypeNames();
    
    /**
     * @brief Print device capabilities to standard output
     * @param caps Device capabilities
     */
    static void printDeviceCapabilities(const DeviceCapabilities& caps);
};

} // namespace signal_processing

#endif // SIGNAL_PROCESSING_DEVICE_DETECTION_H