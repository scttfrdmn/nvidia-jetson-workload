/**
 * @file gpu_adaptability.hpp
 * @brief GPU adaptability pattern for geospatial workloads
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#ifndef GEOSPATIAL_GPU_ADAPTABILITY_HPP
#define GEOSPATIAL_GPU_ADAPTABILITY_HPP

#include <string>
#include <memory>
#include <vector>

namespace geospatial {

/**
 * @enum DeviceType
 * @brief Enumeration of supported device types
 */
enum class DeviceType {
    CPU,            ///< CPU-only execution
    NVIDIA_SM_7_5,  ///< NVIDIA GPU with SM 7.5 (e.g., T4)
    NVIDIA_SM_8_7,  ///< NVIDIA GPU with SM 8.7 (e.g., Jetson Orin NX)
    NVIDIA_OTHER    ///< Other NVIDIA GPU
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
    bool unified_memory_support;    ///< Whether the device supports unified memory
    DeviceType device_type;         ///< Device type classification
};

/**
 * @class DeviceAdaptor
 * @brief Provides device-specific adaptations for geospatial workloads
 */
class DeviceAdaptor {
public:
    /**
     * @brief Get the device capabilities for the specified device
     * @param device_id The CUDA device ID
     * @return DeviceCapabilities structure with device information
     */
    static DeviceCapabilities getDeviceCapabilities(int device_id = 0);
    
    /**
     * @brief Get optimal kernel configuration for DEM operations
     * @param width Width of the DEM raster
     * @param height Height of the DEM raster
     * @param device_id The CUDA device ID
     * @return Tuple of (block_dim_x, block_dim_y, tile_size)
     */
    static std::tuple<int, int, int> getOptimalDEMKernelConfig(
        int width, int height, int device_id = 0);
    
    /**
     * @brief Get optimal kernel configuration for point cloud operations
     * @param num_points Number of points in the point cloud
     * @param device_id The CUDA device ID
     * @return Tuple of (block_size, points_per_thread)
     */
    static std::tuple<int, int> getOptimalPointCloudKernelConfig(
        int num_points, int device_id = 0);
    
    /**
     * @brief Get optimal kernel configuration for image operations
     * @param width Width of the image
     * @param height Height of the image
     * @param num_bands Number of image bands
     * @param device_id The CUDA device ID
     * @return Tuple of (block_dim_x, block_dim_y, tile_size)
     */
    static std::tuple<int, int, int> getOptimalImageKernelConfig(
        int width, int height, int num_bands, int device_id = 0);
    
    /**
     * @brief Get optimal kernel configuration for vector operations
     * @param num_features Number of vector features
     * @param avg_vertices_per_feature Average number of vertices per feature
     * @param device_id The CUDA device ID
     * @return Tuple of (block_size, features_per_block)
     */
    static std::tuple<int, int> getOptimalVectorKernelConfig(
        int num_features, int avg_vertices_per_feature, int device_id = 0);
    
    /**
     * @brief Determine if the device supports unified memory
     * @param device_id The CUDA device ID
     * @return True if unified memory is supported, false otherwise
     */
    static bool hasUnifiedMemory(int device_id = 0);
    
    /**
     * @brief Get recommended tile size for processing large rasters
     * @param total_width Total width of the raster
     * @param total_height Total height of the raster
     * @param bytes_per_pixel Bytes per pixel
     * @param device_id The CUDA device ID
     * @return Recommended tile size in pixels
     */
    static std::tuple<int, int> getRecommendedTileSize(
        int total_width, int total_height, int bytes_per_pixel, int device_id = 0);
};

} // namespace geospatial

#endif // GEOSPATIAL_GPU_ADAPTABILITY_HPP