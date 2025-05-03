/**
 * @file point_cloud.hpp
 * @brief Point cloud processing for geospatial data (LiDAR)
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#ifndef GEOSPATIAL_POINT_CLOUD_HPP
#define GEOSPATIAL_POINT_CLOUD_HPP

#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include <array>
#include <unordered_map>

#include "dem_processing.hpp" // For GeoTransform

namespace geospatial {

/**
 * @enum PointClass
 * @brief Standard LiDAR point classifications
 * Based on ASPRS LAS specification
 */
enum class PointClass : uint8_t {
    Created = 0,        ///< Created, unclassified
    Unclassified = 1,   ///< Unclassified
    Ground = 2,         ///< Ground
    LowVegetation = 3,  ///< Low vegetation
    MedVegetation = 4,  ///< Medium vegetation
    HighVegetation = 5, ///< High vegetation
    Building = 6,       ///< Building
    LowPoint = 7,       ///< Low point (noise)
    Water = 9,          ///< Water
    Rail = 10,          ///< Rail
    RoadSurface = 11,   ///< Road surface
    Bridge = 12,        ///< Bridge
    WireGuard = 13,     ///< Wire - guard
    WireConductor = 14, ///< Wire - conductor
    TransmissionTower = 15, ///< Transmission tower
    WireStructure = 16, ///< Wire-structure connector
    Bridge_Deck = 17,   ///< Bridge deck
    HighNoise = 18      ///< High noise
};

/**
 * @struct PointAttributes
 * @brief Additional attributes for point cloud points
 */
struct PointAttributes {
    uint8_t intensity;      ///< Intensity value (0-255)
    uint8_t return_number;  ///< Return number (1-15)
    uint8_t num_returns;    ///< Number of returns (1-15)
    uint8_t scan_direction; ///< Scan direction flag (0-1)
    uint8_t edge_of_flight; ///< Edge of flight line flag (0-1)
    PointClass classification; ///< Point classification
    uint8_t scan_angle_rank; ///< Scan angle rank (-90 to +90)
    uint8_t user_data;      ///< User data
    uint16_t point_source_id; ///< Point source ID
    double gps_time;        ///< GPS time
};

/**
 * @class PointCloud
 * @brief Class for processing LiDAR point cloud data
 */
class PointCloud {
public:
    /**
     * @brief Constructor
     * @param point_cloud_file Path to point cloud file (LAS/LAZ format)
     * @param device_id CUDA device ID (default: 0)
     */
    PointCloud(const std::string& point_cloud_file, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~PointCloud();
    
    /**
     * @brief Get number of points in the point cloud
     * @return Number of points
     */
    size_t getNumPoints() const;
    
    /**
     * @brief Get point cloud bounds
     * @return Tuple of (min_x, min_y, min_z, max_x, max_y, max_z)
     */
    std::tuple<double, double, double, double, double, double> getBounds() const;
    
    /**
     * @brief Classify points into standard LiDAR classes
     * @param algorithm Classification algorithm (0=default, 1=progressive, 2=deep learning)
     * @return New PointCloud with classified points
     */
    std::shared_ptr<PointCloud> classifyPoints(int algorithm = 0);
    
    /**
     * @brief Filter points by classification
     * @param classification Point classification to filter
     * @return New PointCloud with filtered points
     */
    std::shared_ptr<PointCloud> filterByClass(PointClass classification);
    
    /**
     * @brief Filter points by return number
     * @param return_number Return number to filter (1 = first return, etc.)
     * @return New PointCloud with filtered points
     */
    std::shared_ptr<PointCloud> filterByReturnNumber(int return_number);
    
    /**
     * @brief Create Digital Elevation Model from point cloud
     * @param resolution Resolution of the output DEM in the same units as the point cloud
     * @param algorithm Algorithm for DEM creation (0=TIN, 1=IDW, 2=natural neighbor)
     * @return DEMProcessor with created DEM
     */
    std::shared_ptr<DEMProcessor> createDEM(double resolution, int algorithm = 0);
    
    /**
     * @brief Create Digital Surface Model from point cloud
     * @param resolution Resolution of the output DSM in the same units as the point cloud
     * @param algorithm Algorithm for DSM creation (0=highest point, 1=percentile)
     * @return DEMProcessor with created DSM
     */
    std::shared_ptr<DEMProcessor> createDSM(double resolution, int algorithm = 0);
    
    /**
     * @brief Extract building footprints from point cloud
     * @param min_height Minimum height difference for building detection
     * @param min_area Minimum area for building footprint (square units)
     * @return Vector data with building polygons
     */
    std::string extractBuildings(double min_height = 2.0, double min_area = 10.0);
    
    /**
     * @brief Extract vegetation cover from point cloud
     * @param height_classes Vector of height class thresholds
     * @param resolution Resolution of the output raster
     * @return Raster with vegetation heights
     */
    std::shared_ptr<float> extractVegetation(
        const std::vector<double>& height_classes = {0.5, 2.0, 5.0, 15.0},
        double resolution = 1.0);
    
    /**
     * @brief Segment point cloud into coherent objects
     * @param max_distance Maximum distance between points in the same segment
     * @param min_points Minimum number of points in a segment
     * @return Map of segment IDs to point indices
     */
    std::unordered_map<int, std::vector<size_t>> segmentPoints(
        double max_distance = 1.0, int min_points = 10);
    
    /**
     * @brief Save point cloud to file
     * @param output_file Output file path (LAS/LAZ format)
     * @return True if successful, false otherwise
     */
    bool save(const std::string& output_file);
    
    /**
     * @brief Get point at specified index
     * @param index Point index
     * @return Tuple of (x, y, z, attributes)
     */
    std::tuple<double, double, double, PointAttributes> getPoint(size_t index) const;
    
    /**
     * @brief Spatially subsample the point cloud
     * @param voxel_size Voxel size for subsampling
     * @return New PointCloud with subsampled points
     */
    std::shared_ptr<PointCloud> voxelDownsample(double voxel_size);
    
    /**
     * @brief Compute normal vectors for the point cloud
     * @param radius Radius for normal estimation
     * @return Array of normal vectors (x, y, z) for each point
     */
    std::vector<std::array<float, 3>> computeNormals(double radius = 1.0);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace geospatial

#endif // GEOSPATIAL_POINT_CLOUD_HPP