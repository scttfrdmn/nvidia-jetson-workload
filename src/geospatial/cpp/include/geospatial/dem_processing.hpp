/**
 * @file dem_processing.hpp
 * @brief Digital Elevation Model (DEM) processing algorithms
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#ifndef GEOSPATIAL_DEM_PROCESSING_HPP
#define GEOSPATIAL_DEM_PROCESSING_HPP

#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include <functional>

namespace geospatial {

/**
 * @struct GeoTransform
 * @brief Geospatial transformation parameters (affine transformation)
 * 
 * Follows the GDAL GeoTransform convention:
 * - gt[0]: top-left x
 * - gt[1]: w-e pixel resolution
 * - gt[2]: row rotation (typically 0)
 * - gt[3]: top-left y
 * - gt[4]: column rotation (typically 0)
 * - gt[5]: n-s pixel resolution (negative)
 */
struct GeoTransform {
    double parameters[6];
    
    /**
     * @brief Convert pixel coordinates to geographic coordinates
     * @param pixel_x Pixel X coordinate (column)
     * @param pixel_y Pixel Y coordinate (row)
     * @return Tuple of (geo_x, geo_y) in geographic coordinates
     */
    std::tuple<double, double> pixelToGeo(int pixel_x, int pixel_y) const;
    
    /**
     * @brief Convert geographic coordinates to pixel coordinates
     * @param geo_x Geographic X coordinate
     * @param geo_y Geographic Y coordinate
     * @return Tuple of (pixel_x, pixel_y) in pixel coordinates
     */
    std::tuple<int, int> geoToPixel(double geo_x, double geo_y) const;
};

/**
 * @struct Point3D
 * @brief 3D point with x, y, z coordinates
 */
struct Point3D {
    double x;
    double y;
    double z;
};

/**
 * @struct DEMStatistics
 * @brief Statistics for a Digital Elevation Model
 */
struct DEMStatistics {
    double min_elevation;
    double max_elevation;
    double mean_elevation;
    double std_dev;
    
    // Additional terrain metrics
    double mean_slope;
    double mean_aspect;
    double ruggedness_index;
    double hypsometric_integral;
};

/**
 * @class DEMProcessor
 * @brief Processor for Digital Elevation Model operations
 */
class DEMProcessor {
public:
    /**
     * @brief Constructor
     * @param dem_file Path to DEM file (GeoTIFF format)
     * @param device_id CUDA device ID (default: 0)
     */
    DEMProcessor(const std::string& dem_file, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~DEMProcessor();
    
    /**
     * @brief Get DEM dimensions
     * @return Tuple of (width, height)
     */
    std::tuple<int, int> getDimensions() const;
    
    /**
     * @brief Get DEM geotransform parameters
     * @return GeoTransform structure
     */
    GeoTransform getGeoTransform() const;
    
    /**
     * @brief Compute viewshed from a specified observer point
     * @param observer_point Geographic coordinates of the observer (x, y)
     * @param observer_height Height of the observer above the terrain (meters)
     * @param radius Maximum viewshed radius (meters, 0 for unlimited)
     * @return Binary raster (1 = visible, 0 = not visible)
     */
    std::shared_ptr<float> computeViewshed(
        const std::tuple<double, double>& observer_point,
        float observer_height = 1.8f,
        float radius = 0.0f);
    
    /**
     * @brief Compute terrain derivatives (slope, aspect, curvature)
     * @param z_factor Vertical exaggeration factor
     * @return Tuple of (slope_raster, aspect_raster, curvature_raster)
     */
    std::tuple<std::shared_ptr<float>, std::shared_ptr<float>, std::shared_ptr<float>>
    computeTerrainDerivatives(float z_factor = 1.0f);
    
    /**
     * @brief Compute hydrological features (flow direction, flow accumulation)
     * @return Tuple of (flow_dir_raster, flow_acc_raster)
     */
    std::tuple<std::shared_ptr<uint8_t>, std::shared_ptr<float>>
    computeHydrologicalFeatures();
    
    /**
     * @brief Compute least-cost path between two points
     * @param start_point Start point in geographic coordinates (x, y)
     * @param end_point End point in geographic coordinates (x, y)
     * @param cost_function Function that computes transition cost between adjacent cells
     * @return Vector of points representing the path
     */
    std::vector<Point3D> computeLeastCostPath(
        const std::tuple<double, double>& start_point,
        const std::tuple<double, double>& end_point,
        std::function<float(float, float, float, float)> cost_function = nullptr);
    
    /**
     * @brief Fill DEM sinks/depressions for hydrological analysis
     * @param z_limit Maximum z-value difference for fill
     * @return Filled DEM raster
     */
    std::shared_ptr<float> fillSinks(float z_limit = std::numeric_limits<float>::max());
    
    /**
     * @brief Compute DEM statistics
     * @return DEMStatistics structure
     */
    DEMStatistics computeStatistics();
    
    /**
     * @brief Save result to GeoTIFF file
     * @param data Data to save
     * @param output_file Output file path
     * @param data_type GDAL data type (default: Float32)
     * @return True if successful, false otherwise
     */
    bool saveResult(std::shared_ptr<void> data, const std::string& output_file, 
                    int data_type = 6); // Default to GDT_Float32
    
    /**
     * @brief Resample DEM to different resolution
     * @param target_resolution Target resolution in units of the DEM's CRS
     * @param resampling_method Resampling method (0=nearest, 1=bilinear, 2=cubic, 3=cubicspline, 4=lanczos)
     * @return Resampled DEM
     */
    std::shared_ptr<float> resample(float target_resolution, int resampling_method = 1);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace geospatial

#endif // GEOSPATIAL_DEM_PROCESSING_HPP