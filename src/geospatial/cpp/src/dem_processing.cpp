/**
 * @file dem_processing.cpp
 * @brief Implementation of Digital Elevation Model (DEM) processing algorithms
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include "geospatial/dem_processing.hpp"
#include "geospatial/gpu_adaptability.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <queue>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <string>
#include <functional>

// GDAL includes
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <gdalwarper.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

namespace geospatial {

// Forward declarations of CUDA kernels
namespace kernels {
#ifdef WITH_CUDA
extern "C" {
    void launchViewshedKernel(
        const float* d_dem,
        float* d_viewshed,
        int width,
        int height,
        float observer_x,
        float observer_y,
        float observer_height,
        float max_radius,
        float cell_size,
        float curvature_coeff,
        cudaStream_t stream = 0);
        
    void launchTerrainDerivativesKernel(
        const float* d_dem,
        float* d_slope,
        float* d_aspect,
        float* d_curvature,
        int width,
        int height,
        float cell_size,
        float z_factor,
        cudaStream_t stream = 0);
        
    void launchFillSinksKernel(
        const float* d_dem,
        float* d_filled,
        int width,
        int height,
        float z_limit,
        cudaStream_t stream = 0);
}
#endif
} // namespace kernels

// Implementation of GeoTransform methods
std::tuple<double, double> GeoTransform::pixelToGeo(int pixel_x, int pixel_y) const {
    double geo_x = parameters[0] + pixel_x * parameters[1] + pixel_y * parameters[2];
    double geo_y = parameters[3] + pixel_x * parameters[4] + pixel_y * parameters[5];
    return std::make_tuple(geo_x, geo_y);
}

std::tuple<int, int> GeoTransform::geoToPixel(double geo_x, double geo_y) const {
    // For simplicity, assume no rotation (parameters[2] and parameters[4] are 0)
    if (parameters[2] != 0 || parameters[4] != 0) {
        throw std::runtime_error("Rotated geotransforms are not currently supported");
    }
    
    int pixel_x = static_cast<int>((geo_x - parameters[0]) / parameters[1] + 0.5);
    int pixel_y = static_cast<int>((geo_y - parameters[3]) / parameters[5] + 0.5);
    return std::make_tuple(pixel_x, pixel_y);
}

// Implementation class for DEMProcessor
class DEMProcessor::Impl {
public:
    Impl(const std::string& dem_file, int device_id)
        : device_id_(device_id),
          dem_file_(dem_file),
          width_(0),
          height_(0),
          has_cuda_(false),
          h_dem_(nullptr),
          d_dem_(nullptr),
          nodata_value_(-9999.0f)
    {
        // Initialize GDAL
        GDALAllRegister();
        
        // Open the DEM file
        dataset_ = std::unique_ptr<GDALDataset>((GDALDataset*)GDALOpen(dem_file.c_str(), GA_ReadOnly));
        if (dataset_ == nullptr) {
            throw std::runtime_error("Failed to open DEM file: " + dem_file);
        }
        
        // Get DEM dimensions
        width_ = dataset_->GetRasterXSize();
        height_ = dataset_->GetRasterYSize();
        
        // Get geotransform
        dataset_->GetGeoTransform(geo_transform_.parameters);
        
        // Get nodata value
        GDALRasterBand* band = dataset_->GetRasterBand(1);
        int has_nodata = 0;
        nodata_value_ = static_cast<float>(band->GetNoDataValue(&has_nodata));
        if (!has_nodata) {
            nodata_value_ = -9999.0f;
        }
        
        // Calculate cell size in meters
        cell_size_ = std::abs(geo_transform_.parameters[1]);
        
        // Check if we need to reproject to a projected CRS
        OGRSpatialReference src_srs;
        src_srs.importFromWkt(dataset_->GetProjectionRef());
        
        is_geographic_ = !src_srs.IsProjected();
        if (is_geographic_) {
            // Geographic coordinates, convert cell size to approximate meters
            double center_lat = geo_transform_.parameters[3] + height_ / 2.0 * geo_transform_.parameters[5];
            cell_size_ = std::abs(geo_transform_.parameters[1]) * 111320.0 * std::cos(center_lat * M_PI / 180.0);
        }
        
        // Check CUDA availability
#ifdef WITH_CUDA
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error == cudaSuccess && device_count > 0) {
            if (device_id_ >= device_count) {
                std::cerr << "Warning: Device ID " << device_id_ << " is not available. Using device 0." << std::endl;
                device_id_ = 0;
            }
            
            cudaSetDevice(device_id_);
            has_cuda_ = true;
            
            // Get device capabilities
            DeviceCapabilities caps = DeviceAdaptor::getDeviceCapabilities(device_id_);
            std::cout << "Using GPU: " << caps.name << " (SM " << caps.compute_capability << ")" << std::endl;
            
            // Get optimal kernel configuration
            std::tie(block_dim_x_, block_dim_y_, tile_size_) = 
                DeviceAdaptor::getOptimalDEMKernelConfig(width_, height_, device_id_);
                
            std::cout << "Optimal DEM kernel config: " 
                      << block_dim_x_ << "x" << block_dim_y_ << " blocks, " 
                      << tile_size_ << " tile size" << std::endl;
        } else {
            std::cout << "CUDA not available, using CPU implementation." << std::endl;
            has_cuda_ = false;
        }
#endif
        
        // Load DEM data
        loadDEM();
    }
    
    ~Impl() {
        // Free DEM data
        if (h_dem_) {
            delete[] h_dem_;
            h_dem_ = nullptr;
        }
        
#ifdef WITH_CUDA
        if (d_dem_ && has_cuda_) {
            cudaFree(d_dem_);
            d_dem_ = nullptr;
        }
#endif
    }
    
    std::tuple<int, int> getDimensions() const {
        return std::make_tuple(width_, height_);
    }
    
    GeoTransform getGeoTransform() const {
        return geo_transform_;
    }
    
    std::shared_ptr<float> computeViewshed(
        const std::tuple<double, double>& observer_point,
        float observer_height,
        float radius)
    {
        // Convert observer point from geographic to pixel coordinates
        int observer_pixel_x, observer_pixel_y;
        std::tie(observer_pixel_x, observer_pixel_y) = geo_transform_.geoToPixel(
            std::get<0>(observer_point), std::get<1>(observer_point));
        
        // Check if observer is within the DEM
        if (observer_pixel_x < 0 || observer_pixel_x >= width_ ||
            observer_pixel_y < 0 || observer_pixel_y >= height_) {
            throw std::runtime_error("Observer location is outside the DEM bounds");
        }
        
        // Convert radius from meters to pixels if specified
        float pixel_radius = 0.0f;
        if (radius > 0.0f) {
            pixel_radius = radius / cell_size_;
        }
        
        // Earth curvature coefficient (approximation)
        float curvature_coeff = 0.87f / (2.0f * 6371000.0f); // Earth radius in meters
        
        // Allocate output array
        float* h_viewshed = new float[width_ * height_];
        std::fill(h_viewshed, h_viewshed + width_ * height_, 0.0f);
        
        // Compute viewshed
        if (has_cuda_) {
#ifdef WITH_CUDA
            // Allocate device memory for viewshed
            float* d_viewshed = nullptr;
            cudaMalloc(&d_viewshed, width_ * height_ * sizeof(float));
            cudaMemset(d_viewshed, 0, width_ * height_ * sizeof(float));
            
            // Launch kernel
            kernels::launchViewshedKernel(
                d_dem_,
                d_viewshed,
                width_,
                height_,
                static_cast<float>(observer_pixel_x),
                static_cast<float>(observer_pixel_y),
                observer_height,
                pixel_radius,
                static_cast<float>(cell_size_),
                curvature_coeff
            );
            
            // Copy result back to host
            cudaMemcpy(h_viewshed, d_viewshed, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Free device memory
            cudaFree(d_viewshed);
#endif
        } else {
            // CPU implementation of viewshed analysis
            computeViewshedCPU(
                h_dem_,
                h_viewshed,
                width_,
                height_,
                observer_pixel_x,
                observer_pixel_y,
                observer_height,
                pixel_radius,
                static_cast<float>(cell_size_),
                curvature_coeff
            );
        }
        
        // Return result as shared pointer
        return std::shared_ptr<float>(h_viewshed, std::default_delete<float[]>());
    }
    
    std::tuple<std::shared_ptr<float>, std::shared_ptr<float>, std::shared_ptr<float>>
    computeTerrainDerivatives(float z_factor)
    {
        // Allocate output arrays
        float* h_slope = new float[width_ * height_];
        float* h_aspect = new float[width_ * height_];
        float* h_curvature = new float[width_ * height_];
        
        std::fill(h_slope, h_slope + width_ * height_, 0.0f);
        std::fill(h_aspect, h_aspect + width_ * height_, 0.0f);
        std::fill(h_curvature, h_curvature + width_ * height_, 0.0f);
        
        // Compute terrain derivatives
        if (has_cuda_) {
#ifdef WITH_CUDA
            // Allocate device memory
            float* d_slope = nullptr;
            float* d_aspect = nullptr;
            float* d_curvature = nullptr;
            
            cudaMalloc(&d_slope, width_ * height_ * sizeof(float));
            cudaMalloc(&d_aspect, width_ * height_ * sizeof(float));
            cudaMalloc(&d_curvature, width_ * height_ * sizeof(float));
            
            // Launch kernel
            kernels::launchTerrainDerivativesKernel(
                d_dem_,
                d_slope,
                d_aspect,
                d_curvature,
                width_,
                height_,
                static_cast<float>(cell_size_),
                z_factor
            );
            
            // Copy results back to host
            cudaMemcpy(h_slope, d_slope, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_aspect, d_aspect, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_curvature, d_curvature, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Free device memory
            cudaFree(d_slope);
            cudaFree(d_aspect);
            cudaFree(d_curvature);
#endif
        } else {
            // CPU implementation of terrain derivatives
            computeTerrainDerivativesCPU(
                h_dem_,
                h_slope,
                h_aspect,
                h_curvature,
                width_,
                height_,
                static_cast<float>(cell_size_),
                z_factor
            );
        }
        
        // Return results as shared pointers
        return std::make_tuple(
            std::shared_ptr<float>(h_slope, std::default_delete<float[]>()),
            std::shared_ptr<float>(h_aspect, std::default_delete<float[]>()),
            std::shared_ptr<float>(h_curvature, std::default_delete<float[]>())
        );
    }
    
    std::tuple<std::shared_ptr<uint8_t>, std::shared_ptr<float>>
    computeHydrologicalFeatures()
    {
        // Allocate output arrays
        uint8_t* h_flow_dir = new uint8_t[width_ * height_];
        float* h_flow_acc = new float[width_ * height_];
        
        std::fill(h_flow_dir, h_flow_dir + width_ * height_, 0);
        std::fill(h_flow_acc, h_flow_acc + width_ * height_, 0.0f);
        
        // Compute hydrological features (CPU implementation for now)
        computeHydrologicalFeaturesCPU(
            h_dem_,
            h_flow_dir,
            h_flow_acc,
            width_,
            height_,
            static_cast<float>(cell_size_)
        );
        
        // Return results as shared pointers
        return std::make_tuple(
            std::shared_ptr<uint8_t>(h_flow_dir, std::default_delete<uint8_t[]>()),
            std::shared_ptr<float>(h_flow_acc, std::default_delete<float[]>())
        );
    }
    
    std::vector<Point3D> computeLeastCostPath(
        const std::tuple<double, double>& start_point,
        const std::tuple<double, double>& end_point,
        std::function<float(float, float, float, float)> cost_function)
    {
        // Convert points from geographic to pixel coordinates
        int start_pixel_x, start_pixel_y;
        int end_pixel_x, end_pixel_y;
        
        std::tie(start_pixel_x, start_pixel_y) = geo_transform_.geoToPixel(
            std::get<0>(start_point), std::get<1>(start_point));
            
        std::tie(end_pixel_x, end_pixel_y) = geo_transform_.geoToPixel(
            std::get<0>(end_point), std::get<1>(end_point));
        
        // Check if points are within the DEM
        if (start_pixel_x < 0 || start_pixel_x >= width_ ||
            start_pixel_y < 0 || start_pixel_y >= height_ ||
            end_pixel_x < 0 || end_pixel_x >= width_ ||
            end_pixel_y < 0 || end_pixel_y >= height_) {
            throw std::runtime_error("Start or end point is outside the DEM bounds");
        }
        
        // Use a default cost function if none is provided
        std::function<float(float, float, float, float)> cost_func;
        if (cost_function) {
            cost_func = cost_function;
        } else {
            // Default cost function: weighted sum of elevation difference and distance
            cost_func = [](float elev1, float elev2, float slope, float aspect) {
                float elev_diff = std::abs(elev2 - elev1);
                return 1.0f + 5.0f * elev_diff + 10.0f * slope;
            };
        }
        
        // Compute terrain derivatives for slope and aspect
        float* h_slope = new float[width_ * height_];
        float* h_aspect = new float[width_ * height_];
        float* h_curvature = new float[width_ * height_]; // Not used
        
        computeTerrainDerivativesCPU(
            h_dem_,
            h_slope,
            h_aspect,
            h_curvature,
            width_,
            height_,
            static_cast<float>(cell_size_),
            1.0f
        );
        
        // Compute least-cost path using A* algorithm
        std::vector<std::tuple<int, int>> path = computeLeastCostPathCPU(
            h_dem_,
            h_slope,
            h_aspect,
            width_,
            height_,
            start_pixel_x,
            start_pixel_y,
            end_pixel_x,
            end_pixel_y,
            cost_func
        );
        
        // Clean up
        delete[] h_slope;
        delete[] h_aspect;
        delete[] h_curvature;
        
        // Convert pixel path to geographic coordinates
        std::vector<Point3D> geo_path;
        geo_path.reserve(path.size());
        
        for (const auto& point : path) {
            int px = std::get<0>(point);
            int py = std::get<1>(point);
            
            double geo_x, geo_y;
            std::tie(geo_x, geo_y) = geo_transform_.pixelToGeo(px, py);
            
            // Get elevation at this point
            float elevation = h_dem_[py * width_ + px];
            if (elevation == nodata_value_) {
                elevation = 0.0f;
            }
            
            geo_path.push_back({geo_x, geo_y, static_cast<double>(elevation)});
        }
        
        return geo_path;
    }
    
    std::shared_ptr<float> fillSinks(float z_limit)
    {
        // Allocate output array
        float* h_filled = new float[width_ * height_];
        
        // Copy DEM to output array
        std::copy(h_dem_, h_dem_ + width_ * height_, h_filled);
        
        if (has_cuda_) {
#ifdef WITH_CUDA
            // Allocate device memory
            float* d_filled = nullptr;
            cudaMalloc(&d_filled, width_ * height_ * sizeof(float));
            
            // Copy input DEM to device
            cudaMemcpy(d_filled, h_dem_, width_ * height_ * sizeof(float), cudaMemcpyHostToDevice);
            
            // Launch kernel
            kernels::launchFillSinksKernel(
                d_dem_,
                d_filled,
                width_,
                height_,
                z_limit
            );
            
            // Copy result back to host
            cudaMemcpy(h_filled, d_filled, width_ * height_ * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Free device memory
            cudaFree(d_filled);
#endif
        } else {
            // CPU implementation of sink filling
            fillSinksCPU(
                h_dem_,
                h_filled,
                width_,
                height_,
                z_limit
            );
        }
        
        // Return result as shared pointer
        return std::shared_ptr<float>(h_filled, std::default_delete<float[]>());
    }
    
    DEMStatistics computeStatistics()
    {
        DEMStatistics stats;
        
        // Initialize with default values
        stats.min_elevation = std::numeric_limits<double>::max();
        stats.max_elevation = -std::numeric_limits<double>::max();
        stats.mean_elevation = 0.0;
        stats.std_dev = 0.0;
        stats.mean_slope = 0.0;
        stats.mean_aspect = 0.0;
        stats.ruggedness_index = 0.0;
        stats.hypsometric_integral = 0.0;
        
        // Compute basic statistics
        double sum = 0.0;
        double sum_squared = 0.0;
        int valid_count = 0;
        
        for (int i = 0; i < width_ * height_; i++) {
            float elevation = h_dem_[i];
            if (elevation != nodata_value_) {
                stats.min_elevation = std::min(stats.min_elevation, static_cast<double>(elevation));
                stats.max_elevation = std::max(stats.max_elevation, static_cast<double>(elevation));
                sum += elevation;
                sum_squared += elevation * elevation;
                valid_count++;
            }
        }
        
        if (valid_count > 0) {
            stats.mean_elevation = sum / valid_count;
            stats.std_dev = std::sqrt(sum_squared / valid_count - stats.mean_elevation * stats.mean_elevation);
        }
        
        // Compute terrain metrics
        float* h_slope = new float[width_ * height_];
        float* h_aspect = new float[width_ * height_];
        float* h_curvature = new float[width_ * height_];
        
        computeTerrainDerivativesCPU(
            h_dem_,
            h_slope,
            h_aspect,
            h_curvature,
            width_,
            height_,
            static_cast<float>(cell_size_),
            1.0f
        );
        
        double slope_sum = 0.0;
        double aspect_sum = 0.0;
        double ruggedness_sum = 0.0;
        
        for (int i = 0; i < width_ * height_; i++) {
            if (h_dem_[i] != nodata_value_ && h_slope[i] != nodata_value_) {
                slope_sum += h_slope[i];
                aspect_sum += h_aspect[i];
                ruggedness_sum += h_slope[i] / 100.0; // Simplified ruggedness index
            }
        }
        
        if (valid_count > 0) {
            stats.mean_slope = slope_sum / valid_count;
            stats.mean_aspect = aspect_sum / valid_count;
            stats.ruggedness_index = ruggedness_sum / valid_count;
            
            // Compute hypsometric integral (simplified)
            if (stats.max_elevation > stats.min_elevation) {
                stats.hypsometric_integral = (stats.mean_elevation - stats.min_elevation) / 
                                           (stats.max_elevation - stats.min_elevation);
            } else {
                stats.hypsometric_integral = 0.0;
            }
        }
        
        // Clean up
        delete[] h_slope;
        delete[] h_aspect;
        delete[] h_curvature;
        
        return stats;
    }
    
    bool saveResult(std::shared_ptr<void> data, const std::string& output_file, int data_type)
    {
        // Create output dataset
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        if (driver == nullptr) {
            std::cerr << "Error: Failed to get GTiff driver" << std::endl;
            return false;
        }
        
        // Set creation options
        char** options = nullptr;
        options = CSLSetNameValue(options, "COMPRESS", "DEFLATE");
        options = CSLSetNameValue(options, "TILED", "YES");
        
        // Create output dataset
        GDALDataset* output_dataset = driver->Create(
            output_file.c_str(),
            width_,
            height_,
            1,  // Number of bands
            static_cast<GDALDataType>(data_type),
            options
        );
        
        if (output_dataset == nullptr) {
            std::cerr << "Error: Failed to create output dataset: " << output_file << std::endl;
            CSLDestroy(options);
            return false;
        }
        
        // Set geotransform and projection
        output_dataset->SetGeoTransform(geo_transform_.parameters);
        output_dataset->SetProjection(dataset_->GetProjectionRef());
        
        // Write data
        GDALRasterBand* band = output_dataset->GetRasterBand(1);
        band->SetNoDataValue(nodata_value_);
        
        CPLErr err;
        switch (data_type) {
            case GDT_Byte:
                err = band->RasterIO(GF_Write, 0, 0, width_, height_,
                                  static_cast<uint8_t*>(data.get()), width_, height_,
                                  GDT_Byte, 0, 0);
                break;
            case GDT_UInt16:
                err = band->RasterIO(GF_Write, 0, 0, width_, height_,
                                  static_cast<uint16_t*>(data.get()), width_, height_,
                                  GDT_UInt16, 0, 0);
                break;
            case GDT_Int16:
                err = band->RasterIO(GF_Write, 0, 0, width_, height_,
                                  static_cast<int16_t*>(data.get()), width_, height_,
                                  GDT_Int16, 0, 0);
                break;
            case GDT_UInt32:
                err = band->RasterIO(GF_Write, 0, 0, width_, height_,
                                  static_cast<uint32_t*>(data.get()), width_, height_,
                                  GDT_UInt32, 0, 0);
                break;
            case GDT_Int32:
                err = band->RasterIO(GF_Write, 0, 0, width_, height_,
                                  static_cast<int32_t*>(data.get()), width_, height_,
                                  GDT_Int32, 0, 0);
                break;
            case GDT_Float32:
                err = band->RasterIO(GF_Write, 0, 0, width_, height_,
                                  static_cast<float*>(data.get()), width_, height_,
                                  GDT_Float32, 0, 0);
                break;
            case GDT_Float64:
                err = band->RasterIO(GF_Write, 0, 0, width_, height_,
                                  static_cast<double*>(data.get()), width_, height_,
                                  GDT_Float64, 0, 0);
                break;
            default:
                std::cerr << "Error: Unsupported data type: " << data_type << std::endl;
                GDALClose(output_dataset);
                CSLDestroy(options);
                return false;
        }
        
        if (err != CE_None) {
            std::cerr << "Error: Failed to write data to output dataset" << std::endl;
            GDALClose(output_dataset);
            CSLDestroy(options);
            return false;
        }
        
        // Clean up
        GDALClose(output_dataset);
        CSLDestroy(options);
        
        return true;
    }
    
    std::shared_ptr<float> resample(float target_resolution, int resampling_method)
    {
        // Calculate new dimensions
        double current_resolution = std::abs(geo_transform_.parameters[1]);
        double ratio = current_resolution / target_resolution;
        
        int new_width = static_cast<int>(width_ * ratio + 0.5);
        int new_height = static_cast<int>(height_ * ratio + 0.5);
        
        // Create a memory dataset for resampling
        GDALDriver* mem_driver = GetGDALDriverManager()->GetDriverByName("MEM");
        if (mem_driver == nullptr) {
            throw std::runtime_error("Failed to get MEM driver");
        }
        
        // Create warping options
        GDALWarpOptions* warp_options = GDALCreateWarpOptions();
        warp_options->eResampleAlg = static_cast<GDALResampleAlg>(resampling_method);
        warp_options->hSrcDS = dataset_.get();
        
        // Create output dataset
        GDALDataset* dst_ds = mem_driver->Create("", new_width, new_height, 1, GDT_Float32, nullptr);
        if (dst_ds == nullptr) {
            GDALDestroyWarpOptions(warp_options);
            throw std::runtime_error("Failed to create output dataset for resampling");
        }
        
        // Calculate new geotransform
        double new_geo_transform[6];
        std::copy(geo_transform_.parameters, geo_transform_.parameters + 6, new_geo_transform);
        new_geo_transform[1] = target_resolution;
        new_geo_transform[5] = -target_resolution;
        
        dst_ds->SetGeoTransform(new_geo_transform);
        dst_ds->SetProjection(dataset_->GetProjectionRef());
        
        // Set output band nodata value
        GDALRasterBand* dst_band = dst_ds->GetRasterBand(1);
        dst_band->SetNoDataValue(nodata_value_);
        
        // Perform warping
        GDALWarpOperation warp_op;
        warp_options->hDstDS = dst_ds;
        if (warp_op.Initialize(warp_options) != CE_None) {
            GDALDestroyWarpOptions(warp_options);
            GDALClose(dst_ds);
            throw std::runtime_error("Failed to initialize warping operation");
        }
        
        if (warp_op.ChunkAndWarpImage(0, 0, new_width, new_height) != CE_None) {
            GDALDestroyWarpOptions(warp_options);
            GDALClose(dst_ds);
            throw std::runtime_error("Failed to warp image");
        }
        
        // Read resampled data
        float* resampled_data = new float[new_width * new_height];
        if (dst_band->RasterIO(GF_Read, 0, 0, new_width, new_height,
                             resampled_data, new_width, new_height,
                             GDT_Float32, 0, 0) != CE_None) {
            delete[] resampled_data;
            GDALDestroyWarpOptions(warp_options);
            GDALClose(dst_ds);
            throw std::runtime_error("Failed to read resampled data");
        }
        
        // Clean up
        GDALDestroyWarpOptions(warp_options);
        GDALClose(dst_ds);
        
        // Return resampled data
        return std::shared_ptr<float>(resampled_data, std::default_delete<float[]>());
    }
    
    int getDeviceId() const {
        return device_id_;
    }
    
private:
    // Load DEM data from file
    void loadDEM() {
        // Allocate host memory
        h_dem_ = new float[width_ * height_];
        
        // Read DEM data
        GDALRasterBand* band = dataset_->GetRasterBand(1);
        CPLErr err = band->RasterIO(GF_Read, 0, 0, width_, height_,
                                  h_dem_, width_, height_,
                                  GDT_Float32, 0, 0);
        
        if (err != CE_None) {
            delete[] h_dem_;
            h_dem_ = nullptr;
            throw std::runtime_error("Failed to read DEM data");
        }
        
        // Copy data to device if CUDA is available
#ifdef WITH_CUDA
        if (has_cuda_) {
            cudaMalloc(&d_dem_, width_ * height_ * sizeof(float));
            cudaMemcpy(d_dem_, h_dem_, width_ * height_ * sizeof(float), cudaMemcpyHostToDevice);
        }
#endif
    }
    
    // CPU implementation of viewshed algorithm
    void computeViewshedCPU(
        const float* dem,
        float* viewshed,
        int width,
        int height,
        int observer_x,
        int observer_y,
        float observer_height,
        float max_radius,
        float cell_size,
        float curvature_coeff)
    {
        float observer_elevation = dem[observer_y * width + observer_x] + observer_height;
        
        // Process each cell
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int idx = row * width + col;
                
                // Skip nodata cells
                if (dem[idx] == nodata_value_) {
                    viewshed[idx] = 0.0f;
                    continue;
                }
                
                // Skip observer cell (always visible)
                if (col == observer_x && row == observer_y) {
                    viewshed[idx] = 1.0f;
                    continue;
                }
                
                // Calculate distance from observer to cell
                float dx = static_cast<float>(col - observer_x);
                float dy = static_cast<float>(row - observer_y);
                float distance = std::sqrt(dx * dx + dy * dy) * cell_size;
                
                // Apply distance limit if specified
                if (max_radius > 0 && distance > max_radius) {
                    viewshed[idx] = 0.0f;
                    continue;
                }
                
                // For cells very close to the observer, mark as visible
                if (distance < cell_size * 0.5f) {
                    viewshed[idx] = 1.0f;
                    continue;
                }
                
                // Line of sight algorithm
                float max_angle = -std::numeric_limits<float>::infinity();
                
                // Number of samples along the line
                int num_samples = std::ceil(distance / (cell_size * 0.5f));
                
                for (int i = 1; i < num_samples; i++) {
                    float sample_fraction = static_cast<float>(i) / static_cast<float>(num_samples);
                    float sample_x = observer_x + sample_fraction * dx;
                    float sample_y = observer_y + sample_fraction * dy;
                    float sample_distance = sample_fraction * distance;
                    
                    // Calculate earth curvature adjustment
                    float curvature_adjustment = curvature_coeff * sample_distance * sample_distance;
                    
                    // Bilinear interpolation for elevation
                    int x0 = static_cast<int>(sample_x);
                    int y0 = static_cast<int>(sample_y);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;
                    
                    // Bounds check
                    if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height) {
                        continue;
                    }
                    
                    // Interpolation weights
                    float wx = sample_x - x0;
                    float wy = sample_y - y0;
                    
                    // Get elevations
                    float e00 = dem[y0 * width + x0];
                    float e01 = dem[y0 * width + x1];
                    float e10 = dem[y1 * width + x0];
                    float e11 = dem[y1 * width + x1];
                    
                    // Skip if any value is nodata
                    if (e00 == nodata_value_ || e01 == nodata_value_ ||
                        e10 == nodata_value_ || e11 == nodata_value_) {
                        continue;
                    }
                    
                    // Bilinear interpolation
                    float sample_elevation = 
                        (1.0f - wx) * (1.0f - wy) * e00 +
                        wx * (1.0f - wy) * e01 +
                        (1.0f - wx) * wy * e10 +
                        wx * wy * e11;
                    
                    // Adjust for Earth curvature
                    sample_elevation -= curvature_adjustment;
                    
                    // Calculate angle to sample
                    float h_diff = sample_elevation - observer_elevation;
                    float angle = std::atan2(h_diff, sample_distance);
                    
                    // Update maximum angle
                    max_angle = std::max(max_angle, angle);
                }
                
                // Check if target cell is visible
                float target_elevation = dem[idx];
                float target_distance = distance;
                float target_curvature = curvature_coeff * target_distance * target_distance;
                float adjusted_target_elevation = target_elevation - target_curvature;
                float target_angle = std::atan2(adjusted_target_elevation - observer_elevation, target_distance);
                
                viewshed[idx] = (target_angle >= max_angle) ? 1.0f : 0.0f;
            }
        }
    }
    
    // CPU implementation of terrain derivatives
    void computeTerrainDerivativesCPU(
        const float* dem,
        float* slope,
        float* aspect,
        float* curvature,
        int width,
        int height,
        float cell_size,
        float z_factor)
    {
        // Process each cell (skip borders)
        for (int row = 1; row < height - 1; row++) {
            for (int col = 1; col < width - 1; col++) {
                int idx = row * width + col;
                
                // Get 3x3 neighborhood
                float z11 = dem[(row-1)*width + (col-1)] * z_factor;
                float z12 = dem[(row-1)*width + col] * z_factor;
                float z13 = dem[(row-1)*width + (col+1)] * z_factor;
                float z21 = dem[row*width + (col-1)] * z_factor;
                float z22 = dem[idx] * z_factor;
                float z23 = dem[row*width + (col+1)] * z_factor;
                float z31 = dem[(row+1)*width + (col-1)] * z_factor;
                float z32 = dem[(row+1)*width + col] * z_factor;
                float z33 = dem[(row+1)*width + (col+1)] * z_factor;
                
                // Check for nodata
                if (z11 == nodata_value_ * z_factor || z12 == nodata_value_ * z_factor || z13 == nodata_value_ * z_factor ||
                    z21 == nodata_value_ * z_factor || z22 == nodata_value_ * z_factor || z23 == nodata_value_ * z_factor ||
                    z31 == nodata_value_ * z_factor || z32 == nodata_value_ * z_factor || z33 == nodata_value_ * z_factor) {
                    
                    slope[idx] = nodata_value_;
                    aspect[idx] = nodata_value_;
                    curvature[idx] = nodata_value_;
                    continue;
                }
                
                // Calculate gradients (Horn's formula)
                float dx = ((z13 - z11) + 2.0f * (z23 - z21) + (z33 - z31)) / (8.0f * cell_size);
                float dy = ((z31 - z11) + 2.0f * (z32 - z12) + (z33 - z13)) / (8.0f * cell_size);
                
                // Calculate slope (in degrees)
                float slope_rad = std::atan(std::sqrt(dx*dx + dy*dy));
                slope[idx] = slope_rad * 180.0f / M_PI;
                
                // Calculate aspect (in degrees from north)
                float aspect_rad;
                if (dx == 0.0f) {
                    if (dy > 0.0f) {
                        aspect_rad = M_PI / 2.0f;
                    } else if (dy < 0.0f) {
                        aspect_rad = 3.0f * M_PI / 2.0f;
                    } else {
                        aspect_rad = 0.0f;
                    }
                } else {
                    aspect_rad = std::atan2(dy, dx);
                    
                    if (dx < 0.0f) {
                        aspect_rad += M_PI;
                    } else if (dy < 0.0f) {
                        aspect_rad += 2.0f * M_PI;
                    }
                }
                
                // Convert to degrees and adjust to start from north
                aspect[idx] = std::fmod(90.0f - aspect_rad * 180.0f / M_PI + 360.0f, 360.0f);
                
                // Calculate curvature (second derivatives)
                float d2x = (z13 - 2.0f * z22 + z31) / (cell_size * cell_size);
                float d2y = (z31 - 2.0f * z22 + z13) / (cell_size * cell_size);
                float dxy = ((z33 - z31 - z13 + z11) / 4.0f) / (cell_size * cell_size);
                
                // Total curvature (negative for convergent, positive for divergent)
                curvature[idx] = -1.0f * (d2x + d2y);
            }
        }
        
        // Handle borders (set to nodata)
        for (int col = 0; col < width; col++) {
            slope[col] = nodata_value_;
            aspect[col] = nodata_value_;
            curvature[col] = nodata_value_;
            
            slope[(height-1)*width + col] = nodata_value_;
            aspect[(height-1)*width + col] = nodata_value_;
            curvature[(height-1)*width + col] = nodata_value_;
        }
        
        for (int row = 0; row < height; row++) {
            slope[row*width] = nodata_value_;
            aspect[row*width] = nodata_value_;
            curvature[row*width] = nodata_value_;
            
            slope[row*width + width-1] = nodata_value_;
            aspect[row*width + width-1] = nodata_value_;
            curvature[row*width + width-1] = nodata_value_;
        }
    }
    
    // CPU implementation of hydrological features
    void computeHydrologicalFeaturesCPU(
        const float* dem,
        uint8_t* flow_dir,
        float* flow_acc,
        int width,
        int height,
        float cell_size)
    {
        // D8 flow direction values (ESRI style)
        // 32 64 128
        // 16  0   1
        //  8  4   2
        const uint8_t d8_values[8] = {1, 2, 4, 8, 16, 32, 64, 128};
        const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        
        // Initialize flow directions
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int idx = row * width + col;
                
                // Skip nodata cells
                if (dem[idx] == nodata_value_) {
                    flow_dir[idx] = 0;
                    flow_acc[idx] = nodata_value_;
                    continue;
                }
                
                // Find steepest downslope direction
                float max_slope = -std::numeric_limits<float>::infinity();
                int best_dir = -1;
                
                for (int dir = 0; dir < 8; dir++) {
                    int nx = col + dx[dir];
                    int ny = row + dy[dir];
                    
                    // Skip out-of-bounds neighbors
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                        continue;
                    }
                    
                    int nidx = ny * width + nx;
                    
                    // Skip nodata neighbors
                    if (dem[nidx] == nodata_value_) {
                        continue;
                    }
                    
                    // Calculate slope
                    float distance = (dir % 2 == 0) ? cell_size : cell_size * 1.414f;
                    float slope = (dem[idx] - dem[nidx]) / distance;
                    
                    if (slope > max_slope) {
                        max_slope = slope;
                        best_dir = dir;
                    }
                }
                
                // Set flow direction (0 for local minimum/pit)
                flow_dir[idx] = (best_dir >= 0) ? d8_values[best_dir] : 0;
            }
        }
        
        // Compute flow accumulation using O(n log n) algorithm
        
        // First, build a priority queue of cells sorted by elevation
        std::vector<std::tuple<float, int, int>> cells;
        cells.reserve(width * height);
        
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                int idx = row * width + col;
                if (dem[idx] != nodata_value_) {
                    cells.emplace_back(dem[idx], col, row);
                }
            }
        }
        
        // Sort cells by elevation (highest first)
        std::sort(cells.begin(), cells.end(),
                 [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
        
        // Initialize flow accumulation
        for (int i = 0; i < width * height; i++) {
            flow_acc[i] = (dem[i] != nodata_value_) ? 1.0f : nodata_value_;
        }
        
        // Process cells from highest to lowest
        for (const auto& cell : cells) {
            int col = std::get<1>(cell);
            int row = std::get<2>(cell);
            int idx = row * width + col;
            
            // Skip cells with no flow direction (pits/sinks)
            if (flow_dir[idx] == 0) {
                continue;
            }
            
            // Find downslope neighbor
            int dir = -1;
            for (int i = 0; i < 8; i++) {
                if (flow_dir[idx] == d8_values[i]) {
                    dir = i;
                    break;
                }
            }
            
            if (dir >= 0) {
                int nx = col + dx[dir];
                int ny = row + dy[dir];
                
                // Ensure neighbor is within bounds
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    int nidx = ny * width + nx;
                    
                    // Add flow to downstream cell
                    if (flow_acc[nidx] != nodata_value_) {
                        flow_acc[nidx] += flow_acc[idx];
                    }
                }
            }
        }
    }
    
    // CPU implementation of least-cost path
    std::vector<std::tuple<int, int>> computeLeastCostPathCPU(
        const float* dem,
        const float* slope,
        const float* aspect,
        int width,
        int height,
        int start_x,
        int start_y,
        int end_x,
        int end_y,
        std::function<float(float, float, float, float)> cost_function)
    {
        // A* search algorithm
        struct Node {
            int x, y;
            float g_cost;  // Cost from start
            float h_cost;  // Heuristic (estimated cost to goal)
            float f_cost;  // Total cost (g + h)
            int parent_x, parent_y;
            
            Node(int x_, int y_, float g_, float h_, int px_, int py_)
                : x(x_), y(y_), g_cost(g_), h_cost(h_), f_cost(g_ + h_), parent_x(px_), parent_y(py_) {}
            
            bool operator<(const Node& other) const {
                return f_cost > other.f_cost;  // For min-heap
            }
        };
        
        // Direction vectors (8-connected neighborhood)
        const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        
        // Heuristic function (Euclidean distance)
        auto heuristic = [](int x1, int y1, int x2, int y2) {
            return std::sqrt(static_cast<float>((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)));
        };
        
        // Initialize open and closed lists
        std::priority_queue<Node> open_list;
        std::vector<std::vector<bool>> closed_list(height, std::vector<bool>(width, false));
        std::vector<std::vector<std::tuple<int, int>>> parent_map(height, std::vector<std::tuple<int, int>>(width, {-1, -1}));
        
        // Start with the initial node
        float h = heuristic(start_x, start_y, end_x, end_y);
        open_list.emplace(start_x, start_y, 0.0f, h, -1, -1);
        
        // A* search
        bool path_found = false;
        
        while (!open_list.empty()) {
            // Get node with lowest f_cost
            Node current = open_list.top();
            open_list.pop();
            
            int x = current.x;
            int y = current.y;
            
            // If already processed, skip
            if (closed_list[y][x]) {
                continue;
            }
            
            // Mark as processed
            closed_list[y][x] = true;
            parent_map[y][x] = {current.parent_x, current.parent_y};
            
            // Check if we reached the goal
            if (x == end_x && y == end_y) {
                path_found = true;
                break;
            }
            
            // Process neighbors
            for (int dir = 0; dir < 8; dir++) {
                int nx = x + dx[dir];
                int ny = y + dy[dir];
                
                // Check bounds
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }
                
                // Skip processed nodes
                if (closed_list[ny][nx]) {
                    continue;
                }
                
                // Skip nodata cells
                int idx = y * width + x;
                int nidx = ny * width + nx;
                if (dem[idx] == nodata_value_ || dem[nidx] == nodata_value_) {
                    continue;
                }
                
                // Calculate transition cost
                float transition_cost;
                if (cost_function) {
                    transition_cost = cost_function(
                        dem[idx], dem[nidx],
                        slope[nidx], aspect[nidx]);
                } else {
                    // Default cost: weighted sum of elevation difference and distance
                    float elev_diff = std::abs(dem[nidx] - dem[idx]);
                    float dist = (dir % 2 == 0) ? 1.0f : 1.414f;  // Diagonal is ~1.414 times longer
                    transition_cost = dist * (1.0f + 5.0f * elev_diff);
                }
                
                // Calculate new g_cost
                float new_g = current.g_cost + transition_cost;
                
                // Calculate heuristic
                float h = heuristic(nx, ny, end_x, end_y);
                
                // Add to open list
                open_list.emplace(nx, ny, new_g, h, x, y);
            }
        }
        
        // Reconstruct path if found
        std::vector<std::tuple<int, int>> path;
        
        if (path_found) {
            int x = end_x;
            int y = end_y;
            
            while (x != -1 && y != -1) {
                path.emplace_back(x, y);
                std::tie(x, y) = parent_map[y][x];
            }
            
            // Reverse path to get start-to-end order
            std::reverse(path.begin(), path.end());
        }
        
        return path;
    }
    
    // CPU implementation of sink filling
    void fillSinksCPU(
        const float* dem,
        float* filled,
        int width,
        int height,
        float z_limit)
    {
        // Direction vectors (8-connected neighborhood)
        const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
        const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};
        
        // Priority flood algorithm
        
        // First, initialize filled with DEM data
        std::copy(dem, dem + width * height, filled);
        
        // Create a priority queue for cells
        // We use a tuple of (elevation, x, y)
        std::vector<std::tuple<float, int, int>> boundary;
        std::vector<std::vector<bool>> processed(height, std::vector<bool>(width, false));
        
        // Initialize boundary with edge cells
        for (int col = 0; col < width; col++) {
            for (int row : {0, height - 1}) {
                int idx = row * width + col;
                if (dem[idx] != nodata_value_) {
                    boundary.emplace_back(dem[idx], col, row);
                    processed[row][col] = true;
                }
            }
        }
        
        for (int row = 1; row < height - 1; row++) {
            for (int col : {0, width - 1}) {
                int idx = row * width + col;
                if (dem[idx] != nodata_value_) {
                    boundary.emplace_back(dem[idx], col, row);
                    processed[row][col] = true;
                }
            }
        }
        
        // Sort boundary cells by elevation (lowest first)
        std::sort(boundary.begin(), boundary.end());
        
        // Process boundary cells
        while (!boundary.empty()) {
            // Get lowest elevation cell
            float elevation;
            int x, y;
            std::tie(elevation, x, y) = boundary.front();
            
            // Remove from boundary (use pop_heap + pop_back for efficiency in real implementation)
            boundary.erase(boundary.begin());
            
            // Process neighbors
            for (int dir = 0; dir < 8; dir++) {
                int nx = x + dx[dir];
                int ny = y + dy[dir];
                
                // Check bounds
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }
                
                // Skip processed cells
                if (processed[ny][nx]) {
                    continue;
                }
                
                // Skip nodata cells
                int nidx = ny * width + nx;
                if (dem[nidx] == nodata_value_) {
                    continue;
                }
                
                // Process cell
                processed[ny][nx] = true;
                
                // If neighbor is higher, add it to boundary with its original elevation
                if (dem[nidx] >= elevation) {
                    boundary.emplace_back(dem[nidx], nx, ny);
                    // Insert in proper order (would use push_heap in real implementation)
                    std::sort(boundary.begin(), boundary.end());
                }
                // Otherwise, fill the sink
                else {
                    float fill_value = elevation;
                    
                    // Apply z-limit if specified
                    if (z_limit < std::numeric_limits<float>::max()) {
                        float diff = elevation - dem[nidx];
                        if (diff > z_limit) {
                            fill_value = dem[nidx] + z_limit;
                        }
                    }
                    
                    filled[nidx] = fill_value;
                    boundary.emplace_back(fill_value, nx, ny);
                    // Insert in proper order
                    std::sort(boundary.begin(), boundary.end());
                }
            }
        }
    }
    
    // Member variables
    int device_id_;
    std::string dem_file_;
    std::unique_ptr<GDALDataset> dataset_;
    GeoTransform geo_transform_;
    int width_;
    int height_;
    double cell_size_;
    bool is_geographic_;
    bool has_cuda_;
    float* h_dem_;
    float* d_dem_;
    float nodata_value_;
    int block_dim_x_;
    int block_dim_y_;
    int tile_size_;
};

// Implementation of DEMProcessor methods using the pImpl idiom
DEMProcessor::DEMProcessor(const std::string& dem_file, int device_id)
    : pImpl(std::make_unique<Impl>(dem_file, device_id))
{
}

DEMProcessor::~DEMProcessor() = default;

std::tuple<int, int> DEMProcessor::getDimensions() const {
    return pImpl->getDimensions();
}

GeoTransform DEMProcessor::getGeoTransform() const {
    return pImpl->getGeoTransform();
}

std::shared_ptr<float> DEMProcessor::computeViewshed(
    const std::tuple<double, double>& observer_point,
    float observer_height,
    float radius)
{
    return pImpl->computeViewshed(observer_point, observer_height, radius);
}

std::tuple<std::shared_ptr<float>, std::shared_ptr<float>, std::shared_ptr<float>>
DEMProcessor::computeTerrainDerivatives(float z_factor)
{
    return pImpl->computeTerrainDerivatives(z_factor);
}

std::tuple<std::shared_ptr<uint8_t>, std::shared_ptr<float>>
DEMProcessor::computeHydrologicalFeatures()
{
    return pImpl->computeHydrologicalFeatures();
}

std::vector<Point3D> DEMProcessor::computeLeastCostPath(
    const std::tuple<double, double>& start_point,
    const std::tuple<double, double>& end_point,
    std::function<float(float, float, float, float)> cost_function)
{
    return pImpl->computeLeastCostPath(start_point, end_point, cost_function);
}

std::shared_ptr<float> DEMProcessor::fillSinks(float z_limit)
{
    return pImpl->fillSinks(z_limit);
}

DEMStatistics DEMProcessor::computeStatistics()
{
    return pImpl->computeStatistics();
}

bool DEMProcessor::saveResult(std::shared_ptr<void> data, const std::string& output_file, int data_type)
{
    return pImpl->saveResult(data, output_file, data_type);
}

std::shared_ptr<float> DEMProcessor::resample(float target_resolution, int resampling_method)
{
    return pImpl->resample(target_resolution, resampling_method);
}

} // namespace geospatial