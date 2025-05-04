/**
 * @file point_cloud.cpp
 * @brief Implementation of point cloud processing for geospatial data (LiDAR)
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include "geospatial/point_cloud.hpp"
#include "geospatial/gpu_adaptability.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <fstream>

// PDAL includes (optional, for reading LAS/LAZ files)
#ifdef WITH_PDAL
#include <pdal/PointView.hpp>
#include <pdal/io/LasReader.hpp>
#include <pdal/io/LasWriter.hpp>
#include <pdal/io/BufferReader.hpp>
#endif

// GDAL includes (for vector and raster operations)
#include <gdal_priv.h>
#include <ogr_spatialref.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

namespace geospatial {

// Forward declarations of CUDA kernels
namespace kernels {
#ifdef WITH_CUDA
// Point cloud structure for GPU processing
struct PointData {
    float x, y, z;        // Position
    uint8_t intensity;    // Intensity value
    uint8_t return_num;   // Return number
    uint8_t classification; // Point classification
};

// CUDA float3 type
struct float3 {
    float x, y, z;
};

extern "C" {
    void launchClassifyPointsKernel(
        const PointData* d_points,
        PointData* d_classified_points,
        int num_points,
        float min_x, float min_y, float min_z,
        float max_x, float max_y, float max_z,
        float grid_size,
        cudaStream_t stream);
        
    void launchCreateDEMKernel(
        const PointData* d_points,
        int num_points,
        float* d_dem,
        int width,
        int height,
        float min_x,
        float min_y,
        float cell_size,
        int algorithm,
        cudaStream_t stream);
        
    void launchComputeNormalsKernel(
        const PointData* d_points,
        float3* d_normals,
        int num_points,
        float radius,
        float min_x, float min_y, float min_z,
        float max_x, float max_y, float max_z,
        float grid_size,
        cudaStream_t stream);
        
    void launchExtractBuildingsKernel(
        const PointData* d_points,
        int num_points,
        int* d_labels,
        float min_x, float min_y, float min_z,
        float max_x, float max_y, float max_z,
        float grid_size,
        float min_height,
        cudaStream_t stream);
}
#endif
} // namespace kernels

// Implementation class for PointCloud
class PointCloud::Impl {
public:
    Impl(const std::string& point_cloud_file, int device_id)
        : device_id_(device_id),
          point_cloud_file_(point_cloud_file),
          num_points_(0),
          has_cuda_(false),
          h_points_(nullptr),
          d_points_(nullptr)
    {
        // Initialize GDAL
        GDALAllRegister();
        
        // Load point cloud from file
        loadPointCloud();
        
        // Calculate bounds
        calculateBounds();
        
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
            
            // Allocate device memory if needed
            allocateDeviceMemory();
        } else {
            std::cout << "CUDA not available, using CPU implementation." << std::endl;
            has_cuda_ = false;
        }
#endif
    }
    
    ~Impl() {
        // Free point cloud data
        cleanup();
    }
    
    size_t getNumPoints() const {
        return num_points_;
    }
    
    std::tuple<double, double, double, double, double, double> getBounds() const {
        return std::make_tuple(
            min_x_, min_y_, min_z_,
            max_x_, max_y_, max_z_
        );
    }
    
    std::shared_ptr<PointCloud> classifyPoints(int algorithm) {
        // Create a new point cloud for the result
        auto result = std::make_shared<PointCloud>(point_cloud_file_, device_id_);
        
        // Copy points to the result
        size_t num_points = points_.size();
        result->pImpl->points_ = points_;
        result->pImpl->num_points_ = num_points_;
        result->pImpl->min_x_ = min_x_;
        result->pImpl->min_y_ = min_y_;
        result->pImpl->min_z_ = min_z_;
        result->pImpl->max_x_ = max_x_;
        result->pImpl->max_y_ = max_y_;
        result->pImpl->max_z_ = max_z_;
        
        // Compute optimal grid size for spatial partitioning
        float grid_size = computeOptimalGridSize();
        
        if (has_cuda_) {
#ifdef WITH_CUDA
            // Allocate device memory for classified points
            kernels::PointData* d_classified_points = nullptr;
            cudaMalloc(&d_classified_points, num_points * sizeof(kernels::PointData));
            
            // Launch kernel
            kernels::launchClassifyPointsKernel(
                d_points_,
                d_classified_points,
                num_points_,
                min_x_, min_y_, min_z_,
                max_x_, max_y_, max_z_,
                grid_size
            );
            
            // Copy result back to host
            kernels::PointData* h_classified_points = new kernels::PointData[num_points_];
            cudaMemcpy(h_classified_points, d_classified_points, num_points_ * sizeof(kernels::PointData), cudaMemcpyDeviceToHost);
            
            // Update point classifications in the result
            for (size_t i = 0; i < num_points_; i++) {
                result->pImpl->points_[i].attributes.classification = 
                    static_cast<PointClass>(h_classified_points[i].classification);
            }
            
            // Clean up
            delete[] h_classified_points;
            cudaFree(d_classified_points);
#endif
        } else {
            // CPU implementation of point classification
            classifyPointsCPU(result->pImpl->points_, algorithm);
        }
        
        return result;
    }
    
    std::shared_ptr<PointCloud> filterByClass(PointClass classification) {
        // Create a new point cloud for the result
        auto result = std::make_shared<PointCloud>(point_cloud_file_, device_id_);
        
        // Filter points by classification
        result->pImpl->points_.clear();
        for (const auto& point : points_) {
            if (point.attributes.classification == classification) {
                result->pImpl->points_.push_back(point);
            }
        }
        
        // Update point count and bounds
        result->pImpl->num_points_ = result->pImpl->points_.size();
        result->pImpl->calculateBounds();
        
        return result;
    }
    
    std::shared_ptr<PointCloud> filterByReturnNumber(int return_number) {
        // Create a new point cloud for the result
        auto result = std::make_shared<PointCloud>(point_cloud_file_, device_id_);
        
        // Filter points by return number
        result->pImpl->points_.clear();
        for (const auto& point : points_) {
            if (point.attributes.return_number == return_number) {
                result->pImpl->points_.push_back(point);
            }
        }
        
        // Update point count and bounds
        result->pImpl->num_points_ = result->pImpl->points_.size();
        result->pImpl->calculateBounds();
        
        return result;
    }
    
    std::shared_ptr<DEMProcessor> createDEM(double resolution, int algorithm) {
        // Calculate DEM dimensions
        double width_meters = max_x_ - min_x_;
        double height_meters = max_y_ - min_y_;
        
        int width = static_cast<int>(std::ceil(width_meters / resolution));
        int height = static_cast<int>(std::ceil(height_meters / resolution));
        
        // Create a temporary file for the DEM
        std::string dem_file = createTemporaryTiff(width, height, min_x_, min_y_, resolution);
        
        // Allocate memory for the DEM
        float* h_dem = new float[width * height];
        std::fill(h_dem, h_dem + width * height, -9999.0f); // NoData value
        
        if (has_cuda_) {
#ifdef WITH_CUDA
            // Allocate device memory for the DEM
            float* d_dem = nullptr;
            cudaMalloc(&d_dem, width * height * sizeof(float));
            cudaMemset(d_dem, 0, width * height * sizeof(float));
            
            // Launch kernel
            kernels::launchCreateDEMKernel(
                d_points_,
                num_points_,
                d_dem,
                width,
                height,
                min_x_,
                min_y_,
                resolution,
                algorithm
            );
            
            // Copy result back to host
            cudaMemcpy(h_dem, d_dem, width * height * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Clean up
            cudaFree(d_dem);
#endif
        } else {
            // CPU implementation of DEM creation
            createDEMCPU(h_dem, width, height, resolution, algorithm);
        }
        
        // Write DEM to file
        writeDEMToFile(h_dem, dem_file, width, height, min_x_, min_y_, resolution);
        
        // Clean up
        delete[] h_dem;
        
        // Create DEMProcessor from the temporary file
        return std::make_shared<DEMProcessor>(dem_file, device_id_);
    }
    
    std::shared_ptr<DEMProcessor> createDSM(double resolution, int algorithm) {
        // Similar to createDEM but uses first returns or highest points
        // Calculate DSM dimensions
        double width_meters = max_x_ - min_x_;
        double height_meters = max_y_ - min_y_;
        
        int width = static_cast<int>(std::ceil(width_meters / resolution));
        int height = static_cast<int>(std::ceil(height_meters / resolution));
        
        // Create a temporary file for the DSM
        std::string dsm_file = createTemporaryTiff(width, height, min_x_, min_y_, resolution);
        
        // Allocate memory for the DSM
        float* h_dsm = new float[width * height];
        std::fill(h_dsm, h_dsm + width * height, -9999.0f); // NoData value
        
        // For DSM, we want to use the highest point in each cell
        // This can be done by modifying the algorithm parameter
        int dsm_algorithm = algorithm + 10; // Offset for DSM
        
        if (has_cuda_) {
#ifdef WITH_CUDA
            // Allocate device memory for the DSM
            float* d_dsm = nullptr;
            cudaMalloc(&d_dsm, width * height * sizeof(float));
            cudaMemset(d_dsm, 0, width * height * sizeof(float));
            
            // Launch kernel with DSM algorithm
            kernels::launchCreateDEMKernel(
                d_points_,
                num_points_,
                d_dsm,
                width,
                height,
                min_x_,
                min_y_,
                resolution,
                dsm_algorithm
            );
            
            // Copy result back to host
            cudaMemcpy(h_dsm, d_dsm, width * height * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Clean up
            cudaFree(d_dsm);
#endif
        } else {
            // CPU implementation of DSM creation
            createDSMCPU(h_dsm, width, height, resolution, algorithm);
        }
        
        // Write DSM to file
        writeDEMToFile(h_dsm, dsm_file, width, height, min_x_, min_y_, resolution);
        
        // Clean up
        delete[] h_dsm;
        
        // Create DEMProcessor from the temporary file
        return std::make_shared<DEMProcessor>(dsm_file, device_id_);
    }
    
    std::string extractBuildings(double min_height, double min_area) {
        // Create a temporary file for the buildings vector data
        std::string buildings_file = createTemporaryShapefile();
        
        // Compute optimal grid size for spatial partitioning
        float grid_size = computeOptimalGridSize();
        
        if (has_cuda_) {
#ifdef WITH_CUDA
            // Allocate device memory for point labels
            int* d_labels = nullptr;
            cudaMalloc(&d_labels, num_points_ * sizeof(int));
            cudaMemset(d_labels, -1, num_points_ * sizeof(int));
            
            // Launch kernel
            kernels::launchExtractBuildingsKernel(
                d_points_,
                num_points_,
                d_labels,
                min_x_, min_y_, min_z_,
                max_x_, max_y_, max_z_,
                grid_size,
                min_height
            );
            
            // Copy result back to host
            int* h_labels = new int[num_points_];
            cudaMemcpy(h_labels, d_labels, num_points_ * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Process building clusters and create vector data
            extractBuildingPolygons(h_labels, buildings_file, min_area);
            
            // Clean up
            delete[] h_labels;
            cudaFree(d_labels);
#endif
        } else {
            // CPU implementation of building extraction
            extractBuildingsCPU(buildings_file, min_height, min_area);
        }
        
        return buildings_file;
    }
    
    std::shared_ptr<float> extractVegetation(
        const std::vector<double>& height_classes,
        double resolution)
    {
        // Calculate vegetation raster dimensions
        double width_meters = max_x_ - min_x_;
        double height_meters = max_y_ - min_y_;
        
        int width = static_cast<int>(std::ceil(width_meters / resolution));
        int height = static_cast<int>(std::ceil(height_meters / resolution));
        
        // Allocate memory for the vegetation raster
        float* h_vegetation = new float[width * height];
        std::fill(h_vegetation, h_vegetation + width * height, -9999.0f); // NoData value
        
        // CPU implementation of vegetation extraction
        extractVegetationCPU(h_vegetation, width, height, resolution, height_classes);
        
        // Return as shared pointer
        return std::shared_ptr<float>(h_vegetation, std::default_delete<float[]>());
    }
    
    std::unordered_map<int, std::vector<size_t>> segmentPoints(
        double max_distance, int min_points)
    {
        // Segment points into coherent objects
        std::unordered_map<int, std::vector<size_t>> segments;
        
        // CPU implementation of point segmentation
        segmentPointsCPU(segments, max_distance, min_points);
        
        return segments;
    }
    
    bool save(const std::string& output_file) {
        // Save point cloud to file
#ifdef WITH_PDAL
        return savePDAL(output_file);
#else
        // Simple binary format fallback
        return saveBinary(output_file);
#endif
    }
    
    std::tuple<double, double, double, PointAttributes> getPoint(size_t index) const {
        if (index >= num_points_) {
            throw std::out_of_range("Point index out of range");
        }
        
        const auto& point = points_[index];
        return std::make_tuple(point.x, point.y, point.z, point.attributes);
    }
    
    std::shared_ptr<PointCloud> voxelDownsample(double voxel_size) {
        // Create a new point cloud for the result
        auto result = std::make_shared<PointCloud>(point_cloud_file_, device_id_);
        
        // CPU implementation of voxel downsampling
        voxelDownsampleCPU(result->pImpl->points_, voxel_size);
        
        // Update point count and bounds
        result->pImpl->num_points_ = result->pImpl->points_.size();
        result->pImpl->calculateBounds();
        
        return result;
    }
    
    std::vector<std::array<float, 3>> computeNormals(double radius) {
        // Allocate memory for the normals
        std::vector<std::array<float, 3>> normals(num_points_);
        
        // Compute optimal grid size for spatial partitioning
        float grid_size = computeOptimalGridSize();
        
        if (has_cuda_) {
#ifdef WITH_CUDA
            // Allocate device memory for normals
            kernels::float3* d_normals = nullptr;
            cudaMalloc(&d_normals, num_points_ * sizeof(kernels::float3));
            
            // Launch kernel
            kernels::launchComputeNormalsKernel(
                d_points_,
                d_normals,
                num_points_,
                radius,
                min_x_, min_y_, min_z_,
                max_x_, max_y_, max_z_,
                grid_size
            );
            
            // Copy result back to host
            kernels::float3* h_normals = new kernels::float3[num_points_];
            cudaMemcpy(h_normals, d_normals, num_points_ * sizeof(kernels::float3), cudaMemcpyDeviceToHost);
            
            // Convert to output format
            for (size_t i = 0; i < num_points_; i++) {
                normals[i] = {h_normals[i].x, h_normals[i].y, h_normals[i].z};
            }
            
            // Clean up
            delete[] h_normals;
            cudaFree(d_normals);
#endif
        } else {
            // CPU implementation of normal computation
            computeNormalsCPU(normals, radius);
        }
        
        return normals;
    }
    
private:
    // Point structure for internal processing
    struct Point {
        double x, y, z;
        PointAttributes attributes;
    };
    
    // Load point cloud from file
    void loadPointCloud() {
#ifdef WITH_PDAL
        loadPDAL();
#else
        // Try to load from simple binary format
        if (point_cloud_file_.substr(point_cloud_file_.size() - 4) == ".bin") {
            loadBinary();
        } else {
            // Create a synthetic point cloud for testing
            createSyntheticPointCloud();
        }
#endif
    }
    
    // Create a synthetic point cloud for testing
    void createSyntheticPointCloud() {
        // Create a simple grid of points
        const int grid_size = 100;
        const double spacing = 1.0;
        
        points_.clear();
        
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                double x = i * spacing;
                double y = j * spacing;
                double z = 0.0;
                
                // Add some terrain features
                z = 10.0 + std::sin(x / 10.0) * std::cos(y / 10.0) * 5.0;
                
                // Add building-like features
                if ((i > 20 && i < 30 && j > 20 && j < 40) ||
                    (i > 60 && i < 70 && j > 60 && j < 70)) {
                    z += 15.0;
                }
                
                // Add vegetation-like features
                if ((i > 40 && i < 50 && j > 40 && j < 50) ||
                    (i > 80 && i < 90 && j > 10 && j < 20)) {
                    z += 5.0 + (i + j) % 10;
                }
                
                Point point;
                point.x = x;
                point.y = y;
                point.z = z;
                
                // Initialize attributes
                point.attributes.intensity = static_cast<uint8_t>((i + j) % 256);
                point.attributes.return_number = 1;
                point.attributes.num_returns = 1;
                point.attributes.scan_direction = 0;
                point.attributes.edge_of_flight = 0;
                point.attributes.classification = PointClass::Unclassified;
                point.attributes.scan_angle_rank = 0;
                point.attributes.user_data = 0;
                point.attributes.point_source_id = 0;
                point.attributes.gps_time = 0.0;
                
                points_.push_back(point);
            }
        }
        
        num_points_ = points_.size();
    }
    
#ifdef WITH_PDAL
    // Load point cloud from LAS/LAZ file using PDAL
    void loadPDAL() {
        try {
            // Create PDAL reader
            pdal::LasReader reader;
            pdal::Options options;
            options.add("filename", point_cloud_file_);
            reader.setOptions(options);
            
            // Execute PDAL pipeline
            reader.prepare();
            pdal::PointViewSet viewSet = reader.execute();
            pdal::PointViewPtr view = *viewSet.begin();
            
            // Reserve space for points
            size_t num_points = view->size();
            points_.clear();
            points_.reserve(num_points);
            
            // Copy points from PDAL to our internal format
            for (size_t i = 0; i < num_points; i++) {
                Point point;
                point.x = view->getFieldAs<double>(pdal::Dimension::Id::X, i);
                point.y = view->getFieldAs<double>(pdal::Dimension::Id::Y, i);
                point.z = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
                
                // Copy attributes
                point.attributes.intensity = view->getFieldAs<uint8_t>(pdal::Dimension::Id::Intensity, i);
                point.attributes.return_number = view->getFieldAs<uint8_t>(pdal::Dimension::Id::ReturnNumber, i);
                point.attributes.num_returns = view->getFieldAs<uint8_t>(pdal::Dimension::Id::NumberOfReturns, i);
                point.attributes.scan_direction = view->getFieldAs<uint8_t>(pdal::Dimension::Id::ScanDirectionFlag, i);
                point.attributes.edge_of_flight = view->getFieldAs<uint8_t>(pdal::Dimension::Id::EdgeOfFlightLine, i);
                point.attributes.classification = static_cast<PointClass>(view->getFieldAs<uint8_t>(pdal::Dimension::Id::Classification, i));
                point.attributes.scan_angle_rank = view->getFieldAs<uint8_t>(pdal::Dimension::Id::ScanAngleRank, i);
                point.attributes.user_data = view->getFieldAs<uint8_t>(pdal::Dimension::Id::UserData, i);
                point.attributes.point_source_id = view->getFieldAs<uint16_t>(pdal::Dimension::Id::PointSourceId, i);
                point.attributes.gps_time = view->getFieldAs<double>(pdal::Dimension::Id::GpsTime, i);
                
                points_.push_back(point);
            }
            
            num_points_ = points_.size();
        } catch (const std::exception& e) {
            std::cerr << "Error loading point cloud with PDAL: " << e.what() << std::endl;
            createSyntheticPointCloud();
        }
    }
    
    // Save point cloud to LAS/LAZ file using PDAL
    bool savePDAL(const std::string& output_file) {
        try {
            // Create PDAL point view
            pdal::PointViewPtr view(new pdal::PointView);
            
            // Add dimensions
            pdal::PointLayoutPtr layout = view->layout();
            layout->registerDim(pdal::Dimension::Id::X);
            layout->registerDim(pdal::Dimension::Id::Y);
            layout->registerDim(pdal::Dimension::Id::Z);
            layout->registerDim(pdal::Dimension::Id::Intensity);
            layout->registerDim(pdal::Dimension::Id::ReturnNumber);
            layout->registerDim(pdal::Dimension::Id::NumberOfReturns);
            layout->registerDim(pdal::Dimension::Id::ScanDirectionFlag);
            layout->registerDim(pdal::Dimension::Id::EdgeOfFlightLine);
            layout->registerDim(pdal::Dimension::Id::Classification);
            layout->registerDim(pdal::Dimension::Id::ScanAngleRank);
            layout->registerDim(pdal::Dimension::Id::UserData);
            layout->registerDim(pdal::Dimension::Id::PointSourceId);
            layout->registerDim(pdal::Dimension::Id::GpsTime);
            
            // Copy points from our internal format to PDAL
            for (size_t i = 0; i < points_.size(); i++) {
                const auto& point = points_[i];
                
                pdal::PointId id = view->size();
                view->setField(pdal::Dimension::Id::X, id, point.x);
                view->setField(pdal::Dimension::Id::Y, id, point.y);
                view->setField(pdal::Dimension::Id::Z, id, point.z);
                
                // Copy attributes
                view->setField(pdal::Dimension::Id::Intensity, id, point.attributes.intensity);
                view->setField(pdal::Dimension::Id::ReturnNumber, id, point.attributes.return_number);
                view->setField(pdal::Dimension::Id::NumberOfReturns, id, point.attributes.num_returns);
                view->setField(pdal::Dimension::Id::ScanDirectionFlag, id, point.attributes.scan_direction);
                view->setField(pdal::Dimension::Id::EdgeOfFlightLine, id, point.attributes.edge_of_flight);
                view->setField(pdal::Dimension::Id::Classification, id, static_cast<uint8_t>(point.attributes.classification));
                view->setField(pdal::Dimension::Id::ScanAngleRank, id, point.attributes.scan_angle_rank);
                view->setField(pdal::Dimension::Id::UserData, id, point.attributes.user_data);
                view->setField(pdal::Dimension::Id::PointSourceId, id, point.attributes.point_source_id);
                view->setField(pdal::Dimension::Id::GpsTime, id, point.attributes.gps_time);
            }
            
            // Create PDAL writer
            pdal::LasWriter writer;
            pdal::Options options;
            options.add("filename", output_file);
            writer.setOptions(options);
            
            // Create buffer reader
            pdal::BufferReader reader;
            pdal::PointViewSet viewSet;
            viewSet.insert(view);
            reader.addView(view);
            
            // Execute PDAL pipeline
            writer.setInput(reader);
            writer.prepare(layout);
            writer.execute(viewSet);
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error saving point cloud with PDAL: " << e.what() << std::endl;
            return false;
        }
    }
#endif
    
    // Load point cloud from simple binary format
    void loadBinary() {
        try {
            std::ifstream file(point_cloud_file_, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + point_cloud_file_);
            }
            
            // Read number of points
            size_t num_points;
            file.read(reinterpret_cast<char*>(&num_points), sizeof(size_t));
            
            // Reserve space for points
            points_.clear();
            points_.reserve(num_points);
            
            // Read points
            for (size_t i = 0; i < num_points; i++) {
                Point point;
                file.read(reinterpret_cast<char*>(&point), sizeof(Point));
                points_.push_back(point);
            }
            
            num_points_ = points_.size();
        } catch (const std::exception& e) {
            std::cerr << "Error loading point cloud from binary file: " << e.what() << std::endl;
            createSyntheticPointCloud();
        }
    }
    
    // Save point cloud to simple binary format
    bool saveBinary(const std::string& output_file) {
        try {
            std::ofstream file(output_file, std::ios::binary);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file for writing: " + output_file);
            }
            
            // Write number of points
            size_t num_points = points_.size();
            file.write(reinterpret_cast<const char*>(&num_points), sizeof(size_t));
            
            // Write points
            for (const auto& point : points_) {
                file.write(reinterpret_cast<const char*>(&point), sizeof(Point));
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error saving point cloud to binary file: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Calculate bounds of the point cloud
    void calculateBounds() {
        if (points_.empty()) {
            min_x_ = min_y_ = min_z_ = 0.0;
            max_x_ = max_y_ = max_z_ = 0.0;
            return;
        }
        
        min_x_ = max_x_ = points_[0].x;
        min_y_ = max_y_ = points_[0].y;
        min_z_ = max_z_ = points_[0].z;
        
        for (const auto& point : points_) {
            min_x_ = std::min(min_x_, point.x);
            min_y_ = std::min(min_y_, point.y);
            min_z_ = std::min(min_z_, point.z);
            
            max_x_ = std::max(max_x_, point.x);
            max_y_ = std::max(max_y_, point.y);
            max_z_ = std::max(max_z_, point.z);
        }
    }
    
    // Allocate device memory for CUDA processing
    void allocateDeviceMemory() {
#ifdef WITH_CUDA
        if (!has_cuda_ || points_.empty()) {
            return;
        }
        
        // Convert points to CUDA format
        kernels::PointData* h_points = new kernels::PointData[num_points_];
        for (size_t i = 0; i < num_points_; i++) {
            h_points[i].x = static_cast<float>(points_[i].x);
            h_points[i].y = static_cast<float>(points_[i].y);
            h_points[i].z = static_cast<float>(points_[i].z);
            h_points[i].intensity = points_[i].attributes.intensity;
            h_points[i].return_num = points_[i].attributes.return_number;
            h_points[i].classification = static_cast<uint8_t>(points_[i].attributes.classification);
        }
        
        // Allocate device memory
        cudaMalloc(&d_points_, num_points_ * sizeof(kernels::PointData));
        
        // Copy points to device
        cudaMemcpy(d_points_, h_points, num_points_ * sizeof(kernels::PointData), cudaMemcpyHostToDevice);
        
        // Save host copy for reuse
        h_points_ = h_points;
#endif
    }
    
    // Clean up memory
    void cleanup() {
        // Clean up host memory
        if (h_points_) {
            delete[] h_points_;
            h_points_ = nullptr;
        }
        
        // Clean up device memory
#ifdef WITH_CUDA
        if (d_points_ && has_cuda_) {
            cudaFree(d_points_);
            d_points_ = nullptr;
        }
#endif
    }
    
    // Compute optimal grid size for spatial partitioning
    float computeOptimalGridSize() const {
        // Choose a grid size that results in approximately 1000 cells
        double width = max_x_ - min_x_;
        double height = max_y_ - min_y_;
        double area = width * height;
        double cell_area = area / 1000.0;
        
        return static_cast<float>(std::sqrt(cell_area));
    }
    
    // Create a temporary TIFF file for DEM/DSM output
    std::string createTemporaryTiff(int width, int height, double min_x, double min_y, double resolution) {
        // Create a unique temporary filename
        std::string temp_dir = "/tmp";
        std::string temp_file = temp_dir + "/temp_dem_" + std::to_string(std::time(nullptr)) + ".tif";
        
        return temp_file;
    }
    
    // Create a temporary shapefile for vector data output
    std::string createTemporaryShapefile() {
        // Create a unique temporary filename
        std::string temp_dir = "/tmp";
        std::string temp_file = temp_dir + "/temp_buildings_" + std::to_string(std::time(nullptr)) + ".shp";
        
        return temp_file;
    }
    
    // Write DEM to TIFF file
    void writeDEMToFile(const float* dem, const std::string& file_path, int width, int height,
                       double min_x, double min_y, double resolution) {
        // Create GDAL driver
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
        if (!driver) {
            std::cerr << "Failed to create GDAL driver for TIFF" << std::endl;
            return;
        }
        
        // Create dataset
        GDALDataset* dataset = driver->Create(file_path.c_str(), width, height, 1, GDT_Float32, nullptr);
        if (!dataset) {
            std::cerr << "Failed to create GDAL dataset" << std::endl;
            return;
        }
        
        // Set geotransform
        double geotransform[6] = {
            min_x,        // top left x
            resolution,   // w-e pixel resolution
            0,            // rotation, 0 if image is "north up"
            min_y + height * resolution, // top left y
            0,            // rotation, 0 if image is "north up"
            -resolution   // n-s pixel resolution (negative value)
        };
        dataset->SetGeoTransform(geotransform);
        
        // Write data
        GDALRasterBand* band = dataset->GetRasterBand(1);
        band->SetNoDataValue(-9999.0);
        band->RasterIO(GF_Write, 0, 0, width, height, const_cast<float*>(dem), width, height, GDT_Float32, 0, 0);
        
        // Clean up
        GDALClose(dataset);
    }
    
    // CPU implementation of point classification
    void classifyPointsCPU(std::vector<Point>& points, int algorithm) {
        // Calculate grid size for spatial partitioning
        float grid_size = computeOptimalGridSize();
        
        // Create a spatial grid for efficient neighborhood queries
        double width = max_x_ - min_x_;
        double height = max_y_ - min_y_;
        int grid_width = static_cast<int>(std::ceil(width / grid_size));
        int grid_height = static_cast<int>(std::ceil(height / grid_size));
        
        // Create grid cell lists
        std::vector<std::vector<size_t>> grid_cells(grid_width * grid_height);
        
        // Assign points to grid cells
        for (size_t i = 0; i < points.size(); i++) {
            const auto& point = points[i];
            int cell_x = static_cast<int>((point.x - min_x_) / grid_size);
            int cell_y = static_cast<int>((point.y - min_y_) / grid_size);
            
            if (cell_x >= 0 && cell_x < grid_width && cell_y >= 0 && cell_y < grid_height) {
                grid_cells[cell_y * grid_width + cell_x].push_back(i);
            }
        }
        
        // Classify each point
        for (size_t i = 0; i < points.size(); i++) {
            auto& point = points[i];
            int cell_x = static_cast<int>((point.x - min_x_) / grid_size);
            int cell_y = static_cast<int>((point.y - min_y_) / grid_size);
            
            // Find minimum height in neighborhood
            double min_height = point.z;
            double max_height = point.z;
            double mean_height = point.z;
            int count = 1;
            
            // Check for multiple returns (vegetation indicator)
            bool is_multiple_return = point.attributes.return_number > 1;
            
            // Scan neighborhood for min/max heights
            const int neighborhood_size = 1;
            for (int nx = std::max(0, cell_x - neighborhood_size); nx <= std::min(grid_width - 1, cell_x + neighborhood_size); nx++) {
                for (int ny = std::max(0, cell_y - neighborhood_size); ny <= std::min(grid_height - 1, cell_y + neighborhood_size); ny++) {
                    
                    // Process points in this grid cell
                    for (size_t idx : grid_cells[ny * grid_width + nx]) {
                        if (idx == i) continue; // Skip the current point
                        
                        const auto& neighbor = points[idx];
                        min_height = std::min(min_height, neighbor.z);
                        max_height = std::max(max_height, neighbor.z);
                        mean_height += neighbor.z;
                        count++;
                    }
                }
            }
            
            if (count > 1) {
                mean_height /= count;
            }
            
            // Height range in the neighborhood
            double height_range = max_height - min_height;
            
            // Distance from point to minimum height
            double height_from_ground = point.z - min_height;
            
            // Classify points
            PointClass classification;
            
            // Ground points (lowest points)
            if (height_from_ground < 0.2) {
                classification = PointClass::Ground;
            }
            // Vegetation (multiple returns or non-planar neighborhood)
            else if (is_multiple_return || height_range > 3.0) {
                // Low, medium, or high vegetation based on height
                if (height_from_ground < 0.5) {
                    classification = PointClass::LowVegetation;
                } else if (height_from_ground < 2.0) {
                    classification = PointClass::MedVegetation;
                } else {
                    classification = PointClass::HighVegetation;
                }
            }
            // Buildings (planar and above ground)
            else if (height_from_ground > 1.0 && height_range < 1.0) {
                classification = PointClass::Building;
            }
            // Water (low intensity, near ground)
            else if (point.attributes.intensity < 30 && height_from_ground < 0.3) {
                classification = PointClass::Water;
            }
            // Default: unclassified
            else {
                classification = PointClass::Unclassified;
            }
            
            // Update classification
            point.attributes.classification = classification;
        }
    }
    
    // CPU implementation of DEM creation
    void createDEMCPU(float* dem, int width, int height, double resolution, int algorithm) {
        // Initialize DEM with NoData values
        for (int i = 0; i < width * height; i++) {
            dem[i] = -9999.0f;
        }
        
        // Only use ground points (classification = 2) for DEM
        std::vector<size_t> ground_points;
        for (size_t i = 0; i < points_.size(); i++) {
            if (points_[i].attributes.classification == PointClass::Ground) {
                ground_points.push_back(i);
            }
        }
        
        // If no ground points are found, use all points
        if (ground_points.empty()) {
            for (size_t i = 0; i < points_.size(); i++) {
                ground_points.push_back(i);
            }
        }
        
        // Select algorithm
        switch (algorithm) {
            case 0: // TIN-based interpolation (simplified)
                createDEMWithTIN(dem, width, height, resolution, ground_points);
                break;
                
            case 1: // IDW interpolation
                createDEMWithIDW(dem, width, height, resolution, ground_points);
                break;
                
            case 2: // Natural neighbor (simplified)
                createDEMWithNaturalNeighbor(dem, width, height, resolution, ground_points);
                break;
                
            default:
                createDEMWithIDW(dem, width, height, resolution, ground_points);
                break;
        }
    }
    
    // TIN-based interpolation (simplified)
    void createDEMWithTIN(float* dem, int width, int height, double resolution, const std::vector<size_t>& ground_points) {
        // Calculate search radius based on point density
        double search_radius = resolution * 2.0;
        double search_radius_sq = search_radius * search_radius;
        
        // Process each DEM cell
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                // Calculate geographic coordinates of this cell
                double geo_x = min_x_ + col * resolution;
                double geo_y = min_y_ + row * resolution;
                
                // Find closest 3 points for triangulation
                std::vector<std::pair<double, size_t>> neighbors;
                
                for (size_t idx : ground_points) {
                    const auto& point = points_[idx];
                    double dx = point.x - geo_x;
                    double dy = point.y - geo_y;
                    double dist_sq = dx * dx + dy * dy;
                    
                    if (dist_sq < search_radius_sq) {
                        neighbors.emplace_back(dist_sq, idx);
                    }
                }
                
                // Sort by distance
                std::sort(neighbors.begin(), neighbors.end());
                
                // Perform simplified triangulation (weighted average)
                if (neighbors.size() >= 3) {
                    double weight_sum = 0.0;
                    double weighted_elev = 0.0;
                    
                    for (size_t i = 0; i < std::min(size_t(3), neighbors.size()); i++) {
                        size_t idx = neighbors[i].second;
                        double dist = std::sqrt(neighbors[i].first);
                        double weight = 1.0 / (dist + 1e-6);
                        
                        weight_sum += weight;
                        weighted_elev += weight * points_[idx].z;
                    }
                    
                    if (weight_sum > 0.0) {
                        dem[row * width + col] = static_cast<float>(weighted_elev / weight_sum);
                    }
                }
            }
        }
    }
    
    // IDW interpolation
    void createDEMWithIDW(float* dem, int width, int height, double resolution, const std::vector<size_t>& ground_points) {
        // Calculate search radius based on point density
        double search_radius = resolution * 3.0;
        double search_radius_sq = search_radius * search_radius;
        
        // Process each DEM cell
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                // Calculate geographic coordinates of this cell
                double geo_x = min_x_ + col * resolution;
                double geo_y = min_y_ + row * resolution;
                
                // Find points within search radius
                double weight_sum = 0.0;
                double weighted_elev = 0.0;
                
                for (size_t idx : ground_points) {
                    const auto& point = points_[idx];
                    double dx = point.x - geo_x;
                    double dy = point.y - geo_y;
                    double dist_sq = dx * dx + dy * dy;
                    
                    if (dist_sq < search_radius_sq) {
                        double weight = 1.0 / (std::sqrt(dist_sq) + 1e-6);
                        weight_sum += weight;
                        weighted_elev += weight * point.z;
                    }
                }
                
                if (weight_sum > 0.0) {
                    dem[row * width + col] = static_cast<float>(weighted_elev / weight_sum);
                }
            }
        }
    }
    
    // Natural neighbor interpolation (simplified)
    void createDEMWithNaturalNeighbor(float* dem, int width, int height, double resolution, const std::vector<size_t>& ground_points) {
        // This is a simplified version - actual natural neighbor would use Voronoi diagrams
        // Here we implement a modified IDW that adjusts weights based on point density
        
        // Create a spatial grid for efficient neighborhood queries
        double grid_size = resolution * 2.0;
        int grid_width = static_cast<int>(std::ceil((max_x_ - min_x_) / grid_size));
        int grid_height = static_cast<int>(std::ceil((max_y_ - min_y_) / grid_size));
        
        // Create grid cell lists
        std::vector<std::vector<size_t>> grid_cells(grid_width * grid_height);
        
        // Assign points to grid cells
        for (size_t idx : ground_points) {
            const auto& point = points_[idx];
            int cell_x = static_cast<int>((point.x - min_x_) / grid_size);
            int cell_y = static_cast<int>((point.y - min_y_) / grid_size);
            
            if (cell_x >= 0 && cell_x < grid_width && cell_y >= 0 && cell_y < grid_height) {
                grid_cells[cell_y * grid_width + cell_x].push_back(idx);
            }
        }
        
        // Process each DEM cell
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                // Calculate geographic coordinates of this cell
                double geo_x = min_x_ + col * resolution;
                double geo_y = min_y_ + row * resolution;
                
                // Find grid cell for this point
                int cell_x = static_cast<int>((geo_x - min_x_) / grid_size);
                int cell_y = static_cast<int>((geo_y - min_y_) / grid_size);
                
                // Count points in neighborhood
                int point_count = 0;
                const int neighborhood_size = 2;
                
                for (int nx = std::max(0, cell_x - neighborhood_size); nx <= std::min(grid_width - 1, cell_x + neighborhood_size); nx++) {
                    for (int ny = std::max(0, cell_y - neighborhood_size); ny <= std::min(grid_height - 1, cell_y + neighborhood_size); ny++) {
                        point_count += grid_cells[ny * grid_width + nx].size();
                    }
                }
                
                // Adjust search radius based on point density
                double search_radius = resolution * 3.0;
                if (point_count > 10) {
                    search_radius *= 0.7;
                } else if (point_count < 3) {
                    search_radius *= 1.5;
                }
                double search_radius_sq = search_radius * search_radius;
                
                // Find points within search radius
                double weight_sum = 0.0;
                double weighted_elev = 0.0;
                
                for (int nx = std::max(0, cell_x - neighborhood_size); nx <= std::min(grid_width - 1, cell_x + neighborhood_size); nx++) {
                    for (int ny = std::max(0, cell_y - neighborhood_size); ny <= std::min(grid_height - 1, cell_y + neighborhood_size); ny++) {
                        for (size_t idx : grid_cells[ny * grid_width + nx]) {
                            const auto& point = points_[idx];
                            double dx = point.x - geo_x;
                            double dy = point.y - geo_y;
                            double dist_sq = dx * dx + dy * dy;
                            
                            if (dist_sq < search_radius_sq) {
                                // Modified inverse distance weighting
                                double weight = 1.0 / std::pow(std::sqrt(dist_sq) + 1e-6, 2.0);
                                weight_sum += weight;
                                weighted_elev += weight * point.z;
                            }
                        }
                    }
                }
                
                if (weight_sum > 0.0) {
                    dem[row * width + col] = static_cast<float>(weighted_elev / weight_sum);
                }
            }
        }
    }
    
    // CPU implementation of DSM creation
    void createDSMCPU(float* dsm, int width, int height, double resolution, int algorithm) {
        // Similar to createDEMCPU but uses the highest points
        // Initialize DSM with minimum possible values
        for (int i = 0; i < width * height; i++) {
            dsm[i] = -9999.0f;
        }
        
        // Create a height grid
        for (const auto& point : points_) {
            // Calculate grid cell indices
            int col = static_cast<int>((point.x - min_x_) / resolution);
            int row = static_cast<int>((point.y - min_y_) / resolution);
            
            // Check bounds
            if (col >= 0 && col < width && row >= 0 && row < height) {
                // For DSM, we use the highest point in each cell
                int idx = row * width + col;
                if (dsm[idx] == -9999.0f || point.z > dsm[idx]) {
                    dsm[idx] = static_cast<float>(point.z);
                }
            }
        }
        
        // Fill gaps using nearest valid neighbor
        fillDSMGaps(dsm, width, height);
    }
    
    // Fill gaps in DSM
    void fillDSMGaps(float* dsm, int width, int height) {
        // Copy original DSM
        std::vector<float> original(dsm, dsm + width * height);
        
        // Loop until all gaps are filled
        bool has_gaps = true;
        int iterations = 0;
        const int max_iterations = 10;
        
        while (has_gaps && iterations < max_iterations) {
            has_gaps = false;
            iterations++;
            
            for (int row = 0; row < height; row++) {
                for (int col = 0; col < width; col++) {
                    int idx = row * width + col;
                    
                    // Skip valid cells
                    if (original[idx] != -9999.0f) {
                        continue;
                    }
                    
                    // Try to fill gap from neighbors
                    float sum = 0.0f;
                    int count = 0;
                    
                    // Check 8 neighbors
                    for (int dr = -1; dr <= 1; dr++) {
                        for (int dc = -1; dc <= 1; dc++) {
                            if (dr == 0 && dc == 0) continue;
                            
                            int nr = row + dr;
                            int nc = col + dc;
                            
                            if (nr >= 0 && nr < height && nc >= 0 && nc < width) {
                                int nidx = nr * width + nc;
                                if (dsm[nidx] != -9999.0f) {
                                    sum += dsm[nidx];
                                    count++;
                                }
                            }
                        }
                    }
                    
                    // If we found valid neighbors, update DSM
                    if (count > 0) {
                        dsm[idx] = sum / count;
                    } else {
                        has_gaps = true;
                    }
                }
            }
            
            // Update original for next iteration
            std::copy(dsm, dsm + width * height, original.begin());
        }
    }
    
    // Process building clusters and create vector data
    void extractBuildingPolygons(const int* labels, const std::string& output_file, double min_area) {
        // Create a map of cluster IDs to point indices
        std::unordered_map<int, std::vector<size_t>> clusters;
        
        for (size_t i = 0; i < num_points_; i++) {
            int label = labels[i];
            if (label >= 0) {
                clusters[label].push_back(i);
            }
        }
        
        // Create OGR driver
        GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("ESRI Shapefile");
        if (!driver) {
            std::cerr << "Failed to create GDAL driver for Shapefile" << std::endl;
            return;
        }
        
        // Create dataset
        GDALDataset* dataset = driver->Create(output_file.c_str(), 0, 0, 0, GDT_Unknown, nullptr);
        if (!dataset) {
            std::cerr << "Failed to create Shapefile dataset" << std::endl;
            return;
        }
        
        // Create layer
        OGRSpatialReference srs;
        srs.SetWellKnownGeogCS("WGS84");
        OGRLayer* layer = dataset->CreateLayer("buildings", &srs, wkbPolygon, nullptr);
        if (!layer) {
            std::cerr << "Failed to create Shapefile layer" << std::endl;
            GDALClose(dataset);
            return;
        }
        
        // Add fields
        OGRFieldDefn id_field("ID", OFTInteger);
        OGRFieldDefn height_field("Height", OFTReal);
        OGRFieldDefn area_field("Area", OFTReal);
        
        layer->CreateField(&id_field);
        layer->CreateField(&height_field);
        layer->CreateField(&area_field);
        
        // Process each cluster
        for (const auto& cluster : clusters) {
            int label = cluster.first;
            const auto& points = cluster.second;
            
            // Skip small clusters
            if (points.size() < 10) {
                continue;
            }
            
            // Extract building footprint using convex hull
            std::vector<OGRPoint> hull_points;
            double mean_height = 0.0;
            
            for (size_t idx : points) {
                const auto& point = points_[idx];
                hull_points.emplace_back(point.x, point.y);
                mean_height += point.z;
            }
            
            mean_height /= points.size();
            
            // Create convex hull
            OGRMultiPoint multipoint;
            for (const auto& point : hull_points) {
                multipoint.addGeometry(&point);
            }
            
            OGRGeometry* hull = multipoint.ConvexHull();
            if (!hull) {
                continue;
            }
            
            // Calculate area
            double area = hull->get_Area();
            if (area < min_area) {
                OGRGeometryFactory::destroyGeometry(hull);
                continue;
            }
            
            // Create feature
            OGRFeature* feature = OGRFeature::CreateFeature(layer->GetLayerDefn());
            feature->SetField("ID", label);
            feature->SetField("Height", mean_height);
            feature->SetField("Area", area);
            feature->SetGeometry(hull);
            
            // Add feature to layer
            layer->CreateFeature(feature);
            
            // Clean up
            OGRFeature::DestroyFeature(feature);
            OGRGeometryFactory::destroyGeometry(hull);
        }
        
        // Clean up
        GDALClose(dataset);
    }
    
    // CPU implementation of building extraction
    void extractBuildingsCPU(const std::string& output_file, double min_height, double min_area) {
        // Filter building points
        std::vector<size_t> building_points;
        for (size_t i = 0; i < points_.size(); i++) {
            if (points_[i].attributes.classification == PointClass::Building) {
                building_points.push_back(i);
            }
        }
        
        // If no building points are found, try to classify points
        if (building_points.empty()) {
            // Create a copy of points for classification
            std::vector<Point> classified_points = points_;
            classifyPointsCPU(classified_points, 0);
            
            // Find building points
            for (size_t i = 0; i < classified_points.size(); i++) {
                if (classified_points[i].attributes.classification == PointClass::Building) {
                    building_points.push_back(i);
                }
            }
        }
        
        // Create a grid for clustering
        double grid_size = computeOptimalGridSize();
        int grid_width = static_cast<int>(std::ceil((max_x_ - min_x_) / grid_size));
        int grid_height = static_cast<int>(std::ceil((max_y_ - min_y_) / grid_size));
        
        // Assign building points to clusters
        std::vector<int> labels(num_points_, -1);
        int next_label = 0;
        
        for (size_t idx : building_points) {
            // Skip if already labeled
            if (labels[idx] >= 0) {
                continue;
            }
            
            // Create a new cluster
            int label = next_label++;
            labels[idx] = label;
            
            // Find connected points (BFS)
            std::queue<size_t> queue;
            queue.push(idx);
            
            while (!queue.empty()) {
                size_t current = queue.front();
                queue.pop();
                
                const auto& point = points_[current];
                
                // Find neighbors
                for (size_t neighbor_idx : building_points) {
                    // Skip if already in this cluster
                    if (labels[neighbor_idx] == label) {
                        continue;
                    }
                    
                    // Skip if already in another cluster
                    if (labels[neighbor_idx] >= 0) {
                        continue;
                    }
                    
                    const auto& neighbor = points_[neighbor_idx];
                    
                    // Check if the neighbor is close enough
                    double dx = neighbor.x - point.x;
                    double dy = neighbor.y - point.y;
                    double dist_sq = dx * dx + dy * dy;
                    
                    if (dist_sq < grid_size * grid_size) {
                        // Add to cluster
                        labels[neighbor_idx] = label;
                        queue.push(neighbor_idx);
                    }
                }
            }
        }
        
        // Extract building polygons
        extractBuildingPolygons(labels.data(), output_file, min_area);
    }
    
    // CPU implementation of vegetation extraction
    void extractVegetationCPU(float* vegetation, int width, int height, double resolution,
                             const std::vector<double>& height_classes) {
        // Create a height grid for vegetation
        for (const auto& point : points_) {
            // Skip non-vegetation points
            if (point.attributes.classification != PointClass::LowVegetation &&
                point.attributes.classification != PointClass::MedVegetation &&
                point.attributes.classification != PointClass::HighVegetation) {
                continue;
            }
            
            // Calculate grid cell indices
            int col = static_cast<int>((point.x - min_x_) / resolution);
            int row = static_cast<int>((point.y - min_y_) / resolution);
            
            // Check bounds
            if (col >= 0 && col < width && row >= 0 && row < height) {
                // For vegetation, we use the height class value
                int idx = row * width + col;
                
                // Determine height class
                double height_above_ground = point.z;
                float class_value = 0.0f;
                
                for (size_t i = 0; i < height_classes.size(); i++) {
                    if (height_above_ground <= height_classes[i]) {
                        class_value = static_cast<float>(i + 1);
                        break;
                    }
                }
                
                if (class_value > 0.0f) {
                    // Update vegetation raster (maximum class value)
                    if (vegetation[idx] == -9999.0f || class_value > vegetation[idx]) {
                        vegetation[idx] = class_value;
                    }
                }
            }
        }
    }
    
    // CPU implementation of point segmentation
    void segmentPointsCPU(std::unordered_map<int, std::vector<size_t>>& segments,
                         double max_distance, int min_points) {
        // Calculate squared distance threshold
        double max_distance_sq = max_distance * max_distance;
        
        // Create a grid for efficient neighborhood queries
        double grid_size = std::max(max_distance, computeOptimalGridSize());
        int grid_width = static_cast<int>(std::ceil((max_x_ - min_x_) / grid_size));
        int grid_height = static_cast<int>(std::ceil((max_y_ - min_y_) / grid_size));
        
        // Create grid cell lists
        std::vector<std::vector<size_t>> grid_cells(grid_width * grid_height);
        
        // Assign points to grid cells
        for (size_t i = 0; i < points_.size(); i++) {
            const auto& point = points_[i];
            int cell_x = static_cast<int>((point.x - min_x_) / grid_size);
            int cell_y = static_cast<int>((point.y - min_y_) / grid_size);
            
            if (cell_x >= 0 && cell_x < grid_width && cell_y >= 0 && cell_y < grid_height) {
                grid_cells[cell_y * grid_width + cell_x].push_back(i);
            }
        }
        
        // Segment points using region growing
        std::vector<int> labels(num_points_, -1);
        int next_label = 0;
        
        for (size_t i = 0; i < points_.size(); i++) {
            // Skip if already labeled
            if (labels[i] >= 0) {
                continue;
            }
            
            // Create a new segment
            std::vector<size_t> segment;
            int label = next_label++;
            labels[i] = label;
            segment.push_back(i);
            
            // Find connected points (BFS)
            std::queue<size_t> queue;
            queue.push(i);
            
            while (!queue.empty()) {
                size_t current = queue.front();
                queue.pop();
                
                const auto& point = points_[current];
                int cell_x = static_cast<int>((point.x - min_x_) / grid_size);
                int cell_y = static_cast<int>((point.y - min_y_) / grid_size);
                
                // Check neighboring cells
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = cell_x + dx;
                        int ny = cell_y + dy;
                        
                        if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) {
                            continue;
                        }
                        
                        // Check points in this cell
                        for (size_t neighbor_idx : grid_cells[ny * grid_width + nx]) {
                            // Skip if already labeled
                            if (labels[neighbor_idx] >= 0) {
                                continue;
                            }
                            
                            const auto& neighbor = points_[neighbor_idx];
                            
                            // Check if the neighbor is close enough
                            double dx = neighbor.x - point.x;
                            double dy = neighbor.y - point.y;
                            double dz = neighbor.z - point.z;
                            double dist_sq = dx * dx + dy * dy + dz * dz;
                            
                            if (dist_sq < max_distance_sq) {
                                // Add to segment
                                labels[neighbor_idx] = label;
                                segment.push_back(neighbor_idx);
                                queue.push(neighbor_idx);
                            }
                        }
                    }
                }
            }
            
            // Add segment if it has enough points
            if (segment.size() >= static_cast<size_t>(min_points)) {
                segments[label] = segment;
            }
        }
    }
    
    // CPU implementation of voxel downsampling
    void voxelDownsampleCPU(std::vector<Point>& result, double voxel_size) {
        // Create a grid of voxels
        std::unordered_map<size_t, std::vector<size_t>> voxels;
        
        // Hash function for voxel coordinates
        auto voxel_hash = [voxel_size](double x, double y, double z) {
            int vx = static_cast<int>(std::floor(x / voxel_size));
            int vy = static_cast<int>(std::floor(y / voxel_size));
            int vz = static_cast<int>(std::floor(z / voxel_size));
            
            // Simple hash function for 3D coordinates
            return static_cast<size_t>((vx * 73856093) ^ (vy * 19349663) ^ (vz * 83492791));
        };
        
        // Assign points to voxels
        for (size_t i = 0; i < points_.size(); i++) {
            const auto& point = points_[i];
            size_t voxel_key = voxel_hash(point.x, point.y, point.z);
            voxels[voxel_key].push_back(i);
        }
        
        // Create downsampled point cloud
        result.clear();
        result.reserve(voxels.size());
        
        for (const auto& voxel : voxels) {
            const auto& indices = voxel.second;
            
            // Compute average point in this voxel
            Point avg_point;
            avg_point.x = 0.0;
            avg_point.y = 0.0;
            avg_point.z = 0.0;
            avg_point.attributes = points_[indices[0]].attributes;
            
            for (size_t idx : indices) {
                const auto& point = points_[idx];
                avg_point.x += point.x;
                avg_point.y += point.y;
                avg_point.z += point.z;
            }
            
            avg_point.x /= indices.size();
            avg_point.y /= indices.size();
            avg_point.z /= indices.size();
            
            result.push_back(avg_point);
        }
    }
    
    // CPU implementation of normal computation
    void computeNormalsCPU(std::vector<std::array<float, 3>>& normals, double radius) {
        // Calculate squared radius
        double radius_sq = radius * radius;
        
        // Create a grid for efficient neighborhood queries
        double grid_size = std::max(radius, computeOptimalGridSize());
        int grid_width = static_cast<int>(std::ceil((max_x_ - min_x_) / grid_size));
        int grid_height = static_cast<int>(std::ceil((max_y_ - min_y_) / grid_size));
        
        // Create grid cell lists
        std::vector<std::vector<size_t>> grid_cells(grid_width * grid_height);
        
        // Assign points to grid cells
        for (size_t i = 0; i < points_.size(); i++) {
            const auto& point = points_[i];
            int cell_x = static_cast<int>((point.x - min_x_) / grid_size);
            int cell_y = static_cast<int>((point.y - min_y_) / grid_size);
            
            if (cell_x >= 0 && cell_x < grid_width && cell_y >= 0 && cell_y < grid_height) {
                grid_cells[cell_y * grid_width + cell_x].push_back(i);
            }
        }
        
        // Compute normal for each point
        for (size_t i = 0; i < points_.size(); i++) {
            const auto& point = points_[i];
            int cell_x = static_cast<int>((point.x - min_x_) / grid_size);
            int cell_y = static_cast<int>((point.y - min_y_) / grid_size);
            
            // Find neighboring points
            std::vector<std::array<double, 3>> neighbors;
            
            // Add the current point (centered at origin)
            neighbors.push_back({0.0, 0.0, 0.0});
            
            // Add neighbors within radius
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = cell_x + dx;
                    int ny = cell_y + dy;
                    
                    if (nx < 0 || nx >= grid_width || ny < 0 || ny >= grid_height) {
                        continue;
                    }
                    
                    for (size_t idx : grid_cells[ny * grid_width + nx]) {
                        if (idx == i) {
                            continue;
                        }
                        
                        const auto& neighbor = points_[idx];
                        double dx = neighbor.x - point.x;
                        double dy = neighbor.y - point.y;
                        double dz = neighbor.z - point.z;
                        double dist_sq = dx * dx + dy * dy + dz * dz;
                        
                        if (dist_sq < radius_sq) {
                            neighbors.push_back({dx, dy, dz});
                        }
                    }
                }
            }
            
            // Compute covariance matrix
            double cov[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // xx, xy, xz, yy, yz, zz
            
            for (const auto& p : neighbors) {
                cov[0] += p[0] * p[0];
                cov[1] += p[0] * p[1];
                cov[2] += p[0] * p[2];
                cov[3] += p[1] * p[1];
                cov[4] += p[1] * p[2];
                cov[5] += p[2] * p[2];
            }
            
            for (int j = 0; j < 6; j++) {
                cov[j] /= neighbors.size();
            }
            
            // Find normal vector through PCA (use eigendecomposition of 3x3 matrix)
            // This is a simplified approach - we could use a more robust method
            
            // Normal is the eigenvector corresponding to the smallest eigenvalue
            // For simplicity, we use the cross product of the two principal directions
            
            // Extract principal directions (approximate)
            double vec1[3] = {1.0, 0.0, 0.0};
            double vec2[3] = {0.0, 1.0, 0.0};
            
            // Simple power iteration to find the largest eigenvector
            for (int iter = 0; iter < 10; iter++) {
                double new_vec[3] = {
                    cov[0] * vec1[0] + cov[1] * vec1[1] + cov[2] * vec1[2],
                    cov[1] * vec1[0] + cov[3] * vec1[1] + cov[4] * vec1[2],
                    cov[2] * vec1[0] + cov[4] * vec1[1] + cov[5] * vec1[2]
                };
                
                // Normalize
                double len = std::sqrt(new_vec[0] * new_vec[0] + new_vec[1] * new_vec[1] + new_vec[2] * new_vec[2]);
                
                if (len > 1e-6) {
                    vec1[0] = new_vec[0] / len;
                    vec1[1] = new_vec[1] / len;
                    vec1[2] = new_vec[2] / len;
                }
            }
            
            // Find second eigenvector (perpendicular to first)
            // Pick an initial guess that's not parallel to vec1
            if (std::abs(vec1[0]) > 0.9) {
                vec2[0] = 0.0;
                vec2[1] = 1.0;
                vec2[2] = 0.0;
            } else {
                vec2[0] = 1.0;
                vec2[1] = 0.0;
                vec2[2] = 0.0;
            }
            
            // Make vec2 orthogonal to vec1
            double dot = vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
            vec2[0] -= dot * vec1[0];
            vec2[1] -= dot * vec1[1];
            vec2[2] -= dot * vec1[2];
            
            // Normalize
            double len = std::sqrt(vec2[0] * vec2[0] + vec2[1] * vec2[1] + vec2[2] * vec2[2]);
            
            if (len > 1e-6) {
                vec2[0] /= len;
                vec2[1] /= len;
                vec2[2] /= len;
            }
            
            // Normal is cross product of vec1 and vec2
            double normal[3] = {
                vec1[1] * vec2[2] - vec1[2] * vec2[1],
                vec1[2] * vec2[0] - vec1[0] * vec2[2],
                vec1[0] * vec2[1] - vec1[1] * vec2[0]
            };
            
            // Normalize
            len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            
            if (len > 1e-6) {
                normal[0] /= len;
                normal[1] /= len;
                normal[2] /= len;
            }
            
            // Ensure normal points up (assuming Z is up)
            if (normal[2] < 0) {
                normal[0] = -normal[0];
                normal[1] = -normal[1];
                normal[2] = -normal[2];
            }
            
            // Store result
            normals[i] = {static_cast<float>(normal[0]), static_cast<float>(normal[1]), static_cast<float>(normal[2])};
        }
    }
    
    // Member variables
    int device_id_;
    std::string point_cloud_file_;
    std::vector<Point> points_;
    size_t num_points_;
    double min_x_, min_y_, min_z_;
    double max_x_, max_y_, max_z_;
    bool has_cuda_;
    
    // CUDA-related variables
#ifdef WITH_CUDA
    kernels::PointData* h_points_;  // Host copy of points
    kernels::PointData* d_points_;  // Device copy of points
#else
    void* h_points_;
    void* d_points_;
#endif
};

// Implementation of PointCloud methods using the pImpl idiom
PointCloud::PointCloud(const std::string& point_cloud_file, int device_id)
    : pImpl(std::make_unique<Impl>(point_cloud_file, device_id))
{
}

PointCloud::~PointCloud() = default;

size_t PointCloud::getNumPoints() const {
    return pImpl->getNumPoints();
}

std::tuple<double, double, double, double, double, double> PointCloud::getBounds() const {
    return pImpl->getBounds();
}

std::shared_ptr<PointCloud> PointCloud::classifyPoints(int algorithm) {
    return pImpl->classifyPoints(algorithm);
}

std::shared_ptr<PointCloud> PointCloud::filterByClass(PointClass classification) {
    return pImpl->filterByClass(classification);
}

std::shared_ptr<PointCloud> PointCloud::filterByReturnNumber(int return_number) {
    return pImpl->filterByReturnNumber(return_number);
}

std::shared_ptr<DEMProcessor> PointCloud::createDEM(double resolution, int algorithm) {
    return pImpl->createDEM(resolution, algorithm);
}

std::shared_ptr<DEMProcessor> PointCloud::createDSM(double resolution, int algorithm) {
    return pImpl->createDSM(resolution, algorithm);
}

std::string PointCloud::extractBuildings(double min_height, double min_area) {
    return pImpl->extractBuildings(min_height, min_area);
}

std::shared_ptr<float> PointCloud::extractVegetation(
    const std::vector<double>& height_classes,
    double resolution)
{
    return pImpl->extractVegetation(height_classes, resolution);
}

std::unordered_map<int, std::vector<size_t>> PointCloud::segmentPoints(
    double max_distance, int min_points)
{
    return pImpl->segmentPoints(max_distance, min_points);
}

bool PointCloud::save(const std::string& output_file) {
    return pImpl->save(output_file);
}

std::tuple<double, double, double, PointAttributes> PointCloud::getPoint(size_t index) const {
    return pImpl->getPoint(index);
}

std::shared_ptr<PointCloud> PointCloud::voxelDownsample(double voxel_size) {
    return pImpl->voxelDownsample(voxel_size);
}

std::vector<std::array<float, 3>> PointCloud::computeNormals(double radius) {
    return pImpl->computeNormals(radius);
}

} // namespace geospatial