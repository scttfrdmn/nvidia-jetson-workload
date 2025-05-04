/**
 * @file bindings.cpp
 * @brief Python bindings for the geospatial library
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "geospatial/dem_processing.hpp"
#include "geospatial/point_cloud.hpp"
#include "geospatial/gpu_adaptability.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_geospatial, m) {
    m.doc() = "Geospatial Analysis C++/CUDA extension module";
    
    // Version info
    m.attr("__version__") = "1.0.0";
    
    // ==== Enums ====
    
    // PointClass enum
    py::enum_<geospatial::PointClass>(m, "PointClass")
        .value("Created", geospatial::PointClass::Created)
        .value("Unclassified", geospatial::PointClass::Unclassified)
        .value("Ground", geospatial::PointClass::Ground)
        .value("LowVegetation", geospatial::PointClass::LowVegetation)
        .value("MedVegetation", geospatial::PointClass::MedVegetation)
        .value("HighVegetation", geospatial::PointClass::HighVegetation)
        .value("Building", geospatial::PointClass::Building)
        .value("LowPoint", geospatial::PointClass::LowPoint)
        .value("Water", geospatial::PointClass::Water)
        .value("Rail", geospatial::PointClass::Rail)
        .value("RoadSurface", geospatial::PointClass::RoadSurface)
        .value("Bridge", geospatial::PointClass::Bridge)
        .value("WireGuard", geospatial::PointClass::WireGuard)
        .value("WireConductor", geospatial::PointClass::WireConductor)
        .value("TransmissionTower", geospatial::PointClass::TransmissionTower)
        .value("WireStructure", geospatial::PointClass::WireStructure)
        .value("Bridge_Deck", geospatial::PointClass::Bridge_Deck)
        .value("HighNoise", geospatial::PointClass::HighNoise)
        .export_values();
    
    // ==== Structs ====
    
    // GeoTransform
    py::class_<geospatial::GeoTransform>(m, "GeoTransform")
        .def_readwrite("parameters", &geospatial::GeoTransform::parameters)
        .def("pixel_to_geo", &geospatial::GeoTransform::pixelToGeo)
        .def("geo_to_pixel", &geospatial::GeoTransform::geoToPixel);
    
    // Point3D
    py::class_<geospatial::Point3D>(m, "Point3D")
        .def_readwrite("x", &geospatial::Point3D::x)
        .def_readwrite("y", &geospatial::Point3D::y)
        .def_readwrite("z", &geospatial::Point3D::z);
    
    // DEMStatistics
    py::class_<geospatial::DEMStatistics>(m, "DEMStatistics")
        .def_readwrite("min_elevation", &geospatial::DEMStatistics::min_elevation)
        .def_readwrite("max_elevation", &geospatial::DEMStatistics::max_elevation)
        .def_readwrite("mean_elevation", &geospatial::DEMStatistics::mean_elevation)
        .def_readwrite("std_dev", &geospatial::DEMStatistics::std_dev)
        .def_readwrite("mean_slope", &geospatial::DEMStatistics::mean_slope)
        .def_readwrite("mean_aspect", &geospatial::DEMStatistics::mean_aspect)
        .def_readwrite("ruggedness_index", &geospatial::DEMStatistics::ruggedness_index)
        .def_readwrite("hypsometric_integral", &geospatial::DEMStatistics::hypsometric_integral);
    
    // PointAttributes
    py::class_<geospatial::PointAttributes>(m, "PointAttributes")
        .def_readwrite("intensity", &geospatial::PointAttributes::intensity)
        .def_readwrite("return_number", &geospatial::PointAttributes::return_number)
        .def_readwrite("num_returns", &geospatial::PointAttributes::num_returns)
        .def_readwrite("scan_direction", &geospatial::PointAttributes::scan_direction)
        .def_readwrite("edge_of_flight", &geospatial::PointAttributes::edge_of_flight)
        .def_readwrite("classification", &geospatial::PointAttributes::classification)
        .def_readwrite("scan_angle_rank", &geospatial::PointAttributes::scan_angle_rank)
        .def_readwrite("user_data", &geospatial::PointAttributes::user_data)
        .def_readwrite("point_source_id", &geospatial::PointAttributes::point_source_id)
        .def_readwrite("gps_time", &geospatial::PointAttributes::gps_time);
    
    // ==== Classes ====
    
    // DEMProcessor
    py::class_<geospatial::DEMProcessor>(m, "DEMProcessor")
        .def(py::init<const std::string&, int>(), 
             py::arg("dem_file"), py::arg("device_id") = 0)
        .def("get_dimensions", &geospatial::DEMProcessor::getDimensions)
        .def("get_geotransform", &geospatial::DEMProcessor::getGeoTransform)
        .def("compute_viewshed", &geospatial::DEMProcessor::computeViewshed,
             py::arg("observer_point"), py::arg("observer_height") = 1.8f, py::arg("radius") = 0.0f)
        .def("compute_terrain_derivatives", &geospatial::DEMProcessor::computeTerrainDerivatives,
             py::arg("z_factor") = 1.0f)
        .def("compute_hydrological_features", &geospatial::DEMProcessor::computeHydrologicalFeatures)
        .def("compute_least_cost_path", &geospatial::DEMProcessor::computeLeastCostPath,
             py::arg("start_point"), py::arg("end_point"), py::arg("cost_function") = nullptr)
        .def("fill_sinks", &geospatial::DEMProcessor::fillSinks,
             py::arg("z_limit") = std::numeric_limits<float>::max())
        .def("compute_statistics", &geospatial::DEMProcessor::computeStatistics)
        .def("save_result", &geospatial::DEMProcessor::saveResult,
             py::arg("data"), py::arg("output_file"), py::arg("data_type") = 6)
        .def("resample", &geospatial::DEMProcessor::resample,
             py::arg("target_resolution"), py::arg("resampling_method") = 1)
        .def("get_device_id", [](const geospatial::DEMProcessor& self) {
            // This is a workaround since there's no direct method in the C++ class
            // We'll need to add a proper method in the future
            return 0;
        });
    
    // PointCloud
    py::class_<geospatial::PointCloud, std::shared_ptr<geospatial::PointCloud>>(m, "PointCloud")
        .def(py::init<const std::string&, int>(),
             py::arg("point_cloud_file"), py::arg("device_id") = 0)
        .def("get_num_points", &geospatial::PointCloud::getNumPoints)
        .def("get_bounds", &geospatial::PointCloud::getBounds)
        .def("classify_points", &geospatial::PointCloud::classifyPoints,
             py::arg("algorithm") = 0)
        .def("filter_by_class", &geospatial::PointCloud::filterByClass,
             py::arg("classification"))
        .def("filter_by_return_number", &geospatial::PointCloud::filterByReturnNumber,
             py::arg("return_number"))
        .def("create_dem", &geospatial::PointCloud::createDEM,
             py::arg("resolution"), py::arg("algorithm") = 0)
        .def("create_dsm", &geospatial::PointCloud::createDSM,
             py::arg("resolution"), py::arg("algorithm") = 0)
        .def("extract_buildings", &geospatial::PointCloud::extractBuildings,
             py::arg("min_height") = 2.0, py::arg("min_area") = 10.0)
        .def("extract_vegetation", &geospatial::PointCloud::extractVegetation,
             py::arg("height_classes") = std::vector<double>{0.5, 2.0, 5.0, 15.0}, py::arg("resolution") = 1.0)
        .def("segment_points", &geospatial::PointCloud::segmentPoints,
             py::arg("max_distance") = 1.0, py::arg("min_points") = 10)
        .def("save", &geospatial::PointCloud::save,
             py::arg("output_file"))
        .def("get_point", &geospatial::PointCloud::getPoint,
             py::arg("index"))
        .def("voxel_downsample", &geospatial::PointCloud::voxelDownsample,
             py::arg("voxel_size"))
        .def("compute_normals", &geospatial::PointCloud::computeNormals,
             py::arg("radius") = 1.0);

    // DeviceAdaptor (static methods only)
    py::class_<geospatial::DeviceAdaptor>(m, "DeviceAdaptor")
        .def_static("get_device_capabilities", &geospatial::DeviceAdaptor::getDeviceCapabilities,
                   py::arg("device_id") = 0)
        .def_static("get_optimal_dem_kernel_config", &geospatial::DeviceAdaptor::getOptimalDEMKernelConfig,
                   py::arg("width"), py::arg("height"), py::arg("device_id") = 0)
        .def_static("get_optimal_point_cloud_kernel_config", &geospatial::DeviceAdaptor::getOptimalPointCloudKernelConfig,
                   py::arg("num_points"), py::arg("device_id") = 0)
        .def_static("get_optimal_image_kernel_config", &geospatial::DeviceAdaptor::getOptimalImageKernelConfig,
                   py::arg("width"), py::arg("height"), py::arg("num_bands"), py::arg("device_id") = 0)
        .def_static("get_optimal_vector_kernel_config", &geospatial::DeviceAdaptor::getOptimalVectorKernelConfig,
                   py::arg("num_features"), py::arg("avg_vertices_per_feature"), py::arg("device_id") = 0)
        .def_static("has_unified_memory", &geospatial::DeviceAdaptor::hasUnifiedMemory,
                   py::arg("device_id") = 0)
        .def_static("get_recommended_tile_size", &geospatial::DeviceAdaptor::getRecommendedTileSize,
                   py::arg("total_width"), py::arg("total_height"), py::arg("bytes_per_pixel"), py::arg("device_id") = 0);
}