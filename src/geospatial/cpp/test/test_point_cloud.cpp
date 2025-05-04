/**
 * @file test_point_cloud.cpp
 * @brief Tests for point cloud processing functionality
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include <catch2/catch.hpp>
#include "geospatial/point_cloud.hpp"
#include "geospatial/dem_processing.hpp"
#include <memory>
#include <vector>
#include <cmath>
#include <random>

// Create a synthetic point cloud for testing
std::string createSyntheticPointCloud() {
    std::string tempfile = std::string("/tmp/test_cloud_") + 
                          std::to_string(std::time(nullptr)) + ".synthetic";
    
    // This file doesn't actually need to exist, as our implementation
    // will generate a synthetic point cloud if the file has .synthetic extension
    
    // Create an empty file as a marker
    std::ofstream outfile(tempfile);
    outfile.close();
    
    return tempfile;
}

// Clean up temporary files
void cleanupTestFile(const std::string& filename) {
    std::remove(filename.c_str());
}

TEST_CASE("PointCloud construction and basic properties", "[PointCloud]") {
    // Create a synthetic point cloud
    std::string test_cloud = createSyntheticPointCloud();
    
    // Create PointCloud
    geospatial::PointCloud cloud(test_cloud);
    
    // Test point count
    REQUIRE(cloud.getNumPoints() > 0);
    
    // Test bounds
    auto bounds = cloud.getBounds();
    double min_x = std::get<0>(bounds);
    double min_y = std::get<1>(bounds);
    double min_z = std::get<2>(bounds);
    double max_x = std::get<3>(bounds);
    double max_y = std::get<4>(bounds);
    double max_z = std::get<5>(bounds);
    
    REQUIRE(min_x < max_x);
    REQUIRE(min_y < max_y);
    REQUIRE(min_z < max_z);
    
    // Clean up
    cleanupTestFile(test_cloud);
}

TEST_CASE("PointCloud classification", "[PointCloud]") {
    // Create a synthetic point cloud
    std::string test_cloud = createSyntheticPointCloud();
    
    // Create PointCloud
    geospatial::PointCloud cloud(test_cloud);
    
    // Classify points
    auto classified = cloud.classifyPoints();
    
    // Verify result
    REQUIRE(classified != nullptr);
    REQUIRE(classified->getNumPoints() == cloud.getNumPoints());
    
    // Clean up
    cleanupTestFile(test_cloud);
}

TEST_CASE("PointCloud filtering", "[PointCloud]") {
    // Create a synthetic point cloud
    std::string test_cloud = createSyntheticPointCloud();
    
    // Create PointCloud
    geospatial::PointCloud cloud(test_cloud);
    
    // First classify points
    auto classified = cloud.classifyPoints();
    
    // Filter by ground class
    auto ground = classified->filterByClass(geospatial::PointClass::Ground);
    
    // Verify result
    REQUIRE(ground != nullptr);
    
    // Get a point and check its classification
    auto point = ground->getPoint(0);
    REQUIRE(std::get<3>(point).classification == geospatial::PointClass::Ground);
    
    // Clean up
    cleanupTestFile(test_cloud);
}

TEST_CASE("PointCloud DEM creation", "[PointCloud]") {
    // Create a synthetic point cloud
    std::string test_cloud = createSyntheticPointCloud();
    
    // Create PointCloud
    geospatial::PointCloud cloud(test_cloud);
    
    // Create DEM
    auto dem = cloud.createDEM(1.0);
    
    // Verify result
    REQUIRE(dem != nullptr);
    
    // Get dimensions
    auto dimensions = dem->getDimensions();
    REQUIRE(std::get<0>(dimensions) > 0);
    REQUIRE(std::get<1>(dimensions) > 0);
    
    // Clean up
    cleanupTestFile(test_cloud);
}

TEST_CASE("PointCloud DSM creation", "[PointCloud]") {
    // Create a synthetic point cloud
    std::string test_cloud = createSyntheticPointCloud();
    
    // Create PointCloud
    geospatial::PointCloud cloud(test_cloud);
    
    // Create DSM
    auto dsm = cloud.createDSM(1.0);
    
    // Verify result
    REQUIRE(dsm != nullptr);
    
    // Get dimensions
    auto dimensions = dsm->getDimensions();
    REQUIRE(std::get<0>(dimensions) > 0);
    REQUIRE(std::get<1>(dimensions) > 0);
    
    // Clean up
    cleanupTestFile(test_cloud);
}

TEST_CASE("PointCloud normal computation", "[PointCloud]") {
    // Create a synthetic point cloud
    std::string test_cloud = createSyntheticPointCloud();
    
    // Create PointCloud
    geospatial::PointCloud cloud(test_cloud);
    
    // Compute normals
    auto normals = cloud.computeNormals();
    
    // Verify result
    REQUIRE(normals.size() == cloud.getNumPoints());
    
    // Check that normals are normalized
    for (const auto& normal : normals) {
        float norm = std::sqrt(normal[0] * normal[0] + 
                              normal[1] * normal[1] + 
                              normal[2] * normal[2]);
        REQUIRE(norm == Approx(1.0f).epsilon(0.01f));
    }
    
    // Clean up
    cleanupTestFile(test_cloud);
}

TEST_CASE("PointCloud downsampling", "[PointCloud]") {
    // Create a synthetic point cloud
    std::string test_cloud = createSyntheticPointCloud();
    
    // Create PointCloud
    geospatial::PointCloud cloud(test_cloud);
    
    // Get original point count
    size_t original_count = cloud.getNumPoints();
    
    // Downsample
    auto downsampled = cloud.voxelDownsample(2.0);
    
    // Verify result
    REQUIRE(downsampled != nullptr);
    REQUIRE(downsampled->getNumPoints() < original_count);
    
    // Clean up
    cleanupTestFile(test_cloud);
}

TEST_CASE("PointCloud segmentation", "[PointCloud]") {
    // Create a synthetic point cloud
    std::string test_cloud = createSyntheticPointCloud();
    
    // Create PointCloud
    geospatial::PointCloud cloud(test_cloud);
    
    // Segment points
    auto segments = cloud.segmentPoints(2.0, 5);
    
    // Verify result
    REQUIRE(!segments.empty());
    
    // At least one segment should have multiple points
    bool has_multi_point_segment = false;
    for (const auto& segment : segments) {
        if (segment.second.size() >= 5) {
            has_multi_point_segment = true;
            break;
        }
    }
    
    REQUIRE(has_multi_point_segment);
    
    // Clean up
    cleanupTestFile(test_cloud);
}