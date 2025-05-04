/**
 * @file test_dem_processing.cpp
 * @brief Tests for DEM processing functionality
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include <catch2/catch.hpp>
#include "geospatial/dem_processing.hpp"
#include <memory>
#include <vector>
#include <cmath>

// Create a temporary DEM file for testing
std::string createTestDEM(int width, int height) {
    std::string tempfile = std::string("/tmp/test_dem_") + 
                          std::to_string(std::time(nullptr)) + ".tif";
    
    // Create a simple DEM using GDAL
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    GDALDataset* dataset = driver->Create(tempfile.c_str(), width, height, 1, GDT_Float32, nullptr);
    
    // Set geotransform
    double geotransform[6] = {0.0, 1.0, 0.0, 0.0, 0.0, -1.0};
    dataset->SetGeoTransform(geotransform);
    
    // Create a simple DEM
    float* data = new float[width * height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Create a simple sinusoidal surface
            data[y * width + x] = 100.0f + 
                                 10.0f * std::sin(x * 0.1f) * 
                                 std::cos(y * 0.1f);
            
            // Add a "hill" in the center
            float dx = x - width / 2;
            float dy = y - height / 2;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist < width / 4) {
                data[y * width + x] += 20.0f * (1.0f - dist / (width / 4));
            }
        }
    }
    
    // Write data
    GDALRasterBand* band = dataset->GetRasterBand(1);
    band->RasterIO(GF_Write, 0, 0, width, height, data, width, height, GDT_Float32, 0, 0);
    
    // Clean up
    GDALClose(dataset);
    delete[] data;
    
    return tempfile;
}

// Clean up temporary files
void cleanupTestFile(const std::string& filename) {
    std::remove(filename.c_str());
}

TEST_CASE("DEMProcessor construction and basic properties", "[DEMProcessor]") {
    // Create a test DEM
    int width = 100;
    int height = 100;
    std::string test_dem = createTestDEM(width, height);
    
    // Create DEMProcessor
    geospatial::DEMProcessor processor(test_dem);
    
    // Test dimensions
    auto dimensions = processor.getDimensions();
    REQUIRE(std::get<0>(dimensions) == width);
    REQUIRE(std::get<1>(dimensions) == height);
    
    // Test geotransform
    auto geotransform = processor.getGeoTransform();
    REQUIRE(geotransform.parameters[0] == Approx(0.0));
    REQUIRE(geotransform.parameters[1] == Approx(1.0));
    
    // Clean up
    cleanupTestFile(test_dem);
}

TEST_CASE("DEMProcessor viewshed computation", "[DEMProcessor]") {
    // Create a test DEM
    int width = 100;
    int height = 100;
    std::string test_dem = createTestDEM(width, height);
    
    // Create DEMProcessor
    geospatial::DEMProcessor processor(test_dem);
    
    // Compute viewshed
    auto viewshed = processor.computeViewshed(
        std::make_tuple(50.0, 50.0),  // Observer at center
        10.0,                        // Observer height
        50.0                         // Max radius
    );
    
    // Verify viewshed is not null
    REQUIRE(viewshed != nullptr);
    
    // Verify observer point is visible (should be 1.0)
    int center_idx = height / 2 * width + width / 2;
    REQUIRE(viewshed.get()[center_idx] == Approx(1.0f));
    
    // Clean up
    cleanupTestFile(test_dem);
}

TEST_CASE("DEMProcessor terrain derivatives", "[DEMProcessor]") {
    // Create a test DEM
    int width = 100;
    int height = 100;
    std::string test_dem = createTestDEM(width, height);
    
    // Create DEMProcessor
    geospatial::DEMProcessor processor(test_dem);
    
    // Compute terrain derivatives
    auto derivatives = processor.computeTerrainDerivatives(1.0f);
    
    // Unpack results
    auto slope = std::get<0>(derivatives);
    auto aspect = std::get<1>(derivatives);
    auto curvature = std::get<2>(derivatives);
    
    // Verify results are not null
    REQUIRE(slope != nullptr);
    REQUIRE(aspect != nullptr);
    REQUIRE(curvature != nullptr);
    
    // Clean up
    cleanupTestFile(test_dem);
}

TEST_CASE("DEMProcessor statistics", "[DEMProcessor]") {
    // Create a test DEM
    int width = 100;
    int height = 100;
    std::string test_dem = createTestDEM(width, height);
    
    // Create DEMProcessor
    geospatial::DEMProcessor processor(test_dem);
    
    // Compute statistics
    auto stats = processor.computeStatistics();
    
    // Verify statistics are reasonable
    REQUIRE(stats.min_elevation < stats.max_elevation);
    REQUIRE(stats.mean_elevation >= stats.min_elevation);
    REQUIRE(stats.mean_elevation <= stats.max_elevation);
    REQUIRE(stats.std_dev >= 0.0);
    
    // Clean up
    cleanupTestFile(test_dem);
}

TEST_CASE("DEMProcessor resampling", "[DEMProcessor]") {
    // Create a test DEM
    int width = 100;
    int height = 100;
    std::string test_dem = createTestDEM(width, height);
    
    // Create DEMProcessor
    geospatial::DEMProcessor processor(test_dem);
    
    // Resample to half resolution
    auto resampled = processor.resample(2.0f);
    
    // Verify resampled DEM
    REQUIRE(resampled != nullptr);
    
    // Clean up
    cleanupTestFile(test_dem);
}

TEST_CASE("GeoTransform coordinate conversion", "[GeoTransform]") {
    // Create a test DEM
    int width = 100;
    int height = 100;
    std::string test_dem = createTestDEM(width, height);
    
    // Create DEMProcessor
    geospatial::DEMProcessor processor(test_dem);
    
    // Get geotransform
    auto geotransform = processor.getGeoTransform();
    
    // Test pixelToGeo and geoToPixel
    int test_x = 25;
    int test_y = 25;
    
    auto geo_coords = geotransform.pixelToGeo(test_x, test_y);
    auto pixel_coords = geotransform.geoToPixel(std::get<0>(geo_coords), std::get<1>(geo_coords));
    
    REQUIRE(std::get<0>(pixel_coords) == Approx(test_x).epsilon(0.5));
    REQUIRE(std::get<1>(pixel_coords) == Approx(test_y).epsilon(0.5));
    
    // Clean up
    cleanupTestFile(test_dem);
}