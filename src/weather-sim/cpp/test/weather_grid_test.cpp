/**
 * @file weather_grid_test.cpp
 * @brief Unit tests for WeatherGrid class.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <gtest/gtest.h>
#include <weather_sim/weather_sim.hpp>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace weather_sim {
namespace test {

class WeatherGridTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default grid is 10x10
        grid = std::make_unique<WeatherGrid>(10, 10);
    }
    
    std::unique_ptr<WeatherGrid> grid;
};

// Test grid creation with valid dimensions
TEST_F(WeatherGridTest, CreateWithValidDimensions) {
    EXPECT_EQ(grid->getWidth(), 10);
    EXPECT_EQ(grid->getHeight(), 10);
    EXPECT_EQ(grid->getNumLevels(), 1);
    EXPECT_FLOAT_EQ(grid->getDx(), 1.0f);
    EXPECT_FLOAT_EQ(grid->getDy(), 1.0f);
}

// Test grid creation with invalid dimensions
TEST_F(WeatherGridTest, CreateWithInvalidDimensions) {
    EXPECT_THROW(WeatherGrid invalid_grid(0, 10), std::invalid_argument);
    EXPECT_THROW(WeatherGrid invalid_grid(10, 0), std::invalid_argument);
    EXPECT_THROW(WeatherGrid invalid_grid(10, 10, 0), std::invalid_argument);
}

// Test grid spacing
TEST_F(WeatherGridTest, SetSpacing) {
    grid->setSpacing(2.0f, 3.0f);
    EXPECT_FLOAT_EQ(grid->getDx(), 2.0f);
    EXPECT_FLOAT_EQ(grid->getDy(), 3.0f);
    
    EXPECT_THROW(grid->setSpacing(0.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(grid->setSpacing(1.0f, 0.0f), std::invalid_argument);
    EXPECT_THROW(grid->setSpacing(-1.0f, 1.0f), std::invalid_argument);
}

// Test grid reset
TEST_F(WeatherGridTest, Reset) {
    // Modify grid values
    auto& velocity = grid->getVelocityField();
    auto& height = grid->getHeightField();
    
    velocity.set(0, 0, 5.0f, 10.0f);
    height(0, 0) = 20.0f;
    
    // Verify modifications
    scalar_t u, v;
    velocity.get(0, 0, u, v);
    EXPECT_FLOAT_EQ(u, 5.0f);
    EXPECT_FLOAT_EQ(v, 10.0f);
    EXPECT_FLOAT_EQ(height(0, 0), 20.0f);
    
    // Reset grid
    grid->reset();
    
    // Verify reset values
    velocity.get(0, 0, u, v);
    EXPECT_FLOAT_EQ(u, 0.0f);
    EXPECT_FLOAT_EQ(v, 0.0f);
    EXPECT_FLOAT_EQ(height(0, 0), 10.0f);
}

// Test vorticity calculation
TEST_F(WeatherGridTest, CalculateVorticity) {
    // Set up a simple vortex pattern
    auto& velocity = grid->getVelocityField();
    auto& vorticity = grid->getVorticityField();
    
    // Set up a counter-clockwise rotation
    for (index_t y = 0; y < 10; ++y) {
        for (index_t x = 0; x < 10; ++x) {
            // Center of the grid
            scalar_t center_x = 4.5f;
            scalar_t center_y = 4.5f;
            
            // Vector from center to current point
            scalar_t dx = x - center_x;
            scalar_t dy = y - center_y;
            scalar_t r = std::sqrt(dx * dx + dy * dy);
            
            // Tangential velocities (counter-clockwise)
            scalar_t u = -dy / std::max(r, 0.1f);
            scalar_t v = dx / std::max(r, 0.1f);
            
            velocity.set(x, y, u, v);
        }
    }
    
    // Calculate vorticity
    grid->calculateDiagnostics();
    
    // Verify vorticity (should be positive in center, since rotation is counter-clockwise)
    EXPECT_GT(vorticity(5, 5), 0.0f);
}

// Test grid swapping
TEST_F(WeatherGridTest, Swap) {
    // Create two grids
    WeatherGrid grid1(10, 10);
    WeatherGrid grid2(10, 10);
    
    // Set different values in each grid
    grid1.getVelocityField().set(0, 0, 1.0f, 2.0f);
    grid1.getHeightField()(0, 0) = 15.0f;
    
    grid2.getVelocityField().set(0, 0, 3.0f, 4.0f);
    grid2.getHeightField()(0, 0) = 25.0f;
    
    // Swap grids
    grid1.swap(grid2);
    
    // Verify swapped values
    scalar_t u1, v1, u2, v2;
    grid1.getVelocityField().get(0, 0, u1, v1);
    grid2.getVelocityField().get(0, 0, u2, v2);
    
    EXPECT_FLOAT_EQ(u1, 3.0f);
    EXPECT_FLOAT_EQ(v1, 4.0f);
    EXPECT_FLOAT_EQ(grid1.getHeightField()(0, 0), 25.0f);
    
    EXPECT_FLOAT_EQ(u2, 1.0f);
    EXPECT_FLOAT_EQ(v2, 2.0f);
    EXPECT_FLOAT_EQ(grid2.getHeightField()(0, 0), 15.0f);
    
    // Test swapping grids with different dimensions
    WeatherGrid grid3(20, 20);
    EXPECT_THROW(grid1.swap(grid3), std::invalid_argument);
}

} // namespace test
} // namespace weather_sim