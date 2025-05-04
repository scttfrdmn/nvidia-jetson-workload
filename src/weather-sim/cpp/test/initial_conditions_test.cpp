/**
 * @file initial_conditions_test.cpp
 * @brief Unit tests for initial conditions classes.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <gtest/gtest.h>
#include <weather_sim/weather_sim.hpp>
#include <weather_sim/initial_conditions.hpp>
#include <memory>
#include <cmath>

namespace weather_sim {
namespace test {

class InitialConditionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Register all initial conditions
        registerAllInitialConditions();
        
        // Create a 20x20 grid for testing
        grid = std::make_unique<WeatherGrid>(20, 20);
    }
    
    std::unique_ptr<WeatherGrid> grid;
};

// Test uniform initial condition
TEST_F(InitialConditionsTest, UniformInitialCondition) {
    UniformInitialCondition uniform(2.0f, 3.0f, 15.0f, 1000.0f, 290.0f, 0.5f);
    
    uniform.initialize(*grid);
    
    // Check a few points to verify uniform values
    auto& velocity = grid->getVelocityField();
    auto& height = grid->getHeightField();
    auto& temp = grid->getTemperatureField();
    auto& humidity = grid->getHumidityField();
    
    scalar_t u, v;
    
    velocity.get(0, 0, u, v);
    EXPECT_FLOAT_EQ(u, 2.0f);
    EXPECT_FLOAT_EQ(v, 3.0f);
    EXPECT_FLOAT_EQ(height(0, 0), 15.0f);
    EXPECT_FLOAT_EQ(temp(0, 0), 290.0f);
    EXPECT_FLOAT_EQ(humidity(0, 0), 0.5f);
    
    velocity.get(10, 10, u, v);
    EXPECT_FLOAT_EQ(u, 2.0f);
    EXPECT_FLOAT_EQ(v, 3.0f);
    EXPECT_FLOAT_EQ(height(10, 10), 15.0f);
    EXPECT_FLOAT_EQ(temp(10, 10), 290.0f);
    EXPECT_FLOAT_EQ(humidity(10, 10), 0.5f);
}

// Test random initial condition
TEST_F(InitialConditionsTest, RandomInitialCondition) {
    RandomInitialCondition random(42, 2.0f); // Fixed seed for reproducibility
    
    random.initialize(*grid);
    
    // Since it's random, we can't test exact values, but we can check if values are in the expected range
    auto& velocity = grid->getVelocityField();
    auto& height = grid->getHeightField();
    
    bool found_non_zero = false;
    
    for (index_t y = 0; y < grid->getHeight(); ++y) {
        for (index_t x = 0; x < grid->getWidth(); ++x) {
            scalar_t u, v;
            velocity.get(x, y, u, v);
            
            // Check bounds
            EXPECT_LE(std::abs(u), 2.0f);
            EXPECT_LE(std::abs(v), 2.0f);
            EXPECT_GT(height(x, y), 0.0f);
            
            // Check if we found any non-zero values (to ensure randomness)
            if (std::abs(u) > 0.01f || std::abs(v) > 0.01f) {
                found_non_zero = true;
            }
        }
    }
    
    EXPECT_TRUE(found_non_zero);
}

// Test factory creation
TEST_F(InitialConditionsTest, FactoryCreation) {
    auto& factory = InitialConditionFactory::getInstance();
    
    // Test creation of existing initial condition
    auto uniform = factory.createInitialCondition("uniform");
    EXPECT_NE(uniform, nullptr);
    EXPECT_EQ(uniform->getName(), "uniform");
    
    // Test creation of non-existent initial condition
    auto non_existent = factory.createInitialCondition("non_existent");
    EXPECT_EQ(non_existent, nullptr);
    
    // Test getAvailableInitialConditions
    auto available = factory.getAvailableInitialConditions();
    EXPECT_GT(available.size(), 0);
    
    // Check that "uniform" is in the list
    bool found_uniform = false;
    for (const auto& name : available) {
        if (name == "uniform") {
            found_uniform = true;
            break;
        }
    }
    EXPECT_TRUE(found_uniform);
}

// Test zonal flow initial condition
TEST_F(InitialConditionsTest, ZonalFlowInitialCondition) {
    ZonalFlowInitialCondition zonal(10.0f, 10.0f, 0.1f);
    
    zonal.initialize(*grid);
    
    // Check if velocity field has a zonal (east-west) flow
    auto& velocity = grid->getVelocityField();
    auto& height = grid->getHeightField();
    
    scalar_t u, v;
    
    // Check at the center (should have maximum velocity)
    velocity.get(10, 10, u, v);
    EXPECT_GT(std::abs(u), 0.0f);
    EXPECT_NEAR(v, 0.0f, 1e-5f);
    
    // Check that height field varies with latitude
    scalar_t h_middle = height(10, 10);
    scalar_t h_top = height(10, 0);
    scalar_t h_bottom = height(10, 19);
    
    // Height should vary with latitude (but we don't know exact pattern without knowing implementation details)
    EXPECT_NE(h_middle, h_top);
    EXPECT_NE(h_middle, h_bottom);
}

// Test vortex initial condition
TEST_F(InitialConditionsTest, VortexInitialCondition) {
    // Create a vortex at the center
    VortexInitialCondition vortex(0.5f, 0.5f, 0.2f, 10.0f, 10.0f);
    
    vortex.initialize(*grid);
    
    // Check if the velocity field forms a vortex
    auto& velocity = grid->getVelocityField();
    auto& vorticity = grid->getVorticityField();
    
    // Get center coordinates
    index_t center_x = grid->getWidth() / 2;
    index_t center_y = grid->getHeight() / 2;
    
    // At the center, velocity should be near zero
    scalar_t u_center, v_center;
    velocity.get(center_x, center_y, u_center, v_center);
    EXPECT_NEAR(u_center, 0.0f, 1e-1f);
    EXPECT_NEAR(v_center, 0.0f, 1e-1f);
    
    // Around the center, velocity should be non-zero
    scalar_t u_offset, v_offset;
    velocity.get(center_x + 2, center_y, u_offset, v_offset);
    EXPECT_NE(u_offset, 0.0f);
    EXPECT_NE(v_offset, 0.0f);
    
    // Check vorticity
    // Vorticity should be strongest at the center
    EXPECT_GT(std::abs(vorticity(center_x, center_y)), 0.0f);
}

// Test jet stream initial condition
TEST_F(InitialConditionsTest, JetStreamInitialCondition) {
    JetStreamInitialCondition jet(0.5f, 0.1f, 10.0f, 10.0f);
    
    jet.initialize(*grid);
    
    // Check if velocity field has a jet stream
    auto& velocity = grid->getVelocityField();
    
    // Get middle row (where jet is centered)
    index_t middle_y = grid->getHeight() / 2;
    
    // Check u velocity along middle row
    scalar_t u_middle, v_middle;
    velocity.get(10, middle_y, u_middle, v_middle);
    
    // Check a row above and below
    scalar_t u_above, v_above;
    velocity.get(10, middle_y - 5, u_above, v_above);
    
    scalar_t u_below, v_below;
    velocity.get(10, middle_y + 5, u_below, v_below);
    
    // Middle velocity should be greater than velocities away from the jet
    EXPECT_GT(std::abs(u_middle), std::abs(u_above));
    EXPECT_GT(std::abs(u_middle), std::abs(u_below));
    
    // v velocity should be near zero (jet is zonal)
    EXPECT_NEAR(v_middle, 0.0f, 1e-5f);
}

} // namespace test
} // namespace weather_sim