/**
 * @file weather_simulation_test.cpp
 * @brief Unit tests for WeatherSimulation class.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <gtest/gtest.h>
#include <weather_sim/weather_sim.hpp>
#include <weather_sim/initial_conditions.hpp>
#include <memory>

namespace weather_sim {
namespace test {

// Mock initial condition for testing
class MockInitialCondition : public InitialCondition {
public:
    void initialize(WeatherGrid& grid) const override {
        // Set simple initial condition
        auto& velocity = grid.getVelocityField();
        auto& height = grid.getHeightField();
        
        for (index_t y = 0; y < grid.getHeight(); ++y) {
            for (index_t x = 0; x < grid.getWidth(); ++x) {
                velocity.set(x, y, 1.0f, 0.0f);
                height(x, y) = 10.0f;
            }
        }
        
        grid.calculateDiagnostics();
    }
    
    std::string getName() const override {
        return "mock";
    }
};

// Mock output manager for testing
class MockOutputManager : public OutputManager {
public:
    void initialize(const WeatherSimulation& simulation) override {
        initialize_called_ = true;
    }
    
    void writeOutput(const WeatherSimulation& simulation) override {
        write_count_++;
    }
    
    void finalize(const WeatherSimulation& simulation) override {
        finalize_called_ = true;
    }
    
    bool initialize_called_ = false;
    int write_count_ = 0;
    bool finalize_called_ = false;
};

class WeatherSimulationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a small grid for fast testing
        config.grid_width = 10;
        config.grid_height = 10;
        config.dt = 0.1f;
        config.model = SimulationModel::ShallowWater;
        config.integration_method = IntegrationMethod::ExplicitEuler;
        config.compute_backend = ComputeBackend::CPU; // Use CPU for consistent tests
        
        simulation = std::make_unique<WeatherSimulation>(config);
        mock_initial = std::make_shared<MockInitialCondition>();
        mock_output = std::make_shared<MockOutputManager>();
    }
    
    SimulationConfig config;
    std::unique_ptr<WeatherSimulation> simulation;
    std::shared_ptr<MockInitialCondition> mock_initial;
    std::shared_ptr<MockOutputManager> mock_output;
};

// Test basic simulation creation
TEST_F(WeatherSimulationTest, Create) {
    EXPECT_EQ(simulation->getCurrentTime(), 0.0f);
    EXPECT_EQ(simulation->getCurrentStep(), 0);
    EXPECT_EQ(simulation->getDt(), 0.1f);
}

// Test simulation initialization
TEST_F(WeatherSimulationTest, Initialize) {
    simulation->setInitialCondition(mock_initial);
    simulation->setOutputManager(mock_output);
    simulation->initialize();
    
    EXPECT_EQ(simulation->getCurrentTime(), 0.0f);
    EXPECT_EQ(simulation->getCurrentStep(), 0);
    EXPECT_TRUE(mock_output->initialize_called_);
    
    // Initial velocity should be set by MockInitialCondition
    auto& velocity = simulation->getCurrentGrid().getVelocityField();
    scalar_t u, v;
    velocity.get(0, 0, u, v);
    EXPECT_FLOAT_EQ(u, 1.0f);
    EXPECT_FLOAT_EQ(v, 0.0f);
}

// Test single step
TEST_F(WeatherSimulationTest, Step) {
    simulation->setInitialCondition(mock_initial);
    simulation->initialize();
    
    // Take a single step
    simulation->step();
    
    EXPECT_EQ(simulation->getCurrentStep(), 1);
    EXPECT_FLOAT_EQ(simulation->getCurrentTime(), 0.1f);
    
    // Velocity should change due to gravity and height gradient
    auto& velocity = simulation->getCurrentGrid().getVelocityField();
    scalar_t u, v;
    velocity.get(0, 0, u, v);
    EXPECT_NE(u, 1.0f); // Should change from initial value
}

// Test multiple steps
TEST_F(WeatherSimulationTest, Run) {
    simulation->setInitialCondition(mock_initial);
    simulation->initialize();
    
    // Run for 10 steps
    simulation->run(10);
    
    EXPECT_EQ(simulation->getCurrentStep(), 10);
    EXPECT_FLOAT_EQ(simulation->getCurrentTime(), 1.0f);
}

// Test output generation
TEST_F(WeatherSimulationTest, Output) {
    simulation->setInitialCondition(mock_initial);
    simulation->setOutputManager(mock_output);
    simulation->initialize();
    
    // Set output interval
    config.output_interval = 2;
    
    // Run for 10 steps
    simulation->run(10);
    
    // Output should be called 5 times (at steps 0, 2, 4, 6, 8)
    EXPECT_EQ(mock_output->write_count_, 5);
}

// Test different integration methods
TEST_F(WeatherSimulationTest, IntegrationMethods) {
    // Test each integration method
    std::vector<IntegrationMethod> methods = {
        IntegrationMethod::ExplicitEuler,
        IntegrationMethod::RungeKutta2,
        IntegrationMethod::RungeKutta4,
        IntegrationMethod::AdamsBashforth,
        IntegrationMethod::SemiImplicit
    };
    
    for (auto method : methods) {
        config.integration_method = method;
        auto sim = std::make_unique<WeatherSimulation>(config);
        sim->setInitialCondition(mock_initial);
        sim->initialize();
        
        // Run for a few steps
        EXPECT_NO_THROW(sim->run(5));
        EXPECT_EQ(sim->getCurrentStep(), 5);
    }
}

// Test RungeKutta4 more extensively
TEST_F(WeatherSimulationTest, RungeKutta4Method) {
    config.integration_method = IntegrationMethod::RungeKutta4;
    auto sim = std::make_unique<WeatherSimulation>(config);
    sim->setInitialCondition(mock_initial);
    sim->initialize();
    
    // Get initial state
    auto& initial_velocity = sim->getCurrentGrid().getVelocityField();
    scalar_t initial_u, initial_v;
    initial_velocity.get(5, 5, initial_u, initial_v);
    
    // Run for 10 steps
    sim->run(10);
    
    // Get final state
    auto& final_velocity = sim->getCurrentGrid().getVelocityField();
    scalar_t final_u, final_v;
    final_velocity.get(5, 5, final_u, final_v);
    
    // Verify that the state has changed
    EXPECT_NE(final_u, initial_u);
}

// Test runUntil method
TEST_F(WeatherSimulationTest, RunUntil) {
    simulation->setInitialCondition(mock_initial);
    simulation->initialize();
    
    // Run until time 0.5
    simulation->runUntil(0.5f);
    
    // Should be 5 steps with dt=0.1
    EXPECT_EQ(simulation->getCurrentStep(), 5);
    EXPECT_FLOAT_EQ(simulation->getCurrentTime(), 0.5f);
    
    // Run until time 1.0
    simulation->runUntil(1.0f);
    
    // Should be 10 steps total
    EXPECT_EQ(simulation->getCurrentStep(), 10);
    EXPECT_FLOAT_EQ(simulation->getCurrentTime(), 1.0f);
    
    // Running until current time should do nothing
    simulation->runUntil(1.0f);
    EXPECT_EQ(simulation->getCurrentStep(), 10);
}

// Test setting time step
TEST_F(WeatherSimulationTest, SetTimeStep) {
    simulation->setInitialCondition(mock_initial);
    simulation->initialize();
    
    // Change time step
    simulation->setDt(0.2f);
    EXPECT_FLOAT_EQ(simulation->getDt(), 0.2f);
    
    // Run for 5 steps
    simulation->run(5);
    
    // Time should be 5 * 0.2 = 1.0
    EXPECT_FLOAT_EQ(simulation->getCurrentTime(), 1.0f);
}

// Test performance metrics
TEST_F(WeatherSimulationTest, PerformanceMetrics) {
    simulation->setInitialCondition(mock_initial);
    simulation->initialize();
    
    // Reset metrics
    simulation->resetPerformanceMetrics();
    
    // Run for 10 steps
    simulation->run(10);
    
    // Check that metrics were recorded
    const auto& metrics = simulation->getPerformanceMetrics();
    EXPECT_GT(metrics.compute_time_ms, 0.0);
    EXPECT_EQ(metrics.num_steps, 10);
}

} // namespace test
} // namespace weather_sim