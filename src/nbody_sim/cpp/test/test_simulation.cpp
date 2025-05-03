// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include <gtest/gtest.h>
#include "nbody_sim/simulation.hpp"
#include <memory>
#include <cmath>
#include <chrono>
#include <fstream>
#include <thread>
#include <filesystem>

namespace nbody_sim {
namespace test {

// Test basic simulation construction
TEST(SimulationTest, Constructor) {
    // Default constructor
    Simulation sim1;
    EXPECT_EQ(sim1.current_time(), 0.0);
    EXPECT_EQ(sim1.current_step(), 0);
    EXPECT_EQ(sim1.dt(), DEFAULT_TIMESTEP);
    EXPECT_EQ(sim1.duration(), 10.0);
    EXPECT_EQ(sim1.system().size(), 0);
    
    // Custom constructor
    auto system = ParticleSystem::create_random_system(100, 10.0, 1.0, 0.1, 1.0, 42);
    auto integrator = Integrator::create(IntegrationType::Leapfrog);
    scalar_t dt = 0.005;
    scalar_t duration = 5.0;
    
    Simulation sim2(
        std::move(system),
        std::move(integrator),
        dt,
        duration
    );
    
    EXPECT_EQ(sim2.current_time(), 0.0);
    EXPECT_EQ(sim2.current_step(), 0);
    EXPECT_EQ(sim2.dt(), dt);
    EXPECT_EQ(sim2.duration(), duration);
    EXPECT_EQ(sim2.total_steps(), static_cast<index_t>(duration / dt));
    EXPECT_EQ(sim2.system().size(), 100);
}

// Test single step functionality
TEST(SimulationTest, SingleStep) {
    // Create a simple simulation
    auto system = ParticleSystem::create_random_system(10, 10.0, 1.0, 0.1, 1.0, 42);
    auto integrator = Integrator::create(IntegrationType::Euler);
    scalar_t dt = 0.01;
    
    Simulation sim(
        std::move(system),
        std::move(integrator),
        dt,
        1.0
    );
    
    // Take a single step
    sim.step();
    
    // Check that time and step count advanced
    EXPECT_EQ(sim.current_time(), dt);
    EXPECT_EQ(sim.current_step(), 1);
}

// Test full run functionality
TEST(SimulationTest, FullRun) {
    // Create a simple simulation with a small number of steps
    auto system = ParticleSystem::create_random_system(10, 10.0, 1.0, 0.1, 1.0, 42);
    auto integrator = Integrator::create(IntegrationType::Euler);
    scalar_t dt = 0.01;
    scalar_t duration = 0.1; // 10 steps
    
    Simulation sim(
        std::move(system),
        std::move(integrator),
        dt,
        duration
    );
    
    // Run the full simulation
    int callback_count = 0;
    sim.run([&callback_count](const Simulation& s) {
        callback_count++;
    }, 2); // Call every 2 steps
    
    // Check that simulation completed
    EXPECT_EQ(sim.current_time(), duration);
    EXPECT_EQ(sim.current_step(), 10);
    EXPECT_EQ(callback_count, 5); // 10 steps / 2 = 5 callbacks
}

// Test saving and loading simulation state
TEST(SimulationTest, SaveLoadState) {
    // Create a temporary file path
    std::string temp_filename = "sim_state_test.bin";
    
    // Create a simulation and run a few steps
    auto system = ParticleSystem::create_random_system(10, 10.0, 1.0, 0.1, 1.0, 42);
    auto integrator = Integrator::create(IntegrationType::Leapfrog);
    scalar_t dt = 0.01;
    scalar_t duration = 0.1;
    
    Simulation sim1(
        std::move(system),
        std::move(integrator),
        dt,
        duration
    );
    
    // Run for a few steps
    for (int i = 0; i < 5; i++) {
        sim1.step();
    }
    
    // Save state
    sim1.save_state(temp_filename);
    
    // Create a new simulation and load the state
    Simulation sim2;
    sim2.load_state(temp_filename);
    
    // Check that states match
    EXPECT_EQ(sim2.current_time(), sim1.current_time());
    EXPECT_EQ(sim2.current_step(), sim1.current_step());
    EXPECT_EQ(sim2.dt(), sim1.dt());
    EXPECT_EQ(sim2.duration(), sim1.duration());
    EXPECT_EQ(sim2.system().size(), sim1.system().size());
    
    // Cleanup
    std::remove(temp_filename.c_str());
}

// Test performance metrics
TEST(SimulationTest, PerformanceMetrics) {
    // Create a simple simulation
    auto system = ParticleSystem::create_random_system(10, 10.0, 1.0, 0.1, 1.0, 42);
    auto integrator = Integrator::create(IntegrationType::Euler);
    scalar_t dt = 0.01;
    scalar_t duration = 0.1;
    
    Simulation sim(
        std::move(system),
        std::move(integrator),
        dt,
        duration
    );
    
    // Run the simulation
    sim.run();
    
    // Get performance metrics
    auto metrics = sim.get_performance_metrics();
    
    // Check that all expected metrics are present
    EXPECT_TRUE(metrics.find("total_time_ms") != metrics.end());
    EXPECT_TRUE(metrics.find("steps_per_second") != metrics.end());
    EXPECT_TRUE(metrics.find("initial_energy") != metrics.end());
    EXPECT_TRUE(metrics.find("final_energy") != metrics.end());
    EXPECT_TRUE(metrics.find("energy_conservation_error") != metrics.end());
    EXPECT_TRUE(metrics.find("particle_count") != metrics.end());
    EXPECT_TRUE(metrics.find("current_time") != metrics.end());
    EXPECT_TRUE(metrics.find("current_step") != metrics.end());
    
    // Check that step count matches
    EXPECT_EQ(metrics["current_step"], static_cast<double>(sim.current_step()));
    EXPECT_EQ(metrics["particle_count"], 10.0);
}

// Test visualization data
TEST(SimulationTest, VisualizationData) {
    // Create a simple simulation
    auto system = ParticleSystem::create_random_system(10, 10.0, 1.0, 0.1, 1.0, 42);
    auto integrator = Integrator::create(IntegrationType::Euler);
    scalar_t dt = 0.01;
    
    Simulation sim(
        std::move(system),
        std::move(integrator),
        dt,
        1.0
    );
    
    // Generate visualization data without velocities
    auto data1 = sim.create_visualization_data(false);
    
    // Check that position data is present
    EXPECT_TRUE(data1.find("positions_x") != data1.end());
    EXPECT_TRUE(data1.find("positions_y") != data1.end());
    EXPECT_TRUE(data1.find("positions_z") != data1.end());
    EXPECT_TRUE(data1.find("masses") != data1.end());
    EXPECT_TRUE(data1.find("ids") != data1.end());
    EXPECT_TRUE(data1.find("metadata") != data1.end());
    
    // Velocity data should not be present
    EXPECT_TRUE(data1.find("velocities_x") == data1.end());
    
    // Generate visualization data with velocities
    auto data2 = sim.create_visualization_data(true);
    
    // Check that velocity data is now present
    EXPECT_TRUE(data2.find("velocities_x") != data2.end());
    EXPECT_TRUE(data2.find("velocities_y") != data2.end());
    EXPECT_TRUE(data2.find("velocities_z") != data2.end());
    
    // Check array sizes
    EXPECT_EQ(data2["positions_x"].size(), 10);
    EXPECT_EQ(data2["velocities_x"].size(), 10);
    EXPECT_EQ(data2["masses"].size(), 10);
}

// Test simulation creation methods
TEST(SimulationTest, CreationMethods) {
    // Test random simulation creation
    auto random_sim = Simulation::create_random_simulation(100, IntegrationType::Euler, 0.01, 1.0, 42);
    EXPECT_EQ(random_sim->system().size(), 100);
    EXPECT_EQ(random_sim->dt(), 0.01);
    EXPECT_EQ(random_sim->duration(), 1.0);
    EXPECT_EQ(random_sim->integrator().type(), IntegrationType::Euler);
    
    // Test solar system creation
    auto solar_sim = Simulation::create_solar_system_simulation(IntegrationType::Leapfrog, 0.01, 1.0, 2.0);
    EXPECT_GT(solar_sim->system().size(), 1); // At least Sun + planets
    EXPECT_EQ(solar_sim->dt(), 0.01);
    EXPECT_EQ(solar_sim->duration(), 1.0);
    EXPECT_EQ(solar_sim->integrator().type(), IntegrationType::Leapfrog);
    
    // Test galaxy creation
    auto galaxy_sim = Simulation::create_galaxy_simulation(100, IntegrationType::Verlet, 0.01, 1.0, 42);
    EXPECT_EQ(galaxy_sim->system().size(), 100);
    EXPECT_EQ(galaxy_sim->dt(), 0.01);
    EXPECT_EQ(galaxy_sim->duration(), 1.0);
    EXPECT_EQ(galaxy_sim->integrator().type(), IntegrationType::Verlet);
}

// Test energy conservation in simulation
TEST(SimulationTest, EnergyConservation) {
    // Create a simple two-body system
    std::vector<Particle> particles;
    particles.emplace_back(
        Vec3(1.0, 0.0, 0.0),
        Vec3(0.0, 0.5, 0.0),
        1.0,
        0
    );
    particles.emplace_back(
        Vec3(-1.0, 0.0, 0.0),
        Vec3(0.0, -0.5, 0.0),
        1.0,
        1
    );
    
    auto system = std::make_unique<ParticleSystem>(particles, 1.0);
    auto integrator = Integrator::create(IntegrationType::Leapfrog);
    
    Simulation sim(
        std::move(system),
        std::move(integrator),
        0.01,
        1.0
    );
    
    // Get initial energy
    scalar_t initial_energy = sim.system().total_energy();
    
    // Run for 100 steps
    for (int i = 0; i < 100; i++) {
        sim.step();
    }
    
    // Check energy conservation
    scalar_t final_energy = sim.system().total_energy();
    scalar_t relative_error = std::abs(final_energy - initial_energy) / std::abs(initial_energy);
    
    // Leapfrog should conserve energy well
    EXPECT_LT(relative_error, 0.01);
    
    // Check that particles have moved in orbit
    const Particle& p1 = sim.system().particle(0);
    const Particle& p2 = sim.system().particle(1);
    
    // Both particles should have moved
    EXPECT_NE(p1.position().x, 1.0);
    EXPECT_NE(p1.position().y, 0.0);
    EXPECT_NE(p2.position().x, -1.0);
    EXPECT_NE(p2.position().y, 0.0);
}

} // namespace test
} // namespace nbody_sim