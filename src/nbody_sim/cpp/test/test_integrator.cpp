// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include <gtest/gtest.h>
#include "nbody_sim/integrator.hpp"
#include "nbody_sim/particle.hpp"
#include <memory>
#include <cmath>

namespace nbody_sim {
namespace test {

// Helper function to create a simple two-body system for testing
std::unique_ptr<ParticleSystem> create_two_body_system() {
    std::vector<Particle> particles;
    particles.emplace_back(
        Vec3(1.0, 0.0, 0.0),    // position
        Vec3(0.0, 0.1, 0.0),    // velocity
        1.0,                    // mass
        0                       // id
    );
    particles.emplace_back(
        Vec3(-1.0, 0.0, 0.0),   // position
        Vec3(0.0, -0.1, 0.0),   // velocity
        1.0,                    // mass
        1                       // id
    );
    
    return std::make_unique<ParticleSystem>(particles, 1.0);
}

// Test the Euler integrator
TEST(IntegratorTest, EulerIntegrator) {
    auto system = create_two_body_system();
    EulerIntegrator integrator;
    
    // Get initial state
    scalar_t initial_energy = system->total_energy();
    
    // Take a step
    scalar_t dt = 0.01;
    integrator.step(*system, dt);
    
    // Basic checks to ensure the system has changed
    const Particle& p1 = system->particle(0);
    const Particle& p2 = system->particle(1);
    
    EXPECT_NE(p1.acceleration().x, 0.0);
    EXPECT_NE(p2.acceleration().x, 0.0);
    
    // Euler method has energy conservation errors
    // but energy should not explode for small timesteps
    scalar_t final_energy = system->total_energy();
    scalar_t relative_error = std::abs(final_energy - initial_energy) / std::abs(initial_energy);
    EXPECT_LT(relative_error, 0.01);
}

// Test the Leapfrog integrator
TEST(IntegratorTest, LeapfrogIntegrator) {
    auto system = create_two_body_system();
    LeapfrogIntegrator integrator;
    
    // Get initial state
    scalar_t initial_energy = system->total_energy();
    
    // Take a step
    scalar_t dt = 0.01;
    integrator.step(*system, dt);
    
    // Leapfrog should conserve energy better than Euler
    scalar_t final_energy = system->total_energy();
    scalar_t relative_error = std::abs(final_energy - initial_energy) / std::abs(initial_energy);
    EXPECT_LT(relative_error, 0.005);
}

// Test the Verlet integrator
TEST(IntegratorTest, VerletIntegrator) {
    auto system = create_two_body_system();
    VerletIntegrator integrator;
    
    // Get initial state
    scalar_t initial_energy = system->total_energy();
    
    // Initialize integrator
    integrator.initialize(*system);
    
    // Take a step
    scalar_t dt = 0.01;
    integrator.step(*system, dt);
    
    // Verlet should conserve energy well
    scalar_t final_energy = system->total_energy();
    scalar_t relative_error = std::abs(final_energy - initial_energy) / std::abs(initial_energy);
    EXPECT_LT(relative_error, 0.005);
}

// Test the Runge-Kutta 4 integrator
TEST(IntegratorTest, RungeKutta4Integrator) {
    auto system = create_two_body_system();
    RungeKutta4Integrator integrator;
    
    // Get initial state
    scalar_t initial_energy = system->total_energy();
    Vec3 initial_pos1 = system->particle(0).position();
    Vec3 initial_pos2 = system->particle(1).position();
    
    // Take a step
    scalar_t dt = 0.01;
    integrator.step(*system, dt);
    
    // Positions should have changed
    EXPECT_NE(system->particle(0).position().x, initial_pos1.x);
    EXPECT_NE(system->particle(1).position().x, initial_pos1.x);
    
    // RK4 should conserve energy well for small timesteps
    scalar_t final_energy = system->total_energy();
    scalar_t relative_error = std::abs(final_energy - initial_energy) / std::abs(initial_energy);
    EXPECT_LT(relative_error, 0.001);
}

// Test the energy conservation of different integrators
TEST(IntegratorTest, EnergyConservationComparison) {
    // Create systems for different integrators
    auto system_euler = create_two_body_system();
    auto system_leapfrog = create_two_body_system();
    auto system_verlet = create_two_body_system();
    auto system_rk4 = create_two_body_system();
    
    // Create integrators
    EulerIntegrator euler;
    LeapfrogIntegrator leapfrog;
    VerletIntegrator verlet;
    RungeKutta4Integrator rk4;
    
    // Initialize Verlet integrator
    verlet.initialize(*system_verlet);
    
    // Get initial energies
    scalar_t initial_energy_euler = system_euler->total_energy();
    scalar_t initial_energy_leapfrog = system_leapfrog->total_energy();
    scalar_t initial_energy_verlet = system_verlet->total_energy();
    scalar_t initial_energy_rk4 = system_rk4->total_energy();
    
    // Perform 10 steps with each integrator
    scalar_t dt = 0.01;
    for (int i = 0; i < 10; ++i) {
        euler.step(*system_euler, dt);
        leapfrog.step(*system_leapfrog, dt);
        verlet.step(*system_verlet, dt);
        rk4.step(*system_rk4, dt);
    }
    
    // Calculate relative energy errors
    scalar_t error_euler = std::abs(system_euler->total_energy() - initial_energy_euler) / std::abs(initial_energy_euler);
    scalar_t error_leapfrog = std::abs(system_leapfrog->total_energy() - initial_energy_leapfrog) / std::abs(initial_energy_leapfrog);
    scalar_t error_verlet = std::abs(system_verlet->total_energy() - initial_energy_verlet) / std::abs(initial_energy_verlet);
    scalar_t error_rk4 = std::abs(system_rk4->total_energy() - initial_energy_rk4) / std::abs(initial_energy_rk4);
    
    // Higher order methods should conserve energy better
    EXPECT_GT(error_euler, error_leapfrog);
    EXPECT_GT(error_leapfrog, error_rk4);
    
    // Print the errors for information
    std::cout << "Energy conservation errors after 10 steps:" << std::endl;
    std::cout << "  Euler: " << error_euler << std::endl;
    std::cout << "  Leapfrog: " << error_leapfrog << std::endl;
    std::cout << "  Verlet: " << error_verlet << std::endl;
    std::cout << "  RK4: " << error_rk4 << std::endl;
}

// Test the integrator factory
TEST(IntegratorTest, IntegratorFactory) {
    // Create integrators using factory method
    auto euler = Integrator::create(IntegrationType::Euler);
    auto leapfrog = Integrator::create(IntegrationType::Leapfrog);
    auto verlet = Integrator::create(IntegrationType::Verlet);
    auto rk4 = Integrator::create(IntegrationType::RungeKutta4);
    
    // Check types
    EXPECT_EQ(euler->type(), IntegrationType::Euler);
    EXPECT_EQ(leapfrog->type(), IntegrationType::Leapfrog);
    EXPECT_EQ(verlet->type(), IntegrationType::Verlet);
    EXPECT_EQ(rk4->type(), IntegrationType::RungeKutta4);
    
    // Check names
    EXPECT_EQ(euler->name(), "Euler");
    EXPECT_EQ(leapfrog->name(), "Leapfrog");
    EXPECT_EQ(verlet->name(), "Verlet");
    EXPECT_EQ(rk4->name(), "RK4");
    
    // Try to create with invalid type
    EXPECT_THROW(Integrator::create(static_cast<IntegrationType>(999)), std::invalid_argument);
}

// Test the cloning of integrators
TEST(IntegratorTest, CloneIntegrator) {
    // Create an integrator and clone it
    auto original = Integrator::create(IntegrationType::Leapfrog);
    auto clone = original->clone();
    
    // Check that the clone has the same type
    EXPECT_EQ(clone->type(), original->type());
    EXPECT_EQ(clone->name(), original->name());
}

} // namespace test
} // namespace nbody_sim