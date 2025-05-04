// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include <gtest/gtest.h>
#include "nbody_sim/particle.hpp"
#include <cmath>
#include <memory>

namespace nbody_sim {
namespace test {

// Test the Vec3 class
TEST(Vec3Test, BasicOperations) {
    Vec3 v1(1.0, 2.0, 3.0);
    Vec3 v2(4.0, 5.0, 6.0);
    
    // Test addition
    Vec3 sum = v1 + v2;
    EXPECT_DOUBLE_EQ(sum.x, 5.0);
    EXPECT_DOUBLE_EQ(sum.y, 7.0);
    EXPECT_DOUBLE_EQ(sum.z, 9.0);
    
    // Test subtraction
    Vec3 diff = v2 - v1;
    EXPECT_DOUBLE_EQ(diff.x, 3.0);
    EXPECT_DOUBLE_EQ(diff.y, 3.0);
    EXPECT_DOUBLE_EQ(diff.z, 3.0);
    
    // Test scalar multiplication
    Vec3 prod = v1 * 2.0;
    EXPECT_DOUBLE_EQ(prod.x, 2.0);
    EXPECT_DOUBLE_EQ(prod.y, 4.0);
    EXPECT_DOUBLE_EQ(prod.z, 6.0);
    
    // Test scalar division
    Vec3 quot = v1 / 2.0;
    EXPECT_DOUBLE_EQ(quot.x, 0.5);
    EXPECT_DOUBLE_EQ(quot.y, 1.0);
    EXPECT_DOUBLE_EQ(quot.z, 1.5);
}

TEST(Vec3Test, DotProduct) {
    Vec3 v1(1.0, 2.0, 3.0);
    Vec3 v2(4.0, 5.0, 6.0);
    
    scalar_t dot = v1.dot(v2);
    EXPECT_DOUBLE_EQ(dot, 32.0);  // 1*4 + 2*5 + 3*6
}

TEST(Vec3Test, CrossProduct) {
    Vec3 v1(1.0, 0.0, 0.0);
    Vec3 v2(0.0, 1.0, 0.0);
    
    Vec3 cross = v1.cross(v2);
    EXPECT_DOUBLE_EQ(cross.x, 0.0);
    EXPECT_DOUBLE_EQ(cross.y, 0.0);
    EXPECT_DOUBLE_EQ(cross.z, 1.0);
}

TEST(Vec3Test, Length) {
    Vec3 v(3.0, 4.0, 0.0);
    
    scalar_t length = v.length();
    EXPECT_DOUBLE_EQ(length, 5.0);
}

TEST(Vec3Test, Normalize) {
    Vec3 v(3.0, 4.0, 0.0);
    Vec3 normalized = v.normalized();
    
    EXPECT_DOUBLE_EQ(normalized.x, 0.6);
    EXPECT_DOUBLE_EQ(normalized.y, 0.8);
    EXPECT_DOUBLE_EQ(normalized.z, 0.0);
    EXPECT_DOUBLE_EQ(normalized.length(), 1.0);
}

// Test the Particle class
TEST(ParticleTest, Initialization) {
    Particle p(
        Vec3(1.0, 2.0, 3.0),
        Vec3(4.0, 5.0, 6.0),
        7.0,
        42
    );
    
    EXPECT_DOUBLE_EQ(p.position().x, 1.0);
    EXPECT_DOUBLE_EQ(p.position().y, 2.0);
    EXPECT_DOUBLE_EQ(p.position().z, 3.0);
    
    EXPECT_DOUBLE_EQ(p.velocity().x, 4.0);
    EXPECT_DOUBLE_EQ(p.velocity().y, 5.0);
    EXPECT_DOUBLE_EQ(p.velocity().z, 6.0);
    
    EXPECT_DOUBLE_EQ(p.mass(), 7.0);
    EXPECT_EQ(p.id(), 42);
    
    // Acceleration should be initialized to zero
    EXPECT_DOUBLE_EQ(p.acceleration().x, 0.0);
    EXPECT_DOUBLE_EQ(p.acceleration().y, 0.0);
    EXPECT_DOUBLE_EQ(p.acceleration().z, 0.0);
}

TEST(ParticleTest, UpdatePosition) {
    Particle p(
        Vec3(1.0, 2.0, 3.0),
        Vec3(4.0, 5.0, 6.0),
        7.0,
        42
    );
    
    scalar_t dt = 0.1;
    p.update_position(dt);
    
    // New position = old position + velocity * dt
    EXPECT_DOUBLE_EQ(p.position().x, 1.0 + 4.0 * dt);
    EXPECT_DOUBLE_EQ(p.position().y, 2.0 + 5.0 * dt);
    EXPECT_DOUBLE_EQ(p.position().z, 3.0 + 6.0 * dt);
}

TEST(ParticleTest, UpdateVelocity) {
    Particle p(
        Vec3(1.0, 2.0, 3.0),
        Vec3(4.0, 5.0, 6.0),
        7.0,
        42
    );
    
    // Set acceleration
    p.set_acceleration(Vec3(0.1, 0.2, 0.3));
    
    scalar_t dt = 0.1;
    p.update_velocity(dt);
    
    // New velocity = old velocity + acceleration * dt
    EXPECT_DOUBLE_EQ(p.velocity().x, 4.0 + 0.1 * dt);
    EXPECT_DOUBLE_EQ(p.velocity().y, 5.0 + 0.2 * dt);
    EXPECT_DOUBLE_EQ(p.velocity().z, 6.0 + 0.3 * dt);
}

TEST(ParticleTest, KineticEnergy) {
    Particle p(
        Vec3(1.0, 2.0, 3.0),
        Vec3(4.0, 5.0, 6.0),
        2.0,
        42
    );
    
    // KE = 0.5 * m * v^2
    scalar_t expected_ke = 0.5 * 2.0 * (4.0*4.0 + 5.0*5.0 + 6.0*6.0);
    scalar_t ke = p.kinetic_energy();
    
    EXPECT_DOUBLE_EQ(ke, expected_ke);
}

// Test the ParticleSystem class
TEST(ParticleSystemTest, Initialization) {
    // Create some particles
    std::vector<Particle> particles;
    for (int i = 0; i < 10; ++i) {
        particles.emplace_back(
            Vec3(static_cast<scalar_t>(i), 0.0, 0.0),
            Vec3(0.0, static_cast<scalar_t>(i), 0.0),
            static_cast<scalar_t>(i + 1),
            i
        );
    }
    
    // Create particle system
    ParticleSystem system(particles);
    
    // Check particle count
    EXPECT_EQ(system.size(), 10);
    
    // Check gravitational constant
    EXPECT_DOUBLE_EQ(system.gravitational_constant(), DEFAULT_G);
    
    // Check particles
    for (size_t i = 0; i < system.size(); ++i) {
        const Particle& p = system.particle(i);
        EXPECT_DOUBLE_EQ(p.position().x, static_cast<scalar_t>(i));
        EXPECT_DOUBLE_EQ(p.velocity().y, static_cast<scalar_t>(i));
        EXPECT_DOUBLE_EQ(p.mass(), static_cast<scalar_t>(i + 1));
        EXPECT_EQ(p.id(), static_cast<index_t>(i));
    }
}

TEST(ParticleSystemTest, AddParticle) {
    // Create empty system
    ParticleSystem system;
    
    // Add a particle
    system.add_particle(Particle(
        Vec3(1.0, 2.0, 3.0),
        Vec3(4.0, 5.0, 6.0),
        7.0,
        42
    ));
    
    // Check particle count
    EXPECT_EQ(system.size(), 1);
    
    // Check particle data
    const Particle& p = system.particle(0);
    EXPECT_DOUBLE_EQ(p.position().x, 1.0);
    EXPECT_DOUBLE_EQ(p.position().y, 2.0);
    EXPECT_DOUBLE_EQ(p.position().z, 3.0);
    EXPECT_DOUBLE_EQ(p.velocity().x, 4.0);
    EXPECT_DOUBLE_EQ(p.velocity().y, 5.0);
    EXPECT_DOUBLE_EQ(p.velocity().z, 6.0);
    EXPECT_DOUBLE_EQ(p.mass(), 7.0);
    EXPECT_EQ(p.id(), 42);
}

TEST(ParticleSystemTest, TotalMass) {
    // Create particles with different masses
    std::vector<Particle> particles;
    scalar_t total_mass = 0.0;
    
    for (int i = 0; i < 10; ++i) {
        scalar_t mass = static_cast<scalar_t>(i + 1);
        particles.emplace_back(
            Vec3(), Vec3(), mass, i
        );
        total_mass += mass;
    }
    
    // Create particle system
    ParticleSystem system(particles);
    
    // Check total mass
    EXPECT_DOUBLE_EQ(system.total_mass(), total_mass);
}

TEST(ParticleSystemTest, CenterOfMass) {
    // Create two particles with equal mass, opposite positions
    std::vector<Particle> particles;
    particles.emplace_back(
        Vec3(1.0, 0.0, 0.0),  // position
        Vec3(),               // velocity
        1.0,                  // mass
        0                     // id
    );
    particles.emplace_back(
        Vec3(-1.0, 0.0, 0.0), // position
        Vec3(),               // velocity
        1.0,                  // mass
        1                     // id
    );
    
    // Create particle system
    ParticleSystem system(particles);
    
    // Center of mass should be at the origin
    Vec3 com = system.center_of_mass();
    EXPECT_DOUBLE_EQ(com.x, 0.0);
    EXPECT_DOUBLE_EQ(com.y, 0.0);
    EXPECT_DOUBLE_EQ(com.z, 0.0);
    
    // Change the mass of the first particle
    particles[0].set_mass(2.0);
    
    // Create new system with updated particles
    ParticleSystem system2(particles);
    
    // Center of mass should now be at (1/3, 0, 0)
    com = system2.center_of_mass();
    EXPECT_DOUBLE_EQ(com.x, 1.0/3.0);
    EXPECT_DOUBLE_EQ(com.y, 0.0);
    EXPECT_DOUBLE_EQ(com.z, 0.0);
}

TEST(ParticleSystemTest, TotalEnergy) {
    // Create two particles
    std::vector<Particle> particles;
    particles.emplace_back(
        Vec3(1.0, 0.0, 0.0),    // position
        Vec3(0.0, 1.0, 0.0),    // velocity
        1.0,                    // mass
        0                       // id
    );
    particles.emplace_back(
        Vec3(-1.0, 0.0, 0.0),   // position
        Vec3(0.0, -1.0, 0.0),   // velocity
        1.0,                    // mass
        1                       // id
    );
    
    // Create particle system with G = 1.0
    ParticleSystem system(particles, 1.0);
    
    // Calculate expected energies
    // Kinetic energy = 0.5 * m * v^2 for each particle
    scalar_t ke1 = 0.5 * 1.0 * 1.0;  // 0.5 * m * v^2
    scalar_t ke2 = 0.5 * 1.0 * 1.0;
    scalar_t total_ke = ke1 + ke2;
    
    // Potential energy = -G * m1 * m2 / r
    // Distance between particles is 2.0
    scalar_t pe = -1.0 * 1.0 * 1.0 / 2.0;
    
    scalar_t expected_total_energy = total_ke + pe;
    
    // Check energies
    EXPECT_DOUBLE_EQ(system.total_kinetic_energy(), total_ke);
    EXPECT_DOUBLE_EQ(system.total_potential_energy(), pe);
    EXPECT_DOUBLE_EQ(system.total_energy(), expected_total_energy);
}

TEST(ParticleSystemTest, CloneFunctionality) {
    // Create original particle system
    auto original = ParticleSystem::create_random_system(
        100,    // num_particles
        10.0,   // box_size
        1.0,    // max_mass
        0.1,    // max_velocity
        2.0,    // G
        42      // seed
    );
    
    // Clone the system
    auto clone = original->clone();
    
    // Check that the clone has the same properties
    EXPECT_EQ(clone->size(), original->size());
    EXPECT_DOUBLE_EQ(clone->gravitational_constant(), original->gravitational_constant());
    
    // Check all particles
    for (size_t i = 0; i < original->size(); ++i) {
        const Particle& p_orig = original->particle(i);
        const Particle& p_clone = clone->particle(i);
        
        EXPECT_DOUBLE_EQ(p_clone.position().x, p_orig.position().x);
        EXPECT_DOUBLE_EQ(p_clone.position().y, p_orig.position().y);
        EXPECT_DOUBLE_EQ(p_clone.position().z, p_orig.position().z);
        
        EXPECT_DOUBLE_EQ(p_clone.velocity().x, p_orig.velocity().x);
        EXPECT_DOUBLE_EQ(p_clone.velocity().y, p_orig.velocity().y);
        EXPECT_DOUBLE_EQ(p_clone.velocity().z, p_orig.velocity().z);
        
        EXPECT_DOUBLE_EQ(p_clone.mass(), p_orig.mass());
        EXPECT_EQ(p_clone.id(), p_orig.id());
    }
}

TEST(ParticleSystemTest, CreateRandomSystem) {
    size_t num_particles = 1000;
    scalar_t box_size = 10.0;
    scalar_t max_mass = 2.0;
    scalar_t G = 3.0;
    
    // Create random system
    auto system = ParticleSystem::create_random_system(
        num_particles,
        box_size,
        max_mass,
        0.1,
        G,
        42  // seed
    );
    
    // Check system properties
    EXPECT_EQ(system->size(), num_particles);
    EXPECT_DOUBLE_EQ(system->gravitational_constant(), G);
    
    // Check that particles are within bounds
    for (size_t i = 0; i < system->size(); ++i) {
        const Particle& p = system->particle(i);
        
        // Check position bounds
        EXPECT_LE(std::abs(p.position().x), box_size);
        EXPECT_LE(std::abs(p.position().y), box_size);
        EXPECT_LE(std::abs(p.position().z), box_size);
        
        // Check mass bounds
        EXPECT_GT(p.mass(), 0.0);
        EXPECT_LE(p.mass(), max_mass);
    }
}

TEST(ParticleSystemTest, CreateSolarSystem) {
    scalar_t scale_factor = 2.0;
    
    // Create solar system
    auto system = ParticleSystem::create_solar_system(scale_factor);
    
    // Check system properties
    EXPECT_GT(system->size(), 1);  // At least Sun + planets
    
    // Check Sun properties (first particle)
    const Particle& sun = system->particle(0);
    EXPECT_DOUBLE_EQ(sun.position().x, 0.0);
    EXPECT_DOUBLE_EQ(sun.position().y, 0.0);
    EXPECT_DOUBLE_EQ(sun.position().z, 0.0);
    EXPECT_DOUBLE_EQ(sun.velocity().x, 0.0);
    EXPECT_DOUBLE_EQ(sun.velocity().y, 0.0);
    EXPECT_DOUBLE_EQ(sun.velocity().z, 0.0);
    EXPECT_DOUBLE_EQ(sun.mass(), 1.0);  // 1 solar mass
    
    // Check Earth properties (should be the third planet from the Sun)
    const Particle& earth = system->particle(3);
    EXPECT_DOUBLE_EQ(earth.position().length(), 1.0 * scale_factor);
}

TEST(ParticleSystemTest, CreateGalaxyModel) {
    size_t num_particles = 1000;
    scalar_t radius = 10.0;
    scalar_t height = 1.0;
    scalar_t min_mass = 0.1;
    scalar_t max_mass = 2.0;
    scalar_t G = 1.0;
    
    // Create galaxy model
    auto system = ParticleSystem::create_galaxy_model(
        num_particles,
        radius,
        height,
        min_mass,
        max_mass,
        G,
        42  // seed
    );
    
    // Check system properties
    EXPECT_EQ(system->size(), num_particles);
    EXPECT_DOUBLE_EQ(system->gravitational_constant(), G);
    
    // Check central black hole properties (first particle)
    const Particle& black_hole = system->particle(0);
    EXPECT_DOUBLE_EQ(black_hole.position().x, 0.0);
    EXPECT_DOUBLE_EQ(black_hole.position().y, 0.0);
    EXPECT_DOUBLE_EQ(black_hole.position().z, 0.0);
    EXPECT_DOUBLE_EQ(black_hole.velocity().x, 0.0);
    EXPECT_DOUBLE_EQ(black_hole.velocity().y, 0.0);
    EXPECT_DOUBLE_EQ(black_hole.velocity().z, 0.0);
    EXPECT_GT(black_hole.mass(), max_mass);  // Should be more massive than other particles
    
    // Check that particles are within bounds
    for (size_t i = 1; i < system->size(); ++i) {
        const Particle& p = system->particle(i);
        
        // Check radial distance (xy-plane)
        scalar_t r = std::sqrt(p.position().x * p.position().x + p.position().y * p.position().y);
        EXPECT_LE(r, radius);
        
        // Check height
        EXPECT_LE(std::abs(p.position().z), height * 5.0);  // Allow some outliers
        
        // Check mass bounds
        EXPECT_GE(p.mass(), min_mass);
        EXPECT_LE(p.mass(), max_mass);
    }
}

TEST(ParticleSystemTest, UpdateAccelerationsCPU) {
    // Create two particles
    std::vector<Particle> particles;
    particles.emplace_back(
        Vec3(1.0, 0.0, 0.0),    // position
        Vec3(0.0, 0.0, 0.0),    // velocity
        1.0,                    // mass
        0                       // id
    );
    particles.emplace_back(
        Vec3(-1.0, 0.0, 0.0),   // position
        Vec3(0.0, 0.0, 0.0),    // velocity
        1.0,                    // mass
        1                       // id
    );
    
    // Create particle system with G = 1.0
    ParticleSystem system(particles, 1.0);
    
    // Update accelerations
    system.update_accelerations_cpu();
    
    // Check accelerations
    // Each particle should accelerate towards the other
    // with magnitude G * m / r^2
    scalar_t expected_acc = 1.0 * 1.0 / (2.0 * 2.0);
    
    const Particle& p1 = system.particle(0);
    EXPECT_DOUBLE_EQ(p1.acceleration().x, -expected_acc);
    EXPECT_DOUBLE_EQ(p1.acceleration().y, 0.0);
    EXPECT_DOUBLE_EQ(p1.acceleration().z, 0.0);
    
    const Particle& p2 = system.particle(1);
    EXPECT_DOUBLE_EQ(p2.acceleration().x, expected_acc);
    EXPECT_DOUBLE_EQ(p2.acceleration().y, 0.0);
    EXPECT_DOUBLE_EQ(p2.acceleration().z, 0.0);
}

// Add more tests as needed...

} // namespace test
} // namespace nbody_sim