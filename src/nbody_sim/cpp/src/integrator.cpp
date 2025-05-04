// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "nbody_sim/integrator.hpp"
#include "nbody_sim/cuda_kernels.cuh"
#include <vector>
#include <stdexcept>

namespace nbody_sim {

std::unique_ptr<Integrator> Integrator::create(IntegrationType type) {
    switch (type) {
        case IntegrationType::Euler:
            return std::make_unique<EulerIntegrator>();
        case IntegrationType::Leapfrog:
            return std::make_unique<LeapfrogIntegrator>();
        case IntegrationType::Verlet:
            return std::make_unique<VerletIntegrator>();
        case IntegrationType::RungeKutta4:
            return std::make_unique<RungeKutta4Integrator>();
        default:
            throw std::invalid_argument("Unknown integrator type");
    }
}

// EulerIntegrator implementation

void EulerIntegrator::step(ParticleSystem& system, scalar_t dt) {
    // Update accelerations based on current positions
    system.update_accelerations();
    
    // Get particle count
    const size_t n = system.size();
    
    // Check if GPU memory is initialized and system is large enough for GPU
    bool use_gpu = false;
    if (n > 100) {
        try {
            // Check if the system has GPU memory initialized
            if (system.particles().size() > 0 && 
                system.particles()[0].acceleration().x == 0.0 &&
                system.particles()[0].acceleration().y == 0.0 &&
                system.particles()[0].acceleration().z == 0.0) {
                // Force accelerations update to initialize GPU
                system.update_accelerations();
            }
            use_gpu = true;
        } catch (const std::runtime_error&) {
            use_gpu = false;
        }
    }
    
    if (use_gpu) {
        // GPU implementation: update velocities and positions with CUDA kernels
        // This requires the system to have GPU memory already initialized
        
        // Get device pointers from the system (defined in the implementation)
        // Note: These methods are not defined in the header to keep the interface clean
        // They will be defined in the ParticleSystem implementation as private methods
        
        // Here we would launch the CUDA kernels to update velocities and positions
        // For now, we'll fall back to CPU implementation
        
        // For each particle, update velocity and position
        for (size_t i = 0; i < n; ++i) {
            Particle& p = system.particle(i);
            p.update_velocity(dt);
            p.update_position(dt);
        }
    } else {
        // CPU implementation: update velocities and positions
        for (size_t i = 0; i < n; ++i) {
            Particle& p = system.particle(i);
            p.update_velocity(dt);
            p.update_position(dt);
        }
    }
}

// LeapfrogIntegrator implementation

void LeapfrogIntegrator::step(ParticleSystem& system, scalar_t dt) {
    // Get particle count
    const size_t n = system.size();
    
    // First half-step: update positions based on current velocities
    for (size_t i = 0; i < n; ++i) {
        Particle& p = system.particle(i);
        p.set_position(p.position() + p.velocity() * (dt * 0.5));
    }
    
    // Update accelerations based on new positions
    system.update_accelerations();
    
    // Update velocities based on new accelerations
    for (size_t i = 0; i < n; ++i) {
        Particle& p = system.particle(i);
        p.update_velocity(dt);
    }
    
    // Second half-step: update positions based on new velocities
    for (size_t i = 0; i < n; ++i) {
        Particle& p = system.particle(i);
        p.set_position(p.position() + p.velocity() * (dt * 0.5));
    }
}

// VerletIntegrator implementation

void VerletIntegrator::initialize(ParticleSystem& system) {
    // Compute initial accelerations
    system.update_accelerations();
    initialized_ = true;
}

void VerletIntegrator::step(ParticleSystem& system, scalar_t dt) {
    if (!initialized_) {
        initialize(system);
    }
    
    // Get particle count
    const size_t n = system.size();
    
    // Store current accelerations
    std::vector<Vec3> old_accelerations(n);
    for (size_t i = 0; i < n; ++i) {
        old_accelerations[i] = system.particle(i).acceleration();
    }
    
    // Update positions using current velocities and accelerations
    for (size_t i = 0; i < n; ++i) {
        Particle& p = system.particle(i);
        p.set_position(p.position() + p.velocity() * dt + old_accelerations[i] * (0.5 * dt * dt));
    }
    
    // Update accelerations based on new positions
    system.update_accelerations();
    
    // Update velocities using average of old and new accelerations
    for (size_t i = 0; i < n; ++i) {
        Particle& p = system.particle(i);
        p.set_velocity(p.velocity() + (old_accelerations[i] + p.acceleration()) * (0.5 * dt));
    }
}

// RungeKutta4Integrator implementation

void RungeKutta4Integrator::step(ParticleSystem& system, scalar_t dt) {
    // Get particle count
    const size_t n = system.size();
    
    // Store initial state
    std::vector<Vec3> initial_positions(n);
    std::vector<Vec3> initial_velocities(n);
    
    for (size_t i = 0; i < n; ++i) {
        const Particle& p = system.particle(i);
        initial_positions[i] = p.position();
        initial_velocities[i] = p.velocity();
    }
    
    // Stage 1: Evaluate derivatives at the initial point
    system.update_accelerations();
    
    std::vector<Vec3> k1_vel(n);
    std::vector<Vec3> k1_pos(n);
    
    for (size_t i = 0; i < n; ++i) {
        const Particle& p = system.particle(i);
        k1_vel[i] = p.acceleration();
        k1_pos[i] = p.velocity();
    }
    
    // Stage 2: Evaluate derivatives at t + dt/2 using k1
    for (size_t i = 0; i < n; ++i) {
        Particle& p = system.particle(i);
        p.set_position(initial_positions[i] + k1_pos[i] * (dt * 0.5));
        p.set_velocity(initial_velocities[i] + k1_vel[i] * (dt * 0.5));
    }
    
    system.update_accelerations();
    
    std::vector<Vec3> k2_vel(n);
    std::vector<Vec3> k2_pos(n);
    
    for (size_t i = 0; i < n; ++i) {
        const Particle& p = system.particle(i);
        k2_vel[i] = p.acceleration();
        k2_pos[i] = p.velocity();
    }
    
    // Stage 3: Evaluate derivatives at t + dt/2 using k2
    for (size_t i = 0; i < n; ++i) {
        Particle& p = system.particle(i);
        p.set_position(initial_positions[i] + k2_pos[i] * (dt * 0.5));
        p.set_velocity(initial_velocities[i] + k2_vel[i] * (dt * 0.5));
    }
    
    system.update_accelerations();
    
    std::vector<Vec3> k3_vel(n);
    std::vector<Vec3> k3_pos(n);
    
    for (size_t i = 0; i < n; ++i) {
        const Particle& p = system.particle(i);
        k3_vel[i] = p.acceleration();
        k3_pos[i] = p.velocity();
    }
    
    // Stage 4: Evaluate derivatives at t + dt using k3
    for (size_t i = 0; i < n; ++i) {
        Particle& p = system.particle(i);
        p.set_position(initial_positions[i] + k3_pos[i] * dt);
        p.set_velocity(initial_velocities[i] + k3_vel[i] * dt);
    }
    
    system.update_accelerations();
    
    std::vector<Vec3> k4_vel(n);
    std::vector<Vec3> k4_pos(n);
    
    for (size_t i = 0; i < n; ++i) {
        const Particle& p = system.particle(i);
        k4_vel[i] = p.acceleration();
        k4_pos[i] = p.velocity();
    }
    
    // Final update: Combine all stages with weights
    for (size_t i = 0; i < n; ++i) {
        Particle& p = system.particle(i);
        
        // Update position: y_n+1 = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        p.set_position(initial_positions[i] + (k1_pos[i] + k2_pos[i] * 2.0 + k3_pos[i] * 2.0 + k4_pos[i]) * (dt / 6.0));
        
        // Update velocity: v_n+1 = v_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        p.set_velocity(initial_velocities[i] + (k1_vel[i] + k2_vel[i] * 2.0 + k3_vel[i] * 2.0 + k4_vel[i]) * (dt / 6.0));
    }
    
    // Update accelerations for the final state
    system.update_accelerations();
}

} // namespace nbody_sim