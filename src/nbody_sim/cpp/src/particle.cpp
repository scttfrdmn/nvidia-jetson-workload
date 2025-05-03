// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "nbody_sim/particle.hpp"
#include "nbody_sim/cuda_kernels.cuh"
#include <random>
#include <algorithm>
#include <cmath>

namespace nbody_sim {

// Particle implementation

Particle::Particle() 
    : position_(0, 0, 0), 
      velocity_(0, 0, 0), 
      acceleration_(0, 0, 0), 
      mass_(1.0), 
      id_(0) {}

Particle::Particle(const Vec3& position, const Vec3& velocity, scalar_t mass, index_t id)
    : position_(position), 
      velocity_(velocity), 
      acceleration_(0, 0, 0), 
      mass_(mass), 
      id_(id) {}

void Particle::update_position(scalar_t dt) {
    position_ += velocity_ * dt;
}

void Particle::update_velocity(scalar_t dt) {
    velocity_ += acceleration_ * dt;
}

scalar_t Particle::kinetic_energy() const {
    return 0.5 * mass_ * velocity_.length_squared();
}

// ParticleSystem implementation

ParticleSystem::ParticleSystem() : G_(DEFAULT_G), gpu_initialized_(false) {}

ParticleSystem::ParticleSystem(scalar_t G) : G_(G), gpu_initialized_(false) {}

ParticleSystem::ParticleSystem(const std::vector<Particle>& particles, scalar_t G)
    : particles_(particles), G_(G), gpu_initialized_(false) {}

void ParticleSystem::add_particle(const Particle& particle) {
    particles_.push_back(particle);
    // If GPU memory is already initialized, we need to reinitialize it
    if (gpu_initialized_) {
        free_gpu_memory();
        gpu_initialized_ = false;
    }
}

const Particle& ParticleSystem::particle(size_t index) const {
    return particles_.at(index);
}

Particle& ParticleSystem::particle(size_t index) {
    return particles_.at(index);
}

void ParticleSystem::update_accelerations_cpu() {
    const size_t n = particles_.size();
    
    // Reset all accelerations to zero
    for (auto& p : particles_) {
        p.set_acceleration(Vec3(0, 0, 0));
    }
    
    // Compute pairwise gravitational interactions
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            
            const Vec3& pos_i = particles_[i].position();
            const Vec3& pos_j = particles_[j].position();
            const scalar_t mass_j = particles_[j].mass();
            
            // Displacement vector
            Vec3 r = pos_j - pos_i;
            
            // Distance squared
            scalar_t r_squared = r.length_squared();
            
            // Add softening to prevent singularities
            r_squared += DEFAULT_SOFTENING * DEFAULT_SOFTENING;
            
            // Inverse distance cubed
            scalar_t inv_r_cubed = 1.0 / (sqrt(r_squared) * r_squared);
            
            // Accumulate acceleration: a = G * m * r / |r|^3
            Vec3 acceleration = r * (G_ * mass_j * inv_r_cubed);
            
            // Update particle acceleration
            particles_[i].set_acceleration(particles_[i].acceleration() + acceleration);
        }
    }
}

void ParticleSystem::initialize_gpu_memory() {
    if (gpu_initialized_) {
        return;
    }
    
    const size_t n = particles_.size();
    if (n == 0) {
        return;
    }
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_positions_, n * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_velocities_, n * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_accelerations_, n * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_masses_, n * sizeof(scalar_t)));
    
    // Mark as initialized
    gpu_initialized_ = true;
    
    // Copy data to device
    copy_to_device();
}

void ParticleSystem::free_gpu_memory() {
    if (!gpu_initialized_) {
        return;
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_positions_));
    CUDA_CHECK(cudaFree(d_velocities_));
    CUDA_CHECK(cudaFree(d_accelerations_));
    CUDA_CHECK(cudaFree(d_masses_));
    
    // Reset pointers
    d_positions_ = nullptr;
    d_velocities_ = nullptr;
    d_accelerations_ = nullptr;
    d_masses_ = nullptr;
    
    // Mark as not initialized
    gpu_initialized_ = false;
}

void ParticleSystem::copy_to_device() {
    if (!gpu_initialized_) {
        initialize_gpu_memory();
    }
    
    const size_t n = particles_.size();
    if (n == 0) {
        return;
    }
    
    // Create host arrays for positions, velocities, and masses
    std::vector<Vec3> h_positions(n);
    std::vector<Vec3> h_velocities(n);
    std::vector<scalar_t> h_masses(n);
    
    for (size_t i = 0; i < n; ++i) {
        h_positions[i] = particles_[i].position();
        h_velocities[i] = particles_[i].velocity();
        h_masses[i] = particles_[i].mass();
    }
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_positions_, h_positions.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velocities_, h_velocities.data(), n * sizeof(Vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_masses_, h_masses.data(), n * sizeof(scalar_t), cudaMemcpyHostToDevice));
    
    // Initialize accelerations to zero
    CUDA_CHECK(cudaMemset(d_accelerations_, 0, n * sizeof(Vec3)));
}

void ParticleSystem::copy_from_device() {
    if (!gpu_initialized_) {
        return;
    }
    
    const size_t n = particles_.size();
    if (n == 0) {
        return;
    }
    
    // Create host arrays for positions, velocities, and accelerations
    std::vector<Vec3> h_positions(n);
    std::vector<Vec3> h_velocities(n);
    std::vector<Vec3> h_accelerations(n);
    
    // Copy data from device
    CUDA_CHECK(cudaMemcpy(h_positions.data(), d_positions_, n * sizeof(Vec3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velocities.data(), d_velocities_, n * sizeof(Vec3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_accelerations.data(), d_accelerations_, n * sizeof(Vec3), cudaMemcpyDeviceToHost));
    
    // Update particle data
    for (size_t i = 0; i < n; ++i) {
        particles_[i].set_position(h_positions[i]);
        particles_[i].set_velocity(h_velocities[i]);
        particles_[i].set_acceleration(h_accelerations[i]);
    }
}

void ParticleSystem::update_accelerations_gpu() {
    const size_t n = particles_.size();
    if (n == 0) {
        return;
    }
    
    // Initialize GPU memory if needed
    if (!gpu_initialized_) {
        initialize_gpu_memory();
    }
    
    // Copy current positions and masses to device
    copy_to_device();
    
    // Compute accelerations on GPU
    // Use tiled kernel for better performance with larger particle counts
    if (n > 1024) {
        cuda::launch_compute_accelerations_tiled(
            d_positions_,
            d_masses_,
            d_accelerations_,
            static_cast<index_t>(n),
            G_,
            DEFAULT_SOFTENING
        );
    } else {
        cuda::launch_compute_accelerations(
            d_positions_,
            d_masses_,
            d_accelerations_,
            static_cast<index_t>(n),
            G_,
            DEFAULT_SOFTENING
        );
    }
    
    // Copy accelerations back to host
    copy_from_device();
}

void ParticleSystem::update_accelerations() {
    // Choose best method based on system size and available hardware
    const size_t n = particles_.size();
    
    // For very small systems, CPU might be faster due to GPU overhead
    if (n < 100) {
        update_accelerations_cpu();
    } else {
        // Try to use GPU if available
        try {
            update_accelerations_gpu();
        } catch (const std::runtime_error&) {
            // Fall back to CPU if GPU failed
            update_accelerations_cpu();
        }
    }
}

scalar_t ParticleSystem::total_mass() const {
    scalar_t total = 0.0;
    for (const auto& p : particles_) {
        total += p.mass();
    }
    return total;
}

Vec3 ParticleSystem::center_of_mass() const {
    const scalar_t total_m = total_mass();
    if (total_m == 0.0) {
        return Vec3(0, 0, 0);
    }
    
    Vec3 com(0, 0, 0);
    for (const auto& p : particles_) {
        com += p.position() * p.mass();
    }
    
    return com / total_m;
}

Vec3 ParticleSystem::total_momentum() const {
    Vec3 momentum(0, 0, 0);
    for (const auto& p : particles_) {
        momentum += p.velocity() * p.mass();
    }
    return momentum;
}

Vec3 ParticleSystem::total_angular_momentum() const {
    Vec3 angular_momentum(0, 0, 0);
    for (const auto& p : particles_) {
        angular_momentum += p.position().cross(p.velocity() * p.mass());
    }
    return angular_momentum;
}

scalar_t ParticleSystem::total_kinetic_energy() const {
    scalar_t energy = 0.0;
    for (const auto& p : particles_) {
        energy += p.kinetic_energy();
    }
    return energy;
}

scalar_t ParticleSystem::total_potential_energy() const {
    scalar_t energy = 0.0;
    const size_t n = particles_.size();
    
    // Sum over all unique pairs
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const Vec3& pos_i = particles_[i].position();
            const Vec3& pos_j = particles_[j].position();
            const scalar_t mass_i = particles_[i].mass();
            const scalar_t mass_j = particles_[j].mass();
            
            // Distance between particles
            scalar_t distance = (pos_j - pos_i).length();
            
            // Add softening to prevent singularities
            if (distance < DEFAULT_SOFTENING) {
                distance = DEFAULT_SOFTENING;
            }
            
            // Potential energy: U = -G * m1 * m2 / r
            energy -= G_ * mass_i * mass_j / distance;
        }
    }
    
    return energy;
}

scalar_t ParticleSystem::total_energy() const {
    return total_kinetic_energy() + total_potential_energy();
}

std::unique_ptr<ParticleSystem> ParticleSystem::clone() const {
    auto new_system = std::make_unique<ParticleSystem>(particles_, G_);
    return new_system;
}

std::unique_ptr<ParticleSystem> ParticleSystem::create_random_system(
    size_t num_particles,
    scalar_t box_size,
    scalar_t max_mass,
    scalar_t max_velocity,
    scalar_t G,
    unsigned int seed
) {
    // Create random number generator
    std::mt19937 rng(seed);
    std::uniform_real_distribution<scalar_t> pos_dist(-box_size, box_size);
    std::uniform_real_distribution<scalar_t> vel_dist(-max_velocity, max_velocity);
    std::uniform_real_distribution<scalar_t> mass_dist(0.1 * max_mass, max_mass);
    
    // Create particles
    std::vector<Particle> particles;
    particles.reserve(num_particles);
    
    for (size_t i = 0; i < num_particles; ++i) {
        // Random position
        Vec3 position(
            pos_dist(rng),
            pos_dist(rng),
            pos_dist(rng)
        );
        
        // Random velocity
        Vec3 velocity(
            vel_dist(rng),
            vel_dist(rng),
            vel_dist(rng)
        );
        
        // Random mass
        scalar_t mass = mass_dist(rng);
        
        // Create particle
        particles.emplace_back(position, velocity, mass, static_cast<index_t>(i));
    }
    
    // Create particle system
    return std::make_unique<ParticleSystem>(particles, G);
}

std::unique_ptr<ParticleSystem> ParticleSystem::create_solar_system(
    scalar_t scale_factor,
    scalar_t G
) {
    // Create vector for particles
    std::vector<Particle> particles;
    
    // Add the Sun at the center
    particles.emplace_back(
        Vec3(0, 0, 0),                       // position
        Vec3(0, 0, 0),                       // velocity
        1.0,                                 // mass (1 solar mass)
        0                                    // ID
    );
    
    // Planet data: [name, distance(AU), mass(solar masses), orbital_velocity(AU/year)]
    const std::vector<std::tuple<std::string, scalar_t, scalar_t, scalar_t>> planet_data = {
        {"Mercury", 0.39, 1.65e-7, 10.0},
        {"Venus", 0.72, 2.45e-6, 7.4},
        {"Earth", 1.0, 3.0e-6, 6.28},
        {"Mars", 1.52, 3.2e-7, 5.1},
        {"Jupiter", 5.2, 9.5e-4, 2.76},
        {"Saturn", 9.54, 2.85e-4, 2.04},
        {"Uranus", 19.2, 4.4e-5, 1.44},
        {"Neptune", 30.06, 5.15e-5, 1.14}
    };
    
    // Add planets
    for (size_t i = 0; i < planet_data.size(); ++i) {
        const auto& [name, distance, mass, velocity] = planet_data[i];
        
        // Scale the distance for better visualization
        scalar_t scaled_distance = distance * scale_factor;
        
        // Start planets at different angles
        scalar_t angle = i * M_PI / 4;
        
        // Position in orbital plane
        Vec3 position(
            scaled_distance * cos(angle),
            scaled_distance * sin(angle),
            0.0
        );
        
        // Orbital velocity perpendicular to position vector
        Vec3 velocity_vector(
            -sin(angle),
            cos(angle),
            0.0
        );
        
        // Scale velocity for visualization
        velocity_vector *= velocity / scale_factor;
        
        // Add planet to system
        particles.emplace_back(
            position,
            velocity_vector,
            mass,
            static_cast<index_t>(i + 1)
        );
    }
    
    // Create particle system
    return std::make_unique<ParticleSystem>(particles, G);
}

std::unique_ptr<ParticleSystem> ParticleSystem::create_galaxy_model(
    size_t num_particles,
    scalar_t radius,
    scalar_t height,
    scalar_t min_mass,
    scalar_t max_mass,
    scalar_t G,
    unsigned int seed
) {
    // Create random number generator
    std::mt19937 rng(seed);
    std::exponential_distribution<scalar_t> radius_dist(3.0 / radius);
    std::normal_distribution<scalar_t> height_dist(0.0, height);
    std::uniform_real_distribution<scalar_t> angle_dist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<scalar_t> mass_dist(min_mass, max_mass);
    std::normal_distribution<scalar_t> vel_disp_dist(0.0, 0.1);
    
    // Create vector for particles
    std::vector<Particle> particles;
    particles.reserve(num_particles);
    
    // Add central massive black hole
    particles.emplace_back(
        Vec3(0, 0, 0),                       // position
        Vec3(0, 0, 0),                       // velocity
        100.0,                               // mass (much more massive than other particles)
        0                                    // ID
    );
    
    // Create disk particles
    for (size_t i = 1; i < num_particles; ++i) {
        // Distance from center (exponential distribution)
        scalar_t distance = radius_dist(rng);
        if (distance > radius) {
            distance = radius;  // Cap at maximum radius
        }
        
        // Angle around disk with some spiral structure
        scalar_t angle = angle_dist(rng);
        scalar_t spiral_factor = 0.5;  // Controls tightness of spiral arms
        scalar_t spiral_angle = angle + spiral_factor * log(distance / 0.1);
        
        // Height above/below disk plane (thinner near center)
        scalar_t z_height = height_dist(rng) * distance / radius;
        
        // Position
        Vec3 position(
            distance * cos(spiral_angle),
            distance * sin(spiral_angle),
            z_height
        );
        
        // Enclosed mass approximation for orbital velocity
        scalar_t enclosed_mass = 100.0 + i * (min_mass + max_mass) / 2.0 / num_particles;
        
        // Orbital velocity (Keplerian approximation)
        scalar_t v_orbital = distance > 0 ? sqrt(G * enclosed_mass / distance) : 0;
        
        // Tangential velocity vector
        Vec3 velocity(
            -sin(spiral_angle),
            cos(spiral_angle),
            0.0
        );
        velocity *= v_orbital;
        
        // Add some velocity dispersion
        velocity.x += vel_disp_dist(rng) * v_orbital;
        velocity.y += vel_disp_dist(rng) * v_orbital;
        velocity.z += vel_disp_dist(rng) * v_orbital;
        
        // Random mass
        scalar_t mass = mass_dist(rng);
        
        // Add particle to system
        particles.emplace_back(
            position,
            velocity,
            mass,
            static_cast<index_t>(i)
        );
    }
    
    // Create particle system
    return std::make_unique<ParticleSystem>(particles, G);
}

} // namespace nbody_sim