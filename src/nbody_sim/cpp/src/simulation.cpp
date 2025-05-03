// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "nbody_sim/simulation.hpp"
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace nbody_sim {

Simulation::Simulation()
    : system_(std::make_unique<ParticleSystem>()),
      integrator_(std::make_unique<LeapfrogIntegrator>()),
      dt_(DEFAULT_TIMESTEP),
      duration_(10.0),
      current_time_(0.0),
      current_step_(0),
      total_steps_(static_cast<index_t>(duration_ / dt_)),
      initial_energy_(0.0),
      final_energy_(0.0) {}

Simulation::Simulation(
    std::unique_ptr<ParticleSystem> system,
    std::unique_ptr<Integrator> integrator,
    scalar_t dt,
    scalar_t duration
)
    : system_(std::move(system)),
      integrator_(std::move(integrator)),
      dt_(dt),
      duration_(duration),
      current_time_(0.0),
      current_step_(0),
      total_steps_(static_cast<index_t>(duration_ / dt_)),
      initial_energy_(0.0),
      final_energy_(0.0) {
    
    // Initialize the integrator
    integrator_->initialize(*system_);
}

void Simulation::step() {
    // Take a step with the integrator
    integrator_->step(*system_, dt_);
    
    // Update time tracking
    current_time_ += dt_;
    current_step_++;
}

void Simulation::run(StepCallback callback, index_t callback_interval) {
    // Record start time
    start_time_ = std::chrono::steady_clock::now();
    
    // Record initial energy
    initial_energy_ = system_->total_energy();
    
    // Run the simulation
    for (; current_step_ < total_steps_; ++current_step_) {
        // Take a step
        integrator_->step(*system_, dt_);
        
        // Update current time
        current_time_ += dt_;
        
        // Call the callback if provided and it's time to do so
        if (callback && current_step_ % callback_interval == 0) {
            callback(*this);
        }
    }
    
    // Record end time
    end_time_ = std::chrono::steady_clock::now();
    
    // Record final energy
    final_energy_ = system_->total_energy();
}

void Simulation::save_state(const std::string& filename) const {
    // Open file for writing
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Write header with simulation parameters
    file.write(reinterpret_cast<const char*>(&current_time_), sizeof(current_time_));
    file.write(reinterpret_cast<const char*>(&current_step_), sizeof(current_step_));
    file.write(reinterpret_cast<const char*>(&dt_), sizeof(dt_));
    file.write(reinterpret_cast<const char*>(&duration_), sizeof(duration_));
    
    // Write integrator type
    IntegrationType integrator_type = integrator_->type();
    file.write(reinterpret_cast<const char*>(&integrator_type), sizeof(integrator_type));
    
    // Write gravitational constant
    scalar_t G = system_->gravitational_constant();
    file.write(reinterpret_cast<const char*>(&G), sizeof(G));
    
    // Write number of particles
    size_t num_particles = system_->size();
    file.write(reinterpret_cast<const char*>(&num_particles), sizeof(num_particles));
    
    // Write particle data
    for (size_t i = 0; i < num_particles; ++i) {
        const Particle& p = system_->particle(i);
        
        // Write position
        const Vec3& pos = p.position();
        file.write(reinterpret_cast<const char*>(&pos), sizeof(pos));
        
        // Write velocity
        const Vec3& vel = p.velocity();
        file.write(reinterpret_cast<const char*>(&vel), sizeof(vel));
        
        // Write mass
        scalar_t mass = p.mass();
        file.write(reinterpret_cast<const char*>(&mass), sizeof(mass));
        
        // Write ID
        index_t id = p.id();
        file.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
}

void Simulation::load_state(const std::string& filename) {
    // Open file for reading
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    
    // Read header with simulation parameters
    file.read(reinterpret_cast<char*>(&current_time_), sizeof(current_time_));
    file.read(reinterpret_cast<char*>(&current_step_), sizeof(current_step_));
    file.read(reinterpret_cast<char*>(&dt_), sizeof(dt_));
    file.read(reinterpret_cast<char*>(&duration_), sizeof(duration_));
    
    // Update total_steps_ based on duration and dt
    total_steps_ = static_cast<index_t>(duration_ / dt_);
    
    // Read integrator type
    IntegrationType integrator_type;
    file.read(reinterpret_cast<char*>(&integrator_type), sizeof(integrator_type));
    
    // Create new integrator of the specified type
    integrator_ = Integrator::create(integrator_type);
    
    // Read gravitational constant
    scalar_t G;
    file.read(reinterpret_cast<char*>(&G), sizeof(G));
    
    // Read number of particles
    size_t num_particles;
    file.read(reinterpret_cast<char*>(&num_particles), sizeof(num_particles));
    
    // Create particles and add them to the system
    std::vector<Particle> particles;
    particles.reserve(num_particles);
    
    for (size_t i = 0; i < num_particles; ++i) {
        // Read position
        Vec3 pos;
        file.read(reinterpret_cast<char*>(&pos), sizeof(pos));
        
        // Read velocity
        Vec3 vel;
        file.read(reinterpret_cast<char*>(&vel), sizeof(vel));
        
        // Read mass
        scalar_t mass;
        file.read(reinterpret_cast<char*>(&mass), sizeof(mass));
        
        // Read ID
        index_t id;
        file.read(reinterpret_cast<char*>(&id), sizeof(id));
        
        // Create particle and add to vector
        particles.emplace_back(pos, vel, mass, id);
    }
    
    // Create new particle system
    system_ = std::make_unique<ParticleSystem>(particles, G);
    
    // Initialize the integrator with the new system
    integrator_->initialize(*system_);
}

std::map<std::string, double> Simulation::get_performance_metrics() const {
    std::map<std::string, double> metrics;
    
    // Calculate elapsed time
    if (end_time_ > start_time_) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_ - start_time_);
        metrics["total_time_ms"] = elapsed.count();
        
        // Calculate steps per second
        if (current_step_ > 0) {
            metrics["steps_per_second"] = 1000.0 * current_step_ / elapsed.count();
        }
    }
    
    // Calculate energy conservation
    if (initial_energy_ != 0.0) {
        metrics["initial_energy"] = initial_energy_;
        metrics["final_energy"] = final_energy_;
        metrics["energy_conservation_error"] = std::abs((final_energy_ - initial_energy_) / initial_energy_);
    }
    
    // Add system metrics
    metrics["particle_count"] = static_cast<double>(system_->size());
    metrics["current_time"] = current_time_;
    metrics["current_step"] = static_cast<double>(current_step_);
    
    return metrics;
}

std::map<std::string, std::vector<double>> Simulation::create_visualization_data(bool include_velocities) const {
    std::map<std::string, std::vector<double>> data;
    
    const size_t n = system_->size();
    
    // Initialize data arrays
    std::vector<double> positions_x(n);
    std::vector<double> positions_y(n);
    std::vector<double> positions_z(n);
    std::vector<double> masses(n);
    std::vector<double> ids(n);
    
    // Add position and mass data
    for (size_t i = 0; i < n; ++i) {
        const Particle& p = system_->particle(i);
        
        positions_x[i] = p.position().x;
        positions_y[i] = p.position().y;
        positions_z[i] = p.position().z;
        masses[i] = p.mass();
        ids[i] = static_cast<double>(p.id());
    }
    
    // Add data to result
    data["positions_x"] = positions_x;
    data["positions_y"] = positions_y;
    data["positions_z"] = positions_z;
    data["masses"] = masses;
    data["ids"] = ids;
    
    // Add velocity data if requested
    if (include_velocities) {
        std::vector<double> velocities_x(n);
        std::vector<double> velocities_y(n);
        std::vector<double> velocities_z(n);
        
        for (size_t i = 0; i < n; ++i) {
            const Particle& p = system_->particle(i);
            
            velocities_x[i] = p.velocity().x;
            velocities_y[i] = p.velocity().y;
            velocities_z[i] = p.velocity().z;
        }
        
        data["velocities_x"] = velocities_x;
        data["velocities_y"] = velocities_y;
        data["velocities_z"] = velocities_z;
    }
    
    // Add metadata
    std::vector<double> metadata = {
        static_cast<double>(n),
        current_time_,
        static_cast<double>(current_step_),
        dt_,
        duration_,
        system_->gravitational_constant()
    };
    
    data["metadata"] = metadata;
    
    return data;
}

std::unique_ptr<Simulation> Simulation::create_random_simulation(
    size_t num_particles,
    IntegrationType integration_type,
    scalar_t dt,
    scalar_t duration,
    unsigned int seed
) {
    // Create random particle system
    auto system = ParticleSystem::create_random_system(
        num_particles,
        10.0,  // box_size
        1.0,   // max_mass
        0.1,   // max_velocity
        1.0,   // G
        seed
    );
    
    // Create integrator
    auto integrator = Integrator::create(integration_type);
    
    // Create and return simulation
    return std::make_unique<Simulation>(
        std::move(system),
        std::move(integrator),
        dt,
        duration
    );
}

std::unique_ptr<Simulation> Simulation::create_solar_system_simulation(
    IntegrationType integration_type,
    scalar_t dt,
    scalar_t duration,
    scalar_t scale_factor
) {
    // Create solar system
    auto system = ParticleSystem::create_solar_system(scale_factor);
    
    // Create integrator
    auto integrator = Integrator::create(integration_type);
    
    // Create and return simulation
    return std::make_unique<Simulation>(
        std::move(system),
        std::move(integrator),
        dt,
        duration
    );
}

std::unique_ptr<Simulation> Simulation::create_galaxy_simulation(
    size_t num_particles,
    IntegrationType integration_type,
    scalar_t dt,
    scalar_t duration,
    unsigned int seed
) {
    // Create galaxy model
    auto system = ParticleSystem::create_galaxy_model(
        num_particles,
        10.0,  // radius
        1.0,   // height
        0.1,   // min_mass
        1.0,   // max_mass
        1.0,   // G
        seed
    );
    
    // Create integrator
    auto integrator = Integrator::create(integration_type);
    
    // Create and return simulation
    return std::make_unique<Simulation>(
        std::move(system),
        std::move(integrator),
        dt,
        duration
    );
}

} // namespace nbody_sim