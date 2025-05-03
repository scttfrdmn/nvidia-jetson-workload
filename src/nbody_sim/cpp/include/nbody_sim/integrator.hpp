// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "nbody_sim/particle.hpp"
#include <memory>

namespace nbody_sim {

/**
 * @brief Base class for numerical integrators.
 */
class Integrator {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Integrator() = default;

    /**
     * @brief Initialize the integrator with a particle system.
     * 
     * @param system Particle system to initialize with
     */
    virtual void initialize(ParticleSystem& system) {}

    /**
     * @brief Advance the system by one time step.
     * 
     * @param system Particle system to advance
     * @param dt Time step size
     */
    virtual void step(ParticleSystem& system, scalar_t dt) = 0;

    /**
     * @brief Get the integrator type.
     * 
     * @return IntegrationType Integrator type
     */
    virtual IntegrationType type() const = 0;

    /**
     * @brief Get a string representation of the integrator.
     * 
     * @return std::string Integrator name
     */
    virtual std::string name() const = 0;

    /**
     * @brief Create a copy of this integrator.
     * 
     * @return std::unique_ptr<Integrator> New integrator
     */
    virtual std::unique_ptr<Integrator> clone() const = 0;

    /**
     * @brief Create an integrator of the specified type.
     * 
     * @param type Integrator type
     * @return std::unique_ptr<Integrator> New integrator
     */
    static std::unique_ptr<Integrator> create(IntegrationType type);
};

/**
 * @brief Euler integrator (first-order).
 */
class EulerIntegrator : public Integrator {
public:
    void step(ParticleSystem& system, scalar_t dt) override;
    IntegrationType type() const override { return IntegrationType::Euler; }
    std::string name() const override { return "Euler"; }
    std::unique_ptr<Integrator> clone() const override {
        return std::make_unique<EulerIntegrator>(*this);
    }
};

/**
 * @brief Leapfrog integrator (second-order symplectic).
 */
class LeapfrogIntegrator : public Integrator {
public:
    void step(ParticleSystem& system, scalar_t dt) override;
    IntegrationType type() const override { return IntegrationType::Leapfrog; }
    std::string name() const override { return "Leapfrog"; }
    std::unique_ptr<Integrator> clone() const override {
        return std::make_unique<LeapfrogIntegrator>(*this);
    }
};

/**
 * @brief Velocity Verlet integrator (second-order symplectic).
 */
class VerletIntegrator : public Integrator {
public:
    /**
     * @brief Constructor.
     */
    VerletIntegrator() : initialized_(false) {}

    void initialize(ParticleSystem& system) override;
    void step(ParticleSystem& system, scalar_t dt) override;
    IntegrationType type() const override { return IntegrationType::Verlet; }
    std::string name() const override { return "Verlet"; }
    std::unique_ptr<Integrator> clone() const override {
        return std::make_unique<VerletIntegrator>(*this);
    }

private:
    bool initialized_;  // Flag to track initialization
};

/**
 * @brief Fourth-order Runge-Kutta integrator.
 */
class RungeKutta4Integrator : public Integrator {
public:
    void step(ParticleSystem& system, scalar_t dt) override;
    IntegrationType type() const override { return IntegrationType::RungeKutta4; }
    std::string name() const override { return "RK4"; }
    std::unique_ptr<Integrator> clone() const override {
        return std::make_unique<RungeKutta4Integrator>(*this);
    }
};

} // namespace nbody_sim