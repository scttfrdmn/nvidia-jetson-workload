// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "molecular_dynamics/molecular_system.hpp"
#include <memory>
#include <string>

namespace molecular_dynamics {

/**
 * @brief Base class for molecular dynamics integrators.
 */
class Integrator {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Integrator() = default;

    /**
     * @brief Initialize the integrator with a molecular system.
     * 
     * @param system Molecular system to initialize with
     */
    virtual void initialize(MolecularSystem& system) {}

    /**
     * @brief Advance the system by one time step.
     * 
     * @param system Molecular system to advance
     * @param dt Time step in picoseconds
     */
    virtual void step(MolecularSystem& system, scalar_t dt) = 0;

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
 * @brief Velocity Verlet integrator.
 */
class VelocityVerletIntegrator : public Integrator {
public:
    void step(MolecularSystem& system, scalar_t dt) override;
    IntegrationType type() const override { return IntegrationType::VelocityVerlet; }
    std::string name() const override { return "Velocity Verlet"; }
    std::unique_ptr<Integrator> clone() const override {
        return std::make_unique<VelocityVerletIntegrator>(*this);
    }
};

/**
 * @brief Leapfrog integrator.
 */
class LeapfrogIntegrator : public Integrator {
public:
    LeapfrogIntegrator() : initialized_(false) {}
    
    void initialize(MolecularSystem& system) override;
    void step(MolecularSystem& system, scalar_t dt) override;
    IntegrationType type() const override { return IntegrationType::Leapfrog; }
    std::string name() const override { return "Leapfrog"; }
    std::unique_ptr<Integrator> clone() const override {
        return std::make_unique<LeapfrogIntegrator>(*this);
    }
    
private:
    bool initialized_;
    std::vector<Vec3> half_step_velocities_;
};

/**
 * @brief Beeman integrator.
 */
class BeemanIntegrator : public Integrator {
public:
    BeemanIntegrator() : initialized_(false) {}
    
    void initialize(MolecularSystem& system) override;
    void step(MolecularSystem& system, scalar_t dt) override;
    IntegrationType type() const override { return IntegrationType::Beeman; }
    std::string name() const override { return "Beeman"; }
    std::unique_ptr<Integrator> clone() const override {
        return std::make_unique<BeemanIntegrator>(*this);
    }
    
private:
    bool initialized_;
    std::vector<Vec3> previous_forces_;
    std::vector<Vec3> current_forces_;
};

/**
 * @brief Base class for thermostats.
 */
class Thermostat {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~Thermostat() = default;
    
    /**
     * @brief Initialize the thermostat.
     * 
     * @param system Molecular system
     */
    virtual void initialize(MolecularSystem& system) {}
    
    /**
     * @brief Apply the thermostat to the system.
     * 
     * @param system Molecular system
     * @param target_temperature Target temperature in Kelvin
     * @param dt Time step in picoseconds
     */
    virtual void apply(MolecularSystem& system, scalar_t target_temperature, scalar_t dt) = 0;
    
    /**
     * @brief Get the thermostat type.
     * 
     * @return ThermostatType Thermostat type
     */
    virtual ThermostatType type() const = 0;
    
    /**
     * @brief Get a string representation of the thermostat.
     * 
     * @return std::string Thermostat name
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Create a copy of this thermostat.
     * 
     * @return std::unique_ptr<Thermostat> New thermostat
     */
    virtual std::unique_ptr<Thermostat> clone() const = 0;
    
    /**
     * @brief Create a thermostat of the specified type.
     * 
     * @param type Thermostat type
     * @return std::unique_ptr<Thermostat> New thermostat
     */
    static std::unique_ptr<Thermostat> create(ThermostatType type);
};

/**
 * @brief Berendsen thermostat.
 */
class BerendsenThermostat : public Thermostat {
public:
    /**
     * @brief Constructor.
     * 
     * @param coupling_constant Coupling constant (tau) in picoseconds
     */
    explicit BerendsenThermostat(scalar_t coupling_constant = 0.1);
    
    void apply(MolecularSystem& system, scalar_t target_temperature, scalar_t dt) override;
    ThermostatType type() const override { return ThermostatType::Berendsen; }
    std::string name() const override { return "Berendsen"; }
    std::unique_ptr<Thermostat> clone() const override {
        return std::make_unique<BerendsenThermostat>(*this);
    }
    
private:
    scalar_t coupling_constant_;
};

/**
 * @brief Andersen thermostat.
 */
class AndersenThermostat : public Thermostat {
public:
    /**
     * @brief Constructor.
     * 
     * @param collision_frequency Collision frequency in ps^-1
     */
    explicit AndersenThermostat(scalar_t collision_frequency = 1.0);
    
    void initialize(MolecularSystem& system) override;
    void apply(MolecularSystem& system, scalar_t target_temperature, scalar_t dt) override;
    ThermostatType type() const override { return ThermostatType::Andersen; }
    std::string name() const override { return "Andersen"; }
    std::unique_ptr<Thermostat> clone() const override {
        return std::make_unique<AndersenThermostat>(*this);
    }
    
private:
    scalar_t collision_frequency_;
    unsigned int seed_;
};

/**
 * @brief Nose-Hoover thermostat.
 */
class NoseHooverThermostat : public Thermostat {
public:
    /**
     * @brief Constructor.
     * 
     * @param relaxation_time Relaxation time in picoseconds
     */
    explicit NoseHooverThermostat(scalar_t relaxation_time = 0.1);
    
    void initialize(MolecularSystem& system) override;
    void apply(MolecularSystem& system, scalar_t target_temperature, scalar_t dt) override;
    ThermostatType type() const override { return ThermostatType::NoseHoover; }
    std::string name() const override { return "Nose-Hoover"; }
    std::unique_ptr<Thermostat> clone() const override {
        return std::make_unique<NoseHooverThermostat>(*this);
    }
    
private:
    scalar_t relaxation_time_;
    scalar_t thermostat_mass_;
    scalar_t thermostat_velocity_;
    scalar_t thermostat_position_;
};

} // namespace molecular_dynamics