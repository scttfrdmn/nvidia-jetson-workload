# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 nvidia-jetson-workload contributors

"""
Tests for the integrators in the N-body simulation.
"""

import numpy as np
import pytest

from src.nbody_sim.python.particle import Particle, ParticleSystem
from src.nbody_sim.python.integrator import (
    Integrator, 
    EulerIntegrator, 
    LeapfrogIntegrator, 
    VerletIntegrator,
    RungeKuttaIntegrator
)


class TestIntegrators:
    """Tests for the various integrator implementations."""
    
    @pytest.fixture
    def two_body_system(self):
        """Create a simple two-body system for testing."""
        particles = [
            Particle(
                position=np.array([1.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.1, 0.0]),
                mass=1.0,
                particle_id=0
            ),
            Particle(
                position=np.array([-1.0, 0.0, 0.0]),
                velocity=np.array([0.0, -0.1, 0.0]),
                mass=1.0,
                particle_id=1
            ),
        ]
        return ParticleSystem(particles, G=1.0)
    
    def test_base_integrator(self, two_body_system):
        """Test that base integrator cannot be used directly."""
        integrator = Integrator()
        with pytest.raises(NotImplementedError):
            integrator.step(two_body_system, 0.01)
    
    def test_euler_integrator(self, two_body_system):
        """Test the Euler integrator for a simple system."""
        integrator = EulerIntegrator()
        
        # Get initial state
        initial_energy = two_body_system.total_energy()
        
        # Perform a single step
        dt = 0.01
        integrator.step(two_body_system, dt)
        
        # Basic checks to ensure the system has changed
        for particle in two_body_system.particles:
            assert not np.array_equal(particle.acceleration, np.zeros(3))
        
        # Euler method has energy conservation errors
        # but energy should not explode for small timesteps
        new_energy = two_body_system.total_energy()
        assert abs(new_energy - initial_energy) / abs(initial_energy) < 0.01
    
    def test_leapfrog_integrator(self, two_body_system):
        """Test the Leapfrog integrator for a simple system."""
        integrator = LeapfrogIntegrator()
        
        # Get initial state
        initial_energy = two_body_system.total_energy()
        
        # Perform a single step
        dt = 0.01
        integrator.step(two_body_system, dt)
        
        # Leapfrog should conserve energy better than Euler
        new_energy = two_body_system.total_energy()
        assert abs(new_energy - initial_energy) / abs(initial_energy) < 0.005
    
    def test_verlet_integrator(self, two_body_system):
        """Test the Verlet integrator for a simple system."""
        integrator = VerletIntegrator()
        
        # Get initial state
        initial_energy = two_body_system.total_energy()
        
        # Initialize half-step velocities (needed for Verlet)
        integrator.initialize(two_body_system)
        
        # Perform a single step
        dt = 0.01
        integrator.step(two_body_system, dt)
        
        # Verlet should conserve energy well
        new_energy = two_body_system.total_energy()
        assert abs(new_energy - initial_energy) / abs(initial_energy) < 0.005
    
    def test_runge_kutta_integrator(self, two_body_system):
        """Test the 4th order Runge-Kutta integrator for a simple system."""
        integrator = RungeKuttaIntegrator()
        
        # Get initial state
        initial_energy = two_body_system.total_energy()
        initial_positions = [np.copy(p.position) for p in two_body_system.particles]
        
        # Perform a single step
        dt = 0.01
        integrator.step(two_body_system, dt)
        
        # Positions should have changed
        for i, particle in enumerate(two_body_system.particles):
            assert not np.array_equal(particle.position, initial_positions[i])
        
        # RK4 should conserve energy well for small timesteps
        new_energy = two_body_system.total_energy()
        assert abs(new_energy - initial_energy) / abs(initial_energy) < 0.001
    
    def test_energy_conservation_comparison(self, two_body_system):
        """Compare energy conservation of different integrators."""
        # Create copies of the system for different integrators
        systems = {
            "euler": two_body_system.copy(),
            "leapfrog": two_body_system.copy(),
            "verlet": two_body_system.copy(),
            "rk4": two_body_system.copy()
        }
        
        integrators = {
            "euler": EulerIntegrator(),
            "leapfrog": LeapfrogIntegrator(),
            "verlet": VerletIntegrator(),
            "rk4": RungeKuttaIntegrator()
        }
        
        # Initialize Verlet integrator
        integrators["verlet"].initialize(systems["verlet"])
        
        # Get initial energies
        initial_energies = {name: system.total_energy() for name, system in systems.items()}
        
        # Perform 10 steps with each integrator
        dt = 0.01
        for _ in range(10):
            for name, integrator in integrators.items():
                integrator.step(systems[name], dt)
        
        # Calculate relative energy errors
        energy_errors = {}
        for name, system in systems.items():
            final_energy = system.total_energy()
            energy_errors[name] = abs(final_energy - initial_energies[name]) / abs(initial_energies[name])
        
        # Higher order methods should conserve energy better
        assert energy_errors["euler"] > energy_errors["leapfrog"]
        assert energy_errors["leapfrog"] > energy_errors["rk4"]