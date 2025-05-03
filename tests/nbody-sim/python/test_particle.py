# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 nvidia-jetson-workload contributors

"""
Tests for the Particle class in the N-body simulation.
"""

import numpy as np
import pytest

from src.nbody_sim.python.particle import Particle, ParticleSystem


class TestParticle:
    """Tests for the Particle class."""

    def test_particle_initialization(self):
        """Test that a particle can be initialized with the correct properties."""
        particle = Particle(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([4.0, 5.0, 6.0]),
            mass=7.0,
            particle_id=1
        )

        assert particle.particle_id == 1
        assert np.array_equal(particle.position, np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(particle.velocity, np.array([4.0, 5.0, 6.0]))
        assert particle.mass == 7.0
        assert np.array_equal(particle.acceleration, np.zeros(3))

    def test_particle_update_position(self):
        """Test that a particle's position can be updated."""
        particle = Particle(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([4.0, 5.0, 6.0]),
            mass=7.0,
            particle_id=1
        )
        
        dt = 0.1
        particle.update_position(dt)
        
        # New position = old position + velocity * dt
        expected_position = np.array([1.0, 2.0, 3.0]) + np.array([4.0, 5.0, 6.0]) * dt
        assert np.allclose(particle.position, expected_position)

    def test_particle_update_velocity(self):
        """Test that a particle's velocity can be updated."""
        particle = Particle(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([4.0, 5.0, 6.0]),
            mass=7.0,
            particle_id=1
        )
        
        # Set acceleration
        particle.acceleration = np.array([0.1, 0.2, 0.3])
        
        dt = 0.1
        particle.update_velocity(dt)
        
        # New velocity = old velocity + acceleration * dt
        expected_velocity = np.array([4.0, 5.0, 6.0]) + np.array([0.1, 0.2, 0.3]) * dt
        assert np.allclose(particle.velocity, expected_velocity)

    def test_particle_kinetic_energy(self):
        """Test calculation of kinetic energy."""
        particle = Particle(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([4.0, 5.0, 6.0]),
            mass=2.0,
            particle_id=1
        )
        
        # KE = 0.5 * m * v^2
        expected_ke = 0.5 * 2.0 * (4.0**2 + 5.0**2 + 6.0**2)
        assert np.isclose(particle.kinetic_energy(), expected_ke)


class TestParticleSystem:
    """Tests for the ParticleSystem class."""
    
    def test_particle_system_initialization(self):
        """Test that a particle system can be initialized with particles."""
        particles = [
            Particle(
                position=np.array([1.0, 2.0, 3.0]),
                velocity=np.array([4.0, 5.0, 6.0]),
                mass=7.0,
                particle_id=i
            )
            for i in range(10)
        ]
        
        system = ParticleSystem(particles)
        
        assert len(system.particles) == 10
        assert system.num_particles == 10

    def test_particle_system_total_mass(self):
        """Test calculation of total mass in the system."""
        particles = [
            Particle(
                position=np.zeros(3),
                velocity=np.zeros(3),
                mass=float(i + 1),
                particle_id=i
            )
            for i in range(5)
        ]
        
        system = ParticleSystem(particles)
        
        # Total mass = sum of all particle masses (1+2+3+4+5 = 15)
        assert np.isclose(system.total_mass(), 15.0)

    def test_particle_system_center_of_mass(self):
        """Test calculation of center of mass."""
        particles = [
            Particle(
                position=np.array([1.0, 0.0, 0.0]),
                velocity=np.zeros(3),
                mass=1.0,
                particle_id=0
            ),
            Particle(
                position=np.array([-1.0, 0.0, 0.0]),
                velocity=np.zeros(3),
                mass=1.0,
                particle_id=1
            ),
        ]
        
        system = ParticleSystem(particles)
        
        # Center of mass should be at (0,0,0) for these two equal mass particles
        com = system.center_of_mass()
        assert np.allclose(com, np.zeros(3))
        
        # Now try with unequal masses
        particles[0].mass = 2.0
        
        # Center of mass should now be at (1/3, 0, 0)
        # (2*1 + 1*(-1))/(2+1) = 1/3
        com = system.center_of_mass()
        assert np.allclose(com, np.array([1/3, 0.0, 0.0]))

    def test_particle_system_total_energy(self):
        """Test calculation of total energy in the system."""
        # Create a two-particle system
        particles = [
            Particle(
                position=np.array([1.0, 0.0, 0.0]),
                velocity=np.array([0.0, 1.0, 0.0]),
                mass=1.0,
                particle_id=0
            ),
            Particle(
                position=np.array([-1.0, 0.0, 0.0]),
                velocity=np.array([0.0, -1.0, 0.0]),
                mass=1.0,
                particle_id=1
            ),
        ]
        
        system = ParticleSystem(particles, G=1.0)
        
        # Calculate expected energies
        # Kinetic energy = 0.5 * m * v^2 for each particle
        ke1 = 0.5 * 1.0 * 1.0  # 0.5 * m * v^2
        ke2 = 0.5 * 1.0 * 1.0
        total_ke = ke1 + ke2
        
        # Potential energy = -G * m1 * m2 / r
        # Distance between particles is 2.0
        pe = -1.0 * 1.0 * 1.0 / 2.0
        
        expected_total_energy = total_ke + pe
        
        assert np.isclose(system.total_energy(), expected_total_energy)