# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Tests for the N-body simulation class.
"""

import numpy as np
import pytest
import os
import tempfile

from src.nbody_sim.python.particle import Particle, ParticleSystem
from src.nbody_sim.python.integrator import LeapfrogIntegrator
from src.nbody_sim.python.simulation import NBodySimulation


class TestNBodySimulation:
    """Tests for the NBodySimulation class."""
    
    @pytest.fixture
    def basic_simulation(self):
        """Create a basic simulation for testing."""
        # Create a simple two-body system
        particles = [
            Particle(
                position=np.array([1.0, 0.0, 0.0]),
                velocity=np.array([0.0, 0.5, 0.0]),
                mass=1.0,
                particle_id=0
            ),
            Particle(
                position=np.array([-1.0, 0.0, 0.0]),
                velocity=np.array([0.0, -0.5, 0.0]),
                mass=1.0,
                particle_id=1
            ),
        ]
        system = ParticleSystem(particles, G=1.0)
        integrator = LeapfrogIntegrator()
        
        return NBodySimulation(
            system=system,
            integrator=integrator,
            dt=0.01,
            duration=1.0
        )
    
    def test_simulation_initialization(self, basic_simulation):
        """Test that a simulation can be initialized correctly."""
        assert basic_simulation.system.num_particles == 2
        assert isinstance(basic_simulation.integrator, LeapfrogIntegrator)
        assert basic_simulation.dt == 0.01
        assert basic_simulation.duration == 1.0
        assert basic_simulation.current_time == 0.0
        assert basic_simulation.current_step == 0
        assert basic_simulation.total_steps == 100  # duration / dt = 1.0 / 0.01 = 100
    
    def test_simulation_step(self, basic_simulation):
        """Test that a simulation step updates the system correctly."""
        # Get initial state
        initial_positions = [np.copy(p.position) for p in basic_simulation.system.particles]
        
        # Perform a step
        basic_simulation.step()
        
        # Check that time and step count have been updated
        assert basic_simulation.current_time == 0.01
        assert basic_simulation.current_step == 1
        
        # Check that particle positions have changed
        for i, particle in enumerate(basic_simulation.system.particles):
            assert not np.array_equal(particle.position, initial_positions[i])
    
    def test_simulation_run(self, basic_simulation):
        """Test that the simulation runs for the specified duration."""
        # Run the simulation
        basic_simulation.run()
        
        # Check that time and step count have been updated
        assert np.isclose(basic_simulation.current_time, 1.0)
        assert basic_simulation.current_step == 100
        
        # Check that the system has evolved (particles have moved)
        for particle in basic_simulation.system.particles:
            # Particles should not be at their initial positions
            assert not np.allclose(particle.position, np.array([1.0, 0.0, 0.0])) and \
                   not np.allclose(particle.position, np.array([-1.0, 0.0, 0.0]))
    
    def test_simulation_with_callback(self, basic_simulation):
        """Test that the simulation can use a callback function."""
        # Create a callback that records positions at each step
        positions_log = []
        
        def record_positions(sim):
            positions = [np.copy(p.position) for p in sim.system.particles]
            positions_log.append(positions)
        
        # Run the simulation with the callback
        basic_simulation.run(callback=record_positions)
        
        # Check that positions were recorded at each step
        assert len(positions_log) == 100  # should have 100 recordings
        
        # Check that particles moved in each step
        for i in range(1, len(positions_log)):
            assert not np.array_equal(positions_log[i], positions_log[i-1])
    
    def test_simulation_save_state(self, basic_simulation, temp_output_dir):
        """Test that the simulation state can be saved."""
        # Create a file to save to
        state_file = os.path.join(temp_output_dir, "sim_state.npz")
        
        # Run a few steps
        for _ in range(10):
            basic_simulation.step()
        
        # Save the state
        basic_simulation.save_state(state_file)
        
        # Check that the file exists
        assert os.path.exists(state_file)
        
        # Load the state and verify
        state_data = np.load(state_file)
        assert "positions" in state_data
        assert "velocities" in state_data
        assert "masses" in state_data
        assert "time" in state_data
        
        # Check that saved time matches simulation time
        assert state_data["time"] == basic_simulation.current_time
        
        # Check that particle data was saved correctly
        positions = state_data["positions"]
        assert positions.shape == (2, 3)  # 2 particles, 3 dimensions
        
        for i, particle in enumerate(basic_simulation.system.particles):
            assert np.array_equal(positions[i], particle.position)
    
    def test_simulation_load_state(self, basic_simulation, temp_output_dir):
        """Test that the simulation state can be loaded."""
        # Create a file to save to
        state_file = os.path.join(temp_output_dir, "sim_state.npz")
        
        # Run a few steps and save the state
        for _ in range(10):
            basic_simulation.step()
        
        basic_simulation.save_state(state_file)
        
        # Record the state
        time_before_load = basic_simulation.current_time
        positions_before_load = [np.copy(p.position) for p in basic_simulation.system.particles]
        
        # Run more steps to change the state
        for _ in range(10):
            basic_simulation.step()
        
        # Load the saved state
        basic_simulation.load_state(state_file)
        
        # Check that time was restored
        assert basic_simulation.current_time == time_before_load
        
        # Check that particle positions were restored
        for i, particle in enumerate(basic_simulation.system.particles):
            assert np.array_equal(particle.position, positions_before_load[i])
    
    def test_simulation_performance_metrics(self, basic_simulation):
        """Test that performance metrics are recorded correctly."""
        # Run the simulation
        basic_simulation.run()
        
        # Check that metrics are available
        metrics = basic_simulation.get_performance_metrics()
        
        assert "total_time_ms" in metrics
        assert "steps_per_second" in metrics
        assert "energy_conservation_error" in metrics
        
        # Basic sanity checks on metrics
        assert metrics["total_time_ms"] > 0
        assert metrics["steps_per_second"] > 0
        
        # Energy conservation error should be relatively small for Leapfrog
        assert metrics["energy_conservation_error"] < 0.01