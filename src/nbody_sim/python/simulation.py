# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 nvidia-jetson-workload contributors

"""
Main simulation class for N-body simulation.
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import time
import os

from src.nbody_sim.python.particle import ParticleSystem, Particle
from src.nbody_sim.python.integrator import Integrator


class NBodySimulation:
    """
    Main class for running an N-body simulation.
    
    This class combines a particle system with an integrator and provides
    methods to run the simulation and collect results.
    
    Attributes:
        system (ParticleSystem): The particle system being simulated
        integrator (Integrator): The integration method to use
        dt (float): Time step size
        duration (float): Total simulation time
        current_time (float): Current simulation time
        current_step (int): Current simulation step
        total_steps (int): Total number of steps to perform
        performance_data (Dict[str, Any]): Performance metrics
    """
    
    def __init__(
        self,
        system: ParticleSystem,
        integrator: Integrator,
        dt: float,
        duration: float
    ) -> None:
        """
        Initialize the simulation.
        
        Args:
            system: The particle system to simulate
            integrator: The integration method to use
            dt: Time step size
            duration: Total simulation time
        """
        self.system = system
        self.integrator = integrator
        self.dt = dt
        self.duration = duration
        
        # Initialize time tracking
        self.current_time = 0.0
        self.current_step = 0
        self.total_steps = int(duration / dt)
        
        # Initialize performance tracking
        self.performance_data = {
            "start_time": None,
            "end_time": None,
            "initial_energy": None,
            "final_energy": None,
        }
        
        # Initialize the integrator if needed
        self.integrator.initialize(system)
    
    def step(self) -> None:
        """
        Advance the simulation by one time step.
        """
        # Take a step with the chosen integrator
        self.integrator.step(self.system, self.dt)
        
        # Update time tracking
        self.current_time += self.dt
        self.current_step += 1
    
    def run(
        self, 
        callback: Optional[Callable[['NBodySimulation'], None]] = None,
        callback_interval: int = 1,
        record_initial_state: bool = True
    ) -> None:
        """
        Run the simulation for the specified duration.
        
        Args:
            callback: Optional function to call at each step
            callback_interval: How often to call the callback (in steps)
            record_initial_state: Whether to record initial energy for conservation checks
        """
        # Record start time
        self.performance_data["start_time"] = time.time()
        
        # Record initial energy if requested
        if record_initial_state:
            self.performance_data["initial_energy"] = self.system.total_energy()
        
        # Run the simulation
        for step in range(self.total_steps):
            # Take a step
            self.step()
            
            # Call the callback if provided and it's time to do so
            if callback is not None and step % callback_interval == 0:
                callback(self)
        
        # Record end time and energy
        self.performance_data["end_time"] = time.time()
        self.performance_data["final_energy"] = self.system.total_energy()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance metrics for the simulation.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Check if we have timing data
        if self.performance_data["start_time"] is not None and self.performance_data["end_time"] is not None:
            # Calculate elapsed time
            elapsed_time = self.performance_data["end_time"] - self.performance_data["start_time"]
            
            # Calculate steps per second
            if self.current_step > 0:
                steps_per_second = self.current_step / elapsed_time
            else:
                steps_per_second = 0.0
            
            metrics["total_time_ms"] = elapsed_time * 1000  # Convert to milliseconds
            metrics["steps_per_second"] = steps_per_second
        
        # Check if we have energy conservation data
        if self.performance_data["initial_energy"] is not None and self.performance_data["final_energy"] is not None:
            # Calculate relative energy error
            initial_energy = self.performance_data["initial_energy"]
            final_energy = self.performance_data["final_energy"]
            
            if initial_energy != 0:
                energy_error = abs((final_energy - initial_energy) / initial_energy)
            else:
                energy_error = abs(final_energy - initial_energy)
            
            metrics["initial_energy"] = initial_energy
            metrics["final_energy"] = final_energy
            metrics["energy_conservation_error"] = energy_error
        
        return metrics
    
    def save_state(self, filename: str) -> None:
        """
        Save the current state of the simulation to a file.
        
        Args:
            filename: Path to save the state file
        """
        # Extract particle data
        num_particles = self.system.num_particles
        positions = np.zeros((num_particles, 3))
        velocities = np.zeros((num_particles, 3))
        masses = np.zeros(num_particles)
        ids = np.zeros(num_particles, dtype=int)
        
        for i, particle in enumerate(self.system.particles):
            positions[i] = particle.position
            velocities[i] = particle.velocity
            masses[i] = particle.mass
            ids[i] = particle.particle_id
        
        # Save to file
        np.savez(
            filename,
            positions=positions,
            velocities=velocities,
            masses=masses,
            ids=ids,
            time=self.current_time,
            step=self.current_step,
            dt=self.dt,
            G=self.system.G
        )
    
    def load_state(self, filename: str) -> None:
        """
        Load a simulation state from a file.
        
        Args:
            filename: Path to the state file
        """
        # Load data from file
        data = np.load(filename)
        
        # Update simulation state
        self.current_time = float(data["time"])
        self.current_step = int(data["step"])
        
        # Update system and particles
        positions = data["positions"]
        velocities = data["velocities"]
        masses = data["masses"]
        ids = data["ids"]
        
        # Create new particles with loaded data
        particles = []
        for i in range(len(masses)):
            particles.append(
                Particle(
                    position=positions[i],
                    velocity=velocities[i],
                    mass=masses[i],
                    particle_id=int(ids[i])
                )
            )
        
        # Update the system
        self.system = ParticleSystem(particles, G=float(data["G"]))
        
        # Initialize the integrator with the new system
        self.integrator.initialize(self.system)
    
    def create_visualization_data(self, include_velocities: bool = True) -> Dict[str, Any]:
        """
        Create a data structure for visualization.
        
        Args:
            include_velocities: Whether to include velocity data
        
        Returns:
            Dictionary with visualization data
        """
        num_particles = self.system.num_particles
        
        # Extract data for visualization
        positions = np.zeros((num_particles, 3))
        masses = np.zeros(num_particles)
        ids = np.zeros(num_particles, dtype=int)
        
        for i, p in enumerate(self.system.particles):
            positions[i] = p.position
            masses[i] = p.mass
            ids[i] = p.particle_id
        
        # Create data structure
        vis_data = {
            "time": self.current_time,
            "step": self.current_step,
            "positions": positions.tolist(),
            "masses": masses.tolist(),
            "ids": ids.tolist(),
        }
        
        # Include velocities if requested
        if include_velocities:
            velocities = np.zeros((num_particles, 3))
            for i, p in enumerate(self.system.particles):
                velocities[i] = p.velocity
            vis_data["velocities"] = velocities.tolist()
        
        return vis_data