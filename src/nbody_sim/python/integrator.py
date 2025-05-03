# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Integration methods for N-body simulation.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from src.nbody_sim.python.particle import ParticleSystem


class Integrator:
    """
    Base class for numerical integrators.
    
    This class defines the interface for all integrators.
    Subclasses should implement the step method.
    """
    
    def __init__(self) -> None:
        """Initialize the integrator."""
        pass
    
    def initialize(self, system: ParticleSystem) -> None:
        """
        Perform any initialization needed before the first step.
        
        Args:
            system: The particle system to initialize
        """
        pass
    
    def step(self, system: ParticleSystem, dt: float) -> None:
        """
        Advance the system by one time step.
        
        Args:
            system: The particle system to advance
            dt: The time step size
        
        Raises:
            NotImplementedError: If not implemented by a subclass
        """
        raise NotImplementedError("Subclasses must implement this method")


class EulerIntegrator(Integrator):
    """
    Simple first-order Euler integrator.
    
    This is the most basic integration method. It has poor
    energy conservation for large time steps but is simple to implement.
    """
    
    def step(self, system: ParticleSystem, dt: float) -> None:
        """
        Advance the system by one time step using the Euler method.
        
        Args:
            system: The particle system to advance
            dt: The time step size
        """
        # Update accelerations based on current positions
        system.update_accelerations()
        
        # Update positions and velocities
        for particle in system.particles:
            # Update velocity based on acceleration
            particle.update_velocity(dt)
            
            # Update position based on new velocity
            particle.update_position(dt)


class LeapfrogIntegrator(Integrator):
    """
    Second-order symplectic Leapfrog integrator.
    
    This method has better energy conservation than Euler.
    It is a symplectic integrator, which means it preserves
    the phase-space volume and has good long-term energy conservation.
    """
    
    def step(self, system: ParticleSystem, dt: float) -> None:
        """
        Advance the system by one time step using the Leapfrog method.
        
        Args:
            system: The particle system to advance
            dt: The time step size
        """
        # Update positions using current velocities (half step)
        for particle in system.particles:
            particle.position += particle.velocity * (dt / 2)
        
        # Update accelerations based on new positions
        system.update_accelerations()
        
        # Update velocities using new accelerations
        for particle in system.particles:
            particle.update_velocity(dt)
        
        # Update positions using new velocities (half step)
        for particle in system.particles:
            particle.position += particle.velocity * (dt / 2)


class VerletIntegrator(Integrator):
    """
    Velocity Verlet integrator.
    
    This is a popular second-order integrator for molecular dynamics and
    N-body simulations. It has good energy conservation properties.
    """
    
    def __init__(self) -> None:
        """Initialize the Verlet integrator."""
        super().__init__()
        self.initialized = False
    
    def initialize(self, system: ParticleSystem) -> None:
        """
        Compute initial accelerations for the Verlet integrator.
        
        Args:
            system: The particle system to initialize
        """
        system.update_accelerations()
        self.initialized = True
    
    def step(self, system: ParticleSystem, dt: float) -> None:
        """
        Advance the system by one time step using the Velocity Verlet method.
        
        Args:
            system: The particle system to advance
            dt: The time step size
        """
        # Ensure accelerations are initialized
        if not self.initialized:
            self.initialize(system)
        
        # Store current accelerations
        old_accelerations = [np.copy(p.acceleration) for p in system.particles]
        
        # Update positions using current velocities and accelerations
        for i, particle in enumerate(system.particles):
            particle.position += particle.velocity * dt + 0.5 * old_accelerations[i] * dt**2
        
        # Update accelerations based on new positions
        system.update_accelerations()
        
        # Update velocities using average of old and new accelerations
        for i, particle in enumerate(system.particles):
            particle.velocity += 0.5 * (old_accelerations[i] + particle.acceleration) * dt


class RungeKuttaIntegrator(Integrator):
    """
    Fourth-order Runge-Kutta integrator.
    
    This is a high-order method with excellent accuracy for smooth problems.
    However, it is more computationally expensive per step.
    """
    
    def step(self, system: ParticleSystem, dt: float) -> None:
        """
        Advance the system by one time step using the RK4 method.
        
        Args:
            system: The particle system to advance
            dt: The time step size
        """
        # Make a copy of the initial state
        initial_positions = [np.copy(p.position) for p in system.particles]
        initial_velocities = [np.copy(p.velocity) for p in system.particles]
        
        # Stage 1: Evaluate at the initial point
        system.update_accelerations()
        k1_vel = [np.copy(p.acceleration) for p in system.particles]
        k1_pos = [np.copy(p.velocity) for p in system.particles]
        
        # Stage 2: Evaluate at t + dt/2 using k1
        for i, particle in enumerate(system.particles):
            particle.position = initial_positions[i] + k1_pos[i] * dt / 2
            particle.velocity = initial_velocities[i] + k1_vel[i] * dt / 2
        
        system.update_accelerations()
        k2_vel = [np.copy(p.acceleration) for p in system.particles]
        k2_pos = [np.copy(p.velocity) for p in system.particles]
        
        # Stage 3: Evaluate at t + dt/2 using k2
        for i, particle in enumerate(system.particles):
            particle.position = initial_positions[i] + k2_pos[i] * dt / 2
            particle.velocity = initial_velocities[i] + k2_vel[i] * dt / 2
        
        system.update_accelerations()
        k3_vel = [np.copy(p.acceleration) for p in system.particles]
        k3_pos = [np.copy(p.velocity) for p in system.particles]
        
        # Stage 4: Evaluate at t + dt using k3
        for i, particle in enumerate(system.particles):
            particle.position = initial_positions[i] + k3_pos[i] * dt
            particle.velocity = initial_velocities[i] + k3_vel[i] * dt
        
        system.update_accelerations()
        k4_vel = [np.copy(p.acceleration) for p in system.particles]
        k4_pos = [np.copy(p.velocity) for p in system.particles]
        
        # Final update: Combine all stages with weights
        for i, particle in enumerate(system.particles):
            # Update position: y_n+1 = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            particle.position = initial_positions[i] + (dt / 6) * (
                k1_pos[i] + 2 * k2_pos[i] + 2 * k3_pos[i] + k4_pos[i]
            )
            
            # Update velocity: v_n+1 = v_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            particle.velocity = initial_velocities[i] + (dt / 6) * (
                k1_vel[i] + 2 * k2_vel[i] + 2 * k3_vel[i] + k4_vel[i]
            )
        
        # Update accelerations for the final state
        system.update_accelerations()