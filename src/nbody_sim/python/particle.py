# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Particle and ParticleSystem classes for N-body simulation.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from copy import deepcopy


class Particle:
    """
    Represents a single particle in the N-body simulation.
    
    Attributes:
        position (np.ndarray): 3D position vector [x, y, z]
        velocity (np.ndarray): 3D velocity vector [vx, vy, vz]
        acceleration (np.ndarray): 3D acceleration vector [ax, ay, az]
        mass (float): Mass of the particle
        particle_id (int): Unique identifier for the particle
    """
    
    def __init__(
        self, 
        position: np.ndarray, 
        velocity: np.ndarray, 
        mass: float, 
        particle_id: int
    ) -> None:
        """
        Initialize a particle with position, velocity, and mass.
        
        Args:
            position: 3D position vector [x, y, z]
            velocity: 3D velocity vector [vx, vy, vz]
            mass: Mass of the particle
            particle_id: Unique identifier for the particle
        """
        self.position = position.astype(np.float64)
        self.velocity = velocity.astype(np.float64)
        self.acceleration = np.zeros(3, dtype=np.float64)
        self.mass = float(mass)
        self.particle_id = particle_id
    
    def update_position(self, dt: float) -> None:
        """
        Update the position based on current velocity.
        
        Args:
            dt: Time step size
        """
        self.position += self.velocity * dt
    
    def update_velocity(self, dt: float) -> None:
        """
        Update the velocity based on current acceleration.
        
        Args:
            dt: Time step size
        """
        self.velocity += self.acceleration * dt
    
    def kinetic_energy(self) -> float:
        """
        Calculate the kinetic energy of the particle.
        
        Returns:
            Kinetic energy (0.5 * m * v^2)
        """
        return 0.5 * self.mass * np.sum(self.velocity**2)
    
    def __repr__(self) -> str:
        """String representation of the particle."""
        return (
            f"Particle(id={self.particle_id}, "
            f"position={self.position}, "
            f"velocity={self.velocity}, "
            f"mass={self.mass})"
        )
    
    def copy(self) -> 'Particle':
        """Create a deep copy of this particle."""
        return Particle(
            position=np.copy(self.position),
            velocity=np.copy(self.velocity),
            mass=self.mass,
            particle_id=self.particle_id
        )


class ParticleSystem:
    """
    System of particles for N-body simulation.
    
    Attributes:
        particles (List[Particle]): List of particles in the system
        num_particles (int): Number of particles in the system
        G (float): Gravitational constant
    """
    
    def __init__(
        self, 
        particles: List[Particle], 
        G: float = 6.67430e-11
    ) -> None:
        """
        Initialize a system with a list of particles.
        
        Args:
            particles: List of particles in the system
            G: Gravitational constant (default is G in SI units)
        """
        self.particles = particles
        self.num_particles = len(particles)
        self.G = G
    
    def update_accelerations(self) -> None:
        """
        Update accelerations of all particles based on gravitational interactions.
        
        This implements the O(n^2) direct summation approach. More efficient methods
        like Barnes-Hut or Fast Multipole Method would be used for large particle counts.
        """
        # Reset all accelerations
        for particle in self.particles:
            particle.acceleration[:] = 0.0
        
        # Calculate accelerations due to pairwise gravitational interactions
        for i in range(self.num_particles):
            for j in range(self.num_particles):
                if i == j:
                    continue
                
                p1 = self.particles[i]
                p2 = self.particles[j]
                
                # Vector from particle i to particle j
                r_vector = p2.position - p1.position
                
                # Distance between particles
                r_squared = np.sum(r_vector**2)
                r = np.sqrt(r_squared)
                
                # Add a small softening factor to prevent singularities
                softening = 1e-6
                
                # Gravitational force: F = G * m1 * m2 * r_vector / r^3
                # Acceleration: a = F / m1 = G * m2 * r_vector / r^3
                if r > softening:
                    p1.acceleration += self.G * p2.mass * r_vector / (r_squared * r)
    
    def total_mass(self) -> float:
        """
        Calculate the total mass of the system.
        
        Returns:
            Sum of all particle masses
        """
        return sum(p.mass for p in self.particles)
    
    def center_of_mass(self) -> np.ndarray:
        """
        Calculate the center of mass of the system.
        
        Returns:
            3D position vector of the center of mass
        """
        total_mass = self.total_mass()
        if total_mass == 0:
            return np.zeros(3)
        
        com = np.zeros(3)
        for p in self.particles:
            com += p.mass * p.position
        
        return com / total_mass
    
    def total_momentum(self) -> np.ndarray:
        """
        Calculate the total momentum of the system.
        
        Returns:
            3D momentum vector
        """
        momentum = np.zeros(3)
        for p in self.particles:
            momentum += p.mass * p.velocity
        
        return momentum
    
    def total_angular_momentum(self) -> np.ndarray:
        """
        Calculate the total angular momentum of the system.
        
        Returns:
            3D angular momentum vector
        """
        angular_momentum = np.zeros(3)
        for p in self.particles:
            angular_momentum += p.mass * np.cross(p.position, p.velocity)
        
        return angular_momentum
    
    def total_kinetic_energy(self) -> float:
        """
        Calculate the total kinetic energy of the system.
        
        Returns:
            Sum of kinetic energies of all particles
        """
        return sum(p.kinetic_energy() for p in self.particles)
    
    def total_potential_energy(self) -> float:
        """
        Calculate the total potential energy of the system.
        
        Returns:
            Sum of potential energies of all particle pairs
        """
        potential_energy = 0.0
        
        # Sum over all unique pairs of particles
        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                p1 = self.particles[i]
                p2 = self.particles[j]
                
                # Vector from particle i to particle j
                r_vector = p2.position - p1.position
                
                # Distance between particles
                r = np.sqrt(np.sum(r_vector**2))
                
                # Add a small softening factor to prevent singularities
                softening = 1e-6
                if r < softening:
                    r = softening
                
                # Gravitational potential energy: U = -G * m1 * m2 / r
                potential_energy += -self.G * p1.mass * p2.mass / r
        
        return potential_energy
    
    def total_energy(self) -> float:
        """
        Calculate the total energy of the system (kinetic + potential).
        
        Returns:
            Total energy
        """
        return self.total_kinetic_energy() + self.total_potential_energy()
    
    def copy(self) -> 'ParticleSystem':
        """Create a deep copy of this particle system."""
        return ParticleSystem(
            particles=[p.copy() for p in self.particles],
            G=self.G
        )
    
    @classmethod
    def create_random_system(
        cls, 
        num_particles: int, 
        box_size: float = 10.0, 
        max_mass: float = 1.0, 
        max_velocity: float = 0.1,
        G: float = 1.0,
        seed: Optional[int] = None
    ) -> 'ParticleSystem':
        """
        Create a random particle system within a box.
        
        Args:
            num_particles: Number of particles to create
            box_size: Size of the cubic box
            max_mass: Maximum particle mass
            max_velocity: Maximum initial velocity
            G: Gravitational constant
            seed: Random seed for reproducibility
        
        Returns:
            A ParticleSystem with randomly positioned particles
        """
        if seed is not None:
            np.random.seed(seed)
        
        particles = []
        for i in range(num_particles):
            # Random position within the box
            position = (np.random.random(3) - 0.5) * 2.0 * box_size
            
            # Random velocity
            velocity = (np.random.random(3) - 0.5) * 2.0 * max_velocity
            
            # Random mass
            mass = np.random.random() * max_mass
            
            particles.append(Particle(position, velocity, mass, i))
        
        return cls(particles, G)
    
    @classmethod
    def create_solar_system(cls, scale_factor: float = 1.0) -> 'ParticleSystem':
        """
        Create a simplified model of the solar system.
        
        Args:
            scale_factor: Factor to scale distances and velocities for visualization
        
        Returns:
            A ParticleSystem representing the solar system
        """
        # Use approximate real values, but scaled for better visualization
        # Distances in AU, velocities in AU/year, masses in solar masses
        G = 4 * np.pi**2  # In AU^3 / (year^2 * solar_mass)
        
        sun = Particle(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            mass=1.0,  # 1 solar mass
            particle_id=0
        )
        
        # Approximate orbital parameters for planets
        # [name, distance(AU), mass(solar masses), orbital_velocity(AU/year)]
        planets_data = [
            ["Mercury", 0.39, 1.65e-7, 10.0],
            ["Venus", 0.72, 2.45e-6, 7.4],
            ["Earth", 1.0, 3.0e-6, 6.28],
            ["Mars", 1.52, 3.2e-7, 5.1],
            ["Jupiter", 5.2, 9.5e-4, 2.76],
            ["Saturn", 9.54, 2.85e-4, 2.04],
            ["Uranus", 19.2, 4.4e-5, 1.44],
            ["Neptune", 30.06, 5.15e-5, 1.14]
        ]
        
        particles = [sun]
        
        for i, (name, distance, mass, velocity) in enumerate(planets_data, 1):
            # Scale the distance for better visualization
            scaled_distance = distance * scale_factor
            
            # Start planets at different angles
            angle = i * np.pi / 4
            
            position = np.array([
                scaled_distance * np.cos(angle),
                scaled_distance * np.sin(angle),
                0.0
            ])
            
            # Orbital velocity is perpendicular to position vector
            velocity_vector = np.array([
                -np.sin(angle),
                np.cos(angle),
                0.0
            ]) * velocity / scale_factor  # Adjust velocity for scale
            
            planet = Particle(
                position=position,
                velocity=velocity_vector,
                mass=mass,
                particle_id=i
            )
            
            particles.append(planet)
        
        return cls(particles, G)
    
    @classmethod
    def create_galaxy_model(
        cls, 
        num_particles: int = 1000, 
        radius: float = 10.0, 
        height: float = 1.0,
        mass_range: Tuple[float, float] = (0.1, 1.0),
        G: float = 1.0
    ) -> 'ParticleSystem':
        """
        Create a simplified spiral galaxy model.
        
        Args:
            num_particles: Number of particles in the galaxy
            radius: Radius of the disk in arbitrary units
            height: Height/thickness of the disk
            mass_range: Range of particle masses (min, max)
            G: Gravitational constant
        
        Returns:
            A ParticleSystem representing a disk galaxy
        """
        particles = []
        
        # Create a central massive black hole
        black_hole = Particle(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            mass=100.0,  # Much more massive than other particles
            particle_id=0
        )
        particles.append(black_hole)
        
        # Create disk particles
        for i in range(1, num_particles):
            # Distance from center follows exponential distribution
            distance = np.random.exponential(scale=radius/3)
            if distance > radius:
                distance = radius  # Cap at maximum radius
            
            # Angle around disk with some spiral structure
            angle = np.random.uniform(0, 2 * np.pi)
            spiral_factor = 0.5  # Controls tightness of spiral arms
            spiral_angle = angle + spiral_factor * np.log(distance / 0.1)
            
            # Height above/below disk plane (thinner near center)
            z_height = np.random.normal(0, height * distance / radius)
            
            # Position
            position = np.array([
                distance * np.cos(spiral_angle),
                distance * np.sin(spiral_angle),
                z_height
            ])
            
            # Orbital velocity (Keplerian approximation)
            # v_orbital = sqrt(G * M_enclosed / r)
            enclosed_mass = black_hole.mass + i * (mass_range[0] + mass_range[1]) / 2 / num_particles
            v_orbital = np.sqrt(G * enclosed_mass / distance) if distance > 0 else 0
            
            # Tangential velocity vector
            velocity = np.array([
                -np.sin(spiral_angle),
                np.cos(spiral_angle),
                0.0
            ]) * v_orbital
            
            # Add some velocity dispersion
            velocity += np.random.normal(0, v_orbital * 0.1, 3)
            
            # Random mass
            mass = np.random.uniform(mass_range[0], mass_range[1])
            
            particles.append(Particle(position, velocity, mass, i))
        
        return cls(particles, G)