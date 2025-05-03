# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Python interface to the molecular dynamics simulation engine.
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional, Union
import os
import time

# Import the C++ extension module
try:
    from pymolecular_dynamics import (
        # Classes
        Vec3, Atom, MolecularSystem, Simulation,
        VelocityVerletIntegrator, LeapfrogIntegrator, BeemanIntegrator,
        BerendsenThermostat, AndersenThermostat, NoseHooverThermostat,
        DeviceCapabilities,
        
        # Enums
        AtomType, ForceFieldType, IntegrationType, ThermostatType, GPUDeviceType,
        
        # Functions
        detect_device_capabilities, vec3_array_to_numpy, numpy_to_vec3_array, atoms_to_numpy,
        
        # Constants
        DEFAULT_TIMESTEP, DEFAULT_TEMPERATURE, DEFAULT_CUTOFF, BOLTZMANN
    )
except ImportError:
    raise ImportError(
        "Failed to import pymolecular_dynamics. Make sure the C++ extension is built properly. "
        "Run 'cd /path/to/molecular-dynamics/cpp && mkdir -p build && cd build && "
        "cmake .. -DMOLECULAR_DYNAMICS_BUILD_PYTHON_BINDINGS=ON && make'"
    )


class MDSystem:
    """High-level wrapper for molecular dynamics system."""
    
    def __init__(self, system: MolecularSystem):
        """Initialize with a MolecularSystem instance."""
        self.system = system
    
    @classmethod
    def create_water_box(cls, box_size: float, density: float = 1.0) -> "MDSystem":
        """Create a water box system."""
        system = MolecularSystem.create_water_box(box_size, density)
        return cls(system)
    
    @classmethod
    def create_lj_fluid(cls, 
                       num_particles: int, 
                       box_size: float,
                       temperature: float = DEFAULT_TEMPERATURE,
                       seed: int = 0) -> "MDSystem":
        """Create a Lennard-Jones fluid system."""
        system = MolecularSystem.create_lj_fluid(num_particles, box_size, temperature, seed)
        return cls(system)
    
    @classmethod
    def load_from_pdb(cls, pdb_file: str) -> "MDSystem":
        """Load a molecular system from a PDB file."""
        system = MolecularSystem.load_from_pdb(pdb_file)
        return cls(system)
    
    @classmethod
    def load_with_forcefield(cls, 
                            pdb_file: str, 
                            topology_file: str,
                            parameter_file: str,
                            force_field_type: ForceFieldType = ForceFieldType.AMBER) -> "MDSystem":
        """Load a molecular system from a PDB file with force field parameters."""
        system = MolecularSystem.load_with_forcefield(
            pdb_file, topology_file, parameter_file, force_field_type)
        return cls(system)
    
    def get_positions(self) -> np.ndarray:
        """Get atom positions as a numpy array."""
        atoms = self.system.atoms()
        return np.array([[atom.position().x, atom.position().y, atom.position().z] 
                        for atom in atoms])
    
    def get_velocities(self) -> np.ndarray:
        """Get atom velocities as a numpy array."""
        atoms = self.system.atoms()
        return np.array([[atom.velocity().x, atom.velocity().y, atom.velocity().z] 
                        for atom in atoms])
    
    def get_forces(self) -> np.ndarray:
        """Get forces on atoms as a numpy array."""
        atoms = self.system.atoms()
        return np.array([[atom.force().x, atom.force().y, atom.force().z] 
                        for atom in atoms])
    
    def get_masses(self) -> np.ndarray:
        """Get atom masses as a numpy array."""
        atoms = self.system.atoms()
        return np.array([atom.mass() for atom in atoms])
    
    def get_charges(self) -> np.ndarray:
        """Get atom charges as a numpy array."""
        atoms = self.system.atoms()
        return np.array([atom.charge() for atom in atoms])
    
    def set_positions(self, positions: np.ndarray) -> None:
        """Set atom positions from a numpy array."""
        atoms = self.system.atoms()
        if len(positions) != len(atoms):
            raise ValueError("Number of positions does not match number of atoms")
        
        for i, atom in enumerate(atoms):
            atom.set_position(Vec3(positions[i, 0], positions[i, 1], positions[i, 2]))
    
    def set_velocities(self, velocities: np.ndarray) -> None:
        """Set atom velocities from a numpy array."""
        atoms = self.system.atoms()
        if len(velocities) != len(atoms):
            raise ValueError("Number of velocities does not match number of atoms")
        
        for i, atom in enumerate(atoms):
            atom.set_velocity(Vec3(velocities[i, 0], velocities[i, 1], velocities[i, 2]))
    
    def total_energy(self) -> float:
        """Get the total energy of the system."""
        return self.system.total_energy()
    
    def kinetic_energy(self) -> float:
        """Get the kinetic energy of the system."""
        return self.system.total_kinetic_energy()
    
    def potential_energy(self) -> float:
        """Get the potential energy of the system."""
        return self.system.total_potential_energy()
    
    def temperature(self) -> float:
        """Get the temperature of the system."""
        return self.system.temperature()
    
    def update_forces(self, use_gpu: bool = True) -> None:
        """Update forces on all atoms."""
        if use_gpu:
            self.system.update_forces_gpu()
        else:
            self.system.update_forces_cpu()


class MDSimulation:
    """High-level wrapper for molecular dynamics simulation."""
    
    def __init__(self, 
                system: Union[MDSystem, MolecularSystem],
                integrator_type: IntegrationType = IntegrationType.VelocityVerlet,
                thermostat_type: ThermostatType = ThermostatType.Berendsen,
                dt: float = DEFAULT_TIMESTEP,
                temperature: float = DEFAULT_TEMPERATURE,
                duration: float = 10.0):
        """Initialize the simulation."""
        # Get the underlying C++ MolecularSystem
        if isinstance(system, MDSystem):
            cpp_system = system.system
        else:
            cpp_system = system
        
        # Create integrator based on type
        if integrator_type == IntegrationType.VelocityVerlet:
            integrator = VelocityVerletIntegrator()
        elif integrator_type == IntegrationType.Leapfrog:
            integrator = LeapfrogIntegrator()
        elif integrator_type == IntegrationType.Beeman:
            integrator = BeemanIntegrator()
        else:
            raise ValueError(f"Unsupported integrator type: {integrator_type}")
        
        # Create thermostat based on type
        if thermostat_type == ThermostatType.None:
            thermostat = None
        elif thermostat_type == ThermostatType.Berendsen:
            thermostat = BerendsenThermostat()
        elif thermostat_type == ThermostatType.Andersen:
            thermostat = AndersenThermostat()
        elif thermostat_type == ThermostatType.NoseHoover:
            thermostat = NoseHooverThermostat()
        else:
            raise ValueError(f"Unsupported thermostat type: {thermostat_type}")
        
        # Create the C++ Simulation object
        self.simulation = Simulation(cpp_system, integrator, thermostat, dt, duration)
        
        # Set the target temperature
        self.simulation.set_temperature(temperature)
        
        # Detect device capabilities
        self.device_capabilities = self.simulation.device_capabilities()
        
        # Store wrapped system
        if isinstance(system, MDSystem):
            self._system = system
        else:
            self._system = MDSystem(cpp_system)
    
    @classmethod
    def create_water_box_simulation(cls, 
                                  box_size: float,
                                  integrator_type: IntegrationType = IntegrationType.VelocityVerlet,
                                  thermostat_type: ThermostatType = ThermostatType.Berendsen,
                                  temperature: float = DEFAULT_TEMPERATURE,
                                  dt: float = DEFAULT_TIMESTEP,
                                  duration: float = 10.0) -> "MDSimulation":
        """Create a simulation with a water box system."""
        sim = Simulation.create_water_box_simulation(
            box_size, integrator_type, thermostat_type, temperature, dt, duration)
        
        result = cls.__new__(cls)  # Create instance without calling __init__
        result.simulation = sim
        result.device_capabilities = sim.device_capabilities()
        result._system = MDSystem(sim.system())
        return result
    
    @classmethod
    def create_lj_fluid_simulation(cls,
                                 num_particles: int,
                                 box_size: float,
                                 integrator_type: IntegrationType = IntegrationType.VelocityVerlet,
                                 thermostat_type: ThermostatType = ThermostatType.Berendsen,
                                 temperature: float = DEFAULT_TEMPERATURE,
                                 dt: float = DEFAULT_TIMESTEP,
                                 duration: float = 10.0,
                                 seed: int = 0) -> "MDSimulation":
        """Create a simulation with a Lennard-Jones fluid system."""
        sim = Simulation.create_lj_fluid_simulation(
            num_particles, box_size, integrator_type, thermostat_type, 
            temperature, dt, duration, seed)
        
        result = cls.__new__(cls)  # Create instance without calling __init__
        result.simulation = sim
        result.device_capabilities = sim.device_capabilities()
        result._system = MDSystem(sim.system())
        return result
    
    @classmethod
    def create_from_pdb(cls,
                      pdb_file: str,
                      integrator_type: IntegrationType = IntegrationType.VelocityVerlet,
                      thermostat_type: ThermostatType = ThermostatType.Berendsen,
                      temperature: float = DEFAULT_TEMPERATURE,
                      dt: float = DEFAULT_TIMESTEP,
                      duration: float = 10.0) -> "MDSimulation":
        """Create a simulation from a PDB file."""
        sim = Simulation.create_from_pdb(
            pdb_file, integrator_type, thermostat_type, temperature, dt, duration)
        
        result = cls.__new__(cls)  # Create instance without calling __init__
        result.simulation = sim
        result.device_capabilities = sim.device_capabilities()
        result._system = MDSystem(sim.system())
        return result
    
    @property
    def system(self) -> MDSystem:
        """Get the simulation system."""
        return self._system
    
    @property
    def dt(self) -> float:
        """Get the simulation time step."""
        return self.simulation.dt()
    
    @property
    def temperature(self) -> float:
        """Get the target temperature."""
        return self.simulation.temperature()
    
    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the target temperature."""
        self.simulation.set_temperature(value)
    
    @property
    def current_time(self) -> float:
        """Get the current simulation time."""
        return self.simulation.current_time()
    
    @property
    def current_step(self) -> int:
        """Get the current simulation step."""
        return self.simulation.current_step()
    
    @property
    def total_steps(self) -> int:
        """Get the total number of simulation steps."""
        return self.simulation.total_steps()
    
    def step(self) -> None:
        """Advance the simulation by one time step."""
        self.simulation.step()
    
    def run(self, callback: Optional[Callable] = None, callback_interval: int = 100) -> Dict[str, float]:
        """Run the simulation with optional progress callback."""
        start_time = time.time()
        
        # Create a wrapper callback that calls the Python callback
        if callback:
            def cpp_callback(sim):
                callback(self)
            
            # Run the simulation with the C++ callback
            self.simulation.run(cpp_callback, callback_interval)
        else:
            # Run without callback
            self.simulation.run(None, callback_interval)
        
        # Get performance metrics
        metrics = self.simulation.get_performance_metrics()
        
        # Add Python-specific metrics
        metrics['python_total_time'] = time.time() - start_time
        
        return metrics
    
    def save_state(self, filename: str) -> None:
        """Save the simulation state to a file."""
        self.simulation.save_state(filename)
    
    def load_state(self, filename: str) -> None:
        """Load a simulation state from a file."""
        self.simulation.load_state(filename)
    
    def save_trajectory(self, filename: str) -> None:
        """Save the trajectory to a DCD file."""
        self.simulation.save_trajectory(filename)
    
    def get_visualization_data(self, include_velocities: bool = True, include_forces: bool = False) -> Dict:
        """Get data for visualization."""
        return self.simulation.create_visualization_data(include_velocities, include_forces)
    
    def get_device_info(self) -> Dict[str, str]:
        """Get information about the computational device being used."""
        caps = self.device_capabilities
        
        device_type_str = str(caps.device_type).split('.')[-1]
        
        return {
            'device_type': device_type_str,
            'compute_capability': f"{caps.compute_capability_major}.{caps.compute_capability_minor}",
            'global_memory': f"{caps.global_memory_bytes / (1024**2):.1f} MB",
            'multiprocessors': str(caps.multiprocessor_count),
            'max_threads_per_block': str(caps.max_threads_per_block),
            'max_shared_memory': f"{caps.max_shared_memory_per_block / 1024:.1f} KB",
            'optimal_block_size': str(caps.get_optimal_block_size()),
            'optimal_tile_size': str(caps.get_optimal_tile_size())
        }


# Function to detect available hardware
def get_device_info() -> Dict[str, str]:
    """Get information about the available computational device."""
    caps = detect_device_capabilities()
    
    device_type_str = str(caps.device_type).split('.')[-1]
    
    return {
        'device_type': device_type_str,
        'compute_capability': f"{caps.compute_capability_major}.{caps.compute_capability_minor}",
        'global_memory': f"{caps.global_memory_bytes / (1024**2):.1f} MB",
        'multiprocessors': str(caps.multiprocessor_count),
        'max_threads_per_block': str(caps.max_threads_per_block),
        'max_shared_memory': f"{caps.max_shared_memory_per_block / 1024:.1f} KB",
        'optimal_block_size': str(caps.get_optimal_block_size()),
        'optimal_tile_size': str(caps.get_optimal_tile_size())
    }