#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Example demonstrating Lennard-Jones fluid simulation using the Python bindings.
"""

import sys
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import our module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from molecular_dynamics import MDSimulation, IntegrationType, ThermostatType, get_device_info
    from visualization import visualize_system, visualize_trajectory, energy_plot, temperature_plot
except ImportError:
    print("Error: Could not import molecular dynamics module. Make sure to build the Python bindings.")
    print("Run: cd /path/to/molecular-dynamics/cpp && mkdir -p build && cd build && cmake .. -DMOLECULAR_DYNAMICS_BUILD_PYTHON_BINDINGS=ON && make")
    sys.exit(1)


def progress_callback(sim):
    """Callback function to report simulation progress."""
    step = sim.current_step
    total = sim.total_steps
    time = sim.current_time
    temp = sim.system.temperature()
    energy = sim.system.total_energy()
    
    print(f"Step {step}/{total} ({100.0 * step / total:.1f}%) - Time: {time:.3f} ps - "
          f"Temperature: {temp:.2f} K - Energy: {energy:.2f} kJ/mol")


def main():
    """Run a Lennard-Jones fluid simulation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Lennard-Jones fluid simulation example")
    parser.add_argument("-n", "--num-particles", type=int, default=1000,
                     help="Number of particles")
    parser.add_argument("-b", "--box-size", type=float, default=20.0,
                     help="Box size in Angstroms")
    parser.add_argument("-t", "--temperature", type=float, default=300.0,
                     help="Temperature in Kelvin")
    parser.add_argument("-d", "--duration", type=float, default=10.0,
                     help="Simulation duration in picoseconds")
    parser.add_argument("-dt", "--timestep", type=float, default=0.001,
                     help="Time step in picoseconds")
    parser.add_argument("-i", "--integrator", type=str, default="velocity-verlet",
                     choices=["velocity-verlet", "leapfrog", "beeman"],
                     help="Integration method")
    parser.add_argument("--thermostat", type=str, default="berendsen",
                     choices=["none", "berendsen", "andersen", "nose-hoover"],
                     help="Thermostat type")
    parser.add_argument("-o", "--output", type=str, default="trajectory.dcd",
                     help="Output trajectory file")
    parser.add_argument("--no-gpu", action="store_true",
                     help="Disable GPU acceleration")
    args = parser.parse_args()
    
    # Get device info
    print("Device information:")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Map string arguments to enum values
    integrator_map = {
        "velocity-verlet": IntegrationType.VelocityVerlet,
        "leapfrog": IntegrationType.Leapfrog,
        "beeman": IntegrationType.Beeman
    }
    
    thermostat_map = {
        "none": ThermostatType.None,
        "berendsen": ThermostatType.Berendsen,
        "andersen": ThermostatType.Andersen,
        "nose-hoover": ThermostatType.NoseHoover
    }
    
    integrator_type = integrator_map[args.integrator]
    thermostat_type = thermostat_map[args.thermostat]
    
    # Print simulation parameters
    print("Simulation parameters:")
    print(f"  Number of particles: {args.num_particles}")
    print(f"  Box size: {args.box_size} Å")
    print(f"  Temperature: {args.temperature} K")
    print(f"  Duration: {args.duration} ps")
    print(f"  Time step: {args.timestep} ps")
    print(f"  Integrator: {args.integrator}")
    print(f"  Thermostat: {args.thermostat}")
    print(f"  Output: {args.output}")
    print(f"  GPU: {'disabled' if args.no_gpu else 'enabled'}")
    print()
    
    # Create simulation
    print("Creating simulation...")
    sim = MDSimulation.create_lj_fluid_simulation(
        args.num_particles,
        args.box_size,
        integrator_type,
        thermostat_type,
        args.temperature,
        args.timestep,
        args.duration
    )
    
    # Setup data collection
    energy_data = {
        'time': [],
        'kinetic': [],
        'potential': [],
        'total': []
    }
    
    temperature_data = {
        'time': [],
        'temperature': []
    }
    
    positions_data = []
    
    # Run simulation
    print("Running simulation...")
    start_time = time.time()
    
    # Collect initial state
    energy_data['time'].append(sim.current_time)
    energy_data['kinetic'].append(sim.system.kinetic_energy())
    energy_data['potential'].append(sim.system.potential_energy())
    energy_data['total'].append(sim.system.total_energy())
    
    temperature_data['time'].append(sim.current_time)
    temperature_data['temperature'].append(sim.system.temperature())
    
    # Store initial positions
    positions_data.append(sim.system.get_positions().copy())
    
    # Define custom callback to collect data
    def data_collection_callback(sim):
        progress_callback(sim)
        
        # Collect energy data
        energy_data['time'].append(sim.current_time)
        energy_data['kinetic'].append(sim.system.kinetic_energy())
        energy_data['potential'].append(sim.system.potential_energy())
        energy_data['total'].append(sim.system.total_energy())
        
        # Collect temperature data
        temperature_data['time'].append(sim.current_time)
        temperature_data['temperature'].append(sim.system.temperature())
        
        # Store positions (every 10 frames)
        if sim.current_step % 10 == 0:
            positions_data.append(sim.system.get_positions().copy())
    
    # Run the simulation with data collection
    metrics = sim.run(data_collection_callback, 100)
    
    elapsed_time = time.time() - start_time
    
    # Print performance metrics
    print("\nPerformance metrics:")
    print(f"  Wall clock time: {elapsed_time:.2f} s")
    print(f"  Simulation steps: {sim.current_step}")
    print(f"  Steps per second: {sim.current_step / elapsed_time:.2f}")
    print(f"  Initial energy: {metrics.get('initial_energy', 'N/A')}")
    print(f"  Final energy: {metrics.get('final_energy', 'N/A')}")
    
    if 'energy_conservation_error' in metrics:
        print(f"  Energy conservation error: {metrics['energy_conservation_error'] * 100:.6f}%")
    
    # Save trajectory
    print(f"\nSaving trajectory to {args.output}...")
    sim.save_trajectory(args.output)
    
    # Generate plots
    print("Generating plots...")
    
    # Energy plot
    energy_plot(energy_data, save_path="energy_plot.png")
    
    # Temperature plot
    temperature_plot(temperature_data, target_temp=args.temperature, save_path="temperature_plot.png")
    
    # Visualize final state
    fig = visualize_system(sim.system, show_axes=True)
    if fig:
        fig.savefig("final_state.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Create trajectory animation (if matplotlib is available)
    try:
        visualize_trajectory(positions_data, save_path="trajectory.gif", fps=10)
        print("Trajectory animation saved to trajectory.gif")
    except Exception as e:
        print(f"Error creating trajectory animation: {e}")
    
    print("\nSimulation completed successfully!")


if __name__ == "__main__":
    main()