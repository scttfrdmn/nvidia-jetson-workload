# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Command-line interface for the N-body simulation.
"""

import argparse
import os
import sys
import time
import json
import numpy as np
from typing import Dict, Any, Optional

from src.nbody_sim.python.particle import ParticleSystem, Particle
from src.nbody_sim.python.integrator import (
    EulerIntegrator, 
    LeapfrogIntegrator, 
    VerletIntegrator,
    RungeKuttaIntegrator
)
from src.nbody_sim.python.simulation import NBodySimulation


def create_particle_system(args: argparse.Namespace) -> ParticleSystem:
    """
    Create a particle system based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        A configured ParticleSystem
    """
    if args.system_type == "random":
        return ParticleSystem.create_random_system(
            num_particles=args.num_particles,
            box_size=args.box_size,
            max_mass=args.max_mass,
            max_velocity=args.max_velocity,
            G=args.g_constant,
            seed=args.seed
        )
    elif args.system_type == "solar":
        return ParticleSystem.create_solar_system(scale_factor=args.scale_factor)
    elif args.system_type == "galaxy":
        return ParticleSystem.create_galaxy_model(
            num_particles=args.num_particles,
            radius=args.galaxy_radius,
            height=args.galaxy_height,
            mass_range=(args.min_mass, args.max_mass),
            G=args.g_constant
        )
    elif args.system_type == "file" and args.input_file:
        # Load from file
        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} not found.", file=sys.stderr)
            sys.exit(1)
        
        data = np.load(args.input_file)
        positions = data["positions"]
        velocities = data["velocities"]
        masses = data["masses"]
        ids = data["ids"]
        
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
        
        return ParticleSystem(particles, G=float(data.get("G", args.g_constant)))
    else:
        print("Error: Invalid system type or missing input file.", file=sys.stderr)
        sys.exit(1)


def create_integrator(args: argparse.Namespace) -> Any:
    """
    Create an integrator based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        An Integrator instance
    """
    if args.integrator == "euler":
        return EulerIntegrator()
    elif args.integrator == "leapfrog":
        return LeapfrogIntegrator()
    elif args.integrator == "verlet":
        return VerletIntegrator()
    elif args.integrator == "rk4":
        return RungeKuttaIntegrator()
    else:
        print(f"Error: Unknown integrator '{args.integrator}'", file=sys.stderr)
        sys.exit(1)


def progress_callback(sim: NBodySimulation) -> None:
    """
    Callback function to display progress during simulation.
    
    Args:
        sim: The simulation instance
    """
    progress_pct = 100 * sim.current_step / sim.total_steps
    print(f"\rProgress: {progress_pct:.1f}% (Step {sim.current_step}/{sim.total_steps})", end="")
    sys.stdout.flush()


def save_results(sim: NBodySimulation, args: argparse.Namespace) -> None:
    """
    Save simulation results to files.
    
    Args:
        sim: The simulation instance
        args: Command-line arguments
    """
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save final state
    state_file = os.path.join(args.output_dir, "final_state.npz")
    sim.save_state(state_file)
    print(f"\nFinal state saved to {state_file}")
    
    # Save performance metrics
    metrics = sim.get_performance_metrics()
    metrics_file = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Performance metrics saved to {metrics_file}")
    
    # Save visualization data if requested
    if args.save_visualization:
        vis_data = sim.create_visualization_data()
        vis_file = os.path.join(args.output_dir, "visualization.json")
        with open(vis_file, "w") as f:
            json.dump(vis_data, f)
        print(f"Visualization data saved to {vis_file}")


def main() -> None:
    """
    Main entry point for the N-body simulation.
    """
    parser = argparse.ArgumentParser(description="N-body gravitational simulation")
    
    # System configuration
    parser.add_argument("--system-type", choices=["random", "solar", "galaxy", "file"],
                        default="random", help="Type of system to simulate")
    parser.add_argument("--num-particles", type=int, default=1000,
                        help="Number of particles (for random and galaxy systems)")
    parser.add_argument("--box-size", type=float, default=10.0,
                        help="Size of box for random distribution")
    parser.add_argument("--min-mass", type=float, default=0.1,
                        help="Minimum particle mass")
    parser.add_argument("--max-mass", type=float, default=1.0,
                        help="Maximum particle mass")
    parser.add_argument("--max-velocity", type=float, default=0.1,
                        help="Maximum initial velocity (for random system)")
    parser.add_argument("--g-constant", type=float, default=1.0,
                        help="Gravitational constant")
    parser.add_argument("--scale-factor", type=float, default=1.0,
                        help="Scale factor for solar system distances")
    parser.add_argument("--galaxy-radius", type=float, default=10.0,
                        help="Radius of galaxy disk")
    parser.add_argument("--galaxy-height", type=float, default=1.0,
                        help="Height of galaxy disk")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Input file for system initialization")
    
    # Simulation parameters
    parser.add_argument("--integrator", choices=["euler", "leapfrog", "verlet", "rk4"],
                        default="leapfrog", help="Integration method")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="Time step size")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Total simulation time")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory for output files")
    parser.add_argument("--save-visualization", action="store_true",
                        help="Save visualization data")
    parser.add_argument("--disable-progress", action="store_true",
                        help="Disable progress display")
    
    args = parser.parse_args()
    
    # Create system, integrator, and simulation
    system = create_particle_system(args)
    integrator = create_integrator(args)
    
    simulation = NBodySimulation(
        system=system,
        integrator=integrator,
        dt=args.dt,
        duration=args.duration
    )
    
    # Display configuration
    print("N-body Simulation Configuration:")
    print(f"  System Type: {args.system_type}")
    print(f"  Particles: {system.num_particles}")
    print(f"  Integrator: {args.integrator}")
    print(f"  Time Step: {args.dt}")
    print(f"  Duration: {args.duration}")
    print(f"  Total Steps: {simulation.total_steps}")
    print()
    
    # Run simulation with or without progress callback
    callback = None if args.disable_progress else progress_callback
    callback_interval = max(1, int(simulation.total_steps / 100))  # Update ~100 times
    
    print("Starting simulation...")
    start_time = time.time()
    
    simulation.run(callback=callback, callback_interval=callback_interval)
    
    elapsed = time.time() - start_time
    print(f"\nSimulation completed in {elapsed:.2f} seconds")
    
    # Save results
    save_results(simulation, args)
    
    # Print performance summary
    metrics = simulation.get_performance_metrics()
    print("\nPerformance Summary:")
    print(f"  Execution Time: {metrics['total_time_ms']/1000:.2f} seconds")
    print(f"  Steps per Second: {metrics['steps_per_second']:.2f}")
    print(f"  Energy Conservation Error: {metrics['energy_conservation_error']:.6f}")


if __name__ == "__main__":
    main()