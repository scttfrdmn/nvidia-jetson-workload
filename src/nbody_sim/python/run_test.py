#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 nvidia-jetson-workload contributors

"""
Simple test script for the N-body simulation.

This script creates a simple N-body system and runs a short simulation
to verify that the code is working correctly.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

from src.nbody_sim.python.particle import ParticleSystem
from src.nbody_sim.python.integrator import LeapfrogIntegrator
from src.nbody_sim.python.simulation import NBodySimulation


def run_test_simulation() -> Tuple[NBodySimulation, List[np.ndarray]]:
    """
    Run a test simulation and return the results.
    
    Returns:
        Tuple of (simulation, positions_history)
    """
    print("Creating a two-body system...")
    system = ParticleSystem.create_random_system(
        num_particles=100,
        box_size=5.0,
        max_mass=1.0,
        max_velocity=0.5,
        G=1.0,
        seed=42
    )
    
    print(f"System created with {system.num_particles} particles")
    
    integrator = LeapfrogIntegrator()
    
    # Configure simulation parameters
    dt = 0.01
    duration = 5.0
    
    simulation = NBodySimulation(
        system=system,
        integrator=integrator,
        dt=dt,
        duration=duration
    )
    
    print(f"Simulation configured with dt={dt}, duration={duration}")
    print(f"Total steps: {simulation.total_steps}")
    
    # Store positions for visualization
    positions_history = []
    
    def record_positions(sim):
        # Store a copy of current positions
        positions = np.array([np.copy(p.position) for p in sim.system.particles])
        positions_history.append(positions)
    
    # Run simulation
    print("Running simulation...")
    start_time = time.time()
    
    simulation.run(
        callback=record_positions,
        callback_interval=10  # Record every 10 steps
    )
    
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    
    # Calculate energy conservation
    metrics = simulation.get_performance_metrics()
    print(f"Energy conservation error: {metrics['energy_conservation_error']:.6f}")
    print(f"Steps per second: {metrics['steps_per_second']:.2f}")
    
    return simulation, positions_history


def plot_results(positions_history: List[np.ndarray], output_dir: str = "./output") -> None:
    """
    Create some basic plots of the simulation results.
    
    Args:
        positions_history: List of particle positions at different time steps
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure
    fig = plt.figure(figsize=(12, 10))
    
    # Plot initial and final positions (2D projection)
    ax1 = fig.add_subplot(221)
    ax1.set_title("Initial particle positions")
    ax1.scatter(
        positions_history[0][:, 0],  # x coordinates
        positions_history[0][:, 1],  # y coordinates
        s=10, alpha=0.7
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.grid(True)
    
    ax2 = fig.add_subplot(222)
    ax2.set_title("Final particle positions")
    ax2.scatter(
        positions_history[-1][:, 0],  # x coordinates
        positions_history[-1][:, 1],  # y coordinates
        s=10, alpha=0.7
    )
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)
    
    # Plot a different projection
    ax3 = fig.add_subplot(223)
    ax3.set_title("Initial particle positions (XZ plane)")
    ax3.scatter(
        positions_history[0][:, 0],  # x coordinates
        positions_history[0][:, 2],  # z coordinates
        s=10, alpha=0.7
    )
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.grid(True)
    
    ax4 = fig.add_subplot(224)
    ax4.set_title("Final particle positions (XZ plane)")
    ax4.scatter(
        positions_history[-1][:, 0],  # x coordinates
        positions_history[-1][:, 2],  # z coordinates
        s=10, alpha=0.7
    )
    ax4.set_xlabel("X")
    ax4.set_ylabel("Z")
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, "nbody_results.png")
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    # Create a simple animation of the first 20 particles
    # (to keep the file size manageable)
    num_particles_to_plot = min(20, len(positions_history[0]))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim([
        np.min([pos[:num_particles_to_plot, 0].min() for pos in positions_history]) - 1,
        np.max([pos[:num_particles_to_plot, 0].max() for pos in positions_history]) + 1
    ])
    ax.set_ylim([
        np.min([pos[:num_particles_to_plot, 1].min() for pos in positions_history]) - 1,
        np.max([pos[:num_particles_to_plot, 1].max() for pos in positions_history]) + 1
    ])
    
    ax.set_title(f"Particle Motion (first {num_particles_to_plot} particles)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    
    # Plot the path of each particle
    for i in range(num_particles_to_plot):
        trajectory_x = [pos[i, 0] for pos in positions_history]
        trajectory_y = [pos[i, 1] for pos in positions_history]
        ax.plot(trajectory_x, trajectory_y, '-', alpha=0.3)
    
    # Plot the final positions
    ax.scatter(
        positions_history[-1][:num_particles_to_plot, 0],
        positions_history[-1][:num_particles_to_plot, 1],
        s=30, alpha=0.7
    )
    
    plt.tight_layout()
    
    # Save figure
    trajectory_file = os.path.join(output_dir, "nbody_trajectories.png")
    plt.savefig(trajectory_file)
    print(f"Trajectory plot saved to {trajectory_file}")


def main():
    parser = argparse.ArgumentParser(description="Run a test N-body simulation")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory for output files")
    args = parser.parse_args()
    
    # Run simulation
    simulation, positions_history = run_test_simulation()
    
    # Plot results
    try:
        import matplotlib
        plot_results(positions_history, args.output_dir)
    except ImportError:
        print("Matplotlib not available - skipping plots")
    
    # Save final state
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "nbody_final_state.npz")
    simulation.save_state(output_file)
    print(f"Final state saved to {output_file}")


if __name__ == "__main__":
    main()