#!/usr/bin/env python3
"""
Example script for running a shallow water simulation.

Author: Scott Friedman
Copyright 2025 Scott Friedman. All rights reserved.
"""

import os
import sys
import argparse
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    # Try to import the module from the package
    from weather_sim.weather_simulation import (
        WeatherSimulationWrapper, 
        is_cuda_available,
        get_device_info,
        get_available_initial_conditions
    )
    from weather_sim.visualization import (
        visualize_height, 
        visualize_velocity, 
        visualize_vorticity,
        animate_simulation
    )
except ImportError:
    print("Error: Could not import weather_sim module.")
    print("Make sure you've built the C++ library and Python bindings.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Shallow Water Simulation Example')
    
    # Grid parameters
    parser.add_argument('--width', type=int, default=256,
                        help='Grid width in cells')
    parser.add_argument('--height', type=int, default=256,
                        help='Grid height in cells')
    
    # Simulation parameters
    parser.add_argument('--dt', type=float, default=0.01,
                        help='Time step size')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of simulation steps')
    parser.add_argument('--method', type=str, default='rk4',
                        choices=['euler', 'rk2', 'rk4', 'adams_bashforth', 'semi_implicit'],
                        help='Integration method')
    
    # Initial condition
    parser.add_argument('--initial', type=str, default='vortex',
                        help='Initial condition')
    
    # Backend selection
    parser.add_argument('--backend', type=str, default='adaptive',
                        choices=['cuda', 'cpu', 'hybrid', 'adaptive'],
                        help='Compute backend')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of CPU threads (0 = auto)')
    
    # Output options
    parser.add_argument('--output-interval', type=int, default=5,
                        help='Interval between outputs (0 = disabled)')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--animate', action='store_true',
                        help='Create animation')
    parser.add_argument('--save-animation', type=str, default='',
                        help='Save animation to file')
    
    # Other options
    parser.add_argument('--list-initial', action='store_true',
                        help='List available initial conditions')
    parser.add_argument('--device-info', action='store_true',
                        help='Show device information')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Show device information if requested
    if args.device_info:
        info = get_device_info()
        print("Device Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print(f"CUDA Available: {is_cuda_available()}")
        return
    
    # List initial conditions if requested
    if args.list_initial:
        conditions = get_available_initial_conditions()
        print("Available Initial Conditions:")
        for condition in conditions:
            print(f"  {condition}")
        return
    
    # Create output directory if it doesn't exist
    if args.output_interval > 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and configure simulation
    sim = WeatherSimulationWrapper(
        width=args.width,
        height=args.height,
        dt=args.dt,
        integration_method=args.method,
        backend=args.backend,
        device_id=args.device,
        threads=args.threads,
        output_interval=args.output_interval,
        output_path=args.output_dir
    )
    
    # Set initial condition
    if args.initial == 'vortex':
        sim.set_initial_condition('vortex', x_center=0.5, y_center=0.5, radius=0.2, strength=10.0)
    elif args.initial == 'jet_stream':
        sim.set_initial_condition('jet_stream', y_center=0.5, width=0.1, strength=20.0)
    elif args.initial == 'breaking_wave':
        sim.set_initial_condition('breaking_wave', amplitude=1.0, wavelength=0.2)
    elif args.initial == 'zonal_flow':
        sim.set_initial_condition('zonal_flow', u_max=20.0, beta=0.2)
    else:
        try:
            sim.set_initial_condition(args.initial)
        except Exception as e:
            print(f"Error setting initial condition '{args.initial}': {e}")
            print("Available initial conditions:")
            for cond in get_available_initial_conditions():
                print(f"  {cond}")
            return
    
    # Initialize simulation
    sim.initialize()
    
    # Print simulation info
    grid = sim.get_grid()
    print(f"Simulation initialized:")
    print(f"  Grid: {grid.get_width()} x {grid.get_height()}")
    print(f"  Time step: {args.dt}")
    print(f"  Integration method: {args.method}")
    print(f"  Initial condition: {args.initial}")
    print(f"  Backend: {args.backend}")
    print(f"  CUDA available: {is_cuda_available()}")
    
    # Run simulation
    print(f"Running simulation for {args.steps} steps...")
    start_time = time.time()
    sim.run(args.steps)
    end_time = time.time()
    
    # Print performance info
    elapsed = end_time - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")
    print(f"Steps per second: {args.steps / elapsed:.2f}")
    
    # Get final state
    grid = sim.get_grid()
    u, v = grid.get_velocity_field()
    height = grid.get_height_field()
    vorticity = grid.get_vorticity_field()
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Height field with velocity vectors
    visualize_height(height, u, v, show_velocity=True, 
                    title=f"Height Field (t={sim.simulation.get_current_time():.2f})",
                    save_path=os.path.join(args.output_dir, "height_field.png"))
    
    # Velocity field
    visualize_velocity(u, v, title=f"Velocity Field (t={sim.simulation.get_current_time():.2f})",
                      streamlines=True, save_path=os.path.join(args.output_dir, "velocity_field.png"))
    
    # Vorticity field
    visualize_vorticity(vorticity, title=f"Vorticity Field (t={sim.simulation.get_current_time():.2f})",
                       save_path=os.path.join(args.output_dir, "vorticity_field.png"))
    
    # Create animation if requested
    if args.animate:
        print("Creating animation...")
        snapshots = sim.get_output_data()
        
        if not snapshots:
            print("Error: No output data available for animation.")
            print("Make sure output_interval > 0 and steps > output_interval.")
            return
        
        anim = animate_simulation(
            snapshots, 
            field_type="height", 
            fps=10, 
            show_velocity=True,
            save_path=args.save_animation if args.save_animation else None
        )
        
        # Show animation if not saving to file
        if not args.save_animation:
            plt.figure(figsize=(10, 8))
            plt.show()
        else:
            print(f"Animation saved to {args.save_animation}")


if __name__ == "__main__":
    main()