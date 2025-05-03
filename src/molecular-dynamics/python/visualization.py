# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Visualization utilities for molecular dynamics simulations.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import os
import sys
import warnings

# Try to import visualization libraries, but make the module usable without them
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib not found. Visualization functions will not work.")

try:
    import nglview as nv
    HAS_NGLVIEW = True
except ImportError:
    HAS_NGLVIEW = False
    warnings.warn("NGLView not found. 3D visualization in notebooks will not work.")


def visualize_system(system, backend='matplotlib', show_axes=True, 
                    width=800, height=600, **kwargs):
    """
    Visualize a molecular system.
    
    Parameters
    ----------
    system : MDSystem or MolecularSystem
        The molecular system to visualize.
    backend : str
        Visualization backend to use ('matplotlib' or 'nglview').
    show_axes : bool
        Whether to show coordinate axes.
    width : int
        Width of the visualization in pixels.
    height : int
        Height of the visualization in pixels.
    **kwargs : dict
        Additional keyword arguments passed to the backend.
    
    Returns
    -------
    object
        The visualization object (plt.Figure or nglview.NGLWidget).
    """
    # Handle different system types
    if hasattr(system, 'system'):
        # It's an MDSystem
        system = system.system
    
    # Get atom positions and properties
    num_atoms = system.size()
    positions = np.zeros((num_atoms, 3))
    atom_types = []
    
    for i in range(num_atoms):
        atom = system.atom(i)
        positions[i, 0] = atom.position().x
        positions[i, 1] = atom.position().y
        positions[i, 2] = atom.position().z
        atom_types.append(str(atom.type()).split('.')[-1])
    
    if backend == 'matplotlib':
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for visualization.")
        
        fig = plt.figure(figsize=(width/100, height/100))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colors for different atom types
        color_map = {
            'Hydrogen': 'white',
            'Carbon': 'black',
            'Nitrogen': 'blue',
            'Oxygen': 'red',
            'Sulfur': 'yellow',
            'Phosphorus': 'orange',
            'Other': 'gray'
        }
        
        # Define sizes for different atom types
        size_map = {
            'Hydrogen': 10,
            'Carbon': 20,
            'Nitrogen': 20,
            'Oxygen': 20,
            'Sulfur': 25,
            'Phosphorus': 25,
            'Other': 15
        }
        
        # Plot atoms
        for i in range(num_atoms):
            atom_type = atom_types[i]
            color = color_map.get(atom_type, 'gray')
            size = size_map.get(atom_type, 15)
            ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], 
                      color=color, s=size)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set axis labels
        if show_axes:
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
        else:
            ax.set_axis_off()
        
        # Set title
        ax.set_title(f'Molecular System ({num_atoms} atoms)')
        
        plt.tight_layout()
        return fig
    
    elif backend == 'nglview':
        if not HAS_NGLVIEW:
            raise ImportError("NGLView is required for 3D visualization in notebooks.")
        
        # Create a structure that NGLView can understand
        try:
            import mdtraj as md
            has_mdtraj = True
        except ImportError:
            has_mdtraj = False
            
        try:
            from biopandas.pdb import PandasPdb
            has_biopandas = True
        except ImportError:
            has_biopandas = False
        
        # TODO: Implement conversion to nglview-compatible structure
        # This will depend on the available packages
        
        # For now, we'll just show a warning
        warnings.warn("NGLView visualization not fully implemented yet.")
        return None
    
    else:
        raise ValueError(f"Unknown visualization backend: {backend}")


def visualize_trajectory(trajectory_data, backend='matplotlib', 
                        show_animation=True, save_path=None, fps=10,
                        **kwargs):
    """
    Visualize a molecular dynamics trajectory.
    
    Parameters
    ----------
    trajectory_data : dict or list of dict
        Trajectory data with positions for each frame.
    backend : str
        Visualization backend to use ('matplotlib' or 'nglview').
    show_animation : bool
        Whether to show the animation (or return the animation object).
    save_path : str, optional
        Path to save the animation (e.g., 'animation.gif', 'animation.mp4').
    fps : int
        Frames per second for the animation.
    **kwargs : dict
        Additional keyword arguments passed to the backend.
    
    Returns
    -------
    object
        The animation object if show_animation is False.
    """
    if backend == 'matplotlib':
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for visualization.")
        
        # Check if trajectory_data is a list or a single frame
        if not isinstance(trajectory_data, list):
            trajectory_data = [trajectory_data]
        
        # Get positions for each frame
        positions_frames = []
        for frame in trajectory_data:
            if 'positions_x' in frame:
                # It's a dict with positions_x, positions_y, positions_z
                x = frame['positions_x']
                y = frame['positions_y']
                z = frame['positions_z']
                positions = np.column_stack((x, y, z))
            elif 'positions' in frame:
                # It's a dict with positions as an array
                positions = frame['positions']
            else:
                # Try to interpret as array of positions
                positions = np.array(frame)
            
            positions_frames.append(positions)
        
        # Create figure and animation
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get limits for all frames
        all_positions = np.vstack(positions_frames)
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()
        
        # Add a small margin
        margin = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)
        
        # Initial scatter plot
        scatter = ax.scatter([], [], [], c='b', s=10)
        
        # Update function for animation
        def update(frame):
            pos = positions_frames[frame]
            scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
            ax.set_title(f'Frame {frame}')
            return scatter,
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=len(positions_frames),
                           blit=True, interval=1000/fps)
        
        # Set axis labels
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        
        # Save animation if requested
        if save_path:
            ani.save(save_path, writer='pillow', fps=fps)
        
        # Show or return animation
        if show_animation:
            plt.tight_layout()
            plt.show()
            return None
        else:
            return ani
    
    elif backend == 'nglview':
        if not HAS_NGLVIEW:
            raise ImportError("NGLView is required for 3D visualization in notebooks.")
        
        # TODO: Implement NGLView trajectory visualization
        warnings.warn("NGLView trajectory visualization not fully implemented yet.")
        return None
    
    else:
        raise ValueError(f"Unknown visualization backend: {backend}")


def energy_plot(energy_data, show=True, save_path=None, **kwargs):
    """
    Plot energy components over time.
    
    Parameters
    ----------
    energy_data : dict
        Dictionary with energy components (kinetic, potential, total).
    show : bool
        Whether to show the plot.
    save_path : str, optional
        Path to save the plot image.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib.
    
    Returns
    -------
    tuple
        Figure and axes objects if show is False.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for visualization.")
    
    # Extract energy components
    time = energy_data.get('time', np.arange(len(energy_data['kinetic'])))
    kinetic = energy_data.get('kinetic', [])
    potential = energy_data.get('potential', [])
    total = energy_data.get('total', [])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(kinetic) > 0:
        ax.plot(time, kinetic, label='Kinetic Energy', color='blue')
    
    if len(potential) > 0:
        ax.plot(time, potential, label='Potential Energy', color='red')
    
    if len(total) > 0:
        ax.plot(time, total, label='Total Energy', color='black', linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Energy (kJ/mol)')
    ax.set_title('Energy Components')
    ax.legend()
    ax.grid(True)
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or return plot
    if show:
        plt.show()
        return None
    else:
        return fig, ax


def temperature_plot(temperature_data, target_temp=None, show=True, save_path=None, **kwargs):
    """
    Plot temperature over time.
    
    Parameters
    ----------
    temperature_data : dict or array-like
        Temperature values over time.
    target_temp : float, optional
        Target temperature to show as a horizontal line.
    show : bool
        Whether to show the plot.
    save_path : str, optional
        Path to save the plot image.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib.
    
    Returns
    -------
    tuple
        Figure and axes objects if show is False.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib is required for visualization.")
    
    # Extract data
    if isinstance(temperature_data, dict):
        time = temperature_data.get('time', np.arange(len(temperature_data['temperature'])))
        temp = temperature_data['temperature']
    else:
        time = np.arange(len(temperature_data))
        temp = temperature_data
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(time, temp, label='Temperature', color='red')
    
    if target_temp is not None:
        ax.axhline(y=target_temp, color='blue', linestyle='--', 
                  label=f'Target: {target_temp} K')
    
    # Set labels and title
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Temperature Over Time')
    ax.legend()
    ax.grid(True)
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show or return plot
    if show:
        plt.show()
        return None
    else:
        return fig, ax