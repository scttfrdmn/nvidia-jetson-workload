"""
Visualization utilities for Weather Simulation.

Author: Scott Friedman
Copyright 2025 Scott Friedman. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple, Union, Optional, Callable
import os
import io
from PIL import Image

try:
    from ..pyweather_sim import WeatherGrid, WeatherSimulation, PerformanceMetrics
except ImportError:
    # Mock classes for documentation without C++ library
    class WeatherGrid:
        pass
    
    class WeatherSimulation:
        pass
    
    class PerformanceMetrics:
        pass


def visualize_field(
    field: np.ndarray,
    title: str = "Scalar Field",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_label: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize a 2D scalar field.
    
    Args:
        field: 2D numpy array containing the field data
        title: Plot title
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        colorbar_label: Label for the colorbar
        ax: Matplotlib axes to plot on (if None, creates new figure)
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, doesn't save)
        show: Whether to show the figure
    
    Returns:
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot the field
    im = ax.imshow(field, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='nearest')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def visualize_velocity(
    u: np.ndarray,
    v: np.ndarray,
    title: str = "Velocity Field",
    cmap: str = "viridis",
    density: int = 20,
    scale: float = 1.0,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    streamlines: bool = True
) -> plt.Figure:
    """
    Visualize a 2D velocity field with streamlines or arrows.
    
    Args:
        u: 2D numpy array containing the x-component of velocity
        v: 2D numpy array containing the y-component of velocity
        title: Plot title
        cmap: Colormap name
        density: Density of streamlines/arrows
        scale: Scaling factor for arrows
        ax: Matplotlib axes to plot on (if None, creates new figure)
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, doesn't save)
        show: Whether to show the figure
        streamlines: Whether to use streamlines (True) or arrows (False)
    
    Returns:
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Calculate velocity magnitude
    speed = np.sqrt(u**2 + v**2)
    
    # Plot the speed as background
    im = ax.imshow(speed, origin='lower', cmap=cmap, aspect='equal',
                   interpolation='nearest')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Speed')
    
    # Create grid for streamlines/arrows
    y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    
    # Skip points for better visualization
    skip = max(1, min(u.shape) // density)
    
    if streamlines:
        # Plot streamlines
        ax.streamplot(x, y, u.T, v.T, color='white', linewidth=1.0, density=density/10,
                     arrowstyle='->', arrowsize=1.5)
    else:
        # Plot arrows
        ax.quiver(x[::skip, ::skip], y[::skip, ::skip], 
                 u[::skip, ::skip], v[::skip, ::skip],
                 color='white', scale=scale, scale_units='inches')
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def visualize_vorticity(
    vorticity: np.ndarray,
    title: str = "Vorticity Field",
    cmap: str = "RdBu_r",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize a vorticity field.
    
    Args:
        vorticity: 2D numpy array containing the vorticity field
        title: Plot title
        cmap: Colormap name
        ax: Matplotlib axes to plot on (if None, creates new figure)
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, doesn't save)
        show: Whether to show the figure
    
    Returns:
        Matplotlib figure object
    """
    # Calculate vmin and vmax symmetrically
    vmax = max(abs(np.min(vorticity)), abs(np.max(vorticity)))
    vmin = -vmax
    
    return visualize_field(
        vorticity, title, cmap, vmin, vmax, 'Vorticity',
        ax, figsize, save_path, show
    )


def visualize_height(
    height: np.ndarray,
    u: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    title: str = "Height Field",
    cmap: str = "terrain",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True,
    show_velocity: bool = False
) -> plt.Figure:
    """
    Visualize a height field, optionally with velocity vectors.
    
    Args:
        height: 2D numpy array containing the height field
        u: Optional 2D numpy array containing the x-component of velocity
        v: Optional 2D numpy array containing the y-component of velocity
        title: Plot title
        cmap: Colormap name
        ax: Matplotlib axes to plot on (if None, creates new figure)
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, doesn't save)
        show: Whether to show the figure
        show_velocity: Whether to overlay velocity vectors
    
    Returns:
        Matplotlib figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot the height field
    im = ax.imshow(height, origin='lower', cmap=cmap, aspect='equal',
                   interpolation='nearest')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Height')
    
    # Add velocity vectors if provided
    if show_velocity and u is not None and v is not None:
        # Create grid
        y, x = np.mgrid[0:height.shape[0], 0:height.shape[1]]
        
        # Skip points for better visualization
        skip = max(1, min(height.shape) // 20)
        
        # Plot arrows
        ax.quiver(x[::skip, ::skip], y[::skip, ::skip],
                 u[::skip, ::skip], v[::skip, ::skip],
                 color='black', scale=20, scale_units='inches')
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def animate_simulation(
    snapshots: List[Dict],
    field_type: str = "height",
    fps: int = 10,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "viridis",
    save_path: Optional[str] = None,
    show_velocity: bool = False,
    repeat: bool = True
) -> FuncAnimation:
    """
    Create an animation of a simulation.
    
    Args:
        snapshots: List of simulation snapshots (dictionaries with field data)
        field_type: Type of field to animate (height, vorticity, etc.)
        fps: Frames per second
        figsize: Figure size (width, height) in inches
        cmap: Colormap name
        save_path: Path to save the animation (if None, doesn't save)
        show_velocity: Whether to overlay velocity vectors
        repeat: Whether to repeat the animation
    
    Returns:
        Matplotlib animation object
    """
    if not snapshots:
        raise ValueError("No snapshots provided")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine field to plot
    if field_type == "height":
        field_key = "height"
        title = "Height Field"
        colorbar_label = "Height"
        plot_cmap = "terrain"
    elif field_type == "vorticity":
        field_key = "vorticity"
        title = "Vorticity Field"
        colorbar_label = "Vorticity"
        plot_cmap = "RdBu_r"
    else:
        raise ValueError(f"Unknown field type: {field_type}")
    
    # Get all field values for colormap normalization
    all_values = np.concatenate([snapshot[field_key].flatten() for snapshot in snapshots])
    
    if field_type == "vorticity":
        # Symmetric colormap for vorticity
        vmax = max(abs(np.min(all_values)), abs(np.max(all_values)))
        vmin = -vmax
    else:
        # Regular colormap for other fields
        vmin = np.min(all_values)
        vmax = np.max(all_values)
    
    # Initialize plot
    im = ax.imshow(
        snapshots[0][field_key],
        origin='lower',
        cmap=plot_cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='equal',
        interpolation='nearest'
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    
    # Add title with time counter
    time_text = ax.set_title(f"{title} (t=0.0)")
    
    # Add velocity quiver if requested
    quiver = None
    if show_velocity:
        # Create grid
        y, x = np.mgrid[0:snapshots[0][field_key].shape[0], 0:snapshots[0][field_key].shape[1]]
        
        # Skip points for better visualization
        skip = max(1, min(snapshots[0][field_key].shape) // 20)
        
        # Initialize quiver
        quiver = ax.quiver(
            x[::skip, ::skip], y[::skip, ::skip],
            snapshots[0]["u"][::skip, ::skip], snapshots[0]["v"][::skip, ::skip],
            color='black', scale=20, scale_units='inches'
        )
    
    # Animation update function
    def update(frame):
        snapshot = snapshots[frame]
        im.set_array(snapshot[field_key])
        time_text.set_text(f"{title} (t={snapshot['time']:.2f})")
        
        if show_velocity and quiver:
            skip = max(1, min(snapshot[field_key].shape) // 20)
            quiver.set_UVC(
                snapshot["u"][::skip, ::skip],
                snapshot["v"][::skip, ::skip]
            )
        
        return [im, time_text] + ([quiver] if quiver else [])
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(snapshots),
        interval=1000/fps, blit=True, repeat=repeat
    )
    
    # Save if requested
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
    
    plt.close(fig)
    
    return anim


def plot_performance(
    metrics: PerformanceMetrics,
    title: str = "Performance Metrics",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot performance metrics.
    
    Args:
        metrics: PerformanceMetrics object
        title: Plot title
        figsize: Figure size (width, height) in inches
        save_path: Path to save the figure (if None, doesn't save)
        show: Whether to show the figure
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract metrics
    labels = ['Compute', 'Memory Transfer', 'I/O', 'Other']
    values = [
        metrics.compute_time_ms,
        metrics.memory_transfer_time_ms,
        metrics.io_time_ms,
        metrics.total_time_ms - metrics.compute_time_ms - metrics.memory_transfer_time_ms - metrics.io_time_ms
    ]
    
    # Filter out zero values
    non_zero_indices = [i for i, v in enumerate(values) if v > 0]
    values = [values[i] for i in non_zero_indices]
    labels = [labels[i] for i in non_zero_indices]
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Make text more readable
    for text in texts + autotexts:
        text.set_fontsize(12)
    
    # Add title and some stats
    ax.set_title(title)
    ax.text(
        0, -1.2,
        f"Total time: {metrics.total_time_ms:.2f} ms, Steps: {metrics.num_steps}, "
        f"Time per step: {metrics.total_time_ms / max(1, metrics.num_steps):.2f} ms",
        ha='center', fontsize=12
    )
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.set_aspect('equal')
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig