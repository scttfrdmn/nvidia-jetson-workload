"""
Visualization module for Medical Imaging workload.

This module provides visualization utilities for medical images.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def create_medical_colormap():
    """Create a medical-style colormap."""
    return LinearSegmentedColormap.from_list(
        'medical',
        ['black', 'blue', 'purple', 'red', 'yellow', 'white'],
        N=256
    )

def plot_image(image, title=None, cmap=None, figsize=(8, 8), colorbar=True, ax=None):
    """Plot a 2D medical image.
    
    Args:
        image: 2D numpy array
        title: Title string (optional)
        cmap: Colormap (default: grayscale)
        figsize: Figure size as tuple
        colorbar: Whether to show colorbar
        ax: Matplotlib axis to plot on (optional)
        
    Returns:
        Figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    im = ax.imshow(image, cmap=cmap or 'gray')
    
    if title:
        ax.set_title(title)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return fig, ax

def plot_images(images, titles=None, cmap=None, figsize=(15, 5), cmaps=None):
    """Plot multiple images in a row.
    
    Args:
        images: List of 2D numpy arrays
        titles: List of title strings (optional)
        cmap: Colormap (default: grayscale)
        figsize: Figure size as tuple
        cmaps: List of colormaps for each image (optional)
        
    Returns:
        Figure and list of axis objects
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        current_cmap = cmaps[i] if cmaps and i < len(cmaps) else cmap or 'gray'
        im = ax.imshow(img, cmap=current_cmap)
        
        if titles and i < len(titles):
            ax.set_title(titles[i])
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig, axes

def plot_overlay(image, segmentation, alpha=0.5, title=None, figsize=(8, 8)):
    """Plot an image with segmentation overlay.
    
    Args:
        image: 2D numpy array of the image
        segmentation: 2D numpy array of the segmentation mask
        alpha: Transparency of the overlay
        title: Title string (optional)
        figsize: Figure size as tuple
        
    Returns:
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot base image
    ax.imshow(image, cmap='gray')
    
    # Create colored segmentation for overlay
    colored_segmentation = np.zeros((*segmentation.shape, 4))
    
    # Regions with value > 0 are colored
    mask = segmentation > 0
    colored_segmentation[mask, 0] = 1.0  # R
    colored_segmentation[mask, 3] = alpha  # Alpha
    
    # Different segmentation regions get different colors
    unique_values = np.unique(segmentation)
    unique_values = unique_values[unique_values > 0]  # Skip background
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_values)))
    
    for i, val in enumerate(unique_values):
        region_mask = segmentation == val
        colored_segmentation[region_mask, 0:3] = colors[i, 0:3]
        colored_segmentation[region_mask, 3] = alpha
    
    # Plot overlay
    ax.imshow(colored_segmentation)
    
    if title:
        ax.set_title(title)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig, ax

def plot_slice(volume, axis=0, slice_index=None, title=None, cmap=None, figsize=(8, 8)):
    """Plot a slice from a 3D volume.
    
    Args:
        volume: 3D numpy array
        axis: Axis along which to take the slice (0, 1, or 2)
        slice_index: Index of the slice (default: middle slice)
        title: Title string (optional)
        cmap: Colormap (default: grayscale)
        figsize: Figure size as tuple
        
    Returns:
        Figure and axis objects
    """
    if slice_index is None:
        slice_index = volume.shape[axis] // 2
    
    if axis == 0:
        slice_data = volume[slice_index, :, :]
    elif axis == 1:
        slice_data = volume[:, slice_index, :]
    else:  # axis == 2
        slice_data = volume[:, :, slice_index]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(slice_data, cmap=cmap or 'gray')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Slice {slice_index} along axis {axis}')
    
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return fig, ax

def plot_orthogonal_slices(volume, center=None, titles=None, cmap=None, figsize=(12, 4)):
    """Plot orthogonal slices from a 3D volume.
    
    Args:
        volume: 3D numpy array
        center: Center coordinates [z, y, x] (default: middle of volume)
        titles: List of title strings (optional)
        cmap: Colormap (default: grayscale)
        figsize: Figure size as tuple
        
    Returns:
        Figure and list of axis objects
    """
    if center is None:
        center = [s // 2 for s in volume.shape]
    
    z, y, x = center
    
    axial = volume[z, :, :]
    coronal = volume[:, y, :]
    sagittal = volume[:, :, x]
    
    if titles is None:
        titles = [f'Axial (z={z})', f'Coronal (y={y})', f'Sagittal (x={x})']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(axial, cmap=cmap or 'gray')
    axes[0].set_title(titles[0])
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    axes[1].imshow(coronal, cmap=cmap or 'gray')
    axes[1].set_title(titles[1])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    axes[2].imshow(sagittal, cmap=cmap or 'gray')
    axes[2].set_title(titles[2])
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    
    plt.tight_layout()
    return fig, axes

def create_volume_animation(volume, axis=0, cmap=None, figsize=(8, 8), interval=50, title=None):
    """Create animation of slices from a 3D volume.
    
    Args:
        volume: 3D numpy array
        axis: Axis along which to take slices (0, 1, or 2)
        cmap: Colormap (default: grayscale)
        figsize: Figure size as tuple
        interval: Delay between frames in milliseconds
        title: Title pattern with {slice_idx} placeholder (optional)
        
    Returns:
        Animation object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_slices = volume.shape[axis]
    
    if axis == 0:
        slices = [volume[i, :, :] for i in range(num_slices)]
    elif axis == 1:
        slices = [volume[:, i, :] for i in range(num_slices)]
    else:  # axis == 2
        slices = [volume[:, :, i] for i in range(num_slices)]
    
    im = ax.imshow(slices[0], cmap=cmap or 'gray')
    ax.set_xticks([])
    ax.set_yticks([])
    
    if title:
        title_text = ax.set_title(title.format(slice_idx=0))
    
    # Update function for animation
    def update(frame):
        im.set_array(slices[frame])
        if title:
            title_text.set_text(title.format(slice_idx=frame))
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=num_slices, 
                                 interval=interval, blit=True)
    
    return ani

def plot_histogram(image, bins=50, title=None, figsize=(8, 6), log_scale=False):
    """Plot histogram of image intensity values.
    
    Args:
        image: 2D or 3D numpy array
        bins: Number of histogram bins
        title: Title string (optional)
        figsize: Figure size as tuple
        log_scale: Whether to use log scale for y-axis
        
    Returns:
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(image.flatten(), bins=bins, color='steelblue', edgecolor='none', alpha=0.7)
    
    if log_scale:
        ax.set_yscale('log')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Histogram of Intensity Values')
    
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Frequency')
    
    return fig, ax

def plot_surface_3d(image, threshold=0.5, title=None, figsize=(10, 8), cmap='viridis'):
    """Create a 3D surface plot of a 2D image.
    
    Args:
        image: 2D numpy array
        threshold: Threshold for surface (optional)
        title: Title string (optional)
        figsize: Figure size as tuple
        cmap: Colormap
        
    Returns:
        Figure and axis objects
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    # Plot surface
    surf = ax.plot_surface(x, y, image, cmap=cmap, linewidth=0, antialiased=True)
    
    if title:
        ax.set_title(title)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig, ax

def plot_ct_projection(sinogram, angles=None, title=None, figsize=(10, 8)):
    """Plot a CT sinogram.
    
    Args:
        sinogram: 2D numpy array (projections x detector elements)
        angles: Array of projection angles in radians (optional)
        title: Title string (optional)
        figsize: Figure size as tuple
        
    Returns:
        Figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if angles is None:
        # Assume angles are evenly spaced from 0 to 180 degrees
        extent = [0, 180, 0, sinogram.shape[1]]
    else:
        # Convert angles to degrees for display
        angles_deg = np.rad2deg(angles)
        extent = [angles_deg.min(), angles_deg.max(), 0, sinogram.shape[1]]
    
    im = ax.imshow(sinogram.T, cmap='gray', aspect='auto', extent=extent)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('CT Sinogram')
    
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Detector Position')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return fig, ax

def plot_comparison(before, after, label1='Before', label2='After', title=None, 
                   figsize=(12, 6), cmap=None):
    """Plot before and after images for comparison.
    
    Args:
        before: 2D numpy array of the before image
        after: 2D numpy array of the after image
        label1: Label for the before image
        label2: Label for the after image
        title: Title string (optional)
        figsize: Figure size as tuple
        cmap: Colormap (default: grayscale)
        
    Returns:
        Figure and axis objects
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    im1 = axes[0].imshow(before, cmap=cmap or 'gray')
    axes[0].set_title(label1)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(after, cmap=cmap or 'gray')
    axes[1].set_title(label2)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig, axes

def plot_difference(image1, image2, title=None, figsize=(8, 8), cmap='RdBu'):
    """Plot the difference between two images.
    
    Args:
        image1: 2D numpy array of the first image
        image2: 2D numpy array of the second image
        title: Title string (optional)
        figsize: Figure size as tuple
        cmap: Colormap for difference image
        
    Returns:
        Figure and axis objects
    """
    # Compute difference
    diff = image1 - image2
    
    # Symmetric colormap range
    vmax = max(abs(diff.min()), abs(diff.max()))
    vmin = -vmax
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Difference')
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return fig, ax