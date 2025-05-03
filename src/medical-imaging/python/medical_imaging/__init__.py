"""
Medical Imaging Workload for GPU-accelerated image processing and analysis.

This package provides Python bindings for the medical imaging C++/CUDA implementation.
"""

import numpy as np
from ._medical_imaging import *

__author__ = "Scott Friedman"
__copyright__ = "Copyright 2025 Scott Friedman. All rights reserved."
__version__ = "0.1.0"

# Initialize the adaptive kernel manager
try:
    AdaptiveKernelManager.get_instance().initialize()
except Exception as e:
    print(f"Warning: Could not initialize GPU: {e}")

# Helper functions for working with NumPy arrays
def load_image(filename):
    """Load an image from file and return as a NumPy array."""
    img = MedicalImage(filename)
    return to_numpy(img)

def save_image(array, filename, format=""):
    """Save a NumPy array as an image."""
    img = from_numpy(array)
    return img.save(filename, format)

def apply_filter(array, filter_method, **kwargs):
    """Apply a filter to a NumPy array.
    
    Args:
        array: Input NumPy array
        filter_method: FilterMethod enum value
        **kwargs: Filter parameters
    
    Returns:
        Filtered NumPy array
    """
    # Create configuration
    config = ProcessingConfig()
    config.compute_backend = ComputeBackend.CUDA
    
    # Set filter parameters
    for key, value in kwargs.items():
        if isinstance(value, float):
            config.scalar_params[key] = value
        elif isinstance(value, str):
            config.string_params[key] = value
        elif isinstance(value, (list, np.ndarray)):
            config.vector_params[key] = list(value)
    
    # Create image filter
    filter_processor = ProcessorFactory.get_instance().create_image_filter(config)
    filter_processor.set_method(filter_method)
    filter_processor.initialize()
    
    # Process image
    input_img = from_numpy(array)
    output_img = filter_processor.process(input_img)
    
    return to_numpy(output_img)

def apply_segmentation(array, segmentation_method, **kwargs):
    """Apply segmentation to a NumPy array.
    
    Args:
        array: Input NumPy array
        segmentation_method: SegmentationMethod enum value
        **kwargs: Segmentation parameters
    
    Returns:
        Segmented NumPy array
    """
    # Create configuration
    config = ProcessingConfig()
    config.compute_backend = ComputeBackend.CUDA
    
    # Set segmentation parameters
    for key, value in kwargs.items():
        if isinstance(value, float):
            config.scalar_params[key] = value
        elif isinstance(value, str):
            config.string_params[key] = value
        elif isinstance(value, (list, np.ndarray)):
            config.vector_params[key] = list(value)
    
    # Create image segmenter
    segmenter = ProcessorFactory.get_instance().create_image_segmenter(config)
    segmenter.set_method(segmentation_method)
    segmenter.initialize()
    
    # Process image
    input_img = from_numpy(array)
    output_img = segmenter.process(input_img)
    
    return to_numpy(output_img)

def reconstruct_ct(projections, angles, method=ReconstructionMethod.FilteredBackProjection, **kwargs):
    """Reconstruct CT image from projections.
    
    Args:
        projections: NumPy array of projections
        angles: List of projection angles (in radians)
        method: ReconstructionMethod enum value
        **kwargs: Reconstruction parameters
    
    Returns:
        Reconstructed CT image as NumPy array
    """
    # Create configuration
    config = ProcessingConfig()
    config.compute_backend = ComputeBackend.CUDA
    
    # Set reconstruction parameters
    for key, value in kwargs.items():
        if isinstance(value, float):
            config.scalar_params[key] = value
        elif isinstance(value, str):
            config.string_params[key] = value
        elif isinstance(value, (list, np.ndarray)):
            config.vector_params[key] = list(value)
    
    # Create CT reconstructor
    reconstructor = ProcessorFactory.get_instance().create_ct_reconstructor(config)
    reconstructor.set_method(method)
    reconstructor.set_projection_angles(angles)
    reconstructor.initialize()
    
    # Process projections
    input_proj = from_numpy(projections)
    output_img = reconstructor.process(input_proj)
    
    return to_numpy(output_img)

def register_images(fixed, moving, **kwargs):
    """Register a moving image to a fixed image.
    
    Args:
        fixed: Fixed image as NumPy array
        moving: Moving image as NumPy array
        **kwargs: Registration parameters
    
    Returns:
        Registered image as NumPy array
    """
    # Create configuration
    config = ProcessingConfig()
    config.compute_backend = ComputeBackend.CUDA
    
    # Set registration parameters
    for key, value in kwargs.items():
        if isinstance(value, float):
            config.scalar_params[key] = value
        elif isinstance(value, str):
            config.string_params[key] = value
        elif isinstance(value, (list, np.ndarray)):
            config.vector_params[key] = list(value)
    
    # Create registration processor
    registration = ProcessorFactory.get_instance().create_image_registration(config)
    registration.initialize()
    
    # Register images
    fixed_img = from_numpy(fixed)
    moving_img = from_numpy(moving)
    registered_img = registration.register_images(fixed_img, moving_img)
    
    return to_numpy(registered_img)