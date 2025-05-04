"""
Digital Elevation Model (DEM) processing module

This module provides GPU-accelerated operations for processing digital
elevation models, including viewshed analysis, terrain derivatives,
hydrological modeling, and path optimization.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Callable, Union

try:
    import _geospatial  # Native C++/CUDA extension
except ImportError:
    raise ImportError(
        "Failed to import _geospatial module. Please ensure the geospatial C++ library "
        "is properly built and installed. See README.md for installation instructions."
    )

class GeoTransform:
    """
    Geospatial transformation parameters (affine transformation).
    
    Follows the GDAL GeoTransform convention:
    - gt[0]: top-left x
    - gt[1]: w-e pixel resolution
    - gt[2]: row rotation (typically 0)
    - gt[3]: top-left y
    - gt[4]: column rotation (typically 0)
    - gt[5]: n-s pixel resolution (negative)
    """
    
    def __init__(self, parameters: List[float]):
        """
        Initialize GeoTransform with GDAL-style parameters.
        
        Args:
            parameters: List of 6 geotransform parameters
        """
        if len(parameters) != 6:
            raise ValueError("GeoTransform requires exactly 6 parameters")
        self.parameters = list(parameters)
    
    def pixel_to_geo(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to geographic coordinates.
        
        Args:
            pixel_x: Pixel X coordinate (column)
            pixel_y: Pixel Y coordinate (row)
            
        Returns:
            Tuple of (geo_x, geo_y) in geographic coordinates
        """
        geo_x = self.parameters[0] + pixel_x * self.parameters[1] + pixel_y * self.parameters[2]
        geo_y = self.parameters[3] + pixel_x * self.parameters[4] + pixel_y * self.parameters[5]
        return (geo_x, geo_y)
    
    def geo_to_pixel(self, geo_x: float, geo_y: float) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates.
        
        Args:
            geo_x: Geographic X coordinate
            geo_y: Geographic Y coordinate
            
        Returns:
            Tuple of (pixel_x, pixel_y) in pixel coordinates
        """
        # For simplicity, assume no rotation (parameters[2] and parameters[4] are 0)
        if self.parameters[2] != 0 or self.parameters[4] != 0:
            raise NotImplementedError("Rotated geotransforms are not currently supported")
        
        pixel_x = int((geo_x - self.parameters[0]) / self.parameters[1])
        pixel_y = int((geo_y - self.parameters[3]) / self.parameters[5])
        return (pixel_x, pixel_y)
    
    def __repr__(self) -> str:
        return f"GeoTransform({self.parameters})"


class DEMProcessor:
    """
    Digital Elevation Model (DEM) processor for geospatial analysis.
    
    This class provides GPU-accelerated operations for DEM processing,
    including viewshed analysis, terrain derivatives calculation,
    hydrological modeling, and path optimization.
    """
    
    def __init__(self, dem_file: str, device_id: int = 0):
        """
        Initialize the DEM processor.
        
        Args:
            dem_file: Path to DEM file (GeoTIFF format)
            device_id: CUDA device ID (default: 0)
        """
        if not os.path.exists(dem_file):
            raise FileNotFoundError(f"DEM file not found: {dem_file}")
        
        self._processor = _geospatial.DEMProcessor(dem_file, device_id)
    
    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get DEM dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        return self._processor.get_dimensions()
    
    def get_geotransform(self) -> GeoTransform:
        """
        Get DEM geotransform parameters.
        
        Returns:
            GeoTransform object
        """
        return GeoTransform(self._processor.get_geotransform())
    
    def compute_viewshed(self, 
                         observer_point: Tuple[float, float], 
                         observer_height: float = 1.8, 
                         radius: float = 0.0) -> np.ndarray:
        """
        Compute viewshed from a specified observer point.
        
        Args:
            observer_point: Geographic coordinates of the observer (x, y)
            observer_height: Height of the observer above the terrain (meters)
            radius: Maximum viewshed radius (meters, 0 for unlimited)
            
        Returns:
            Binary numpy array (1 = visible, 0 = not visible)
        """
        result = self._processor.compute_viewshed(observer_point, observer_height, radius)
        return np.array(result, copy=False)
    
    def compute_terrain_derivatives(self, z_factor: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Compute terrain derivatives (slope, aspect, curvature).
        
        Args:
            z_factor: Vertical exaggeration factor
            
        Returns:
            Dictionary with 'slope', 'aspect', and 'curvature' arrays
        """
        slope, aspect, curvature = self._processor.compute_terrain_derivatives(z_factor)
        
        return {
            'slope': np.array(slope, copy=False),      # degrees
            'aspect': np.array(aspect, copy=False),    # degrees
            'curvature': np.array(curvature, copy=False)
        }
    
    def compute_hydrological_features(self) -> Dict[str, np.ndarray]:
        """
        Compute hydrological features (flow direction, flow accumulation).
        
        Returns:
            Dictionary with 'flow_dir' and 'flow_acc' arrays
        """
        flow_dir, flow_acc = self._processor.compute_hydrological_features()
        
        return {
            'flow_dir': np.array(flow_dir, copy=False),  # D8 flow direction (0-255)
            'flow_acc': np.array(flow_acc, copy=False)   # Flow accumulation
        }
    
    def compute_least_cost_path(self, 
                               start_point: Tuple[float, float], 
                               end_point: Tuple[float, float],
                               cost_function: Optional[Callable[[float, float, float, float], float]] = None) -> np.ndarray:
        """
        Compute least-cost path between two points.
        
        Args:
            start_point: Start point in geographic coordinates (x, y)
            end_point: End point in geographic coordinates (x, y)
            cost_function: Function that computes transition cost between adjacent cells
                           Arguments: (elevation1, elevation2, slope, aspect)
            
        Returns:
            Numpy array of path points as (x, y, z) coordinates
        """
        # If no cost function is provided, use default C++ implementation
        if cost_function is None:
            path = self._processor.compute_least_cost_path(start_point, end_point)
        else:
            # Create a Python-callable wrapper for the cost function
            def cost_wrapper(elev1, elev2, slope, aspect):
                return cost_function(elev1, elev2, slope, aspect)
            
            path = self._processor.compute_least_cost_path_with_callback(
                start_point, end_point, cost_wrapper)
        
        return np.array(path)
    
    def fill_sinks(self, z_limit: float = float('inf')) -> 'DEMProcessor':
        """
        Fill DEM sinks/depressions for hydrological analysis.
        
        Args:
            z_limit: Maximum z-value difference for fill
            
        Returns:
            New DEMProcessor with filled DEM
        """
        # Create temporary file for the filled DEM
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Fill sinks and save to temp file
        filled_data = self._processor.fill_sinks(z_limit)
        self.save_result(filled_data, tmp_path)
        
        # Create new DEMProcessor with the filled DEM
        return DEMProcessor(tmp_path, device_id=self._processor.get_device_id())
    
    def compute_statistics(self) -> Dict[str, float]:
        """
        Compute DEM statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self._processor.compute_statistics()
        
        return {
            'min_elevation': stats.min_elevation,
            'max_elevation': stats.max_elevation,
            'mean_elevation': stats.mean_elevation,
            'std_dev': stats.std_dev,
            'mean_slope': stats.mean_slope,
            'mean_aspect': stats.mean_aspect,
            'ruggedness_index': stats.ruggedness_index,
            'hypsometric_integral': stats.hypsometric_integral
        }
    
    def save_result(self, data: Union[np.ndarray, List[float]], output_file: str, data_type: int = 6) -> bool:
        """
        Save result to GeoTIFF file.
        
        Args:
            data: Data to save (numpy array or list of floats)
            output_file: Output file path
            data_type: GDAL data type (default: 6 = GDT_Float32)
            
        Returns:
            True if successful, False otherwise
        """
        if isinstance(data, np.ndarray):
            # Convert numpy array to list
            data_list = data.flatten().tolist()
        else:
            data_list = data
        
        return self._processor.save_result(data_list, output_file, data_type)
    
    def resample(self, target_resolution: float, resampling_method: int = 1) -> 'DEMProcessor':
        """
        Resample DEM to different resolution.
        
        Args:
            target_resolution: Target resolution in units of the DEM's CRS
            resampling_method: Resampling method (0=nearest, 1=bilinear, 2=cubic, 
                              3=cubicspline, 4=lanczos)
            
        Returns:
            New DEMProcessor with resampled DEM
        """
        # Create temporary file for the resampled DEM
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Resample and save to temp file
        resampled_data = self._processor.resample(target_resolution, resampling_method)
        self.save_result(resampled_data, tmp_path)
        
        # Create new DEMProcessor with the resampled DEM
        return DEMProcessor(tmp_path, device_id=self._processor.get_device_id())