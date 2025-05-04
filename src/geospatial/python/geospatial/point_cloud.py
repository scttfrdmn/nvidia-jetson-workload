"""
Point cloud processing module for geospatial data

This module provides GPU-accelerated operations for processing
point clouds, such as LiDAR data, including classification,
filtering, DEM/DSM creation, and feature extraction.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import numpy as np
from typing import Tuple, List, Dict, Optional, Any, Union
from enum import IntEnum

try:
    import _geospatial  # Native C++/CUDA extension
except ImportError:
    raise ImportError(
        "Failed to import _geospatial module. Please ensure the geospatial C++ library "
        "is properly built and installed. See README.md for installation instructions."
    )

from .dem import DEMProcessor

class PointClass(IntEnum):
    """
    Standard LiDAR point classifications based on ASPRS LAS specification
    """
    Created = 0
    Unclassified = 1
    Ground = 2
    LowVegetation = 3
    MedVegetation = 4
    HighVegetation = 5
    Building = 6
    LowPoint = 7
    Water = 9
    Rail = 10
    RoadSurface = 11
    Bridge = 12
    WireGuard = 13
    WireConductor = 14
    TransmissionTower = 15
    WireStructure = 16
    BridgeDeck = 17
    HighNoise = 18
    
    @classmethod
    def from_string(cls, name: str) -> "PointClass":
        """Convert string name to PointClass enum"""
        name = name.lower().replace(" ", "").replace("_", "")
        for item in cls:
            if item.name.lower().replace("_", "") == name:
                return item
        raise ValueError(f"Unknown point class name: {name}")

class PointAttributes:
    """
    Additional attributes for point cloud points
    """
    
    def __init__(self,
                 intensity: int = 0,
                 return_number: int = 1,
                 num_returns: int = 1,
                 scan_direction: int = 0,
                 edge_of_flight: int = 0,
                 classification: Union[PointClass, int] = PointClass.Unclassified,
                 scan_angle_rank: int = 0,
                 user_data: int = 0,
                 point_source_id: int = 0,
                 gps_time: float = 0.0):
        """
        Initialize point attributes
        
        Args:
            intensity: Intensity value (0-255)
            return_number: Return number (1-15)
            num_returns: Number of returns (1-15)
            scan_direction: Scan direction flag (0-1)
            edge_of_flight: Edge of flight line flag (0-1)
            classification: Point classification
            scan_angle_rank: Scan angle rank (-90 to +90)
            user_data: User data
            point_source_id: Point source ID
            gps_time: GPS time
        """
        self.intensity = intensity
        self.return_number = return_number
        self.num_returns = num_returns
        self.scan_direction = scan_direction
        self.edge_of_flight = edge_of_flight
        
        if isinstance(classification, int):
            self.classification = PointClass(classification)
        else:
            self.classification = classification
            
        self.scan_angle_rank = scan_angle_rank
        self.user_data = user_data
        self.point_source_id = point_source_id
        self.gps_time = gps_time

class PointCloud:
    """
    Class for processing LiDAR point cloud data
    
    This class provides GPU-accelerated operations for processing
    point clouds, such as LiDAR data, including classification,
    filtering, DEM/DSM creation, and feature extraction.
    """
    
    def __init__(self, file_path: str, device_id: int = 0):
        """
        Initialize the point cloud processor
        
        Args:
            file_path: Path to point cloud file (LAS/LAZ format)
            device_id: CUDA device ID (default: 0)
        """
        if not os.path.exists(file_path) and not file_path.endswith(".synthetic"):
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")
        
        self._cloud = _geospatial.PointCloud(file_path, device_id)
    
    def get_num_points(self) -> int:
        """
        Get the number of points in the point cloud
        
        Returns:
            Number of points
        """
        return self._cloud.get_num_points()
    
    def get_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Get the bounds of the point cloud
        
        Returns:
            Tuple of (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        return self._cloud.get_bounds()
    
    def classify_points(self, algorithm: int = 0) -> "PointCloud":
        """
        Classify points into standard LiDAR classes
        
        Args:
            algorithm: Classification algorithm
                      0: Default (geometric features)
                      1: Progressive (iterative refinement)
                      2: Deep learning (if available)
        
        Returns:
            New PointCloud with classified points
        """
        result = self._cloud.classify_points(algorithm)
        
        # Create Python wrapper for the C++ result
        cloud = PointCloud.__new__(PointCloud)
        cloud._cloud = result
        return cloud
    
    def filter_by_class(self, classification: Union[PointClass, int, str]) -> "PointCloud":
        """
        Filter points by classification
        
        Args:
            classification: Point classification to filter
                           Can be PointClass enum, integer value, or string name
        
        Returns:
            New PointCloud with filtered points
        """
        if isinstance(classification, str):
            classification = PointClass.from_string(classification)
        elif isinstance(classification, int):
            classification = PointClass(classification)
        
        result = self._cloud.filter_by_class(classification)
        
        # Create Python wrapper for the C++ result
        cloud = PointCloud.__new__(PointCloud)
        cloud._cloud = result
        return cloud
    
    def filter_by_return_number(self, return_number: int) -> "PointCloud":
        """
        Filter points by return number
        
        Args:
            return_number: Return number to filter (1 = first return, etc.)
        
        Returns:
            New PointCloud with filtered points
        """
        result = self._cloud.filter_by_return_number(return_number)
        
        # Create Python wrapper for the C++ result
        cloud = PointCloud.__new__(PointCloud)
        cloud._cloud = result
        return cloud
    
    def create_dem(self, resolution: float, algorithm: int = 0) -> DEMProcessor:
        """
        Create Digital Elevation Model from point cloud
        
        Args:
            resolution: Resolution of the output DEM in the same units as the point cloud
            algorithm: Algorithm for DEM creation
                      0: TIN-based interpolation
                      1: Inverse Distance Weighting (IDW)
                      2: Natural neighbor interpolation
        
        Returns:
            DEMProcessor with created DEM
        """
        return self._cloud.create_dem(resolution, algorithm)
    
    def create_dsm(self, resolution: float, algorithm: int = 0) -> DEMProcessor:
        """
        Create Digital Surface Model from point cloud
        
        Args:
            resolution: Resolution of the output DSM in the same units as the point cloud
            algorithm: Algorithm for DSM creation
                      0: Highest point in cell
                      1: Percentile-based filtering
        
        Returns:
            DEMProcessor with created DSM
        """
        return self._cloud.create_dsm(resolution, algorithm)
    
    def extract_buildings(self, min_height: float = 2.0, min_area: float = 10.0) -> str:
        """
        Extract building footprints from point cloud
        
        Args:
            min_height: Minimum height difference for building detection
            min_area: Minimum area for building footprint (square units)
        
        Returns:
            Path to the output shapefile with building polygons
        """
        return self._cloud.extract_buildings(min_height, min_area)
    
    def extract_vegetation(self, 
                         height_classes: List[float] = None,
                         resolution: float = 1.0) -> np.ndarray:
        """
        Extract vegetation cover from point cloud
        
        Args:
            height_classes: Vector of height class thresholds
                          Default: [0.5, 2.0, 5.0, 15.0]
            resolution: Resolution of the output raster
        
        Returns:
            Numpy array with vegetation classification
            Values correspond to height classes (1, 2, 3, ...)
        """
        if height_classes is None:
            height_classes = [0.5, 2.0, 5.0, 15.0]
        
        # Convert result to numpy array
        result = self._cloud.extract_vegetation(height_classes, resolution)
        return np.array(result, copy=False)
    
    def segment_points(self, max_distance: float = 1.0, min_points: int = 10) -> Dict[int, List[int]]:
        """
        Segment point cloud into coherent objects
        
        Args:
            max_distance: Maximum distance between points in the same segment
            min_points: Minimum number of points in a segment
        
        Returns:
            Dictionary mapping segment IDs to lists of point indices
        """
        return self._cloud.segment_points(max_distance, min_points)
    
    def save(self, output_file: str) -> bool:
        """
        Save point cloud to file
        
        Args:
            output_file: Output file path (LAS/LAZ format)
        
        Returns:
            True if successful, False otherwise
        """
        return self._cloud.save(output_file)
    
    def get_point(self, index: int) -> Tuple[float, float, float, PointAttributes]:
        """
        Get point at specified index
        
        Args:
            index: Point index
        
        Returns:
            Tuple of (x, y, z, attributes)
        """
        x, y, z, attrs = self._cloud.get_point(index)
        
        # Convert C++ attributes to Python PointAttributes
        attributes = PointAttributes(
            intensity=attrs.intensity,
            return_number=attrs.return_number,
            num_returns=attrs.num_returns,
            scan_direction=attrs.scan_direction,
            edge_of_flight=attrs.edge_of_flight,
            classification=PointClass(attrs.classification),
            scan_angle_rank=attrs.scan_angle_rank,
            user_data=attrs.user_data,
            point_source_id=attrs.point_source_id,
            gps_time=attrs.gps_time
        )
        
        return x, y, z, attributes
    
    def voxel_downsample(self, voxel_size: float) -> "PointCloud":
        """
        Spatially subsample the point cloud
        
        Args:
            voxel_size: Voxel size for subsampling
        
        Returns:
            New PointCloud with subsampled points
        """
        result = self._cloud.voxel_downsample(voxel_size)
        
        # Create Python wrapper for the C++ result
        cloud = PointCloud.__new__(PointCloud)
        cloud._cloud = result
        return cloud
    
    def compute_normals(self, radius: float = 1.0) -> np.ndarray:
        """
        Compute normal vectors for the point cloud
        
        Args:
            radius: Radius for normal estimation
        
        Returns:
            Numpy array of normal vectors (Nx3)
        """
        normals = self._cloud.compute_normals(radius)
        return np.array(normals, copy=False)
    
    def __len__(self) -> int:
        """Get number of points"""
        return self.get_num_points()
    
    def __getitem__(self, index: int) -> Tuple[float, float, float, PointAttributes]:
        """Get point at index"""
        return self.get_point(index)