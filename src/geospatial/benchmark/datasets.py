#!/usr/bin/env python3
"""
Dataset generation utilities for Geospatial Analysis benchmarking.

This module provides functions to generate synthetic datasets for benchmarking
the Geospatial Analysis workload, including DEMs and point clouds with
different characteristics.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import numpy as np
import tempfile
import struct
import shutil
from typing import Tuple, Dict, List, Optional, Any, Union
from pathlib import Path

try:
    from osgeo import gdal, osr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

# Constants for dataset generation
DEFAULT_DATASET_DIR = os.path.join(tempfile.gettempdir(), "geospatial_benchmark_data")

# Terrain types for DEM generation
class TerrainType:
    FLAT = "flat"
    ROLLING_HILLS = "rolling_hills"
    MOUNTAINS = "mountains"
    CANYON = "canyon"
    COASTAL = "coastal"
    URBAN = "urban"
    RANDOM = "random"

# Point cloud density categories (points per square meter)
class PointCloudDensity:
    LOW = 1      # ~1 point/m²
    MEDIUM = 5   # ~5 points/m²
    HIGH = 20    # ~20 points/m²
    VERY_HIGH = 50  # ~50 points/m²

def ensure_dataset_dir() -> str:
    """Ensure the dataset directory exists and return its path."""
    os.makedirs(DEFAULT_DATASET_DIR, exist_ok=True)
    return DEFAULT_DATASET_DIR

def create_synthetic_dem(
    size: int = 1024,
    terrain_type: str = TerrainType.ROLLING_HILLS,
    output_dir: Optional[str] = None,
    noise_level: float = 0.05,
    z_scale: float = 100.0,
    seed: int = 42
) -> str:
    """
    Create a synthetic DEM with specified characteristics.
    
    Args:
        size: Size of the DEM (pixels)
        terrain_type: Type of terrain to generate
        output_dir: Directory to save the DEM (default: temp directory)
        noise_level: Amount of noise to add (0-1)
        z_scale: Vertical scale factor
        seed: Random seed for reproducibility
        
    Returns:
        Path to the generated DEM file
    """
    if not GDAL_AVAILABLE:
        raise ImportError("GDAL is required for DEM generation.")
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = ensure_dataset_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create filename based on parameters
    filename = f"dem_{terrain_type}_{size}px_{int(z_scale)}z_{seed}.tif"
    dem_path = os.path.join(output_dir, filename)
    
    # Check if file already exists
    if os.path.exists(dem_path):
        print(f"Using existing DEM: {dem_path}")
        return dem_path
    
    # Initialize DEM data array
    dem_data = np.zeros((size, size), dtype=np.float32)
    
    # Generate different terrain types
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)
    X, Y = np.meshgrid(x, y)
    
    if terrain_type == TerrainType.FLAT:
        # Flat terrain with very small variations
        dem_data = np.ones((size, size)) * 100
        dem_data += np.random.normal(0, 0.5, size=(size, size))
        
    elif terrain_type == TerrainType.ROLLING_HILLS:
        # Rolling hills with smooth transitions
        for i in range(3):
            freq = 0.5 * (i + 1)
            dem_data += 20 * np.sin(X * freq) * np.cos(Y * freq * 0.8)
        
        # Add some random hills
        for _ in range(5):
            cx, cy = np.random.uniform(0, 10, 2)
            radius = np.random.uniform(1, 3)
            height = np.random.uniform(20, 40)
            dem_data += height * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * radius**2))
    
    elif terrain_type == TerrainType.MOUNTAINS:
        # Start with rolling terrain
        for i in range(3):
            freq = 0.5 * (i + 1)
            dem_data += 20 * np.sin(X * freq) * np.cos(Y * freq * 0.8)
        
        # Add mountain peaks
        for _ in range(10):
            cx, cy = np.random.uniform(0, 10, 2)
            radius = np.random.uniform(0.5, 1.5)
            height = np.random.uniform(100, 250)
            dem_data += height * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * radius**2))
            
        # Add ridgelines
        for _ in range(3):
            angle = np.random.uniform(0, np.pi)
            cx, cy = np.random.uniform(1, 9, 2)
            length = np.random.uniform(3, 7)
            width = np.random.uniform(0.5, 1.5)
            height = np.random.uniform(80, 150)
            
            # Create a line
            dx, dy = length * np.cos(angle), length * np.sin(angle)
            dist = np.abs((Y - cy) * dx - (X - cx) * dy) / np.sqrt(dx**2 + dy**2)
            line_mask = np.exp(-(dist**2) / (2 * width**2))
            
            # Ensure the ridge is within the grid
            extent_mask = (np.abs(X - cx) <= np.abs(dx)) & (np.abs(Y - cy) <= np.abs(dy))
            ridge = height * line_mask * extent_mask
            dem_data += ridge
    
    elif terrain_type == TerrainType.CANYON:
        # Start with elevated terrain
        dem_data = np.ones((size, size)) * 200
        
        # Add base noise
        dem_data += 20 * np.sin(X * 0.5) * np.cos(Y * 0.4)
        
        # Create canyon
        canyon_x = np.random.uniform(3, 7)
        canyon_width = np.random.uniform(0.3, 0.8)
        canyon_depth = np.random.uniform(100, 180)
        
        # Main canyon
        canyon = canyon_depth * np.exp(-((X - canyon_x)**2) / (2 * canyon_width**2))
        dem_data -= canyon
        
        # Add tributaries
        for _ in range(3):
            trib_y = np.random.uniform(2, 8)
            trib_width = np.random.uniform(0.1, 0.3)
            trib_depth = np.random.uniform(50, 100)
            trib_length = np.random.uniform(0.5, 2.0)
            
            # Tributary shape
            dist_to_main = np.abs(X - canyon_x)
            trib = trib_depth * np.exp(-((Y - trib_y)**2) / (2 * trib_width**2)) * np.exp(-dist_to_main / trib_length)
            dem_data -= trib
    
    elif terrain_type == TerrainType.COASTAL:
        # Create a coastal terrain with sea level at 0
        
        # Create basic terrain with an east-west coastline
        coastline_y = size // 2
        elevation_gradient = np.zeros((size, size))
        
        # Land area (above coastline)
        elevation_gradient[:coastline_y, :] = np.linspace(0, 100, coastline_y)[:, np.newaxis]
        
        # Add hills on land
        land_mask = np.zeros((size, size))
        land_mask[:coastline_y, :] = 1
        
        for _ in range(10):
            cx = np.random.uniform(0, 10)
            cy = np.random.uniform(0, 5)  # Hills only on land
            radius = np.random.uniform(0.5, 1.5)
            height = np.random.uniform(30, 80)
            hill = height * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * radius**2))
            dem_data += hill * land_mask
        
        # Add ocean bathymetry
        ocean_mask = 1 - land_mask
        for _ in range(5):
            cx = np.random.uniform(0, 10)
            cy = np.random.uniform(5, 10)  # Bathymetry only in ocean
            radius = np.random.uniform(0.5, 2.0)
            depth = np.random.uniform(20, 100)
            bathymetry = depth * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * radius**2))
            dem_data -= bathymetry * ocean_mask
        
        # Add coastline irregularity
        coastline_irregularity = 0.5 * np.sin(X * 2) + 0.3 * np.sin(X * 5)
        coastline_shift = coastline_irregularity * size // 10
        
        for i in range(size):
            for j in range(size):
                shifted_coastline = coastline_y + int(coastline_shift[i, j])
                if 0 <= shifted_coastline < size:
                    if j < shifted_coastline:
                        # Land
                        dem_data[i, j] += elevation_gradient[i, j]
                    else:
                        # Ocean
                        dem_data[i, j] -= 20  # Ocean depth
    
    elif terrain_type == TerrainType.URBAN:
        # Create an urban terrain with buildings
        
        # Start with slightly variable base terrain
        dem_data = np.ones((size, size)) * 100
        dem_data += np.random.normal(0, 1, size=(size, size))
        
        # Add a grid pattern for streets
        street_spacing = size // 20  # Number of pixels between streets
        street_width = max(1, size // 200)  # Width of streets in pixels
        
        # Create street mask
        street_mask = np.ones((size, size))
        for i in range(0, size, street_spacing):
            min_idx = max(0, i - street_width // 2)
            max_idx = min(size, i + street_width // 2)
            street_mask[min_idx:max_idx, :] = 0
            street_mask[:, min_idx:max_idx] = 0
        
        # Create buildings
        building_mask = np.zeros((size, size))
        
        for i in range(street_spacing, size, street_spacing):
            for j in range(street_spacing, size, street_spacing):
                if np.random.random() > 0.2:  # 80% chance of a building block
                    block_size_x = np.random.randint(street_width*2, street_spacing - street_width*2)
                    block_size_y = np.random.randint(street_width*2, street_spacing - street_width*2)
                    
                    start_x = i - street_spacing + street_width + np.random.randint(0, street_spacing - street_width*2 - block_size_x)
                    start_y = j - street_spacing + street_width + np.random.randint(0, street_spacing - street_width*2 - block_size_y)
                    
                    building_height = np.random.randint(5, 50)
                    building_mask[start_x:start_x + block_size_x, start_y:start_y + block_size_y] = building_height
        
        # Apply streets and buildings
        dem_data = dem_data * street_mask + building_mask
    
    elif terrain_type == TerrainType.RANDOM:
        # Fully random terrain with fractal noise (simplified Perlin noise)
        
        def generate_noise(size, scale=1.0):
            """Generate noise at a specific scale"""
            noise = np.random.normal(0, 1, (size, size))
            return scale * noise
        
        # Generate noise at different scales (octaves)
        scales = [64, 32, 16, 8, 4, 2]
        weights = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        
        for scale, weight in zip(scales, weights):
            # Calculate size of this noise octave
            octave_size = max(1, size // scale)
            
            # Generate noise at this scale
            octave_noise = generate_noise(octave_size)
            
            # Resize to full size
            from scipy.ndimage import zoom
            zoom_factor = size / octave_size
            resized_noise = zoom(octave_noise, zoom_factor)
            
            # Add to DEM with appropriate weight
            dem_data += weight * resized_noise
    
    # Scale the terrain and add random noise
    dem_data = dem_data * z_scale
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * z_scale, size=(size, size))
        dem_data += noise
    
    # Create a new GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    dem_file = driver.Create(dem_path, size, size, 1, gdal.GDT_Float32)
    
    # Set geotransform and projection (10m resolution)
    dem_file.SetGeoTransform((0, 10, 0, 0, 0, 10))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dem_file.SetProjection(srs.ExportToWkt())
    
    # Write data
    dem_file.GetRasterBand(1).WriteArray(dem_data)
    dem_file.GetRasterBand(1).SetNoDataValue(-9999)
    
    # Close file
    dem_file = None
    
    print(f"Created synthetic DEM: {dem_path}")
    return dem_path

def create_synthetic_point_cloud(
    num_points: int = 1000000,
    area_size: float = 1000.0,
    density_pattern: str = "uniform",
    include_buildings: bool = True,
    include_vegetation: bool = True,
    output_dir: Optional[str] = None,
    seed: int = 42
) -> str:
    """
    Create a synthetic point cloud with realistic features.
    
    Args:
        num_points: Number of points in the point cloud
        area_size: Size of the area in meters
        density_pattern: Pattern of point density (uniform, clustered, grid)
        include_buildings: Whether to include building structures
        include_vegetation: Whether to include vegetation
        output_dir: Directory to save the point cloud
        seed: Random seed for reproducibility
        
    Returns:
        Path to the generated point cloud file
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = ensure_dataset_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create filename based on parameters
    features = []
    if include_buildings:
        features.append("buildings")
    if include_vegetation:
        features.append("vegetation")
    
    feature_str = "_".join(features) if features else "plain"
    filename = f"pc_{num_points}_{density_pattern}_{feature_str}_{seed}.bin"
    pc_path = os.path.join(output_dir, filename)
    
    # Check if file already exists
    if os.path.exists(pc_path):
        print(f"Using existing point cloud: {pc_path}")
        return pc_path
    
    # Generate point positions based on density pattern
    if density_pattern == "uniform":
        # Uniform distribution across the area
        x = np.random.uniform(0, area_size, num_points)
        y = np.random.uniform(0, area_size, num_points)
        
    elif density_pattern == "clustered":
        # Create clusters of points
        num_clusters = int(np.sqrt(num_points) / 5)
        points_per_cluster = num_points // num_clusters
        remaining_points = num_points % num_clusters
        
        x = np.zeros(num_points)
        y = np.zeros(num_points)
        
        point_idx = 0
        for _ in range(num_clusters):
            # Create cluster center
            center_x = np.random.uniform(0, area_size)
            center_y = np.random.uniform(0, area_size)
            
            # Cluster size (standard deviation)
            cluster_size = np.random.uniform(10, 50)
            
            # Generate points for this cluster
            cluster_points = points_per_cluster + (1 if remaining_points > 0 else 0)
            if remaining_points > 0:
                remaining_points -= 1
            
            x[point_idx:point_idx + cluster_points] = np.random.normal(center_x, cluster_size, cluster_points)
            y[point_idx:point_idx + cluster_points] = np.random.normal(center_y, cluster_size, cluster_points)
            
            # Clamp to area bounds
            x[point_idx:point_idx + cluster_points] = np.clip(x[point_idx:point_idx + cluster_points], 0, area_size)
            y[point_idx:point_idx + cluster_points] = np.clip(y[point_idx:point_idx + cluster_points], 0, area_size)
            
            point_idx += cluster_points
            
    elif density_pattern == "grid":
        # Create a grid pattern with random jitter
        points_per_side = int(np.sqrt(num_points))
        actual_points = points_per_side * points_per_side
        
        # Create grid coordinates
        grid_spacing = area_size / points_per_side
        x_grid, y_grid = np.meshgrid(
            np.linspace(0, area_size, points_per_side),
            np.linspace(0, area_size, points_per_side)
        )
        
        # Add random jitter
        jitter_scale = grid_spacing * 0.3
        x = x_grid.flatten() + np.random.normal(0, jitter_scale, actual_points)
        y = y_grid.flatten() + np.random.normal(0, jitter_scale, actual_points)
        
        # Add remaining points randomly
        if actual_points < num_points:
            remaining_x = np.random.uniform(0, area_size, num_points - actual_points)
            remaining_y = np.random.uniform(0, area_size, num_points - actual_points)
            x = np.concatenate([x, remaining_x])
            y = np.concatenate([y, remaining_y])
    
    else:
        raise ValueError(f"Unknown density pattern: {density_pattern}")
    
    # Generate Z values (terrain height)
    # Use a simplified terrain model with noise
    terrain_scale = area_size / 10
    z_terrain = (
        10 * np.sin(x / terrain_scale) * np.cos(y / terrain_scale * 0.8) +
        5 * np.sin(x / terrain_scale * 2.5) * np.cos(y / terrain_scale * 2.2) +
        np.random.normal(0, 1, num_points)
    ) + 100  # Base elevation of 100m
    
    # Final point coordinates
    points = np.column_stack([x, y, z_terrain])
    
    # Generate classifications
    classifications = np.zeros(num_points, dtype=np.uint8)
    
    # Default all points to unclassified (class 1)
    classifications[:] = 1
    
    # Select ground points (class 2) - points with lower elevation in their vicinity
    # Define a grid for the area
    grid_size = int(np.sqrt(area_size))
    grid_points = {}
    
    # Assign points to grid cells
    for i, (px, py, pz) in enumerate(points):
        grid_x = int(px / area_size * grid_size)
        grid_y = int(py / area_size * grid_size)
        cell_key = (grid_x, grid_y)
        
        if cell_key not in grid_points:
            grid_points[cell_key] = []
        
        grid_points[cell_key].append((i, pz))
    
    # Find lowest points in each cell and classify as ground
    ground_indices = []
    for cell_points in grid_points.values():
        if cell_points:
            # Sort points by elevation
            cell_points.sort(key=lambda p: p[1])
            # Take the lowest point and some nearby points as ground
            num_ground = max(1, int(len(cell_points) * 0.2))
            for i in range(num_ground):
                ground_indices.append(cell_points[i][0])
    
    classifications[ground_indices] = 2  # Ground
    
    # Add buildings if requested
    if include_buildings:
        # Number of buildings based on area
        num_buildings = int(area_size / 100)
        
        for _ in range(num_buildings):
            # Building position
            building_x = np.random.uniform(0.1 * area_size, 0.9 * area_size)
            building_y = np.random.uniform(0.1 * area_size, 0.9 * area_size)
            
            # Building size
            building_width = np.random.uniform(10, 30)
            building_length = np.random.uniform(10, 30)
            building_height = np.random.uniform(5, 20)
            
            # Building rotation
            rotation = np.random.uniform(0, np.pi)
            rot_matrix = np.array([
                [np.cos(rotation), -np.sin(rotation)],
                [np.sin(rotation), np.cos(rotation)]
            ])
            
            # Find points within the building
            dx = x - building_x
            dy = y - building_y
            coords = np.column_stack([dx, dy])
            
            # Apply rotation
            rotated_coords = np.dot(coords, rot_matrix.T)
            rx, ry = rotated_coords[:, 0], rotated_coords[:, 1]
            
            # Find points inside building footprint
            building_mask = (
                (np.abs(rx) < building_width / 2) &
                (np.abs(ry) < building_length / 2)
            )
            
            # Calculate height based on position in building
            building_indices = np.where(building_mask)[0]
            
            if len(building_indices) > 0:
                # Set building elevations
                building_base = np.mean(z_terrain[building_indices])
                
                # Create a flat roof with some noise
                roof_height = building_base + building_height
                roof_noise = np.random.normal(0, 0.1, len(building_indices))
                
                points[building_indices, 2] = roof_height + roof_noise
                classifications[building_indices] = 6  # Building
    
    # Add vegetation if requested
    if include_vegetation:
        # Number of vegetation clusters
        num_veg_clusters = int(area_size / 50)
        
        for _ in range(num_veg_clusters):
            # Vegetation center
            veg_x = np.random.uniform(0, area_size)
            veg_y = np.random.uniform(0, area_size)
            
            # Cluster radius
            radius = np.random.uniform(5, 20)
            
            # Find points within the radius
            dx = x - veg_x
            dy = y - veg_y
            distances = np.sqrt(dx*dx + dy*dy)
            
            # Points within the vegetation area
            veg_mask = distances < radius
            veg_indices = np.where(veg_mask)[0]
            
            if len(veg_indices) > 0:
                # Get base terrain height in this area
                base_heights = z_terrain[veg_indices]
                
                # Random vegetation heights
                veg_height = np.random.uniform(0.5, 5, len(veg_indices))
                
                # Add vegetation height to terrain
                points[veg_indices, 2] = base_heights + veg_height
                
                # Classify vegetation points
                # Class 3: Low vegetation (<2m)
                # Class 4: Medium vegetation (2-5m)
                # Class 5: High vegetation (>5m)
                low_veg = veg_height < 2
                med_veg = (veg_height >= 2) & (veg_height < 5)
                high_veg = veg_height >= 5
                
                classifications[veg_indices[low_veg]] = 3
                classifications[veg_indices[med_veg]] = 4
                classifications[veg_indices[high_veg]] = 5
    
    # Write to a simple binary format
    with open(pc_path, 'wb') as f:
        # Write header
        f.write(struct.pack('<I', num_points))
        
        # Write points with classification
        for i in range(num_points):
            f.write(struct.pack('<fffB', 
                               points[i, 0], points[i, 1], points[i, 2], 
                               classifications[i]))
    
    print(f"Created synthetic point cloud: {pc_path}")
    return pc_path

def clear_datasets():
    """Clear all generated datasets."""
    if os.path.exists(DEFAULT_DATASET_DIR):
        shutil.rmtree(DEFAULT_DATASET_DIR)
        print(f"Cleared dataset directory: {DEFAULT_DATASET_DIR}")

def create_standard_benchmark_datasets() -> Dict[str, str]:
    """
    Create a standard set of benchmark datasets.
    
    Returns:
        Dictionary with paths to generated datasets
    """
    datasets = {}
    
    # Create DEM datasets
    for terrain_type in [
        TerrainType.FLAT,
        TerrainType.ROLLING_HILLS,
        TerrainType.MOUNTAINS,
        TerrainType.CANYON
    ]:
        # Small DEM (512x512)
        datasets[f"dem_small_{terrain_type}"] = create_synthetic_dem(
            size=512,
            terrain_type=terrain_type,
            z_scale=100.0
        )
        
        # Medium DEM (1024x1024)
        datasets[f"dem_medium_{terrain_type}"] = create_synthetic_dem(
            size=1024,
            terrain_type=terrain_type,
            z_scale=100.0
        )
        
        # Large DEM (2048x2048)
        datasets[f"dem_large_{terrain_type}"] = create_synthetic_dem(
            size=2048,
            terrain_type=terrain_type,
            z_scale=100.0
        )
    
    # Create point cloud datasets
    for density in ["uniform", "clustered"]:
        # Small point cloud (100K points)
        datasets[f"pc_small_{density}"] = create_synthetic_point_cloud(
            num_points=100000,
            density_pattern=density,
            include_buildings=True,
            include_vegetation=True
        )
        
        # Medium point cloud (1M points)
        datasets[f"pc_medium_{density}"] = create_synthetic_point_cloud(
            num_points=1000000,
            density_pattern=density,
            include_buildings=True,
            include_vegetation=True
        )
        
        # Large point cloud (10M points)
        datasets[f"pc_large_{density}"] = create_synthetic_point_cloud(
            num_points=10000000,
            density_pattern=density,
            include_buildings=True,
            include_vegetation=True
        )
    
    return datasets

if __name__ == "__main__":
    # Test dataset generation
    print("Generating test datasets...")
    
    dem_path = create_synthetic_dem(
        size=512,
        terrain_type=TerrainType.MOUNTAINS,
        z_scale=100.0
    )
    print(f"Generated DEM: {dem_path}")
    
    pc_path = create_synthetic_point_cloud(
        num_points=100000,
        density_pattern="clustered",
        include_buildings=True,
        include_vegetation=True
    )
    print(f"Generated point cloud: {pc_path}")