#!/usr/bin/env python3
"""
Viewshed Analysis Example

This example demonstrates how to use the Geospatial workload
to perform viewshed analysis on a Digital Elevation Model (DEM).

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# Add the parent directory to the path so we can import the geospatial module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geospatial import DEMProcessor

def main():
    """Main function for the viewshed analysis example."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Viewshed Analysis Example')
    parser.add_argument('dem_file', help='Path to DEM file (GeoTIFF format)')
    parser.add_argument('--observer_x', type=float, help='Observer X coordinate in geographic units')
    parser.add_argument('--observer_y', type=float, help='Observer Y coordinate in geographic units')
    parser.add_argument('--observer_height', type=float, default=1.8, 
                        help='Observer height above the terrain (meters)')
    parser.add_argument('--radius', type=float, default=0.0, 
                        help='Maximum viewshed radius (meters, 0 for unlimited)')
    parser.add_argument('--output', help='Output viewshed file (GeoTIFF format)')
    parser.add_argument('--device', type=int, default=0, help='CUDA device ID')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    args = parser.parse_args()
    
    # Process the DEM
    try:
        print(f"Loading DEM: {args.dem_file}")
        dem = DEMProcessor(args.dem_file, device_id=args.device)
        
        # Get DEM dimensions
        width, height = dem.get_dimensions()
        print(f"DEM dimensions: {width} x {height}")
        
        # Get DEM geotransform
        geotransform = dem.get_geotransform()
        print(f"Geotransform: {geotransform.parameters}")
        
        # If observer coordinates are not provided, use the center of the DEM
        if args.observer_x is None or args.observer_y is None:
            center_pixel_x = width // 2
            center_pixel_y = height // 2
            observer_x, observer_y = geotransform.pixel_to_geo(center_pixel_x, center_pixel_y)
            print(f"Using center of DEM as observer location: ({observer_x}, {observer_y})")
        else:
            observer_x = args.observer_x
            observer_y = args.observer_y
            print(f"Observer location: ({observer_x}, {observer_y})")
        
        # Compute viewshed
        print(f"Computing viewshed (observer height: {args.observer_height}m, radius: {args.radius}m)...")
        start_time = time.time()
        viewshed = dem.compute_viewshed(
            (observer_x, observer_y), 
            observer_height=args.observer_height,
            radius=args.radius
        )
        elapsed_time = time.time() - start_time
        print(f"Viewshed computation completed in {elapsed_time:.2f} seconds")
        
        # Get pixel coordinates of the observer for visualization
        observer_pixel_x, observer_pixel_y = geotransform.geo_to_pixel(observer_x, observer_y)
        
        # Optionally save the viewshed to a file
        if args.output:
            print(f"Saving viewshed to {args.output}")
            dem.save_result(viewshed, args.output)
        
        # Visualize the results
        if args.visualize:
            print("Visualizing results...")
            
            # Compute terrain derivatives
            terrain = dem.compute_terrain_derivatives()
            slope = terrain['slope']
            aspect = terrain['aspect']
            
            # Create a shaded relief image
            ls = LightSource(azdeg=315, altdeg=45)
            shaded_relief = ls.shade(
                np.array(viewshed), 
                blend_mode='soft', 
                vert_exag=10.0,
                cmap='viridis'
            )
            
            # Create the visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(shaded_relief)
            
            # Mark the observer position
            ax.plot(observer_pixel_x, observer_pixel_y, 'ro', markersize=10, label='Observer')
            
            # Add a colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Visibility')
            
            # Add title and labels
            plt.title('Viewshed Analysis')
            plt.xlabel('Easting (pixels)')
            plt.ylabel('Northing (pixels)')
            plt.legend()
            
            # Show the plot
            plt.tight_layout()
            plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())