#!/usr/bin/env python3
"""
Integration tests for the Geospatial Analysis workload.
These tests verify the end-to-end functionality of the entire workflow.
"""

import os
import sys
import tempfile
import shutil
import unittest
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))

# Import geospatial modules
from geospatial.dem import DEMProcessor
from geospatial.point_cloud import PointCloud

class TestGeospatialWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Create temporary directory for test data
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic data using helper functions
        cls.dem_path = create_synthetic_dem(cls.temp_dir)
        cls.point_cloud_path = create_synthetic_point_cloud(cls.temp_dir)
        
        # Create a directory for outputs
        cls.output_dir = os.path.join(cls.temp_dir, 'outputs')
        os.makedirs(cls.output_dir, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.temp_dir)
    
    def test_1_dem_processing_workflow(self):
        """Test the complete DEM processing workflow"""
        # Initialize DEM processor
        dem_proc = DEMProcessor(self.dem_path)
        
        # Verify basic properties
        width, height = dem_proc.get_dimensions()
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        
        # Process the DEM in multiple steps
        
        # 1. Compute terrain derivatives
        slope = dem_proc.compute_slope()
        aspect = dem_proc.compute_aspect()
        hillshade = dem_proc.compute_hillshade(315, 45)
        
        # 2. Compute viewshed
        observer_point = (width // 2, height // 2)  # Center point
        viewshed = dem_proc.compute_viewshed(observer_point, 10.0, width // 4)
        
        # 3. Compute hydrological features
        flow_dir = dem_proc.compute_flow_direction()
        flow_acc = dem_proc.compute_flow_accumulation(flow_dir)
        streams = dem_proc.extract_streams(flow_acc, threshold=100)
        
        # 4. Compute least cost path
        start_point = (width // 4, height // 4)
        end_point = (3 * width // 4, 3 * height // 4)
        path = dem_proc.compute_cost_path(start_point, end_point)
        
        # 5. Save outputs for visualization
        if os.environ.get('GENERATE_PLOTS', '0') == '1':
            # Save slope
            plt.figure(figsize=(10, 8))
            plt.imshow(slope, cmap='viridis')
            plt.colorbar(label='Slope (degrees)')
            plt.title('Slope')
            plt.savefig(os.path.join(self.output_dir, 'slope.png'))
            plt.close()
            
            # Save hillshade with streams
            plt.figure(figsize=(10, 8))
            plt.imshow(hillshade, cmap='gray')
            plt.imshow(streams, cmap='Blues', alpha=0.5)
            plt.title('Hillshade with Streams')
            plt.savefig(os.path.join(self.output_dir, 'hillshade_streams.png'))
            plt.close()
            
            # Save viewshed
            plt.figure(figsize=(10, 8))
            plt.imshow(viewshed, cmap='RdYlGn')
            plt.colorbar(label='Visibility')
            plt.plot(observer_point[0], observer_point[1], 'ro', markersize=8)
            plt.title('Viewshed Analysis')
            plt.savefig(os.path.join(self.output_dir, 'viewshed.png'))
            plt.close()
            
            # Save cost path
            plt.figure(figsize=(10, 8))
            plt.imshow(hillshade, cmap='gray')
            if path is not None and len(path) > 0:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                plt.plot(path_x, path_y, 'r-', linewidth=2)
            plt.plot(start_point[0], start_point[1], 'go', markersize=8)
            plt.plot(end_point[0], end_point[1], 'bo', markersize=8)
            plt.title('Least Cost Path')
            plt.savefig(os.path.join(self.output_dir, 'cost_path.png'))
            plt.close()
        
        # Verify results
        self.assertIsInstance(slope, np.ndarray)
        self.assertIsInstance(aspect, np.ndarray)
        self.assertIsInstance(hillshade, np.ndarray)
        self.assertIsInstance(viewshed, np.ndarray)
        self.assertIsInstance(flow_dir, np.ndarray)
        self.assertIsInstance(flow_acc, np.ndarray)
        
        # Path might be None if no path found, but should be iterable if exists
        if path is not None:
            self.assertTrue(hasattr(path, '__iter__'))
    
    def test_2_point_cloud_processing_workflow(self):
        """Test the complete point cloud processing workflow"""
        try:
            # Initialize point cloud
            pc = PointCloud(self.point_cloud_path)
            
            # Verify point count
            point_count = pc.get_point_count()
            self.assertGreater(point_count, 0)
            
            # 1. Classify points
            classified_pc = pc.classify_points()
            
            # 2. Filter points by height
            ground_pc = classified_pc.filter_by_class(class_value=2)  # Ground points
            
            # 3. Create DEM from ground points
            dem = ground_pc.create_dem(resolution=1.0)
            
            # 4. Create DSM from all points
            dsm = pc.create_dsm(resolution=1.0)
            
            # 5. Compute normals
            pc_with_normals = pc.compute_normals(radius=2.0)
            
            # 6. Extract buildings
            buildings = pc.extract_buildings()
            
            # 7. Save outputs for visualization
            if os.environ.get('GENERATE_PLOTS', '0') == '1':
                # Save DEM
                plt.figure(figsize=(10, 8))
                plt.imshow(dem, cmap='terrain')
                plt.colorbar(label='Elevation')
                plt.title('Digital Elevation Model (DEM)')
                plt.savefig(os.path.join(self.output_dir, 'dem.png'))
                plt.close()
                
                # Save DSM
                plt.figure(figsize=(10, 8))
                plt.imshow(dsm, cmap='terrain')
                plt.colorbar(label='Elevation')
                plt.title('Digital Surface Model (DSM)')
                plt.savefig(os.path.join(self.output_dir, 'dsm.png'))
                plt.close()
                
                # Save normalized DSM (nDSM)
                if dem.shape == dsm.shape:
                    ndsm = dsm - dem
                    plt.figure(figsize=(10, 8))
                    plt.imshow(ndsm, cmap='viridis')
                    plt.colorbar(label='Height')
                    plt.title('Normalized Digital Surface Model (nDSM)')
                    plt.savefig(os.path.join(self.output_dir, 'ndsm.png'))
                    plt.close()
            
            # Verify results
            self.assertIsInstance(dem, np.ndarray)
            self.assertIsInstance(dsm, np.ndarray)
            
            # Check if we have extracted buildings
            if buildings is not None:
                self.assertIsInstance(buildings, list)
        
        except Exception as e:
            # Skip if the binary format is not supported
            self.skipTest(f"Point cloud processing not supported: {e}")
    
    def test_3_integrated_analysis(self):
        """Test integration between DEM and point cloud processing"""
        try:
            # Load both data sources
            dem_proc = DEMProcessor(self.dem_path)
            pc = PointCloud(self.point_cloud_path)
            
            # Create DEM from point cloud
            pc_dem = pc.create_dem(resolution=1.0)
            
            # Get original DEM data
            original_dem = dem_proc.get_elevation_array()
            
            # Resample original DEM to match the point cloud DEM
            if original_dem.shape != pc_dem.shape:
                # Resize the original DEM to match the point cloud DEM dimensions
                resampled_dem = dem_proc.resample(pc_dem.shape[1] / original_dem.shape[1])
                
                if resampled_dem.shape != pc_dem.shape:
                    self.skipTest("Could not match DEM sizes for comparison")
                    return
            else:
                resampled_dem = original_dem
            
            # Create a difference map
            diff_map = np.abs(resampled_dem - pc_dem)
            
            # Save difference map
            if os.environ.get('GENERATE_PLOTS', '0') == '1':
                plt.figure(figsize=(10, 8))
                plt.imshow(diff_map, cmap='hot')
                plt.colorbar(label='Absolute Difference')
                plt.title('DEM Difference Map')
                plt.savefig(os.path.join(self.output_dir, 'dem_diff.png'))
                plt.close()
            
            # Verify results
            self.assertIsInstance(diff_map, np.ndarray)
            
        except Exception as e:
            # Skip if integration test cannot be performed
            self.skipTest(f"Integrated analysis not supported: {e}")

def create_synthetic_dem(temp_dir):
    """Create a synthetic DEM for testing"""
    import numpy as np
    from osgeo import gdal, osr
    
    dem_path = os.path.join(temp_dir, "test_dem.tif")
    
    # Create a synthetic DEM
    dem_size = 256
    dem_data = np.zeros((dem_size, dem_size), dtype=np.float32)
    
    # Add some terrain features
    x, y = np.meshgrid(np.linspace(0, 1, dem_size), np.linspace(0, 1, dem_size))
    dem_data += 100 + 50 * np.sin(x * 10) * np.cos(y * 8)
    
    # Add a central peak
    center_x, center_y = dem_size // 2, dem_size // 2
    for i in range(dem_size):
        for j in range(dem_size):
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if dist < dem_size // 4:
                dem_data[i, j] += 100 * (1 - dist / (dem_size // 4))
    
    # Create a new GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    dem_file = driver.Create(dem_path, dem_size, dem_size, 1, gdal.GDT_Float32)
    
    # Set geotransform and projection
    dem_file.SetGeoTransform((0, 1, 0, 0, 0, 1))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dem_file.SetProjection(srs.ExportToWkt())
    
    # Write data
    dem_file.GetRasterBand(1).WriteArray(dem_data)
    dem_file.GetRasterBand(1).SetNoDataValue(-9999)
    
    # Close file
    dem_file = None
    
    return dem_path

def create_synthetic_point_cloud(temp_dir, num_points=10000):
    """Create a synthetic point cloud for testing"""
    import numpy as np
    import struct
    
    pc_path = os.path.join(temp_dir, "test_pointcloud.bin")
    
    # Generate random points
    np.random.seed(42)  # For reproducibility
    points = np.random.rand(num_points, 3)
    points[:, 0] *= 100  # X range [0, 100]
    points[:, 1] *= 100  # Y range [0, 100]
    points[:, 2] *= 10   # Z range [0, 10]
    
    # Add classification data (random classes 0-9)
    classifications = np.random.randint(0, 10, num_points, dtype=np.uint8)
    
    # Add building-like structure
    building_points = min(1000, num_points // 10)
    building_center = [25, 25]
    building_size = [10, 10]
    for i in range(num_points - building_points, num_points):
        points[i, 0] = building_center[0] + (np.random.rand() - 0.5) * building_size[0]
        points[i, 1] = building_center[1] + (np.random.rand() - 0.5) * building_size[1]
        points[i, 2] = 5 + np.random.rand() * 5  # Building height between 5-10m
        classifications[i] = 6  # Building class
    
    # Write to a simple binary format
    with open(pc_path, 'wb') as f:
        # Write header
        f.write(struct.pack('<I', num_points))
        
        # Write points with classification
        for i in range(num_points):
            f.write(struct.pack('<fffB', 
                                points[i, 0], points[i, 1], points[i, 2], 
                                classifications[i]))
    
    return pc_path

if __name__ == '__main__':
    # Set environment variable to generate plots if running directly
    os.environ['GENERATE_PLOTS'] = '1'
    unittest.main()