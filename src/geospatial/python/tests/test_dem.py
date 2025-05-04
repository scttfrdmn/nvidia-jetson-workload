import os
import unittest
import numpy as np
import tempfile
import shutil
from osgeo import gdal
from geospatial.dem import DEMProcessor

class TestDEMProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a temporary test DEM file"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_dem_path = os.path.join(cls.temp_dir, "test_dem.tif")
        
        # Create a simple synthetic DEM
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
        dem_file = driver.Create(cls.test_dem_path, dem_size, dem_size, 1, gdal.GDT_Float32)
        
        # Set geotransform and projection
        dem_file.SetGeoTransform((-180, 1/256, 0, 90, 0, -1/256))
        srs = gdal.osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dem_file.SetProjection(srs.ExportToWkt())
        
        # Write data
        dem_file.GetRasterBand(1).WriteArray(dem_data)
        dem_file.GetRasterBand(1).SetNoDataValue(-9999)
        
        # Close file
        dem_file = None
        
        # Store data for later comparisons
        cls.dem_data = dem_data
    
    @classmethod
    def tearDownClass(cls):
        """Remove temporary files"""
        shutil.rmtree(cls.temp_dir)
    
    def test_constructor(self):
        """Test DEM processor initialization"""
        dem_proc = DEMProcessor(self.test_dem_path)
        self.assertIsNotNone(dem_proc)
    
    def test_dimensions(self):
        """Test getting DEM dimensions"""
        dem_proc = DEMProcessor(self.test_dem_path)
        width, height = dem_proc.get_dimensions()
        self.assertEqual(width, 256)
        self.assertEqual(height, 256)
    
    def test_metadata(self):
        """Test getting DEM metadata"""
        dem_proc = DEMProcessor(self.test_dem_path)
        metadata = dem_proc.get_metadata()
        self.assertIsInstance(metadata, dict)
        self.assertIn('projection', metadata)
        self.assertIn('geotransform', metadata)
    
    def test_viewshed(self):
        """Test viewshed computation"""
        dem_proc = DEMProcessor(self.test_dem_path)
        # Observer at center point with height of 10m, radius of 1000m
        viewshed = dem_proc.compute_viewshed((0, 0), 10.0, 1000.0)
        self.assertIsInstance(viewshed, np.ndarray)
        self.assertEqual(viewshed.shape, (256, 256))
        # At least some cells should be visible
        self.assertTrue(np.sum(viewshed > 0) > 0)
    
    def test_terrain_derivatives(self):
        """Test terrain derivatives calculation"""
        dem_proc = DEMProcessor(self.test_dem_path)
        
        # Test slope calculation
        slope = dem_proc.compute_slope()
        self.assertIsInstance(slope, np.ndarray)
        self.assertEqual(slope.shape, (256, 256))
        
        # Test aspect calculation
        aspect = dem_proc.compute_aspect()
        self.assertIsInstance(aspect, np.ndarray)
        self.assertEqual(aspect.shape, (256, 256))
        
        # Test hillshade calculation
        hillshade = dem_proc.compute_hillshade(315, 45)
        self.assertIsInstance(hillshade, np.ndarray)
        self.assertEqual(hillshade.shape, (256, 256))
    
    def test_statistics(self):
        """Test DEM statistics calculation"""
        dem_proc = DEMProcessor(self.test_dem_path)
        stats = dem_proc.compute_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        
        # Basic validation of statistics
        self.assertGreaterEqual(stats['min'], np.min(self.dem_data))
        self.assertLessEqual(stats['max'], np.max(self.dem_data) + 0.001)
    
    def test_resampling(self):
        """Test DEM resampling"""
        dem_proc = DEMProcessor(self.test_dem_path)
        resampled = dem_proc.resample(0.5)  # 50% of original resolution
        
        self.assertIsInstance(resampled, np.ndarray)
        self.assertEqual(resampled.shape, (128, 128))  # Half the original size
    
    def test_contours(self):
        """Test contour extraction"""
        dem_proc = DEMProcessor(self.test_dem_path)
        contours = dem_proc.extract_contours(interval=20.0)
        
        self.assertIsInstance(contours, list)
        # Should have at least one contour
        self.assertGreater(len(contours), 0)
        
        # Each contour should be a list of (x,y) points
        for contour in contours:
            self.assertIsInstance(contour, list)
            if len(contour) > 0:
                self.assertEqual(len(contour[0]), 2)

if __name__ == '__main__':
    unittest.main()