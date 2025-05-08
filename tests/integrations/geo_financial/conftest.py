"""
Pytest configuration file for geospatial financial integration tests.

This file contains shared fixtures and configuration for the tests.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import pytest
import numpy as np
import tempfile
from typing import Dict, List, Tuple

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(project_root)

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test data."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    yield temp_dir
    
    # Clean up after the session
    import shutil
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def dem_data(temp_dir):
    """Generate synthetic DEM data for testing."""
    try:
        from src.integrations.geo_financial.generate_test_data import generate_dem
        
        # Generate synthetic DEM
        dem_size = 100
        dem_data, x_coords, y_coords = generate_dem(
            output_dir=temp_dir,
            size=dem_size,
            save_plot=False,
            seed=42
        )
        
        return dem_data, x_coords, y_coords
    except ImportError:
        # If the module is not available, return a simple array
        dem_size = 100
        dem_data = np.random.rand(dem_size, dem_size) * 100
        x_coords = np.linspace(0, 10, dem_size)
        y_coords = np.linspace(0, 10, dem_size)
        
        return dem_data, x_coords, y_coords

@pytest.fixture(scope="session")
def mock_geo_transform():
    """Create a mock GeoTransform for testing."""
    class MockGeoTransform:
        """Mock GeoTransform for testing."""
        
        def geo_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
            """Convert geographic coordinates to pixel coordinates."""
            return int(x * 50), int(y * 50)
        
        def pixel_to_geo(self, x: int, y: int) -> Tuple[float, float]:
            """Convert pixel coordinates to geographic coordinates."""
            return x / 50, y / 50
    
    return MockGeoTransform()

@pytest.fixture(scope="session")
def assets_df(temp_dir, dem_data):
    """Generate synthetic asset data for testing."""
    try:
        from src.integrations.geo_financial.generate_test_data import generate_assets
        
        # Get DEM data from fixture
        dem_array, x_coords, y_coords = dem_data
        
        # Generate synthetic assets
        num_assets = 10
        assets_df = generate_assets(
            output_dir=temp_dir,
            dem_data=dem_array,
            x_coords=x_coords,
            y_coords=y_coords,
            num_assets=num_assets,
            save_plot=False,
            seed=43
        )
        
        return assets_df
    except ImportError:
        # If the module is not available, return a simple DataFrame
        import pandas as pd
        
        dem_array, x_coords, y_coords = dem_data
        num_assets = 10
        
        # Generate simple asset data
        assets_data = {
            'id': [f"ASSET_{i:03d}" for i in range(num_assets)],
            'name': [f"Test Asset {i}" for i in range(num_assets)],
            'value': np.random.uniform(1000, 10000, num_assets).round(2),
            'x': np.random.uniform(0, 10, num_assets),
            'y': np.random.uniform(0, 10, num_assets),
            'sector': np.random.choice(
                ['Energy', 'Utilities', 'Real Estate', 'Infrastructure', 'Agriculture'],
                num_assets
            )
        }
        
        return pd.DataFrame(assets_data)

@pytest.fixture(scope="session")
def portfolio(assets_df):
    """Create a GeospatialPortfolio for testing."""
    try:
        from src.integrations.geo_financial.geo_risk import GeospatialPortfolio
        
        # Create portfolio
        portfolio = GeospatialPortfolio(device_id=-1)  # Use CPU
        
        # Add assets to portfolio
        for _, row in assets_df.iterrows():
            portfolio.add_asset(
                asset_id=row['id'],
                name=row['name'],
                value=row['value'],
                x=row['x'],
                y=row['y'],
                returns=np.random.normal(0.0005, 0.01, 252),  # Random returns
                metadata={'sector': row['sector']} if 'sector' in row else {}
            )
        
        return portfolio
    except ImportError:
        pytest.skip("GeospatialPortfolio not available")

@pytest.fixture(scope="session")
def risk_factors(dem_data, mock_geo_transform):
    """Create risk factors for testing."""
    try:
        from src.integrations.geo_financial.geo_risk import SpatialRiskFactor
        
        dem_array, _, _ = dem_data
        
        # Flood risk (low elevation = high risk)
        flood_risk = SpatialRiskFactor(
            name="Flood Risk",
            description="Risk based on elevation (lower elevations have higher risk)",
            risk_weight=0.6,
            spatial_data=dem_array,
            geo_transform=mock_geo_transform,
            transform_func=lambda elev: 1.0 - np.clip(
                (elev - np.min(elev)) / (np.max(elev) - np.min(elev)), 0, 1
            )
        )
        
        # Slope risk (calculate slope from DEM)
        from scipy.ndimage import sobel
        dx = sobel(dem_array, axis=1)
        dy = sobel(dem_array, axis=0)
        slope_data = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        slope_risk = SpatialRiskFactor(
            name="Landslide Risk",
            description="Risk based on terrain slope (steeper slopes have higher risk)",
            risk_weight=0.4,
            spatial_data=slope_data,
            geo_transform=mock_geo_transform,
            transform_func=lambda slope: np.clip(slope / 45.0, 0, 1)
        )
        
        return [flood_risk, slope_risk]
    except ImportError:
        pytest.skip("SpatialRiskFactor not available")

@pytest.fixture(scope="session")
def temperature_data(dem_data):
    """Generate synthetic temperature data for testing."""
    dem_array, _, _ = dem_data
    
    # Higher temperatures in lower elevations and certain regions
    temperature_data = 25.0 + 15.0 * (1.0 - np.clip(
        (dem_array - np.min(dem_array)) / 
        (np.max(dem_array) - np.min(dem_array)), 0, 1
    ))
    
    # Add some regional variation
    x_idx, y_idx = np.meshgrid(
        np.arange(dem_array.shape[1]), 
        np.arange(dem_array.shape[0])
    )
    regional_pattern = 5.0 * np.sin(x_idx / 50.0) * np.cos(y_idx / 50.0)
    
    return temperature_data + regional_pattern

@pytest.fixture(scope="session")
def coastline_distance(dem_data):
    """Generate synthetic coastline distance data for testing."""
    dem_array, _, _ = dem_data
    
    # Lower part of the DEM will be the "coastline"
    coastline_y = int(dem_array.shape[0] * 0.8)
    coastline_distance = np.zeros_like(dem_array)
    
    for i in range(dem_array.shape[0]):
        for j in range(dem_array.shape[1]):
            # Distance to coastline in grid cells
            distance = abs(i - coastline_y)
            # Convert to kilometers (assuming 1 grid cell = 0.5 km)
            coastline_distance[i, j] = distance * 0.5
    
    return coastline_distance

@pytest.fixture(scope="session")
def climate_assessor(dem_data, mock_geo_transform):
    """Create a ClimateRiskAssessor for testing."""
    try:
        from src.integrations.geo_financial.climate_risk_assessment import (
            ClimateRiskAssessor,
            ClimateScenario,
            TimeHorizon
        )
        
        # Create assessor
        assessor = ClimateRiskAssessor(
            device_id=-1,
            scenario=ClimateScenario.SSP2_RCP45,
            time_horizon=TimeHorizon.MEDIUM
        )
        
        return assessor
    except ImportError:
        pytest.skip("ClimateRiskAssessor not available")

@pytest.fixture(scope="session")
def climate_risk_factors(dem_data, mock_geo_transform, temperature_data, coastline_distance, climate_assessor):
    """Create climate risk factors for testing."""
    try:
        from src.integrations.geo_financial.climate_risk_assessment import (
            create_flood_risk_factor,
            create_heatwave_risk_factor,
            create_sea_level_rise_risk_factor
        )
        
        dem_array, _, _ = dem_data
        
        # Create flood risk factor
        flood_risk = create_flood_risk_factor(
            dem_path=dem_array,
            risk_weight=0.6,
            low_elevation_threshold=np.percentile(dem_array, 20),
            assessor=climate_assessor
        )
        
        # Create heatwave risk factor
        heatwave_risk = create_heatwave_risk_factor(
            temperature_data=temperature_data,
            geo_transform=mock_geo_transform,
            risk_weight=0.4,
            high_temp_threshold=35.0,
            assessor=climate_assessor
        )
        
        # Create sea level rise risk factor
        slr_risk = create_sea_level_rise_risk_factor(
            dem_path=dem_array,
            coastline_distance=coastline_distance,
            geo_transform=mock_geo_transform,
            risk_weight=0.5,
            elevation_threshold=np.percentile(dem_array, 30),
            distance_threshold=20.0,
            assessor=climate_assessor
        )
        
        return [flood_risk, heatwave_risk, slr_risk]
    except ImportError:
        pytest.skip("Climate risk factors not available")

@pytest.fixture(scope="session")
def dashboard_app(temp_dir):
    """Create a DashboardApp for testing."""
    try:
        from src.integrations.geo_financial.dashboard_app import DashboardApp
        
        # Create dashboard app
        app = DashboardApp(
            data_dir=temp_dir,
            output_dir=temp_dir,
            device_id=-1  # Use CPU for tests
        )
        
        # Generate test data
        app.generate_data(
            dem_size=50,  # Small size for faster tests
            num_assets=5,
            days=30,
            seed=42
        )
        
        # Load data
        app.load_data()
        
        return app
    except ImportError:
        pytest.skip("DashboardApp not available")