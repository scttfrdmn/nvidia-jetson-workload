"""
Geospatial Risk Analysis Module

This module provides classes and functions for analyzing financial risk within
a geospatial context, leveraging both the Financial Modeling and Geospatial Analysis
workloads.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from financial_modeling.risk_metrics import RiskMetricsAnalyzer
from financial_modeling.portfolio_optimization import PortfolioOptimizer
from geospatial.dem import DEMProcessor, GeoTransform
from geospatial.point_cloud import PointCloud


class SpatialRiskFactor:
    """
    A class representing a risk factor with spatial properties.
    
    This class combines a geospatial feature (like elevation, flood risk, or
    proximity to certain features) with financial risk implications.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        risk_weight: float,
        spatial_data: Union[np.ndarray, str],
        geo_transform: Optional[GeoTransform] = None,
        transform_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ):
        """
        Initialize a spatial risk factor.
        
        Args:
            name: Name of the risk factor
            description: Description of what this factor represents
            risk_weight: Weight of this factor in risk calculations (0.0 to 1.0)
            spatial_data: Either a NumPy array of spatial data or a path to a spatial data file
            geo_transform: Optional GeoTransform for geospatial referencing
            transform_func: Optional function to transform raw spatial data to risk scores
        """
        self.name = name
        self.description = description
        self.risk_weight = risk_weight
        
        # Load spatial data if a file path is provided
        if isinstance(spatial_data, str):
            self.dem_processor = DEMProcessor(spatial_data)
            self.spatial_data = self.dem_processor.get_elevation_data()
            self.geo_transform = self.dem_processor.get_geo_transform()
        else:
            self.spatial_data = spatial_data
            self.geo_transform = geo_transform
        
        # Apply transformation if provided
        if transform_func is not None:
            self.risk_data = transform_func(self.spatial_data)
        else:
            # Default transformation: normalize to 0-1 range
            min_val = np.min(self.spatial_data)
            max_val = np.max(self.spatial_data)
            if max_val > min_val:
                self.risk_data = (self.spatial_data - min_val) / (max_val - min_val)
            else:
                self.risk_data = np.zeros_like(self.spatial_data)
    
    def get_risk_at_point(self, x: float, y: float) -> float:
        """
        Get the risk value at a specific geographic coordinate.
        
        Args:
            x: X coordinate (longitude/easting)
            y: Y coordinate (latitude/northing)
            
        Returns:
            Risk value at the specified point
        """
        if self.geo_transform is None:
            raise ValueError("Geographic transformation is required for coordinate lookup")
        
        # Convert geographic coordinates to pixel coordinates
        pixel_x, pixel_y = self.geo_transform.geo_to_pixel(x, y)
        
        # Check if coordinates are within the bounds of the spatial data
        if (0 <= pixel_x < self.risk_data.shape[1] and 
            0 <= pixel_y < self.risk_data.shape[0]):
            return self.risk_data[pixel_y, pixel_x]
        else:
            return 0.0  # Out of bounds
            
    def get_risk_in_region(self, min_x: float, min_y: float, max_x: float, max_y: float) -> np.ndarray:
        """
        Get a subset of risk values within a bounding box.
        
        Args:
            min_x: Minimum X coordinate
            min_y: Minimum Y coordinate
            max_x: Maximum X coordinate
            max_y: Maximum Y coordinate
            
        Returns:
            2D array of risk values in the region
        """
        if self.geo_transform is None:
            raise ValueError("Geographic transformation is required for region lookup")
        
        # Convert geographic coordinates to pixel coordinates
        min_pixel_x, min_pixel_y = self.geo_transform.geo_to_pixel(min_x, min_y)
        max_pixel_x, max_pixel_y = self.geo_transform.geo_to_pixel(max_x, max_y)
        
        # Ensure pixel coordinates are within bounds
        min_pixel_x = max(0, min(min_pixel_x, self.risk_data.shape[1] - 1))
        min_pixel_y = max(0, min(min_pixel_y, self.risk_data.shape[0] - 1))
        max_pixel_x = max(0, min(max_pixel_x, self.risk_data.shape[1] - 1))
        max_pixel_y = max(0, min(max_pixel_y, self.risk_data.shape[0] - 1))
        
        # Extract the region
        return self.risk_data[min_pixel_y:max_pixel_y+1, min_pixel_x:max_pixel_x+1]


class GeospatialRiskModel:
    """
    A model for analyzing financial risk based on geospatial factors.
    
    This class combines multiple SpatialRiskFactor objects to create a comprehensive
    risk model that can be used to evaluate financial risk at different locations.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize a geospatial risk model.
        
        Args:
            device_id: CUDA device ID (negative for CPU)
        """
        self.risk_factors = []
        self.risk_analyzer = RiskMetricsAnalyzer(device_id)
        self.device_id = device_id
    
    def add_risk_factor(self, risk_factor: SpatialRiskFactor) -> None:
        """
        Add a spatial risk factor to the model.
        
        Args:
            risk_factor: The SpatialRiskFactor to add
        """
        self.risk_factors.append(risk_factor)
    
    def get_combined_risk_at_point(self, x: float, y: float) -> float:
        """
        Calculate the combined risk at a specific geographic coordinate.
        
        Args:
            x: X coordinate (longitude/easting)
            y: Y coordinate (latitude/northing)
            
        Returns:
            Combined risk value at the specified point
        """
        if not self.risk_factors:
            return 0.0
        
        # Get individual risk values for each factor
        risk_values = []
        risk_weights = []
        
        for factor in self.risk_factors:
            try:
                risk_values.append(factor.get_risk_at_point(x, y))
                risk_weights.append(factor.risk_weight)
            except ValueError:
                # Skip factors that don't cover this point
                continue
        
        if not risk_values:
            return 0.0
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(risk_weights)
        if weight_sum > 0:
            normalized_weights = [w / weight_sum for w in risk_weights]
        else:
            normalized_weights = [1.0 / len(risk_weights)] * len(risk_weights)
        
        # Calculate weighted average
        return sum(v * w for v, w in zip(risk_values, normalized_weights))
    
    def get_risk_map(self, 
                     min_x: float, 
                     min_y: float, 
                     max_x: float, 
                     max_y: float,
                     resolution: float) -> Tuple[np.ndarray, GeoTransform]:
        """
        Generate a risk map for a region.
        
        Args:
            min_x: Minimum X coordinate
            min_y: Minimum Y coordinate
            max_x: Maximum X coordinate
            max_y: Maximum Y coordinate
            resolution: Spatial resolution of the output map
            
        Returns:
            Tuple of (risk_map_array, geo_transform)
        """
        # Calculate grid dimensions
        width = int((max_x - min_x) / resolution) + 1
        height = int((max_y - min_y) / resolution) + 1
        
        # Create new GeoTransform for the output
        geo_transform = GeoTransform([
            min_x,               # top-left x
            resolution,          # w-e pixel resolution
            0,                   # row rotation
            max_y,               # top-left y
            0,                   # column rotation
            -resolution          # n-s pixel resolution (negative)
        ])
        
        # Create output risk map
        risk_map = np.zeros((height, width), dtype=np.float32)
        
        # Fill risk map with combined risk values
        for y in range(height):
            for x in range(width):
                # Convert pixel coordinates to geographic coordinates
                geo_x, geo_y = geo_transform.pixel_to_geo(x, y)
                # Set risk value
                risk_map[y, x] = self.get_combined_risk_at_point(geo_x, geo_y)
        
        return risk_map, geo_transform


class GeospatialPortfolio:
    """
    A portfolio of assets with geospatial properties.
    
    This class represents a financial portfolio where each asset has a geographic
    location, allowing for geospatial risk analysis.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize a geospatial portfolio.
        
        Args:
            device_id: CUDA device ID (negative for CPU)
        """
        self.assets = []
        self.portfolio_optimizer = PortfolioOptimizer(device_id)
        self.risk_analyzer = RiskMetricsAnalyzer(device_id)
        self.device_id = device_id
    
    def add_asset(self, 
                  asset_id: str, 
                  name: str, 
                  value: float, 
                  x: float, 
                  y: float, 
                  returns: Optional[np.ndarray] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an asset to the portfolio.
        
        Args:
            asset_id: Unique identifier for the asset
            name: Name of the asset
            value: Current value of the asset
            x: X coordinate (longitude/easting)
            y: Y coordinate (latitude/northing)
            returns: Optional historical returns for the asset
            metadata: Optional additional metadata
        """
        asset = {
            'id': asset_id,
            'name': name,
            'value': value,
            'x': x,
            'y': y,
            'returns': returns if returns is not None else np.array([]),
            'metadata': metadata if metadata is not None else {}
        }
        self.assets.append(asset)
    
    def add_assets_from_dataframe(self, 
                                  df: pd.DataFrame, 
                                  id_col: str, 
                                  name_col: str, 
                                  value_col: str,
                                  x_col: str, 
                                  y_col: str,
                                  returns_col: Optional[str] = None,
                                  metadata_cols: Optional[List[str]] = None) -> None:
        """
        Add multiple assets from a DataFrame.
        
        Args:
            df: Pandas DataFrame containing asset information
            id_col: Column name for asset IDs
            name_col: Column name for asset names
            value_col: Column name for asset values
            x_col: Column name for X coordinates
            y_col: Column name for Y coordinates
            returns_col: Optional column name for returns (should contain lists or arrays)
            metadata_cols: Optional list of column names to include in metadata
        """
        for _, row in df.iterrows():
            metadata = {}
            if metadata_cols:
                for col in metadata_cols:
                    if col in df.columns:
                        metadata[col] = row[col]
            
            returns = None
            if returns_col and returns_col in df.columns:
                returns = row[returns_col]
                if not isinstance(returns, np.ndarray):
                    returns = np.array(returns)
            
            self.add_asset(
                asset_id=str(row[id_col]),
                name=str(row[name_col]),
                value=float(row[value_col]),
                x=float(row[x_col]),
                y=float(row[y_col]),
                returns=returns,
                metadata=metadata
            )
    
    def assess_risk(self, risk_model: GeospatialRiskModel) -> Dict[str, float]:
        """
        Assess the geospatial risk of all assets using a risk model.
        
        Args:
            risk_model: GeospatialRiskModel to use for assessment
            
        Returns:
            Dictionary mapping asset IDs to risk scores
        """
        risk_scores = {}
        
        for asset in self.assets:
            risk_score = risk_model.get_combined_risk_at_point(asset['x'], asset['y'])
            risk_scores[asset['id']] = risk_score
        
        return risk_scores
    
    def calculate_portfolio_var(self, 
                               confidence_level: float = 0.95, 
                               lookback_days: int = 252) -> float:
        """
        Calculate portfolio Value-at-Risk (VaR).
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            lookback_days: Number of days to look back for historical returns
            
        Returns:
            VaR value for the portfolio
        """
        # Check if all assets have returns data
        for asset in self.assets:
            if len(asset['returns']) < lookback_days:
                raise ValueError(f"Asset {asset['id']} has insufficient returns data")
        
        # Get historical returns for each asset
        asset_returns = np.array([asset['returns'][-lookback_days:] for asset in self.assets])
        
        # Get current weights
        total_value = sum(asset['value'] for asset in self.assets)
        weights = np.array([asset['value'] / total_value for asset in self.assets])
        
        # Calculate portfolio returns
        portfolio_returns = np.sum(asset_returns * weights[:, np.newaxis], axis=0)
        
        # Calculate VaR
        return self.risk_analyzer.calculate_var(portfolio_returns, confidence_level)
    
    def optimize_for_geo_risk(self, 
                             risk_model: GeospatialRiskModel,
                             target_return: float,
                             max_risk_score: float = 0.5,
                             risk_aversion: float = 1.0) -> Dict[str, float]:
        """
        Optimize the portfolio to minimize exposure to geospatial risk.
        
        Args:
            risk_model: GeospatialRiskModel to use for assessment
            target_return: Target portfolio return
            max_risk_score: Maximum allowable risk score per asset
            risk_aversion: Risk aversion parameter (higher values prioritize risk reduction)
            
        Returns:
            Dictionary mapping asset IDs to optimized weights
        """
        # Assess risk for all assets
        risk_scores = self.assess_risk(risk_model)
        
        # Get historical returns for each asset
        lookback_days = 252  # 1 year
        for asset in self.assets:
            if len(asset['returns']) < lookback_days:
                raise ValueError(f"Asset {asset['id']} has insufficient returns data")
        
        # Prepare data for portfolio optimization
        asset_returns = np.array([asset['returns'][-lookback_days:] for asset in self.assets])
        expected_returns = np.mean(asset_returns, axis=1)
        cov_matrix = np.cov(asset_returns)
        
        # Adjust expected returns based on geospatial risk
        risk_array = np.array([risk_scores[asset['id']] for asset in self.assets])
        adjusted_returns = expected_returns - (risk_aversion * risk_array)
        
        # Perform optimization
        result = self.portfolio_optimizer.optimize(
            expected_returns=adjusted_returns,
            cov_matrix=cov_matrix,
            target_return=target_return,
            constraints={'max_weight': 0.3}  # Example constraint: no more than 30% in any asset
        )
        
        # Map weights to asset IDs
        weights = result.get('weights', np.zeros(len(self.assets)))
        optimized_weights = {asset['id']: weights[i] for i, asset in enumerate(self.assets)}
        
        return optimized_weights
    
    def get_asset_locations(self) -> Tuple[List[float], List[float], List[str], List[float]]:
        """
        Get asset locations for visualization.
        
        Returns:
            Tuple of (x_coordinates, y_coordinates, asset_names, asset_values)
        """
        x_coords = [asset['x'] for asset in self.assets]
        y_coords = [asset['y'] for asset in self.assets]
        names = [asset['name'] for asset in self.assets]
        values = [asset['value'] for asset in self.assets]
        
        return x_coords, y_coords, names, values


# Utility functions

def create_elevation_risk_factor(dem_path: str, 
                                name: str = "Elevation Risk", 
                                risk_weight: float = 0.5,
                                high_risk_threshold: float = 10.0,
                                low_risk_threshold: float = 100.0) -> SpatialRiskFactor:
    """
    Create a risk factor based on elevation.
    
    This helper function creates a risk factor where lower elevations have higher risk
    (e.g., for flood risk assessment).
    
    Args:
        dem_path: Path to the Digital Elevation Model (DEM) file
        name: Name for this risk factor
        risk_weight: Weight of this factor in risk calculations (0.0 to 1.0)
        high_risk_threshold: Elevation below which risk is maximum (1.0)
        low_risk_threshold: Elevation above which risk is minimum (0.0)
        
    Returns:
        A configured SpatialRiskFactor
    """
    def elevation_to_risk(elevation_data: np.ndarray) -> np.ndarray:
        """Transform elevation values to risk scores."""
        risk_data = np.zeros_like(elevation_data, dtype=np.float32)
        
        # Areas below high_risk_threshold have maximum risk
        risk_data[elevation_data <= high_risk_threshold] = 1.0
        
        # Areas above low_risk_threshold have minimum risk
        risk_data[elevation_data >= low_risk_threshold] = 0.0
        
        # Linear interpolation for elevations in between
        mask = (elevation_data > high_risk_threshold) & (elevation_data < low_risk_threshold)
        if np.any(mask):
            risk_data[mask] = 1.0 - ((elevation_data[mask] - high_risk_threshold) / 
                                     (low_risk_threshold - high_risk_threshold))
        
        return risk_data
    
    return SpatialRiskFactor(
        name=name,
        description="Risk based on elevation (lower elevations have higher risk)",
        risk_weight=risk_weight,
        spatial_data=dem_path,
        transform_func=elevation_to_risk
    )


def create_slope_risk_factor(dem_path: str, 
                            name: str = "Slope Risk", 
                            risk_weight: float = 0.5,
                            high_risk_threshold: float = 30.0) -> SpatialRiskFactor:
    """
    Create a risk factor based on terrain slope.
    
    This helper function creates a risk factor where steeper slopes have higher risk
    (e.g., for landslide risk assessment).
    
    Args:
        dem_path: Path to the Digital Elevation Model (DEM) file
        name: Name for this risk factor
        risk_weight: Weight of this factor in risk calculations (0.0 to 1.0)
        high_risk_threshold: Slope angle (in degrees) above which risk is maximum (1.0)
        
    Returns:
        A configured SpatialRiskFactor
    """
    # Load DEM and calculate slopes
    dem_processor = DEMProcessor(dem_path)
    slope_data = dem_processor.calculate_slope()
    
    def slope_to_risk(slope_data: np.ndarray) -> np.ndarray:
        """Transform slope values to risk scores."""
        risk_data = np.zeros_like(slope_data, dtype=np.float32)
        
        # Linear increase in risk with slope
        risk_data = np.clip(slope_data / high_risk_threshold, 0.0, 1.0)
        
        return risk_data
    
    return SpatialRiskFactor(
        name=name,
        description="Risk based on terrain slope (steeper slopes have higher risk)",
        risk_weight=risk_weight,
        spatial_data=slope_data,
        geo_transform=dem_processor.get_geo_transform(),
        transform_func=slope_to_risk
    )


def distance_to_risk(distance_data: np.ndarray, 
                    max_distance: float, 
                    inverse: bool = True) -> np.ndarray:
    """
    Convert distance values to risk scores.
    
    Args:
        distance_data: Array of distance values
        max_distance: Distance beyond which risk is negligible
        inverse: If True, shorter distances have higher risk
        
    Returns:
        Array of risk scores
    """
    # Normalize distances
    normalized_distances = np.clip(distance_data / max_distance, 0.0, 1.0)
    
    # Convert to risk scores
    if inverse:
        # Shorter distances have higher risk
        risk_scores = 1.0 - normalized_distances
    else:
        # Longer distances have higher risk
        risk_scores = normalized_distances
    
    return risk_scores