"""
Data Connectors Module

This module provides utilities for loading, transforming, and connecting geospatial and
financial data sources.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import io
import json
import zipfile
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from urllib.request import urlopen

from financial_modeling.risk_metrics import RiskMetricsAnalyzer
from geospatial.dem import DEMProcessor, GeoTransform
from geospatial.point_cloud import PointCloud


class AssetLocationDataLoader:
    """
    Loader for financial asset location data.
    
    This class helps load asset data from various sources and formats,
    supporting both financial and geospatial attributes.
    """
    
    @staticmethod
    def load_asset_csv(
        file_path: str,
        id_col: str,
        name_col: str,
        value_col: str,
        x_col: str,
        y_col: str,
        delimiter: str = ',',
        encoding: str = 'utf-8'
    ) -> pd.DataFrame:
        """
        Load asset data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            id_col: Column name for asset IDs
            name_col: Column name for asset names
            value_col: Column name for asset values
            x_col: Column name for X coordinates
            y_col: Column name for Y coordinates
            delimiter: CSV delimiter (default: ',')
            encoding: File encoding (default: 'utf-8')
            
        Returns:
            DataFrame containing asset data
        """
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
        
        # Validate required columns
        required_cols = [id_col, name_col, value_col, x_col, y_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        return df
    
    @staticmethod
    def load_asset_geojson(
        file_path: str,
        id_prop: str,
        name_prop: str,
        value_prop: str
    ) -> pd.DataFrame:
        """
        Load asset data from a GeoJSON file.
        
        Args:
            file_path: Path to the GeoJSON file
            id_prop: Property name for asset IDs
            name_prop: Property name for asset names
            value_prop: Property name for asset values
            
        Returns:
            DataFrame containing asset data
        """
        # Read GeoJSON file
        with open(file_path, 'r') as f:
            geojson = json.load(f)
        
        # Extract features
        if 'features' not in geojson:
            raise ValueError("GeoJSON file does not contain features")
        
        features = geojson['features']
        if not features:
            raise ValueError("GeoJSON file contains no features")
        
        # Extract data
        data = []
        for feature in features:
            if 'properties' not in feature or 'geometry' not in feature:
                continue
            
            properties = feature['properties']
            geometry = feature['geometry']
            
            # Check if required properties exist
            if id_prop not in properties or name_prop not in properties or value_prop not in properties:
                continue
            
            # Get coordinates
            if geometry['type'] == 'Point':
                x, y = geometry['coordinates']
            else:
                continue  # Skip non-point geometries
            
            # Add to data
            row = {
                'id': properties[id_prop],
                'name': properties[name_prop],
                'value': float(properties[value_prop]),
                'x': x,
                'y': y
            }
            
            # Add other properties
            for key, value in properties.items():
                if key not in [id_prop, name_prop, value_prop]:
                    row[key] = value
            
            data.append(row)
        
        return pd.DataFrame(data)


class FinancialDataLoader:
    """
    Loader for financial time series data.
    
    This class helps load financial time series data (returns, prices, etc.)
    and attach it to asset data.
    """
    
    @staticmethod
    def load_returns_csv(
        file_path: str,
        asset_id_col: str,
        date_col: str,
        return_col: str,
        delimiter: str = ',',
        encoding: str = 'utf-8'
    ) -> Dict[str, np.ndarray]:
        """
        Load returns data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            asset_id_col: Column name for asset IDs
            date_col: Column name for dates
            return_col: Column name for returns
            delimiter: CSV delimiter (default: ',')
            encoding: File encoding (default: 'utf-8')
            
        Returns:
            Dictionary mapping asset IDs to arrays of returns
        """
        df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
        
        # Validate required columns
        required_cols = [asset_id_col, date_col, return_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Convert dates to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Group by asset ID and extract returns
        returns_dict = {}
        for asset_id, group in df.groupby(asset_id_col):
            returns_dict[str(asset_id)] = group[return_col].values
        
        return returns_dict
    
    @staticmethod
    def calculate_returns_from_prices(
        df: pd.DataFrame,
        asset_id_col: str,
        date_col: str,
        price_col: str
    ) -> Dict[str, np.ndarray]:
        """
        Calculate returns from price data.
        
        Args:
            df: DataFrame containing price data
            asset_id_col: Column name for asset IDs
            date_col: Column name for dates
            price_col: Column name for prices
            
        Returns:
            Dictionary mapping asset IDs to arrays of returns
        """
        # Validate required columns
        required_cols = [asset_id_col, date_col, price_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Convert dates to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values([asset_id_col, date_col])
        
        # Group by asset ID and calculate returns
        returns_dict = {}
        for asset_id, group in df.groupby(asset_id_col):
            prices = group[price_col].values
            # Calculate returns: (p_t / p_{t-1}) - 1
            returns = np.diff(prices) / prices[:-1]
            returns_dict[str(asset_id)] = returns
        
        return returns_dict
    
    @staticmethod
    def attach_returns_to_assets(
        assets_df: pd.DataFrame,
        returns_dict: Dict[str, np.ndarray],
        asset_id_col: str
    ) -> pd.DataFrame:
        """
        Attach returns data to an asset DataFrame.
        
        Args:
            assets_df: DataFrame containing asset data
            returns_dict: Dictionary mapping asset IDs to arrays of returns
            asset_id_col: Column name for asset IDs
            
        Returns:
            DataFrame with added returns column
        """
        # Create a copy of the input DataFrame
        df = assets_df.copy()
        
        # Add returns column
        df['returns'] = df[asset_id_col].apply(
            lambda asset_id: returns_dict.get(str(asset_id), np.array([]))
        )
        
        return df


class GeoRiskDataLoader:
    """
    Loader for geospatial risk data.
    
    This class helps load and process geospatial data for risk analysis.
    """
    
    @staticmethod
    def load_dem(file_path: str) -> DEMProcessor:
        """
        Load a Digital Elevation Model (DEM).
        
        Args:
            file_path: Path to the DEM file
            
        Returns:
            DEMProcessor object
        """
        return DEMProcessor(file_path)
    
    @staticmethod
    def load_geotiff_as_risk_factor(
        file_path: str,
        name: str,
        description: str,
        risk_weight: float,
        transform_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
    ) -> 'SpatialRiskFactor':
        """
        Load a GeoTIFF file as a spatial risk factor.
        
        Args:
            file_path: Path to the GeoTIFF file
            name: Name for the risk factor
            description: Description of the risk factor
            risk_weight: Weight of this factor in risk calculations (0.0 to 1.0)
            transform_func: Optional function to transform raw data to risk scores
            
        Returns:
            SpatialRiskFactor object
        """
        # Import locally to avoid circular imports
        from .geo_risk import SpatialRiskFactor
        
        return SpatialRiskFactor(
            name=name,
            description=description,
            risk_weight=risk_weight,
            spatial_data=file_path,
            transform_func=transform_func
        )
    
    @staticmethod
    def download_sample_data(output_dir: str = './data') -> Dict[str, str]:
        """
        Download sample geospatial and financial data for testing.
        
        Args:
            output_dir: Directory to save downloaded data
            
        Returns:
            Dictionary mapping data type to file path
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # URLs for sample data
        sample_data_urls = {
            'dem': 'https://storage.googleapis.com/nvidia-jetson-sample-data/sample_dem.tif',
            'asset_locations': 'https://storage.googleapis.com/nvidia-jetson-sample-data/sample_assets.csv',
            'asset_returns': 'https://storage.googleapis.com/nvidia-jetson-sample-data/sample_returns.csv'
        }
        
        # Download each file
        file_paths = {}
        for data_type, url in sample_data_urls.items():
            output_path = os.path.join(output_dir, os.path.basename(url))
            
            try:
                with urlopen(url) as response:
                    with open(output_path, 'wb') as out_file:
                        out_file.write(response.read())
                file_paths[data_type] = output_path
            except Exception as e:
                print(f"Error downloading {data_type} data: {e}")
                continue
        
        return file_paths