#!/usr/bin/env python3
"""
Multi-region Geospatial Financial Analysis Module

This module extends the geo_financial integration with capabilities for analyzing
and comparing multiple geographic regions simultaneously. It enables cross-region
risk assessment, portfolio diversification, and comparative analysis.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import json
import logging
import matplotlib.pyplot as plt
from scipy import stats
import multiprocessing as mp
from functools import partial

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Import core modules
from src.integrations.geo_financial.geo_risk import (
    GeospatialRiskModel,
    GeospatialPortfolio,
    SpatialRiskFactor,
    create_elevation_risk_factor,
    create_slope_risk_factor
)
from src.integrations.geo_financial.risk_aggregation import (
    RiskAggregator,
    RiskSurfaceGenerator,
    AggregationMethod,
    create_combined_risk_factor
)
from src.integrations.geo_financial.climate_risk_assessment import (
    ClimateRiskAssessor,
    ClimateScenario,
    TimeHorizon,
    ClimateHazardType,
    TransitionRiskType
)
from src.integrations.geo_financial.visualization import GeoFinancialVisualizer


class RegionDefinition:
    """
    Defines a geographic region for analysis.
    
    A region has spatial boundaries, a name, and optional metadata like
    political jurisdiction, climate characteristics, or economic indicators.
    """
    
    def __init__(self, 
                name: str,
                bounds: Dict[str, float],
                dem_data: Optional[np.ndarray] = None,
                dem_transform: Optional[Any] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a region definition.
        
        Args:
            name: Name of the region
            bounds: Geographic bounds as {min_x, min_y, max_x, max_y}
            dem_data: Optional DEM data for this region
            dem_transform: Optional GeoTransform for the DEM data
            metadata: Optional metadata about the region
        """
        self.name = name
        self.bounds = bounds
        self.dem_data = dem_data
        self.dem_transform = dem_transform
        self.metadata = metadata or {}
    
    def contains_point(self, x: float, y: float) -> bool:
        """
        Check if a point is within this region.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if the point is within the region, False otherwise
        """
        return (self.bounds["min_x"] <= x <= self.bounds["max_x"] and
                self.bounds["min_y"] <= y <= self.bounds["max_y"])
    
    def get_area(self) -> float:
        """
        Calculate the area of the region.
        
        Returns:
            Area in square units
        """
        width = self.bounds["max_x"] - self.bounds["min_x"]
        height = self.bounds["max_y"] - self.bounds["min_y"]
        return width * height
    
    def get_dem_subset(self, 
                     full_dem_data: np.ndarray, 
                     full_transform: Any) -> Tuple[np.ndarray, Any]:
        """
        Extract a subset of DEM data for this region.
        
        Args:
            full_dem_data: Complete DEM data array
            full_transform: GeoTransform for the full DEM
            
        Returns:
            Tuple of (dem_subset, transform)
        """
        if full_dem_data is None or full_transform is None:
            raise ValueError("Full DEM data and transform must be provided")
        
        # Calculate pixel coordinates
        min_x_px = int((self.bounds["min_x"] - full_transform.origin_x) / full_transform.pixel_width)
        min_y_px = int((self.bounds["min_y"] - full_transform.origin_y) / full_transform.pixel_height)
        max_x_px = int((self.bounds["max_x"] - full_transform.origin_x) / full_transform.pixel_width)
        max_y_px = int((self.bounds["max_y"] - full_transform.origin_y) / full_transform.pixel_height)
        
        # Ensure within bounds
        height, width = full_dem_data.shape
        min_x_px = max(0, min_x_px)
        min_y_px = max(0, min_y_px)
        max_x_px = min(width, max_x_px)
        max_y_px = min(height, max_y_px)
        
        # Extract subset
        subset = full_dem_data[min_y_px:max_y_px, min_x_px:max_x_px]
        
        # Create new transform
        from geospatial.dem import GeoTransform
        new_origin_x = full_transform.origin_x + (min_x_px * full_transform.pixel_width)
        new_origin_y = full_transform.origin_y + (min_y_px * full_transform.pixel_height)
        new_transform = GeoTransform(
            new_origin_x, 
            new_origin_y,
            full_transform.pixel_width,
            full_transform.pixel_height
        )
        
        return subset, new_transform
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert region to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the region
        """
        return {
            "name": self.name,
            "bounds": self.bounds,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegionDefinition':
        """
        Create a region from a dictionary.
        
        Args:
            data: Dictionary with region data
            
        Returns:
            RegionDefinition instance
        """
        return cls(
            name=data["name"],
            bounds=data["bounds"],
            metadata=data.get("metadata", {})
        )


class RegionalPortfolio:
    """
    An extension of GeospatialPortfolio that separates assets by region.
    
    This class enables region-specific analysis and comparisons.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize a regional portfolio.
        
        Args:
            device_id: GPU device ID (-1 for CPU only)
        """
        self.device_id = device_id
        self.regions = {}  # name -> RegionDefinition
        self.portfolios = {}  # name -> GeospatialPortfolio
    
    def add_region(self, region: RegionDefinition) -> None:
        """
        Add a region to the portfolio.
        
        Args:
            region: RegionDefinition to add
        """
        if region.name in self.regions:
            raise ValueError(f"Region '{region.name}' already exists")
        
        self.regions[region.name] = region
        self.portfolios[region.name] = GeospatialPortfolio(device_id=self.device_id)
    
    def add_asset(self, 
                region_name: str,
                asset_id: str,
                asset_name: str,
                value: float,
                x: float,
                y: float,
                metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an asset to a specific region.
        
        Args:
            region_name: Name of the region
            asset_id: Unique identifier for the asset
            asset_name: Human-readable name for the asset
            value: Value of the asset
            x: X coordinate
            y: Y coordinate
            metadata: Optional additional data about the asset
        """
        if region_name not in self.regions:
            raise ValueError(f"Region '{region_name}' does not exist")
        
        # Check if point is in region
        region = self.regions[region_name]
        if not region.contains_point(x, y):
            raise ValueError(f"Asset ({x}, {y}) is outside region '{region_name}'")
        
        # Add to regional portfolio
        portfolio = self.portfolios[region_name]
        portfolio.add_asset(asset_id, asset_name, value, x, y, metadata)
    
    def add_assets_from_dataframe(self,
                                assets_df: pd.DataFrame,
                                id_col: str,
                                name_col: str,
                                value_col: str,
                                x_col: str,
                                y_col: str,
                                metadata_cols: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Add assets from a DataFrame, automatically assigning to regions.
        
        Args:
            assets_df: DataFrame with asset data
            id_col: Column name for asset IDs
            name_col: Column name for asset names
            value_col: Column name for asset values
            x_col: Column name for X coordinates
            y_col: Column name for Y coordinates
            metadata_cols: Optional list of columns to include as metadata
            
        Returns:
            Dictionary mapping region names to lists of asset IDs
        """
        if not self.regions:
            raise ValueError("No regions defined")
        
        # Track added assets by region
        added_assets = {region_name: [] for region_name in self.regions}
        
        # Process each asset
        for _, asset in assets_df.iterrows():
            asset_id = str(asset[id_col])
            asset_name = str(asset[name_col])
            value = float(asset[value_col])
            x = float(asset[x_col])
            y = float(asset[y_col])
            
            # Extract metadata if requested
            metadata = {}
            if metadata_cols:
                for col in metadata_cols:
                    if col in assets_df.columns:
                        metadata[col] = asset[col]
            
            # Find containing region
            assigned = False
            for region_name, region in self.regions.items():
                if region.contains_point(x, y):
                    # Add to regional portfolio
                    portfolio = self.portfolios[region_name]
                    portfolio.add_asset(asset_id, asset_name, value, x, y, metadata)
                    added_assets[region_name].append(asset_id)
                    assigned = True
                    break
            
            if not assigned:
                # Asset doesn't fall into any region
                continue
        
        return added_assets
    
    def get_region_portfolio(self, region_name: str) -> GeospatialPortfolio:
        """
        Get the GeospatialPortfolio for a specific region.
        
        Args:
            region_name: Name of the region
            
        Returns:
            GeospatialPortfolio for the specified region
        """
        if region_name not in self.portfolios:
            raise ValueError(f"Region '{region_name}' does not exist")
        
        return self.portfolios[region_name]
    
    def get_regions(self) -> List[str]:
        """
        Get a list of all region names.
        
        Returns:
            List of region names
        """
        return list(self.regions.keys())
    
    def get_region_asset_counts(self) -> Dict[str, int]:
        """
        Get the number of assets in each region.
        
        Returns:
            Dictionary mapping region names to asset counts
        """
        return {
            region_name: len(portfolio.assets)
            for region_name, portfolio in self.portfolios.items()
        }
    
    def get_region_asset_values(self) -> Dict[str, float]:
        """
        Get the total asset value in each region.
        
        Returns:
            Dictionary mapping region names to total asset values
        """
        return {
            region_name: sum(asset["value"] for asset in portfolio.assets)
            for region_name, portfolio in self.portfolios.items()
        }
    
    def get_all_assets(self) -> List[Dict[str, Any]]:
        """
        Get a list of all assets from all regions.
        
        Returns:
            List of asset dictionaries with region added
        """
        all_assets = []
        for region_name, portfolio in self.portfolios.items():
            for asset in portfolio.assets:
                # Create a copy with region added
                asset_copy = asset.copy()
                asset_copy["region"] = region_name
                all_assets.append(asset_copy)
        
        return all_assets
    
    def get_all_assets_as_dataframe(self) -> pd.DataFrame:
        """
        Get all assets as a pandas DataFrame.
        
        Returns:
            DataFrame with all assets from all regions
        """
        all_assets = self.get_all_assets()
        return pd.DataFrame(all_assets)
    
    def save(self, file_path: str) -> None:
        """
        Save the regional portfolio to a file.
        
        Args:
            file_path: Path to save the file
        """
        # Create serializable data
        data = {
            "regions": {
                name: region.to_dict()
                for name, region in self.regions.items()
            },
            "assets": {
                region_name: [asset for asset in portfolio.assets]
                for region_name, portfolio in self.portfolios.items()
            }
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, file_path: str, device_id: int = 0) -> 'RegionalPortfolio':
        """
        Load a regional portfolio from a file.
        
        Args:
            file_path: Path to the saved file
            device_id: GPU device ID
            
        Returns:
            RegionalPortfolio instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Create instance
        portfolio = cls(device_id=device_id)
        
        # Add regions
        for name, region_data in data["regions"].items():
            region = RegionDefinition.from_dict(region_data)
            portfolio.add_region(region)
        
        # Add assets
        for region_name, assets in data["assets"].items():
            if region_name not in portfolio.portfolios:
                continue
                
            region_portfolio = portfolio.portfolios[region_name]
            for asset in assets:
                region_portfolio.add_asset(
                    asset["id"],
                    asset["name"],
                    asset["value"],
                    asset["x"],
                    asset["y"],
                    asset.get("metadata", {})
                )
        
        return portfolio


class MultiRegionRiskModel:
    """
    A risk model that can be applied across multiple regions.
    
    This class extends GeospatialRiskModel to work with RegionalPortfolio,
    enabling consistent risk assessment across regions.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize a multi-region risk model.
        
        Args:
            device_id: GPU device ID (-1 for CPU only)
        """
        self.device_id = device_id
        self.base_model = GeospatialRiskModel(device_id=device_id)
        self.region_models = {}  # name -> GeospatialRiskModel
    
    def add_risk_factor(self, risk_factor: SpatialRiskFactor) -> None:
        """
        Add a risk factor to the base model.
        
        This risk factor will be used as a template for all regions.
        
        Args:
            risk_factor: SpatialRiskFactor to add
        """
        self.base_model.add_risk_factor(risk_factor)
    
    def create_region_risk_model(self, 
                               region: RegionDefinition,
                               dem_data: Optional[np.ndarray] = None,
                               dem_transform: Optional[Any] = None) -> GeospatialRiskModel:
        """
        Create a risk model specific to a region.
        
        This adapts the base risk factors to the specific region.
        
        Args:
            region: RegionDefinition to create a model for
            dem_data: Optional DEM data for this region
            dem_transform: Optional GeoTransform for the DEM data
            
        Returns:
            GeospatialRiskModel for the region
        """
        # Create new risk model
        model = GeospatialRiskModel(device_id=self.device_id)
        
        # Use provided or region's DEM data
        region_dem = dem_data if dem_data is not None else region.dem_data
        region_transform = dem_transform if dem_transform is not None else region.dem_transform
        
        # Create region-specific versions of each risk factor
        for rf in self.base_model.risk_factors:
            # Subclass-specific adaptation would go here
            # For now, just add the original
            model.add_risk_factor(rf)
        
        # Store the model
        self.region_models[region.name] = model
        
        return model
    
    def get_region_risk_model(self, region_name: str) -> GeospatialRiskModel:
        """
        Get the risk model for a specific region.
        
        Args:
            region_name: Name of the region
            
        Returns:
            GeospatialRiskModel for the specified region
        """
        if region_name not in self.region_models:
            raise ValueError(f"No risk model for region '{region_name}'")
        
        return self.region_models[region_name]
    
    def assess_regional_risks(self, 
                           regional_portfolio: RegionalPortfolio) -> Dict[str, Dict[str, float]]:
        """
        Assess risks across all regions in a regional portfolio.
        
        Args:
            regional_portfolio: RegionalPortfolio to assess
            
        Returns:
            Dictionary mapping region names to risk assessment results
        """
        results = {}
        
        # Ensure we have models for all regions
        for region_name in regional_portfolio.get_regions():
            if region_name not in self.region_models:
                # Create a model using the base model as template
                region = regional_portfolio.regions[region_name]
                self.create_region_risk_model(region)
        
        # Assess risks for each region
        for region_name, portfolio in regional_portfolio.portfolios.items():
            if len(portfolio.assets) == 0:
                # Skip empty portfolios
                continue
                
            # Get the risk model for this region
            model = self.region_models[region_name]
            
            # Assess portfolio risk
            risk_scores = portfolio.assess_risk(model)
            
            # Store results
            results[region_name] = risk_scores
        
        return results


class RegionalRiskComparator:
    """
    Compares risk characteristics across multiple regions.
    
    This class provides methods for analyzing and visualizing risk patterns
    across different geographic regions.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize a risk comparator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger("RegionalRiskComparator")
    
    def compare_risk_distributions(self,
                                  regional_risks: Dict[str, Dict[str, float]],
                                  regional_portfolio: RegionalPortfolio) -> Dict[str, Dict[str, float]]:
        """
        Compare statistical properties of risk distributions across regions.
        
        Args:
            regional_risks: Results from assess_regional_risks
            regional_portfolio: RegionalPortfolio used for assessment
            
        Returns:
            Dictionary with statistical comparisons by region
        """
        stats_by_region = {}
        
        for region_name, risk_scores in regional_risks.items():
            if not risk_scores:
                # Skip regions with no risk scores
                continue
            
            # Get scores as array
            scores = np.array(list(risk_scores.values()))
            
            # Calculate basic statistics
            stats = {
                "count": len(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "mean": np.mean(scores),
                "median": np.median(scores),
                "std": np.std(scores),
                "skew": stats.skew(scores) if len(scores) > 2 else 0,
                "kurtosis": stats.kurtosis(scores) if len(scores) > 3 else 0
            }
            
            # Get portfolio for this region
            portfolio = regional_portfolio.portfolios[region_name]
            
            # Calculate value-weighted statistics
            total_value = sum(asset["value"] for asset in portfolio.assets)
            weighted_risk = 0
            
            for asset in portfolio.assets:
                asset_id = asset["id"]
                if asset_id in risk_scores:
                    weighted_risk += asset["value"] * risk_scores[asset_id]
            
            if total_value > 0:
                weighted_risk /= total_value
            
            stats["value_weighted_risk"] = weighted_risk
            stats["total_value"] = total_value
            
            # Store results
            stats_by_region[region_name] = stats
        
        return stats_by_region
    
    def identify_high_risk_assets(self,
                               regional_risks: Dict[str, Dict[str, float]],
                               threshold: float = 0.7,
                               top_n: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify high-risk assets in each region.
        
        Args:
            regional_risks: Results from assess_regional_risks
            threshold: Risk threshold for high risk (0-1)
            top_n: Optional limit on number of assets per region
            
        Returns:
            Dictionary mapping region names to lists of high-risk assets
        """
        high_risk_assets = {}
        
        for region_name, risk_scores in regional_risks.items():
            # Sort assets by risk score
            sorted_assets = sorted(
                [(asset_id, score) for asset_id, score in risk_scores.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Apply threshold
            filtered_assets = [
                {"asset_id": asset_id, "risk_score": score}
                for asset_id, score in sorted_assets
                if score >= threshold
            ]
            
            # Apply top_n limit if specified
            if top_n is not None and top_n > 0:
                filtered_assets = filtered_assets[:top_n]
            
            high_risk_assets[region_name] = filtered_assets
        
        return high_risk_assets
    
    def calculate_diversification_benefit(self,
                                        regional_risks: Dict[str, Dict[str, float]],
                                        regional_portfolio: RegionalPortfolio) -> float:
        """
        Calculate the risk reduction benefit from regional diversification.
        
        Args:
            regional_risks: Results from assess_regional_risks
            regional_portfolio: RegionalPortfolio used for assessment
            
        Returns:
            Diversification benefit as a risk reduction percentage
        """
        # Get all assets
        all_assets = regional_portfolio.get_all_assets()
        if not all_assets:
            return 0.0
        
        # Calculate combined portfolio risk (weighted average)
        total_value = sum(asset["value"] for asset in all_assets)
        if total_value == 0:
            return 0.0
        
        combined_weighted_risk = 0
        for asset in all_assets:
            asset_id = asset["id"]
            region = asset["region"]
            
            if region in regional_risks and asset_id in regional_risks[region]:
                risk = regional_risks[region][asset_id]
                combined_weighted_risk += (asset["value"] / total_value) * risk
        
        # Calculate region-based weighted risk
        region_values = regional_portfolio.get_region_asset_values()
        region_weighted_risk = 0
        
        for region_name, region_value in region_values.items():
            if region_name not in regional_risks or not regional_risks[region_name]:
                continue
                
            region_risk_scores = list(regional_risks[region_name].values())
            avg_region_risk = np.mean(region_risk_scores)
            
            region_weighted_risk += (region_value / total_value) * avg_region_risk
        
        # Calculate diversification benefit
        if combined_weighted_risk == 0:
            return 0.0
            
        benefit = (combined_weighted_risk - region_weighted_risk) / combined_weighted_risk
        return benefit
    
    def perform_cross_region_analysis(self,
                                    regional_risks: Dict[str, Dict[str, float]],
                                    regional_portfolio: RegionalPortfolio) -> Dict[str, Any]:
        """
        Perform comprehensive cross-region risk analysis.
        
        Args:
            regional_risks: Results from assess_regional_risks
            regional_portfolio: RegionalPortfolio used for assessment
            
        Returns:
            Dictionary with analysis results
        """
        # Statistical comparison
        stats_by_region = self.compare_risk_distributions(
            regional_risks, regional_portfolio
        )
        
        # High-risk assets
        high_risk_assets = self.identify_high_risk_assets(
            regional_risks, threshold=0.7, top_n=10
        )
        
        # Risk correlations between regions
        risk_correlations = self._calculate_risk_correlations(
            regional_risks, regional_portfolio
        )
        
        # Diversification benefit
        diversification = self.calculate_diversification_benefit(
            regional_risks, regional_portfolio
        )
        
        # Asset allocation recommendation
        allocation = self._recommend_allocation(
            regional_risks, regional_portfolio, stats_by_region
        )
        
        # Combine results
        return {
            "statistics": stats_by_region,
            "high_risk_assets": high_risk_assets,
            "risk_correlations": risk_correlations,
            "diversification_benefit": diversification,
            "recommended_allocation": allocation
        }
    
    def _calculate_risk_correlations(self,
                                   regional_risks: Dict[str, Dict[str, float]],
                                   regional_portfolio: RegionalPortfolio) -> Dict[str, Dict[str, float]]:
        """
        Calculate risk correlations between regions.
        
        Args:
            regional_risks: Results from assess_regional_risks
            regional_portfolio: RegionalPortfolio used for assessment
            
        Returns:
            Dictionary mapping region pairs to correlation coefficients
        """
        correlations = {}
        regions = list(regional_risks.keys())
        
        # Skip if we have less than 2 regions
        if len(regions) < 2:
            return correlations
        
        # Try to match assets across regions by properties
        common_assets = self._identify_related_assets(regional_portfolio)
        
        # Calculate correlations for each pair of regions
        for i, region1 in enumerate(regions):
            correlations[region1] = {}
            
            for region2 in regions:
                if region1 == region2:
                    # Self correlation is always 1
                    correlations[region1][region2] = 1.0
                    continue
                
                # Find matched assets between these regions
                if (region1, region2) in common_assets:
                    matched = common_assets[(region1, region2)]
                elif (region2, region1) in common_assets:
                    matched = common_assets[(region2, region1)]
                    # Flip the matches
                    matched = [(b, a) for a, b in matched]
                else:
                    # No matches between these regions
                    correlations[region1][region2] = np.nan
                    continue
                
                # Get risk scores for matched assets
                risks1 = []
                risks2 = []
                
                for asset1_id, asset2_id in matched:
                    if asset1_id in regional_risks[region1] and asset2_id in regional_risks[region2]:
                        risks1.append(regional_risks[region1][asset1_id])
                        risks2.append(regional_risks[region2][asset2_id])
                
                # Calculate correlation if we have enough data
                if len(risks1) >= 3:
                    corr, _ = stats.pearsonr(risks1, risks2)
                    correlations[region1][region2] = corr
                else:
                    correlations[region1][region2] = np.nan
        
        return correlations
    
    def _identify_related_assets(self, regional_portfolio: RegionalPortfolio) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        """
        Identify related assets across regions based on properties.
        
        Args:
            regional_portfolio: RegionalPortfolio to analyze
            
        Returns:
            Dictionary mapping region pairs to lists of matched asset pairs
        """
        common_assets = {}
        regions = regional_portfolio.get_regions()
        
        # Skip if we have less than 2 regions
        if len(regions) < 2:
            return common_assets
        
        # For each pair of regions
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if j <= i:
                    # Skip self-comparisons and duplicates
                    continue
                
                portfolio1 = regional_portfolio.portfolios[region1]
                portfolio2 = regional_portfolio.portfolios[region2]
                
                # Match assets by sector if available
                assets1_by_sector = {}
                assets2_by_sector = {}
                
                for asset in portfolio1.assets:
                    sector = asset.get("metadata", {}).get("sector", "Unknown")
                    if sector not in assets1_by_sector:
                        assets1_by_sector[sector] = []
                    assets1_by_sector[sector].append(asset)
                
                for asset in portfolio2.assets:
                    sector = asset.get("metadata", {}).get("sector", "Unknown")
                    if sector not in assets2_by_sector:
                        assets2_by_sector[sector] = []
                    assets2_by_sector[sector].append(asset)
                
                # Find matches within each sector
                matches = []
                for sector in set(assets1_by_sector.keys()).intersection(assets2_by_sector.keys()):
                    sector_assets1 = assets1_by_sector[sector]
                    sector_assets2 = assets2_by_sector[sector]
                    
                    # Match by value (closest)
                    for asset1 in sector_assets1:
                        if not sector_assets2:
                            continue
                            
                        # Find asset2 with closest value
                        asset2 = min(sector_assets2, key=lambda a: abs(a["value"] - asset1["value"]))
                        
                        # Store match
                        matches.append((asset1["id"], asset2["id"]))
                        
                        # Remove matched asset2 to prevent duplicate matches
                        sector_assets2.remove(asset2)
                
                # Store matches for this region pair
                if matches:
                    common_assets[(region1, region2)] = matches
        
        return common_assets
    
    def _recommend_allocation(self,
                            regional_risks: Dict[str, Dict[str, float]],
                            regional_portfolio: RegionalPortfolio,
                            stats_by_region: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Recommend portfolio allocation across regions to minimize risk.
        
        Args:
            regional_risks: Results from assess_regional_risks
            regional_portfolio: RegionalPortfolio used for assessment
            stats_by_region: Statistical comparison results
            
        Returns:
            Dictionary mapping regions to recommended allocation percentages
        """
        # Get region names
        regions = list(stats_by_region.keys())
        if not regions:
            return {}
        
        # Get weighted risks
        weighted_risks = {
            region: stats["value_weighted_risk"]
            for region, stats in stats_by_region.items()
        }
        
        # Simple inverse-risk weighting
        total_inverse_risk = sum(1.0 / risk if risk > 0 else 0 for risk in weighted_risks.values())
        
        if total_inverse_risk == 0:
            # Equal weighting if no risk data
            return {region: 1.0 / len(regions) for region in regions}
        
        allocation = {
            region: (1.0 / risk) / total_inverse_risk if risk > 0 else 0
            for region, risk in weighted_risks.items()
        }
        
        # Ensure allocations sum to 1
        total_alloc = sum(allocation.values())
        if total_alloc > 0:
            allocation = {
                region: alloc / total_alloc
                for region, alloc in allocation.items()
            }
        
        return allocation


class MultiRegionVisualizer(GeoFinancialVisualizer):
    """
    Extends the base visualizer with multi-region visualization capabilities.
    
    This class provides methods for creating visualizations that compare
    risk patterns across multiple geographic regions.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize the multi-region visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        super().__init__(figsize=figsize)
    
    def plot_multi_region_risk_map(self,
                                 regional_portfolio: RegionalPortfolio,
                                 regional_risks: Dict[str, Dict[str, float]],
                                 title: str = "Multi-Region Risk Analysis",
                                 region_colors: Optional[Dict[str, str]] = None) -> plt.Figure:
        """
        Plot a risk map showing multiple regions with their assets.
        
        Args:
            regional_portfolio: RegionalPortfolio to visualize
            regional_risks: Results from assess_regional_risks
            title: Title for the plot
            region_colors: Optional mapping of region names to colors
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Generate colors if not provided
        if region_colors is None:
            regions = regional_portfolio.get_regions()
            cmap = plt.cm.get_cmap('tab10', max(len(regions), 10))
            region_colors = {
                region: f"#{int(cmap(i)[0]*255):02x}{int(cmap(i)[1]*255):02x}{int(cmap(i)[2]*255):02x}"
                for i, region in enumerate(regions)
            }
        
        # Plot regions
        for region_name, region in regional_portfolio.regions.items():
            # Draw region boundary
            rect = plt.Rectangle(
                (region.bounds["min_x"], region.bounds["min_y"]),
                region.bounds["max_x"] - region.bounds["min_x"],
                region.bounds["max_y"] - region.bounds["min_y"],
                linewidth=2,
                edgecolor=region_colors.get(region_name, 'black'),
                facecolor='none',
                alpha=0.7,
                label=region_name
            )
            ax.add_patch(rect)
            
            # Add label
            center_x = (region.bounds["min_x"] + region.bounds["max_x"]) / 2
            center_y = (region.bounds["min_y"] + region.bounds["max_y"]) / 2
            ax.text(
                center_x, center_y, region_name,
                horizontalalignment='center',
                verticalalignment='center',
                color=region_colors.get(region_name, 'black'),
                fontweight='bold'
            )
        
        # Plot assets
        all_assets = regional_portfolio.get_all_assets()
        if all_assets:
            # Extract coordinates and sizes
            x_coords = []
            y_coords = []
            sizes = []
            colors = []
            region_markers = []
            texts = []
            
            for asset in all_assets:
                asset_id = asset["id"]
                region = asset["region"]
                
                # Skip if no risk score
                if (region not in regional_risks or
                    asset_id not in regional_risks[region]):
                    continue
                
                risk_score = regional_risks[region][asset_id]
                
                x_coords.append(asset["x"])
                y_coords.append(asset["y"])
                
                # Size based on value
                size = np.sqrt(asset["value"]) / 10 + 10
                sizes.append(size)
                
                # Color based on risk
                colors.append(risk_score)
                
                # Marker based on region
                region_markers.append(region)
                
                # Hover text
                sector = asset.get("metadata", {}).get("sector", "Unknown")
                texts.append(
                    f"ID: {asset_id}\n"
                    f"Name: {asset['name']}\n"
                    f"Value: ${asset['value']:,.2f}\n"
                    f"Sector: {sector}\n"
                    f"Region: {region}\n"
                    f"Risk Score: {risk_score:.3f}"
                )
            
            # Plot assets with risk-based colors
            scatter = ax.scatter(
                x_coords, y_coords,
                s=sizes,
                c=colors,
                cmap='RdYlGn_r',
                vmin=0, vmax=1,
                alpha=0.8
            )
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', 
                               pad=0.01, fraction=0.05)
            cbar.set_label('Risk Score')
        
        # Add title
        ax.set_title(title, fontsize=14)
        
        # Add legend for regions
        ax.legend(loc='upper right')
        
        # Set axis labels
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Ensure equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        return fig
    
    def plot_region_risk_comparison(self,
                                  stats_by_region: Dict[str, Dict[str, float]],
                                  title: str = "Regional Risk Comparison") -> plt.Figure:
        """
        Plot a comparison of risk statistics across regions.
        
        Args:
            stats_by_region: Statistical comparison results
            title: Title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        if not stats_by_region:
            # Create an empty figure if no data
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title("No regional risk data available")
            return fig
        
        # Create multi-panel figure
        fig, axs = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Get regions and sort by mean risk
        regions = sorted(
            stats_by_region.keys(),
            key=lambda r: stats_by_region[r]["mean"]
        )
        
        # Bar colors
        cmap = plt.cm.get_cmap('RdYlGn_r')
        colors = [cmap(stats_by_region[r]["mean"]) for r in regions]
        
        # Panel 1: Mean Risk by Region
        ax = axs[0, 0]
        mean_risks = [stats_by_region[r]["mean"] for r in regions]
        bars = ax.bar(regions, mean_risks, color=colors)
        ax.set_title("Mean Risk by Region")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Risk Score")
        
        # Add value annotations
        for bar, risk in zip(bars, mean_risks):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{risk:.3f}",
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        # Panel 2: Risk Dispersion (Min, Mean, Max)
        ax = axs[0, 1]
        for i, region in enumerate(regions):
            stats = stats_by_region[region]
            ax.plot([i, i], [stats["min"], stats["max"]], 'k-', alpha=0.7)
            ax.plot(i, stats["mean"], 'ro', markersize=8)
            ax.plot(i, stats["median"], 'bs', markersize=6)
        
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions)
        ax.set_title("Risk Dispersion by Region")
        ax.set_ylabel("Risk Score")
        ax.set_ylim(0, 1)
        ax.legend(['Range', 'Mean', 'Median'], loc='upper right')
        
        # Panel 3: Value-Weighted Risk
        ax = axs[1, 0]
        weighted_risks = [stats_by_region[r]["value_weighted_risk"] for r in regions]
        total_values = [stats_by_region[r]["total_value"] for r in regions]
        normalized_values = [v / max(total_values) for v in total_values]
        
        # Bar plot for weighted risk
        bars = ax.bar(
            regions, weighted_risks,
            color=colors
        )
        
        # Line plot for relative value
        ax2 = ax.twinx()
        ax2.plot(
            regions, normalized_values,
            'bo-', linewidth=2, markersize=8
        )
        
        ax.set_title("Value-Weighted Risk by Region")
        ax.set_ylabel("Value-Weighted Risk")
        ax2.set_ylabel("Relative Value")
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, 1.1)
        
        # Add value annotations
        for bar, risk in zip(bars, weighted_risks):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{risk:.3f}",
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        # Panel 4: Asset Count and Distribution
        ax = axs[1, 1]
        counts = [stats_by_region[r]["count"] for r in regions]
        stds = [stats_by_region[r]["std"] for r in regions]
        
        # Create bar chart for counts
        bars = ax.bar(
            regions, counts,
            alpha=0.7,
            color='lightblue'
        )
        
        # Add std dev as line
        ax2 = ax.twinx()
        ax2.plot(
            regions, stds,
            'ro-', linewidth=2, markersize=8
        )
        
        ax.set_title("Asset Count and Risk Std Dev")
        ax.set_ylabel("Asset Count")
        ax2.set_ylabel("Risk Std Dev")
        ax2.set_ylim(0, max(stds) * 1.2)
        
        # Add count annotations
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                str(count),
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        # Adjust layout
        fig.tight_layout()
        
        return fig
    
    def plot_risk_correlation_matrix(self,
                                   risk_correlations: Dict[str, Dict[str, float]],
                                   title: str = "Risk Correlation Between Regions") -> plt.Figure:
        """
        Plot a correlation matrix of risks between regions.
        
        Args:
            risk_correlations: Correlation results
            title: Title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        if not risk_correlations:
            # Create an empty figure if no data
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title("No correlation data available")
            return fig
        
        # Get regions
        regions = list(risk_correlations.keys())
        
        # Create correlation matrix
        corr_matrix = np.zeros((len(regions), len(regions)))
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                corr_matrix[i, j] = risk_correlations[region1].get(region2, np.nan)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(
            corr_matrix,
            cmap='RdBu_r',
            vmin=-1,
            vmax=1
        )
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', 
                           pad=0.01, fraction=0.05)
        cbar.set_label('Correlation Coefficient')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(regions)))
        ax.set_yticks(np.arange(len(regions)))
        ax.set_xticklabels(regions)
        ax.set_yticklabels(regions)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(regions)):
            for j in range(len(regions)):
                value = corr_matrix[i, j]
                if np.isnan(value):
                    text = "N/A"
                else:
                    text = f"{value:.2f}"
                
                color = "white" if abs(value) > 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color)
        
        # Add title
        ax.set_title(title)
        
        # Ensure layout fits
        fig.tight_layout()
        
        return fig
    
    def plot_recommended_allocation(self,
                                  allocation: Dict[str, float],
                                  stats_by_region: Dict[str, Dict[str, float]],
                                  title: str = "Recommended Regional Allocation") -> plt.Figure:
        """
        Plot recommended portfolio allocation across regions.
        
        Args:
            allocation: Recommended allocation percentages
            stats_by_region: Statistical comparison results
            title: Title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        if not allocation:
            # Create an empty figure if no data
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title("No allocation data available")
            return fig
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16)
        
        # Get regions and sort by allocation
        regions = sorted(
            allocation.keys(),
            key=lambda r: allocation[r],
            reverse=True
        )
        
        # Get allocations
        allocations = [allocation[r] * 100 for r in regions]  # Convert to percent
        
        # Bar colors - use risk levels for coloring if available
        if stats_by_region:
            risks = [stats_by_region.get(r, {}).get("value_weighted_risk", 0.5) for r in regions]
            cmap = plt.cm.get_cmap('RdYlGn_r')
            colors = [cmap(risk) for risk in risks]
        else:
            colors = plt.cm.tab10(np.arange(len(regions)) % 10)
        
        # Plot allocation as bar chart
        bars = ax1.bar(
            regions, allocations,
            color=colors
        )
        
        # Add value annotations
        for bar, alloc in zip(bars, allocations):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f"{alloc:.1f}%",
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        ax1.set_title("Allocation by Region")
        ax1.set_ylabel("Allocation (%)")
        ax1.set_ylim(0, max(allocations) * 1.2)
        
        # Plot allocation as pie chart
        wedges, texts, autotexts = ax2.pie(
            allocations,
            labels=regions,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        
        # Set font size for pie labels
        plt.setp(autotexts, size=9, weight="bold")
        plt.setp(texts, size=9)
        
        ax2.set_title("Allocation Distribution")
        
        # Ensure layout fits
        fig.tight_layout()
        
        return fig
    
    def create_multi_region_dashboard(self,
                                    regional_portfolio: RegionalPortfolio,
                                    regional_risks: Dict[str, Dict[str, float]],
                                    analysis_results: Dict[str, Any],
                                    output_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            regional_portfolio: RegionalPortfolio to visualize
            regional_risks: Results from assess_regional_risks
            analysis_results: Results from perform_cross_region_analysis
            output_path: Optional path to save the dashboard image
            
        Returns:
            Matplotlib Figure object
        """
        # Create large figure with grid
        fig = plt.figure(figsize=(20, 16))
        
        # Define grid
        grid = plt.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)
        
        # Add title
        fig.suptitle("Multi-Region Geospatial Financial Risk Analysis", 
                    fontsize=20, y=0.98)
        
        # Plot 1: Multi-region risk map
        ax1 = fig.add_subplot(grid[0, 0])
        self._plot_risk_map_in_ax(
            ax1, regional_portfolio, regional_risks
        )
        
        # Plot 2: Region risk comparison
        ax2 = fig.add_subplot(grid[0, 1])
        self._plot_risk_comparison_in_ax(
            ax2, analysis_results["statistics"]
        )
        
        # Plot 3: Risk correlation matrix
        ax3 = fig.add_subplot(grid[1, 0])
        self._plot_correlation_in_ax(
            ax3, analysis_results["risk_correlations"]
        )
        
        # Plot 4: Recommended allocation
        ax4 = fig.add_subplot(grid[1, 1])
        self._plot_allocation_in_ax(
            ax4, 
            analysis_results["recommended_allocation"],
            analysis_results["statistics"]
        )
        
        # Add metadata
        fig.text(0.5, 0.01, 
                f"Diversification Benefit: {analysis_results['diversification_benefit']*100:.2f}%", 
                ha='center', fontsize=12)
        
        # Save if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _plot_risk_map_in_ax(self, ax, regional_portfolio, regional_risks):
        """Helper to plot risk map in a given axis."""
        # Clear axis
        ax.clear()
        
        # Generate colors for regions
        regions = regional_portfolio.get_regions()
        cmap = plt.cm.get_cmap('tab10', max(len(regions), 10))
        region_colors = {
            region: f"#{int(cmap(i)[0]*255):02x}{int(cmap(i)[1]*255):02x}{int(cmap(i)[2]*255):02x}"
            for i, region in enumerate(regions)
        }
        
        # Plot regions
        for region_name, region in regional_portfolio.regions.items():
            # Draw region boundary
            rect = plt.Rectangle(
                (region.bounds["min_x"], region.bounds["min_y"]),
                region.bounds["max_x"] - region.bounds["min_x"],
                region.bounds["max_y"] - region.bounds["min_y"],
                linewidth=2,
                edgecolor=region_colors.get(region_name, 'black'),
                facecolor='none',
                alpha=0.7,
                label=region_name
            )
            ax.add_patch(rect)
            
            # Add label
            center_x = (region.bounds["min_x"] + region.bounds["max_x"]) / 2
            center_y = (region.bounds["min_y"] + region.bounds["max_y"]) / 2
            ax.text(
                center_x, center_y, region_name,
                horizontalalignment='center',
                verticalalignment='center',
                color=region_colors.get(region_name, 'black'),
                fontweight='bold'
            )
        
        # Plot assets
        all_assets = regional_portfolio.get_all_assets()
        if all_assets:
            # Extract coordinates and sizes
            x_coords = []
            y_coords = []
            sizes = []
            colors = []
            region_markers = []
            
            for asset in all_assets:
                asset_id = asset["id"]
                region = asset["region"]
                
                # Skip if no risk score
                if (region not in regional_risks or
                    asset_id not in regional_risks[region]):
                    continue
                
                risk_score = regional_risks[region][asset_id]
                
                x_coords.append(asset["x"])
                y_coords.append(asset["y"])
                
                # Size based on value
                size = np.sqrt(asset["value"]) / 10 + 10
                sizes.append(size)
                
                # Color based on risk
                colors.append(risk_score)
                
                # Marker based on region
                region_markers.append(region)
            
            # Plot assets with risk-based colors
            scatter = ax.scatter(
                x_coords, y_coords,
                s=sizes,
                c=colors,
                cmap='RdYlGn_r',
                vmin=0, vmax=1,
                alpha=0.8
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', 
                               pad=0.01, fraction=0.05)
            cbar.set_label('Risk Score')
        
        # Add title
        ax.set_title("Multi-Region Risk Map", fontsize=14)
        
        # Add legend for regions
        ax.legend(loc='upper right')
        
        # Set axis labels
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Ensure equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
    def _plot_risk_comparison_in_ax(self, ax, stats_by_region):
        """Helper to plot risk comparison in a given axis."""
        # Clear axis
        ax.clear()
        
        if not stats_by_region:
            ax.set_title("No regional risk data available")
            return
        
        # Get regions and sort by mean risk
        regions = sorted(
            stats_by_region.keys(),
            key=lambda r: stats_by_region[r]["mean"]
        )
        
        # Bar colors
        cmap = plt.cm.get_cmap('RdYlGn_r')
        colors = [cmap(stats_by_region[r]["mean"]) for r in regions]
        
        # Mean Risk by Region
        mean_risks = [stats_by_region[r]["mean"] for r in regions]
        weighted_risks = [stats_by_region[r]["value_weighted_risk"] for r in regions]
        
        # Create grouped bar chart
        x = np.arange(len(regions))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mean_risks, width, color=colors, alpha=0.7, label='Mean Risk')
        bars2 = ax.bar(x + width/2, weighted_risks, width, color=colors, alpha=1.0, label='Weighted Risk')
        
        # Add value annotations
        for bar, risk in zip(bars1, mean_risks):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{risk:.2f}",
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        for bar, risk in zip(bars2, weighted_risks):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{risk:.2f}",
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        # Set titles and labels
        ax.set_title("Risk Comparison by Region", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(regions)
        ax.set_ylabel("Risk Score")
        ax.set_ylim(0, 1.0)
        ax.legend()
        
        # Add asset counts
        for i, region in enumerate(regions):
            count = stats_by_region[region]["count"]
            ax.text(
                i, -0.05,
                f"{count} assets",
                ha='center',
                va='top',
                fontsize=8,
                rotation=0
            )
    
    def _plot_correlation_in_ax(self, ax, risk_correlations):
        """Helper to plot correlation matrix in a given axis."""
        # Clear axis
        ax.clear()
        
        if not risk_correlations:
            ax.set_title("No correlation data available")
            return
        
        # Get regions
        regions = list(risk_correlations.keys())
        
        # Create correlation matrix
        corr_matrix = np.zeros((len(regions), len(regions)))
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                corr_matrix[i, j] = risk_correlations[region1].get(region2, np.nan)
        
        # Create heatmap
        im = ax.imshow(
            corr_matrix,
            cmap='RdBu_r',
            vmin=-1,
            vmax=1
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', 
                           pad=0.01, fraction=0.05)
        cbar.set_label('Correlation Coefficient')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(regions)))
        ax.set_yticks(np.arange(len(regions)))
        ax.set_xticklabels(regions)
        ax.set_yticklabels(regions)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(regions)):
            for j in range(len(regions)):
                value = corr_matrix[i, j]
                if np.isnan(value):
                    text = "N/A"
                else:
                    text = f"{value:.2f}"
                
                color = "white" if abs(value) > 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color)
        
        # Add title
        ax.set_title("Risk Correlation Between Regions", fontsize=14)
    
    def _plot_allocation_in_ax(self, ax, allocation, stats_by_region):
        """Helper to plot recommended allocation in a given axis."""
        # Clear axis
        ax.clear()
        
        if not allocation:
            ax.set_title("No allocation data available")
            return
        
        # Get regions and sort by allocation
        regions = sorted(
            allocation.keys(),
            key=lambda r: allocation[r],
            reverse=True
        )
        
        # Get allocations
        allocations = [allocation[r] * 100 for r in regions]  # Convert to percent
        
        # Bar colors - use risk levels for coloring if available
        if stats_by_region:
            risks = [stats_by_region.get(r, {}).get("value_weighted_risk", 0.5) for r in regions]
            cmap = plt.cm.get_cmap('RdYlGn_r')
            colors = [cmap(risk) for risk in risks]
        else:
            colors = plt.cm.tab10(np.arange(len(regions)) % 10)
        
        # Plot as pie chart
        wedges, texts, autotexts = ax.pie(
            allocations,
            labels=regions,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'alpha': 0.8}
        )
        
        # Set font size for pie labels
        plt.setp(autotexts, size=9, weight="bold")
        plt.setp(texts, size=9)
        
        # Add title
        ax.set_title("Recommended Regional Allocation", fontsize=14)
        
        # Add a circle at the center for a donut chart effect
        centre_circle = plt.Circle((0, 0), 0.4, fc='white')
        ax.add_artist(centre_circle)
        
        # Add value-weighted risk as text in the center
        if stats_by_region:
            risk_text = "Risks:\n"
            for region in regions:
                if region in stats_by_region:
                    risk = stats_by_region[region]["value_weighted_risk"]
                    risk_text += f"{region}: {risk:.2f}\n"
            
            ax.text(
                0, 0,
                risk_text,
                ha='center',
                va='center',
                fontsize=8
            )


# Utility functions for creating multi-region analysis

def create_region_grid(
    base_bounds: Dict[str, float],
    grid_size: Tuple[int, int] = (2, 2),
    region_names: Optional[List[str]] = None
) -> List[RegionDefinition]:
    """
    Create a grid of regions based on the provided bounds.
    
    Args:
        base_bounds: Base geographic bounds as {min_x, min_y, max_x, max_y}
        grid_size: Tuple of (rows, columns)
        region_names: Optional list of region names (must match grid_size)
        
    Returns:
        List of RegionDefinition objects
    """
    rows, cols = grid_size
    
    # Generate default region names if not provided
    if region_names is None:
        region_names = [f"Region_{r}_{c}" for r in range(rows) for c in range(cols)]
    
    # Ensure we have the right number of names
    if len(region_names) != rows * cols:
        raise ValueError(f"Expected {rows * cols} region names, got {len(region_names)}")
    
    # Calculate region bounds
    width = base_bounds["max_x"] - base_bounds["min_x"]
    height = base_bounds["max_y"] - base_bounds["min_y"]
    
    region_width = width / cols
    region_height = height / rows
    
    # Create regions
    regions = []
    for r in range(rows):
        for c in range(cols):
            region_bounds = {
                "min_x": base_bounds["min_x"] + c * region_width,
                "min_y": base_bounds["min_y"] + r * region_height,
                "max_x": base_bounds["min_x"] + (c + 1) * region_width,
                "max_y": base_bounds["min_y"] + (r + 1) * region_height
            }
            
            region_name = region_names[r * cols + c]
            region = RegionDefinition(name=region_name, bounds=region_bounds)
            regions.append(region)
    
    return regions


def create_regional_portfolio_from_grid(
    regions: List[RegionDefinition],
    assets_df: pd.DataFrame,
    id_col: str,
    name_col: str,
    value_col: str,
    x_col: str,
    y_col: str,
    metadata_cols: Optional[List[str]] = None,
    device_id: int = 0
) -> RegionalPortfolio:
    """
    Create a regional portfolio from a grid of regions and asset data.
    
    Args:
        regions: List of RegionDefinition objects
        assets_df: DataFrame with asset data
        id_col: Column name for asset IDs
        name_col: Column name for asset names
        value_col: Column name for asset values
        x_col: Column name for X coordinates
        y_col: Column name for Y coordinates
        metadata_cols: Optional list of columns to include as metadata
        device_id: GPU device ID
        
    Returns:
        RegionalPortfolio instance
    """
    # Create regional portfolio
    portfolio = RegionalPortfolio(device_id=device_id)
    
    # Add all regions
    for region in regions:
        portfolio.add_region(region)
    
    # Add assets
    portfolio.add_assets_from_dataframe(
        assets_df=assets_df,
        id_col=id_col,
        name_col=name_col,
        value_col=value_col,
        x_col=x_col,
        y_col=y_col,
        metadata_cols=metadata_cols
    )
    
    return portfolio


def perform_multi_region_analysis(
    regional_portfolio: RegionalPortfolio,
    risk_model: MultiRegionRiskModel,
    output_dir: Optional[str] = None,
    create_visualizations: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Perform a comprehensive multi-region risk analysis.
    
    Args:
        regional_portfolio: RegionalPortfolio to analyze
        risk_model: MultiRegionRiskModel to use
        output_dir: Optional directory to save visualizations
        create_visualizations: Whether to create visualizations
        logger: Optional logger instance
        
    Returns:
        Dictionary with analysis results
    """
    # Create logger if not provided
    if logger is None:
        logger = logging.getLogger("MultiRegionAnalysis")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    
    logger.info("Starting multi-region analysis")
    
    # Assess risks across all regions
    logger.info("Assessing regional risks")
    regional_risks = risk_model.assess_regional_risks(regional_portfolio)
    
    # Create regional risk comparator
    comparator = RegionalRiskComparator(logger=logger)
    
    # Perform cross-region analysis
    logger.info("Performing cross-region analysis")
    analysis_results = comparator.perform_cross_region_analysis(
        regional_risks=regional_risks,
        regional_portfolio=regional_portfolio
    )
    
    # Create visualizations if requested
    if create_visualizations and output_dir:
        logger.info("Creating visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizer
        visualizer = MultiRegionVisualizer(figsize=(12, 10))
        
        # Create visualizations
        try:
            # Risk map
            fig = visualizer.plot_multi_region_risk_map(
                regional_portfolio=regional_portfolio,
                regional_risks=regional_risks
            )
            fig.savefig(os.path.join(output_dir, "multi_region_risk_map.png"),
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Risk comparison
            fig = visualizer.plot_region_risk_comparison(
                stats_by_region=analysis_results["statistics"]
            )
            fig.savefig(os.path.join(output_dir, "region_risk_comparison.png"),
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Risk correlation matrix
            fig = visualizer.plot_risk_correlation_matrix(
                risk_correlations=analysis_results["risk_correlations"]
            )
            fig.savefig(os.path.join(output_dir, "risk_correlation_matrix.png"),
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Recommended allocation
            fig = visualizer.plot_recommended_allocation(
                allocation=analysis_results["recommended_allocation"],
                stats_by_region=analysis_results["statistics"]
            )
            fig.savefig(os.path.join(output_dir, "recommended_allocation.png"),
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Comprehensive dashboard
            fig = visualizer.create_multi_region_dashboard(
                regional_portfolio=regional_portfolio,
                regional_risks=regional_risks,
                analysis_results=analysis_results,
                output_path=os.path.join(output_dir, "multi_region_dashboard.png")
            )
            plt.close(fig)
            
            logger.info(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    # Combine all results
    results = {
        "regional_risks": regional_risks,
        "analysis": analysis_results
    }
    
    logger.info("Multi-region analysis complete")
    
    return results