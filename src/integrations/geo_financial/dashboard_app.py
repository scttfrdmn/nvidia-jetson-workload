#!/usr/bin/env python3
"""
Geospatial Financial Dashboard Application

This application provides an interactive dashboard for geospatial financial risk analysis,
including climate risk assessment, portfolio optimization, and scenario analysis.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(project_root)

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback
    import dash_bootstrap_components as dbc
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_DASH = True
except ImportError:
    HAS_DASH = False
    print("Warning: Dash not available. To install required packages:")
    print("pip install dash dash-bootstrap-components plotly")

# Import modules
from src.integrations.geo_financial.geo_risk import (
    GeospatialRiskModel,
    GeospatialPortfolio,
    SpatialRiskFactor,
    create_elevation_risk_factor,
    create_slope_risk_factor
)
from src.integrations.geo_financial.data_connectors import (
    AssetLocationDataLoader,
    FinancialDataLoader,
    GeoRiskDataLoader
)
from src.integrations.geo_financial.visualization import (
    GeoFinancialVisualizer
)
from src.integrations.geo_financial.risk_aggregation import (
    RiskAggregator,
    RiskSurfaceGenerator,
    AggregationMethod,
    create_combined_risk_factor,
    interpolate_asset_risk_surface
)
from src.integrations.geo_financial.climate_risk_assessment import (
    ClimateRiskAssessor,
    ClimateScenario,
    TimeHorizon,
    ClimateHazardType,
    TransitionRiskType,
    create_flood_risk_factor,
    create_heatwave_risk_factor,
    create_sea_level_rise_risk_factor,
    create_default_transition_risks
)
from src.integrations.geo_financial.generate_test_data import (
    generate_dem,
    generate_assets,
    generate_returns
)


class DashboardApp:
    """
    Interactive dashboard application for geospatial financial risk analysis.
    
    This class provides the core functionality for the dashboard, including
    data loading, risk analysis, and visualization components.
    """
    
    def __init__(self, 
                data_dir: str = 'data/geo_financial',
                output_dir: str = 'results/geo_financial/dashboard',
                device_id: int = 0):
        """
        Initialize the dashboard application.
        
        Args:
            data_dir: Directory for input data files
            output_dir: Directory for output files
            device_id: CUDA device ID (negative for CPU)
        """
        self.data_dir = os.path.join(project_root, data_dir)
        self.output_dir = os.path.join(project_root, output_dir)
        self.device_id = device_id
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data
        self.dem_data = None
        self.assets_df = None
        self.returns_df = None
        self.returns_dict = {}
        self.geo_transform = None
        self.portfolio = None
        self.risk_factors = []
        self.climate_risk_factors = []
        self.risk_model = None
        self.climate_assessor = None
        self.combined_risk = None
        self.risk_scores = {}
        self.climate_risk_scores = {}
        
        # State variables
        self.active_scenario = ClimateScenario.SSP2_RCP45
        self.active_time_horizon = TimeHorizon.MEDIUM
        self.active_aggregation_method = AggregationMethod.WEIGHTED_AVERAGE
        
        # Initialize components
        self.visualizer = GeoFinancialVisualizer(figsize=(12, 8))
        self.risk_aggregator = RiskAggregator(device_id=device_id)
    
    def generate_data(self, 
                    dem_size: int = 500,
                    num_assets: int = 30,
                    days: int = 252,
                    seed: int = 42) -> None:
        """
        Generate synthetic data for the dashboard.
        
        Args:
            dem_size: Size of the synthetic DEM (pixels)
            num_assets: Number of assets to generate
            days: Number of days of returns data
            seed: Random seed for reproducibility
        """
        print("Generating synthetic data...")
        
        # Set random seed
        np.random.seed(seed)
        
        # Generate DEM
        self.dem_data, x_coords, y_coords = generate_dem(
            output_dir=self.data_dir,
            size=dem_size,
            save_plot=True,
            seed=seed
        )
        
        # Generate assets
        self.assets_df = generate_assets(
            output_dir=self.data_dir,
            dem_data=self.dem_data,
            x_coords=x_coords,
            y_coords=y_coords,
            num_assets=num_assets,
            save_plot=True,
            seed=seed + 1
        )
        
        # Generate returns
        self.returns_df = generate_returns(
            output_dir=self.data_dir,
            assets_df=self.assets_df,
            days=days,
            save_plot=True,
            seed=seed + 2
        )
        
        # Convert returns to dictionary
        self.returns_dict = {}
        for asset_id in self.assets_df['id'].unique():
            asset_returns = self.returns_df[self.returns_df['asset_id'] == asset_id]['return'].values
            self.returns_dict[asset_id] = asset_returns
        
        # Create geo_transform
        try:
            from src.geospatial.python.geospatial import dem as geospatial_dem
            self.geo_transform = geospatial_dem.GeoTransform([0, 0.02, 0, 10, 0, -0.02])
        except ImportError:
            # Simple mock GeoTransform
            class MockGeoTransform:
                def geo_to_pixel(self, x, y):
                    return int(x * 50), int(y * 50)
                
                def pixel_to_geo(self, x, y):
                    return x / 50, y / 50
            
            self.geo_transform = MockGeoTransform()
    
    def load_data(self) -> None:
        """
        Load data from files if available, otherwise generate synthetic data.
        """
        # Check if required files exist
        dem_file = os.path.join(self.data_dir, 'synthetic_dem.npy')
        assets_file = os.path.join(self.data_dir, 'synthetic_assets.csv')
        returns_file = os.path.join(self.data_dir, 'synthetic_returns.csv')
        
        if all(os.path.exists(f) for f in [dem_file, assets_file, returns_file]):
            print("Loading existing data files...")
            
            # Load DEM
            self.dem_data = np.load(dem_file)
            
            # Load assets
            asset_loader = AssetLocationDataLoader()
            self.assets_df = asset_loader.load_asset_csv(
                file_path=assets_file,
                id_col='id',
                name_col='name',
                value_col='value',
                x_col='x',
                y_col='y'
            )
            
            # Load returns
            financial_loader = FinancialDataLoader()
            self.returns_dict = financial_loader.load_returns_csv(
                file_path=returns_file,
                asset_id_col='asset_id',
                date_col='date',
                return_col='return'
            )
            
            # Load returns dataframe
            self.returns_df = pd.read_csv(returns_file)
            self.returns_df['date'] = pd.to_datetime(self.returns_df['date'])
            
            # Create geo_transform
            try:
                from src.geospatial.python.geospatial import dem as geospatial_dem
                self.geo_transform = geospatial_dem.GeoTransform([0, 0.02, 0, 10, 0, -0.02])
            except ImportError:
                # Simple mock GeoTransform
                class MockGeoTransform:
                    def geo_to_pixel(self, x, y):
                        return int(x * 50), int(y * 50)
                    
                    def pixel_to_geo(self, x, y):
                        return x / 50, y / 50
                
                self.geo_transform = MockGeoTransform()
        else:
            # Generate synthetic data
            self.generate_data()
    
    def create_portfolio(self) -> None:
        """
        Create geospatial portfolio from loaded data.
        """
        # Check if data is loaded
        if self.assets_df is None or not self.returns_dict:
            print("No data loaded. Loading data first...")
            self.load_data()
        
        # Create portfolio
        self.portfolio = GeospatialPortfolio(device_id=self.device_id)
        
        # Add assets to portfolio
        for _, row in self.assets_df.iterrows():
            self.portfolio.add_asset(
                asset_id=row['id'],
                name=row['name'],
                value=float(row['value']),
                x=float(row['x']),
                y=float(row['y']),
                returns=self.returns_dict.get(row['id'], np.array([])),
                metadata={'sector': row['sector']} if 'sector' in row else {}
            )
    
    def create_risk_factors(self) -> None:
        """
        Create risk factors for geospatial risk analysis.
        """
        # Check if data is loaded
        if self.dem_data is None or self.geo_transform is None:
            print("No data loaded. Loading data first...")
            self.load_data()
        
        # Create elevation-based flood risk
        flood_risk = create_elevation_risk_factor(
            dem_path=self.dem_data,
            name="Flood Risk",
            risk_weight=0.6,
            high_risk_threshold=np.percentile(self.dem_data, 20)  # 20th percentile
        )
        
        # Calculate slope data
        from scipy.ndimage import sobel
        dx = sobel(self.dem_data, axis=1)
        dy = sobel(self.dem_data, axis=0)
        slope_data = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        
        # Create slope-based landslide risk
        landslide_risk = SpatialRiskFactor(
            name="Landslide Risk",
            description="Risk based on terrain slope (steeper slopes have higher risk)",
            risk_weight=0.4,
            spatial_data=slope_data,
            geo_transform=self.geo_transform,
            transform_func=lambda slope: np.clip(slope / 45.0, 0, 1)  # Normalize to 0-1 (45 degrees is max risk)
        )
        
        # Add risk factors
        self.risk_factors = [flood_risk, landslide_risk]
        
        # Create combined risk factor
        self.combined_risk = create_combined_risk_factor(
            risk_factors=self.risk_factors,
            name="Combined Natural Hazard Risk",
            description="Combined risk from multiple natural hazards",
            method=self.active_aggregation_method,
            device_id=self.device_id
        )
        
        # Create risk model
        self.risk_model = GeospatialRiskModel(device_id=self.device_id)
        self.risk_model.add_risk_factor(self.combined_risk)
    
    def create_climate_risk_factors(self) -> None:
        """
        Create climate-specific risk factors.
        """
        # Check if data is loaded
        if self.dem_data is None or self.geo_transform is None:
            print("No data loaded. Loading data first...")
            self.load_data()
        
        # Create climate risk assessor
        self.climate_assessor = ClimateRiskAssessor(
            device_id=self.device_id,
            scenario=self.active_scenario,
            time_horizon=self.active_time_horizon
        )
        
        # Create flood risk factor
        flood_risk = create_flood_risk_factor(
            dem_path=self.dem_data,
            risk_weight=0.6,
            low_elevation_threshold=np.percentile(self.dem_data, 20),  # 20th percentile
            assessor=self.climate_assessor
        )
        
        # Create heatwave risk factor
        # For simulation, create synthetic temperature data
        # Higher temperatures in lower elevations and certain regions
        temperature_data = 25.0 + 15.0 * (1.0 - np.clip(
            (self.dem_data - np.min(self.dem_data)) / 
            (np.max(self.dem_data) - np.min(self.dem_data)), 0, 1
        ))
        
        # Add some regional variation
        x_idx, y_idx = np.meshgrid(
            np.arange(self.dem_data.shape[1]), 
            np.arange(self.dem_data.shape[0])
        )
        regional_pattern = 5.0 * np.sin(x_idx / 50.0) * np.cos(y_idx / 50.0)
        temperature_data += regional_pattern
        
        heatwave_risk = create_heatwave_risk_factor(
            temperature_data=temperature_data,
            geo_transform=self.geo_transform,
            risk_weight=0.4,
            high_temp_threshold=35.0,
            assessor=self.climate_assessor
        )
        
        # Create sea level rise risk factor
        # For simulation, create synthetic coastline distance data
        # Lower part of the DEM will be the "coastline"
        coastline_y = int(self.dem_data.shape[0] * 0.8)
        coastline_distance = np.zeros_like(self.dem_data)
        
        for i in range(self.dem_data.shape[0]):
            for j in range(self.dem_data.shape[1]):
                # Distance to coastline in grid cells
                distance = abs(i - coastline_y)
                # Convert to kilometers (assuming 1 grid cell = 0.5 km)
                coastline_distance[i, j] = distance * 0.5
        
        sea_level_risk = create_sea_level_rise_risk_factor(
            dem_path=self.dem_data,
            coastline_distance=coastline_distance,
            geo_transform=self.geo_transform,
            risk_weight=0.5,
            elevation_threshold=np.percentile(self.dem_data, 30),  # 30th percentile
            distance_threshold=20.0,  # 20 km
            assessor=self.climate_assessor
        )
        
        # Add climate risk factors
        self.climate_risk_factors = [flood_risk, heatwave_risk, sea_level_risk]
    
    def assess_portfolio_risk(self) -> Dict[str, float]:
        """
        Assess portfolio risk using the current risk model.
        
        Returns:
            Dictionary of risk scores by asset ID
        """
        # Check if portfolio and risk model are created
        if self.portfolio is None:
            self.create_portfolio()
        
        if self.risk_model is None:
            self.create_risk_factors()
        
        # Assess risk
        self.risk_scores = self.portfolio.assess_risk(self.risk_model)
        
        return self.risk_scores
    
    def assess_climate_risk(self) -> Dict[str, Dict[str, float]]:
        """
        Assess climate risk for the portfolio.
        
        Returns:
            Dictionary of climate risk scores by asset ID
        """
        # Check if portfolio is created
        if self.portfolio is None:
            self.create_portfolio()
        
        # Create climate risk factors if needed
        if not self.climate_risk_factors:
            self.create_climate_risk_factors()
        
        # Assess physical climate risk
        physical_risk_scores = self.climate_assessor.assess_physical_climate_risk(
            portfolio=self.portfolio,
            risk_factors=self.climate_risk_factors,
            aggregation_method=self.active_aggregation_method
        )
        
        # Create transition risks
        transition_risks = create_default_transition_risks()
        
        # Set risk weights for different transition risk types
        sector_risk_weights = {
            TransitionRiskType.POLICY: 0.3,
            TransitionRiskType.LEGAL: 0.2,
            TransitionRiskType.TECHNOLOGY: 0.2,
            TransitionRiskType.MARKET: 0.2,
            TransitionRiskType.REPUTATION: 0.1
        }
        
        # Adjust transition risks for scenario
        adjusted_transition_risks = {}
        for risk_type, sector_risks in transition_risks.items():
            adjusted_sector_risks = self.climate_assessor.create_transition_risk_factor(
                risk_type=risk_type,
                sector_risks=sector_risks,
                risk_weight=sector_risk_weights[risk_type],
                apply_scenario=True
            )
            adjusted_transition_risks[risk_type] = adjusted_sector_risks
        
        # Assess transition climate risk
        transition_risk_scores = self.climate_assessor.assess_transition_climate_risk(
            portfolio=self.portfolio,
            sector_field='sector',
            transition_risks=adjusted_transition_risks,
            risk_weights=sector_risk_weights
        )
        
        # Combine physical and transition risks
        self.climate_risk_scores = self.climate_assessor.assess_combined_climate_risk(
            portfolio=self.portfolio,
            physical_risk_scores=physical_risk_scores,
            transition_risk_scores=transition_risk_scores,
            physical_weight=0.5
        )
        
        return self.climate_risk_scores
    
    def optimize_portfolio(self, 
                          target_return: Optional[float] = None,
                          max_risk_score: float = 0.5,
                          risk_aversion: float = 1.0) -> Dict[str, float]:
        """
        Optimize the portfolio to minimize geospatial risk.
        
        Args:
            target_return: Target portfolio return (if None, will be calculated)
            max_risk_score: Maximum allowable risk score per asset
            risk_aversion: Risk aversion parameter
            
        Returns:
            Dictionary of optimized weights by asset ID
        """
        # Check if portfolio is created
        if self.portfolio is None:
            self.create_portfolio()
        
        # Assess risk if not already done
        if not self.risk_scores:
            self.assess_portfolio_risk()
        
        # Calculate current portfolio return if target not specified
        if target_return is None:
            try:
                # Calculate current portfolio return
                current_returns = np.array([np.mean(asset['returns']) for asset in self.portfolio.assets])
                total_value = sum(asset['value'] for asset in self.portfolio.assets)
                current_weights = np.array([asset['value'] / total_value for asset in self.portfolio.assets])
                current_portfolio_return = np.sum(current_returns * current_weights)
                
                # Set target return slightly lower
                target_return = 0.9 * current_portfolio_return
            except (ValueError, ZeroDivisionError):
                # Default target return if calculation fails
                target_return = 0.0004  # 0.04% daily return
        
        # Optimize portfolio
        try:
            optimized_weights = self.portfolio.optimize_for_geo_risk(
                risk_model=self.risk_model,
                target_return=target_return,
                max_risk_score=max_risk_score,
                risk_aversion=risk_aversion
            )
            
            return optimized_weights
        except Exception as e:
            print(f"Portfolio optimization failed: {e}")
            
            # Create synthetic optimized weights (reduce weights of high-risk assets)
            total_value = sum(asset['value'] for asset in self.portfolio.assets)
            original_weights = {asset['id']: asset['value'] / total_value for asset in self.portfolio.assets}
            
            optimized_weights = {}
            for asset_id, orig_weight in original_weights.items():
                risk = self.risk_scores.get(asset_id, 0.5)
                # Reduce weight based on risk (higher risk = more reduction)
                optimized_weights[asset_id] = orig_weight * (1.0 - 0.5 * risk)
            
            # Normalize weights to sum to 1.0
            weight_sum = sum(optimized_weights.values())
            optimized_weights = {k: v / weight_sum for k, v in optimized_weights.items()}
            
            return optimized_weights
    
    def calculate_climate_var(self, 
                             confidence_level: float = 0.95,
                             risk_weight: float = 0.5) -> float:
        """
        Calculate climate-adjusted Value-at-Risk (VaR).
        
        Args:
            confidence_level: Confidence level for VaR calculation
            risk_weight: Weight for climate risk adjustment
            
        Returns:
            Climate-adjusted VaR
        """
        # Check if portfolio is created
        if self.portfolio is None:
            self.create_portfolio()
        
        # Assess climate risk if not already done
        if not self.climate_risk_scores:
            self.assess_climate_risk()
        
        # Calculate climate-adjusted VaR
        climate_var = self.climate_assessor.calculate_climate_var(
            portfolio=self.portfolio,
            climate_risk_scores=self.climate_risk_scores,
            confidence_level=confidence_level,
            risk_weight=risk_weight
        )
        
        return climate_var
    
    def compare_scenarios(self, 
                         scenarios: List[ClimateScenario],
                         time_horizons: List[TimeHorizon]) -> Dict[str, Dict[str, float]]:
        """
        Compare climate risks across different scenarios and time horizons.
        
        Args:
            scenarios: List of climate scenarios to compare
            time_horizons: List of time horizons to compare
            
        Returns:
            Nested dictionary of risk statistics by scenario and time horizon
        """
        # Check if portfolio is created
        if self.portfolio is None:
            self.create_portfolio()
        
        # Initialize results
        results = {}
        
        # Store current scenario and time horizon
        current_scenario = self.active_scenario
        current_time_horizon = self.active_time_horizon
        
        # Assess risk for each scenario and time horizon
        for scenario in scenarios:
            scenario_results = {}
            
            for time_horizon in time_horizons:
                # Update active scenario and time horizon
                self.active_scenario = scenario
                self.active_time_horizon = time_horizon
                
                # Create new climate risk factors for this scenario
                self.create_climate_risk_factors()
                
                # Assess climate risk
                climate_risk_scores = self.assess_climate_risk()
                
                # Calculate statistics
                risk_values = [scores['combined'] for scores in climate_risk_scores.values()]
                
                stats = {
                    'min': float(np.min(risk_values)),
                    'max': float(np.max(risk_values)),
                    'mean': float(np.mean(risk_values)),
                    'median': float(np.median(risk_values)),
                    'std': float(np.std(risk_values)),
                    'var': self.calculate_climate_var(confidence_level=0.95)
                }
                
                # Store results
                scenario_results[time_horizon.value] = stats
            
            results[scenario.value] = scenario_results
        
        # Restore current scenario and time horizon
        self.active_scenario = current_scenario
        self.active_time_horizon = current_time_horizon
        
        # Recreate climate risk factors with original settings
        self.create_climate_risk_factors()
        
        return results
    
    def save_results(self) -> None:
        """
        Save analysis results to files.
        """
        # Check if output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save risk scores if available
        if self.risk_scores:
            risk_scores_df = pd.DataFrame({
                'asset_id': list(self.risk_scores.keys()),
                'risk_score': list(self.risk_scores.values())
            })
            risk_scores_df.to_csv(
                os.path.join(self.output_dir, 'risk_scores.csv'),
                index=False
            )
        
        # Save climate risk scores if available
        if self.climate_risk_scores:
            # Convert nested dictionary to DataFrame
            climate_data = []
            
            for asset_id, scores in self.climate_risk_scores.items():
                row = {
                    'asset_id': asset_id,
                    'physical_risk': scores['physical'],
                    'transition_risk': scores['transition'],
                    'combined_risk': scores['combined']
                }
                climate_data.append(row)
            
            climate_risk_df = pd.DataFrame(climate_data)
            climate_risk_df.to_csv(
                os.path.join(self.output_dir, 'climate_risk_scores.csv'),
                index=False
            )
        
        # Save scenario comparison results if available
        scenarios = list(ClimateScenario)[:4]  # First 4 scenarios
        time_horizons = list(TimeHorizon)[:3]  # First 3 time horizons
        
        scenario_results = self.compare_scenarios(scenarios, time_horizons)
        
        with open(os.path.join(self.output_dir, 'scenario_comparison.json'), 'w') as f:
            # Convert enum keys to strings for JSON serialization
            serializable_results = {}
            
            for scenario, horizon_results in scenario_results.items():
                serializable_results[scenario] = {
                    horizon: stats for horizon, stats in horizon_results.items()
                }
            
            json.dump(serializable_results, f, indent=2)
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate data for the dashboard.
        
        Returns:
            Dictionary of dashboard data
        """
        # Check if portfolio is created
        if self.portfolio is None:
            self.create_portfolio()
        
        # Assess risks if not already done
        if not self.risk_scores:
            self.assess_portfolio_risk()
        
        if not self.climate_risk_scores:
            self.assess_climate_risk()
        
        # Optimize portfolio
        optimized_weights = self.optimize_portfolio()
        
        # Calculate climate VaR
        climate_var = self.calculate_climate_var()
        
        # Compare scenarios
        scenarios = list(ClimateScenario)[:4]  # First 4 scenarios
        time_horizons = list(TimeHorizon)[:3]  # First 3 time horizons
        
        scenario_results = self.compare_scenarios(scenarios, time_horizons)
        
        # Store asset properties
        asset_data = []
        for asset in self.portfolio.assets:
            asset_id = asset['id']
            
            if asset_id in self.risk_scores and asset_id in self.climate_risk_scores:
                data = {
                    'id': asset_id,
                    'name': asset['name'],
                    'value': asset['value'],
                    'x': asset['x'],
                    'y': asset['y'],
                    'sector': asset['metadata'].get('sector', 'Unknown'),
                    'risk_score': self.risk_scores[asset_id],
                    'physical_risk': self.climate_risk_scores[asset_id]['physical'],
                    'transition_risk': self.climate_risk_scores[asset_id]['transition'],
                    'climate_risk': self.climate_risk_scores[asset_id]['combined'],
                    'original_weight': asset['value'] / sum(a['value'] for a in self.portfolio.assets),
                    'optimized_weight': optimized_weights.get(asset_id, 0.0)
                }
                
                asset_data.append(data)
        
        # Convert to DataFrame for easier manipulation
        assets_df = pd.DataFrame(asset_data)
        
        # Group assets by sector
        sector_data = assets_df.groupby('sector').agg({
            'value': 'sum',
            'risk_score': 'mean',
            'physical_risk': 'mean',
            'transition_risk': 'mean',
            'climate_risk': 'mean'
        }).reset_index()
        
        # Calculate sector weights
        total_value = sector_data['value'].sum()
        sector_data['weight'] = sector_data['value'] / total_value
        
        # Create dashboard data dictionary
        dashboard_data = {
            'assets': assets_df.to_dict('records'),
            'sectors': sector_data.to_dict('records'),
            'risk_map': self.combined_risk.risk_data.tolist(),
            'climate_risk_maps': {
                rf.name: rf.risk_data.tolist() for rf in self.climate_risk_factors
            },
            'statistics': {
                'avg_risk': float(assets_df['risk_score'].mean()),
                'max_risk': float(assets_df['risk_score'].max()),
                'avg_climate_risk': float(assets_df['climate_risk'].mean()),
                'max_climate_risk': float(assets_df['climate_risk'].max()),
                'climate_var': climate_var
            },
            'scenario_comparison': scenario_results
        }
        
        return dashboard_data
    
    def generate_dashboard_plots(self) -> Dict[str, str]:
        """
        Generate plots for the dashboard.
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        # Check if portfolio is created
        if self.portfolio is None:
            self.create_portfolio()
        
        # Assess risks if not already done
        if not self.risk_scores:
            self.assess_portfolio_risk()
        
        if not self.climate_risk_scores:
            self.assess_climate_risk()
        
        # Create output directory for plots
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate plots
        plot_files = {}
        
        # Plot risk map
        risk_map_file = os.path.join(plots_dir, 'risk_map.png')
        fig = self.visualizer.plot_risk_map(
            risk_map=self.combined_risk.risk_data,
            title="Geospatial Risk Map"
        )
        fig.savefig(risk_map_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['risk_map'] = risk_map_file
        
        # Plot portfolio on risk map
        portfolio_risk_file = os.path.join(plots_dir, 'portfolio_risk_map.png')
        fig = self.visualizer.plot_portfolio_on_risk_map(
            risk_map=self.combined_risk.risk_data,
            geo_transform=self.geo_transform,
            portfolio=self.portfolio,
            title="Portfolio Assets on Risk Map"
        )
        fig.savefig(portfolio_risk_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['portfolio_risk_map'] = portfolio_risk_file
        
        # Plot risk heatmap
        risk_heatmap_file = os.path.join(plots_dir, 'risk_heatmap.png')
        fig = self.visualizer.plot_portfolio_risk_heatmap(
            portfolio=self.portfolio,
            risk_scores=self.risk_scores,
            title="Asset Risk Heatmap"
        )
        fig.savefig(risk_heatmap_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['risk_heatmap'] = risk_heatmap_file
        
        # Plot climate risk factors
        for i, rf in enumerate(self.climate_risk_factors):
            rf_file = os.path.join(plots_dir, f'climate_risk_{i}.png')
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(rf.risk_data, cmap='viridis_r', origin='upper')
            ax.set_title(rf.name)
            plt.colorbar(im, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.savefig(rf_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_files[f'climate_risk_{i}'] = rf_file
        
        # Plot portfolio optimization results
        # Calculate original weights
        total_value = sum(asset['value'] for asset in self.portfolio.assets)
        original_weights = {asset['id']: asset['value'] / total_value for asset in self.portfolio.assets}
        
        # Optimize portfolio
        optimized_weights = self.optimize_portfolio()
        
        # Plot optimization results
        optimization_file = os.path.join(plots_dir, 'portfolio_optimization.png')
        fig = self.visualizer.plot_portfolio_optimization_results(
            portfolio=self.portfolio,
            original_weights=original_weights,
            optimized_weights=optimized_weights,
            risk_scores=self.risk_scores,
            title="Portfolio Optimization Results"
        )
        fig.savefig(optimization_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['portfolio_optimization'] = optimization_file
        
        # Plot climate risk comparison
        climate_comparison_file = os.path.join(plots_dir, 'climate_comparison.png')
        
        # Compare physical and transition risks
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get data
        asset_data = []
        for asset in self.portfolio.assets:
            asset_id = asset['id']
            if asset_id in self.climate_risk_scores:
                data = {
                    'name': asset['name'],
                    'sector': asset['metadata'].get('sector', 'Unknown'),
                    'physical': self.climate_risk_scores[asset_id]['physical'],
                    'transition': self.climate_risk_scores[asset_id]['transition']
                }
                asset_data.append(data)
        
        # Convert to DataFrame
        df = pd.DataFrame(asset_data)
        
        # Plot comparison
        ax.scatter(df['physical'], df['transition'], alpha=0.7, s=100)
        
        # Add labels for points
        for i, row in df.iterrows():
            ax.annotate(
                row['name'],
                xy=(row['physical'], row['transition']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel('Physical Risk')
        ax.set_ylabel('Transition Risk')
        ax.set_title('Physical vs. Transition Climate Risk')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        fig.savefig(climate_comparison_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['climate_comparison'] = climate_comparison_file
        
        # Plot sector risks
        sector_risks_file = os.path.join(plots_dir, 'sector_risks.png')
        
        # Group data by sector
        sector_data = df.groupby('sector').agg({
            'physical': 'mean',
            'transition': 'mean'
        }).reset_index()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot sector risks
        bar_width = 0.35
        indices = np.arange(len(sector_data))
        
        ax.bar(indices - bar_width/2, sector_data['physical'], 
              bar_width, alpha=0.7, label='Physical Risk')
        ax.bar(indices + bar_width/2, sector_data['transition'], 
              bar_width, alpha=0.7, label='Transition Risk')
        
        # Add labels and title
        ax.set_xlabel('Sector')
        ax.set_ylabel('Risk Score')
        ax.set_title('Climate Risks by Sector')
        ax.set_xticks(indices)
        ax.set_xticklabels(sector_data['sector'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(sector_risks_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['sector_risks'] = sector_risks_file
        
        return plot_files
    
    def create_dashboard(self) -> None:
        """
        Create an interactive dashboard for geospatial financial risk analysis.
        """
        if not HAS_DASH:
            print("Dash not available. Cannot create interactive dashboard.")
            print("To install required packages:")
            print("pip install dash dash-bootstrap-components plotly")
            return
        
        # Initialize Dash app
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="Geospatial Financial Risk Analysis"
        )
        
        # Load initial data
        if self.portfolio is None:
            self.create_portfolio()
        
        if self.risk_model is None:
            self.create_risk_factors()
        
        if not self.risk_scores:
            self.assess_portfolio_risk()
        
        if not self.climate_risk_scores:
            self.assess_climate_risk()
        
        # Optimize portfolio
        total_value = sum(asset['value'] for asset in self.portfolio.assets)
        original_weights = {asset['id']: asset['value'] / total_value for asset in self.portfolio.assets}
        optimized_weights = self.optimize_portfolio()
        
        # Create asset dataframe
        asset_data = []
        for asset in self.portfolio.assets:
            asset_id = asset['id']
            
            if asset_id in self.risk_scores and asset_id in self.climate_risk_scores:
                data = {
                    'id': asset_id,
                    'name': asset['name'],
                    'value': asset['value'],
                    'x': asset['x'],
                    'y': asset['y'],
                    'sector': asset['metadata'].get('sector', 'Unknown'),
                    'risk_score': self.risk_scores[asset_id],
                    'physical_risk': self.climate_risk_scores[asset_id]['physical'],
                    'transition_risk': self.climate_risk_scores[asset_id]['transition'],
                    'climate_risk': self.climate_risk_scores[asset_id]['combined'],
                    'original_weight': original_weights.get(asset_id, 0.0),
                    'optimized_weight': optimized_weights.get(asset_id, 0.0)
                }
                
                asset_data.append(data)
        
        # Convert to DataFrame
        assets_df = pd.DataFrame(asset_data)
        
        # Create risk maps figure
        risk_map_fig = px.imshow(
            self.combined_risk.risk_data,
            color_continuous_scale='plasma_r',
            title="Geospatial Risk Map"
        )
        
        # Create climate risk maps figures
        climate_map_figs = {}
        for rf in self.climate_risk_factors:
            climate_map_figs[rf.name] = px.imshow(
                rf.risk_data,
                color_continuous_scale='plasma_r',
                title=rf.name
            )
        
        # Create portfolio scatter plot
        portfolio_fig = go.Figure()
        
        # Add risk map as background
        portfolio_fig.add_trace(
            go.Heatmap(
                z=self.combined_risk.risk_data,
                colorscale='plasma_r',
                showscale=True,
                opacity=0.8
            )
        )
        
        # Add assets as scatter points
        for sector in assets_df['sector'].unique():
            sector_df = assets_df[assets_df['sector'] == sector]
            
            # Convert geographic coordinates to pixel coordinates
            pixels = [self.geo_transform.geo_to_pixel(x, y) 
                     for x, y in zip(sector_df['x'], sector_df['y'])]
            pixel_x, pixel_y = zip(*pixels)
            
            portfolio_fig.add_trace(
                go.Scatter(
                    x=pixel_x,
                    y=pixel_y,
                    mode='markers',
                    marker=dict(
                        size=sector_df['value'] / 200,
                        color=sector_df['risk_score'],
                        colorscale='plasma_r',
                        line=dict(width=1, color='white')
                    ),
                    text=sector_df['name'],
                    name=sector
                )
            )
        
        portfolio_fig.update_layout(
            title="Portfolio Assets on Risk Map",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False)
        )
        
        # Create physical vs transition risk scatter plot
        risk_scatter_fig = px.scatter(
            assets_df,
            x='physical_risk',
            y='transition_risk',
            color='sector',
            size='value',
            hover_name='name',
            title="Physical vs. Transition Climate Risk",
            labels={
                'physical_risk': 'Physical Risk',
                'transition_risk': 'Transition Risk'
            }
        )
        
        risk_scatter_fig.update_layout(
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        # Add diagonal line
        risk_scatter_fig.add_shape(
            type="line",
            x0=0, y0=0,
            x1=1, y1=1,
            line=dict(
                color="black",
                width=1,
                dash="dash"
            )
        )
        
        # Create sector risk bar chart
        sector_risk_df = assets_df.groupby('sector').agg({
            'risk_score': 'mean',
            'physical_risk': 'mean',
            'transition_risk': 'mean',
            'climate_risk': 'mean',
            'value': 'sum'
        }).reset_index()
        
        sector_fig = go.Figure()
        
        sector_fig.add_trace(
            go.Bar(
                x=sector_risk_df['sector'],
                y=sector_risk_df['physical_risk'],
                name='Physical Risk',
                marker_color='blue'
            )
        )
        
        sector_fig.add_trace(
            go.Bar(
                x=sector_risk_df['sector'],
                y=sector_risk_df['transition_risk'],
                name='Transition Risk',
                marker_color='orange'
            )
        )
        
        sector_fig.update_layout(
            title="Climate Risks by Sector",
            xaxis_title="Sector",
            yaxis_title="Risk Score",
            barmode='group'
        )
        
        # Create portfolio optimization figure
        optimization_df = pd.DataFrame({
            'asset': assets_df['name'],
            'original': assets_df['original_weight'],
            'optimized': assets_df['optimized_weight'],
            'risk_score': assets_df['risk_score']
        })
        
        # Sort by risk score
        optimization_df = optimization_df.sort_values('risk_score', ascending=False)
        
        # Create figure
        optimization_fig = go.Figure()
        
        optimization_fig.add_trace(
            go.Bar(
                x=optimization_df['asset'],
                y=optimization_df['original'],
                name='Original Weights',
                marker_color='blue'
            )
        )
        
        optimization_fig.add_trace(
            go.Bar(
                x=optimization_df['asset'],
                y=optimization_df['optimized'],
                name='Optimized Weights',
                marker_color='green'
            )
        )
        
        optimization_fig.update_layout(
            title="Portfolio Optimization Results",
            xaxis_title="Asset",
            yaxis_title="Weight",
            barmode='group',
            xaxis=dict(tickangle=45)
        )
        
        # Create app layout
        app.layout = dbc.Container(
            fluid=True,
            children=[
                dbc.Row([
                    dbc.Col(html.H1("Geospatial Financial Risk Analysis Dashboard"), width=12)
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H4("Risk Analysis Settings"),
                        dbc.Card(
                            dbc.CardBody([
                                html.H5("Climate Scenario"),
                                dcc.Dropdown(
                                    id='scenario-dropdown',
                                    options=[
                                        {'label': s.value, 'value': s.value}
                                        for s in ClimateScenario
                                    ],
                                    value=self.active_scenario.value
                                ),
                                
                                html.H5("Time Horizon", className="mt-3"),
                                dcc.Dropdown(
                                    id='time-horizon-dropdown',
                                    options=[
                                        {'label': t.value, 'value': t.value}
                                        for t in TimeHorizon
                                    ],
                                    value=self.active_time_horizon.value
                                ),
                                
                                html.H5("Risk Aggregation Method", className="mt-3"),
                                dcc.Dropdown(
                                    id='aggregation-dropdown',
                                    options=[
                                        {'label': m.value.replace('_', ' ').title(), 
                                         'value': m.value}
                                        for m in AggregationMethod
                                    ],
                                    value=self.active_aggregation_method.value
                                ),
                                
                                dbc.Button(
                                    "Run Analysis",
                                    id="run-button",
                                    color="primary",
                                    className="mt-3"
                                )
                            ])
                        )
                    ], width=3),
                    
                    dbc.Col([
                        html.H4("Portfolio Statistics"),
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Geospatial Risk"),
                                        html.P(id="geo-risk-stats")
                                    ], width=6),
                                    
                                    dbc.Col([
                                        html.H5("Climate Risk"),
                                        html.P(id="climate-risk-stats")
                                    ], width=6)
                                ])
                            ])
                        )
                    ], width=9)
                ], className="mt-3"),
                
                dbc.Tabs([
                    dbc.Tab(
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Portfolio on Risk Map"),
                                        dcc.Graph(
                                            id='portfolio-graph',
                                            figure=portfolio_fig
                                        )
                                    ], width=6),
                                    
                                    dbc.Col([
                                        html.H5("Sector Risk Analysis"),
                                        dcc.Graph(
                                            id='sector-graph',
                                            figure=sector_fig
                                        )
                                    ], width=6)
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Portfolio Optimization"),
                                        dcc.Graph(
                                            id='optimization-graph',
                                            figure=optimization_fig
                                        )
                                    ], width=12)
                                ], className="mt-3")
                            ])
                        ),
                        label="Portfolio Analysis"
                    ),
                    
                    dbc.Tab(
                        dbc.Card(
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Physical vs. Transition Risk"),
                                        dcc.Graph(
                                            id='risk-scatter-graph',
                                            figure=risk_scatter_fig
                                        )
                                    ], width=6),
                                    
                                    dbc.Col([
                                        html.H5("Climate Risk Factors"),
                                        dcc.Dropdown(
                                            id='climate-factor-dropdown',
                                            options=[
                                                {'label': rf.name, 'value': rf.name}
                                                for rf in self.climate_risk_factors
                                            ],
                                            value=self.climate_risk_factors[0].name
                                        ),
                                        dcc.Graph(id='climate-factor-graph')
                                    ], width=6)
                                ]),
                                
                                dbc.Row([
                                    dbc.Col([
                                        html.H5("Scenario Comparison"),
                                        dcc.Graph(id='scenario-graph')
                                    ], width=12)
                                ], className="mt-3")
                            ])
                        ),
                        label="Climate Risk Analysis"
                    )
                ], className="mt-3")
            ]
        )
        
        # Callback to update climate factor graph
        @app.callback(
            Output('climate-factor-graph', 'figure'),
            Input('climate-factor-dropdown', 'value')
        )
        def update_climate_factor_graph(factor_name):
            for rf in self.climate_risk_factors:
                if rf.name == factor_name:
                    return px.imshow(
                        rf.risk_data,
                        color_continuous_scale='plasma_r',
                        title=rf.name
                    )
            
            # Default fallback
            return px.imshow(
                self.climate_risk_factors[0].risk_data,
                color_continuous_scale='plasma_r',
                title=self.climate_risk_factors[0].name
            )
        
        # Callback to update scenario comparison graph
        @app.callback(
            Output('scenario-graph', 'figure'),
            Input('run-button', 'n_clicks')
        )
        def update_scenario_graph(n_clicks):
            # Compare scenarios
            scenarios = list(ClimateScenario)[:4]  # First 4 scenarios
            time_horizons = list(TimeHorizon)[:3]  # First 3 time horizons
            
            scenario_results = self.compare_scenarios(scenarios, time_horizons)
            
            # Create data for plotting
            plot_data = []
            
            for scenario, horizon_results in scenario_results.items():
                for horizon, stats in horizon_results.items():
                    plot_data.append({
                        'scenario': scenario,
                        'horizon': horizon,
                        'avg_risk': stats['mean'],
                        'var': stats['var']
                    })
            
            # Convert to DataFrame
            df = pd.DataFrame(plot_data)
            
            # Create figure
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Average Climate Risk", "Climate-Adjusted VaR")
            )
            
            # Add average risk by scenario and horizon
            for horizon in df['horizon'].unique():
                horizon_df = df[df['horizon'] == horizon]
                
                fig.add_trace(
                    go.Bar(
                        x=horizon_df['scenario'],
                        y=horizon_df['avg_risk'],
                        name=f"{horizon} (Risk)",
                        marker_color=['green', 'yellow', 'orange', 'red'][
                            list(df['horizon'].unique()).index(horizon) % 4
                        ]
                    ),
                    row=1, col=1
                )
            
            # Add VaR by scenario and horizon
            for horizon in df['horizon'].unique():
                horizon_df = df[df['horizon'] == horizon]
                
                fig.add_trace(
                    go.Bar(
                        x=horizon_df['scenario'],
                        y=horizon_df['var'],
                        name=f"{horizon} (VaR)",
                        marker_color=['blue', 'purple', 'magenta', 'cyan'][
                            list(df['horizon'].unique()).index(horizon) % 4
                        ]
                    ),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Climate Risk Comparison Across Scenarios and Time Horizons",
                barmode='group',
                xaxis=dict(tickangle=45),
                xaxis2=dict(tickangle=45)
            )
            
            return fig
        
        # Callback to update statistics
        @app.callback(
            [Output('geo-risk-stats', 'children'),
             Output('climate-risk-stats', 'children')],
            Input('run-button', 'n_clicks'),
            [State('scenario-dropdown', 'value'),
             State('time-horizon-dropdown', 'value'),
             State('aggregation-dropdown', 'value')]
        )
        def update_statistics(n_clicks, scenario, time_horizon, aggregation_method):
            # Update settings if provided
            if scenario:
                self.active_scenario = ClimateScenario(scenario)
            
            if time_horizon:
                self.active_time_horizon = TimeHorizon(time_horizon)
            
            if aggregation_method:
                self.active_aggregation_method = AggregationMethod(aggregation_method)
            
            # Re-run analysis with new settings
            self.create_risk_factors()
            self.create_climate_risk_factors()
            self.assess_portfolio_risk()
            self.assess_climate_risk()
            
            # Calculate statistics
            geo_risk_values = list(self.risk_scores.values())
            climate_risk_values = [scores['combined'] for scores in self.climate_risk_scores.values()]
            
            # Calculate climate VaR
            climate_var = self.calculate_climate_var()
            
            # Format geo-risk stats
            geo_risk_stats = [
                html.P(f"Average Risk: {np.mean(geo_risk_values):.4f}"),
                html.P(f"Maximum Risk: {np.max(geo_risk_values):.4f}"),
                html.P(f"Minimum Risk: {np.min(geo_risk_values):.4f}")
            ]
            
            # Format climate risk stats
            climate_risk_stats = [
                html.P(f"Average Risk: {np.mean(climate_risk_values):.4f}"),
                html.P(f"Maximum Risk: {np.max(climate_risk_values):.4f}"),
                html.P(f"Climate-Adjusted VaR (95%): {climate_var:.2%}")
            ]
            
            return geo_risk_stats, climate_risk_stats
        
        # Run the application
        print("Starting dashboard application...")
        app.run_server(debug=False, port=8050)


def main():
    """Main function for the dashboard application."""
    parser = argparse.ArgumentParser(description='Geospatial Financial Dashboard Application')
    parser.add_argument('--data-dir', type=str, default='data/geo_financial',
                       help='Directory for data files')
    parser.add_argument('--output-dir', type=str, default='results/geo_financial/dashboard',
                       help='Directory for output files')
    parser.add_argument('--device-id', type=int, default=-1,
                       help='CUDA device ID (negative for CPU)')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (generate data without launching dashboard)')
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate new synthetic data')
    parser.add_argument('--dem-size', type=int, default=500,
                       help='Size of the synthetic DEM (pixels)')
    parser.add_argument('--num-assets', type=int, default=30,
                       help='Number of assets to generate')
    parser.add_argument('--days', type=int, default=252,
                       help='Number of days of returns data')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Create dashboard application
    app = DashboardApp(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device_id=args.device_id
    )
    
    # Generate new data if requested
    if args.generate_data:
        app.generate_data(
            dem_size=args.dem_size,
            num_assets=args.num_assets,
            days=args.days,
            seed=args.seed
        )
    else:
        # Load existing data
        app.load_data()
    
    # Create portfolio and risk factors
    app.create_portfolio()
    app.create_risk_factors()
    app.create_climate_risk_factors()
    
    # Run analysis
    app.assess_portfolio_risk()
    app.assess_climate_risk()
    
    # Save results
    app.save_results()
    
    # Generate plots
    plot_files = app.generate_dashboard_plots()
    print("Generated plots:")
    for name, file in plot_files.items():
        print(f"  - {name}: {file}")
    
    # Launch dashboard in non-headless mode
    if not args.headless and HAS_DASH:
        app.create_dashboard()
    elif not HAS_DASH and not args.headless:
        print("Cannot launch dashboard: Dash not available.")
        print("To install required packages:")
        print("pip install dash dash-bootstrap-components plotly")
    else:
        print("Running in headless mode, dashboard not launched.")
        
        # Generate and save dashboard data in JSON format
        data_file = os.path.join(args.output_dir, 'dashboard_data.json')
        dashboard_data = app.generate_dashboard_data()
        
        # Convert numpy arrays to lists for JSON serialization
        with open(data_file, 'w') as f:
            json.dump(dashboard_data, f, cls=NumpyEncoder)
        
        print(f"Dashboard data saved to {data_file}")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, ClimateScenario):
            return obj.value
        elif isinstance(obj, TimeHorizon):
            return obj.value
        elif isinstance(obj, AggregationMethod):
            return obj.value
        return super().default(obj)


if __name__ == '__main__':
    main()