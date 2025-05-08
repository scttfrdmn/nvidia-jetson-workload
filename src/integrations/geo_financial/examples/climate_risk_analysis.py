#!/usr/bin/env python3
"""
Climate Risk Analysis Example

This script demonstrates how to use the climate risk assessment module
to evaluate the financial impact of climate change on investment portfolios.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import Optional, Tuple, Dict, List

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
sys.path.append(project_root)

# Import modules
from src.integrations.geo_financial.geo_risk import (
    GeospatialRiskModel,
    GeospatialPortfolio,
    SpatialRiskFactor
)
from src.integrations.geo_financial.data_connectors import (
    AssetLocationDataLoader,
    FinancialDataLoader
)
from src.integrations.geo_financial.visualization import GeoFinancialVisualizer
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
    TransitionRiskType,
    create_default_transition_risks,
    create_flood_risk_factor,
    create_heatwave_risk_factor,
    create_sea_level_rise_risk_factor
)
from src.integrations.geo_financial.generate_test_data import (
    generate_dem,
    generate_assets,
    generate_returns
)

# Optional imports
try:
    from src.geospatial.python.geospatial import dem as geospatial_dem
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False
    print("Warning: Geospatial module not available, some features will be limited")


def plot_scenario_comparison(portfolio: GeospatialPortfolio,
                           scenarios: List[ClimateScenario],
                           time_horizons: List[TimeHorizon],
                           risk_factors: List[SpatialRiskFactor],
                           figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Plot a comparison of climate risks across different scenarios and time horizons.
    
    Args:
        portfolio: GeospatialPortfolio object
        scenarios: List of climate scenarios to compare
        time_horizons: List of time horizons to compare
        risk_factors: List of risk factors to use in assessment
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, axes = plt.subplots(len(time_horizons), len(scenarios), figsize=figsize)
    
    # Flatten axes if needed
    if len(time_horizons) == 1:
        axes = np.array([axes])
    if len(scenarios) == 1:
        axes = np.array([axes]).T
    
    # Store results for comparison
    avg_risks = np.zeros((len(time_horizons), len(scenarios)))
    max_risks = np.zeros((len(time_horizons), len(scenarios)))
    
    # Assess risk for each scenario and time horizon
    for i, time_horizon in enumerate(time_horizons):
        for j, scenario in enumerate(scenarios):
            # Create climate risk assessor
            assessor = ClimateRiskAssessor(
                scenario=scenario,
                time_horizon=time_horizon,
                device_id=0
            )
            
            # Assess physical climate risk
            risk_scores = assessor.assess_physical_climate_risk(
                portfolio=portfolio,
                risk_factors=risk_factors,
                aggregation_method=AggregationMethod.COPULA_GAUSSIAN
            )
            
            # Calculate statistics
            avg_risk = np.mean(list(risk_scores.values()))
            max_risk = np.max(list(risk_scores.values()))
            
            avg_risks[i, j] = avg_risk
            max_risks[i, j] = max_risk
            
            # Plot risk scores
            sector_risk_data = {}
            for asset in portfolio.assets:
                if asset['id'] in risk_scores and 'metadata' in asset and 'sector' in asset['metadata']:
                    sector = asset['metadata']['sector']
                    if sector not in sector_risk_data:
                        sector_risk_data[sector] = []
                    sector_risk_data[sector].append(risk_scores[asset['id']])
            
            # Plot sector-based risk distribution
            if sector_risk_data:
                ax = axes[i, j]
                
                # Create box plot data
                labels, data = zip(*[(k, v) for k, v in sector_risk_data.items()])
                
                # Plot box plot
                ax.boxplot(data, labels=labels, patch_artist=True)
                ax.set_ylim(0, 1)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
                # Add title with statistics
                title = f"{scenario.value}\n{time_horizon.value} horizon\n"
                title += f"Avg: {avg_risk:.3f}, Max: {max_risk:.3f}"
                ax.set_title(title)
                
                # Set y-axis label only for the first column
                if j == 0:
                    ax.set_ylabel('Risk Score')
    
    # Add overall title
    fig.suptitle("Climate Risk Comparison Across Scenarios and Time Horizons", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    return fig, avg_risks, max_risks


def plot_sector_transition_risks(transition_risks: Dict[TransitionRiskType, Dict[str, float]],
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot transition risks by sector.
    
    Args:
        transition_risks: Dictionary mapping risk types to sector risk dictionaries
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Get all sectors
    sectors = set()
    for risk_type, sector_risks in transition_risks.items():
        sectors.update(sector_risks.keys())
    sectors = sorted(sectors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create data for plotting
    risk_types = list(transition_risks.keys())
    bar_width = 0.8 / len(risk_types)
    positions = np.arange(len(sectors))
    
    # Plot bars for each risk type
    for i, risk_type in enumerate(risk_types):
        sector_risks = transition_risks[risk_type]
        
        # Get risk values for each sector
        risk_values = [sector_risks.get(sector, 0.0) for sector in sectors]
        
        # Calculate bar positions
        bar_positions = positions + (i - len(risk_types) / 2 + 0.5) * bar_width
        
        # Plot bars
        ax.bar(bar_positions, risk_values, width=bar_width, 
              label=risk_type.value.replace('_', ' ').title())
    
    # Add labels and legend
    ax.set_xlabel('Sector')
    ax.set_ylabel('Risk Score')
    ax.set_title('Transition Risks by Sector')
    ax.set_xticks(positions)
    ax.set_xticklabels(sectors, rotation=45, ha='right')
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def plot_climate_var_comparison(portfolio: GeospatialPortfolio,
                              scenarios: List[ClimateScenario],
                              time_horizons: List[TimeHorizon],
                              risk_factors: List[SpatialRiskFactor],
                              confidence_level: float = 0.95,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot a comparison of climate-adjusted VaR across different scenarios and time horizons.
    
    Args:
        portfolio: GeospatialPortfolio object
        scenarios: List of climate scenarios to compare
        time_horizons: List of time horizons to compare
        risk_factors: List of risk factors to use in assessment
        confidence_level: Confidence level for VaR calculation
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate base VaR
    try:
        base_var = portfolio.calculate_portfolio_var(confidence_level)
    except ValueError:
        # If VaR calculation fails, use a synthetic value
        base_var = 0.02  # 2% daily VaR
    
    # Store VaR values for plotting
    var_values = []
    var_labels = []
    
    # Create transition risks
    transition_risks = create_default_transition_risks()
    
    # Calculate VaR for each scenario and time horizon
    for scenario in scenarios:
        for time_horizon in time_horizons:
            # Create climate risk assessor
            assessor = ClimateRiskAssessor(
                scenario=scenario,
                time_horizon=time_horizon,
                device_id=0
            )
            
            # Assess physical climate risk
            physical_risk_scores = assessor.assess_physical_climate_risk(
                portfolio=portfolio,
                risk_factors=risk_factors,
                aggregation_method=AggregationMethod.COPULA_GAUSSIAN
            )
            
            # Assess transition risk
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
                adjusted_sector_risks = assessor.create_transition_risk_factor(
                    risk_type=risk_type,
                    sector_risks=sector_risks,
                    risk_weight=sector_risk_weights[risk_type],
                    apply_scenario=True
                )
                adjusted_transition_risks[risk_type] = adjusted_sector_risks
            
            # Assess transition climate risk
            transition_risk_scores = assessor.assess_transition_climate_risk(
                portfolio=portfolio,
                sector_field='sector',
                transition_risks=adjusted_transition_risks,
                risk_weights=sector_risk_weights
            )
            
            # Combine physical and transition risks
            combined_risk_scores = assessor.assess_combined_climate_risk(
                portfolio=portfolio,
                physical_risk_scores=physical_risk_scores,
                transition_risk_scores=transition_risk_scores,
                physical_weight=0.5
            )
            
            # Calculate climate-adjusted VaR
            climate_var = assessor.calculate_climate_var(
                portfolio=portfolio,
                climate_risk_scores=combined_risk_scores,
                confidence_level=confidence_level,
                risk_weight=1.0
            )
            
            # Store VaR value and label
            var_values.append(climate_var)
            var_labels.append(f"{scenario.value}\n{time_horizon.value}")
    
    # Plot base VaR
    ax.axhline(y=base_var, color='black', linestyle='--', label=f'Base VaR ({confidence_level*100:.0f}%)')
    
    # Plot climate-adjusted VaR values
    bar_positions = np.arange(len(var_labels))
    bars = ax.bar(bar_positions, var_values)
    
    # Color bars based on scenario
    scenario_colors = {
        ClimateScenario.SSP1_RCP26: 'green',
        ClimateScenario.SSP2_RCP45: 'yellow',
        ClimateScenario.SSP3_RCP70: 'orange',
        ClimateScenario.SSP5_RCP85: 'red',
        ClimateScenario.NGFS_ORDERLY: 'lightgreen',
        ClimateScenario.NGFS_DISORDERLY: 'gold',
        ClimateScenario.NGFS_HOTHOUSE: 'darkred'
    }
    
    for i, (scenario, _) in enumerate([(s, t) for s in scenarios for t in time_horizons]):
        if scenario in scenario_colors:
            bars[i].set_color(scenario_colors[scenario])
    
    # Add labels and title
    ax.set_xlabel('Climate Scenario and Time Horizon')
    ax.set_ylabel(f'Value-at-Risk ({confidence_level*100:.0f}%)')
    ax.set_title(f'Climate-Adjusted VaR Comparison')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(var_labels, rotation=45, ha='right')
    
    # Add values on top of bars
    for bar, value in zip(bars, var_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
               f'{value:.2%}', ha='center', va='bottom')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def main():
    """Main function for the climate risk analysis example."""
    parser = argparse.ArgumentParser(description='Climate Risk Analysis Example')
    parser.add_argument('--data-dir', type=str, default='data/geo_financial',
                       help='Directory for data files')
    parser.add_argument('--output-dir', type=str, default='results/geo_financial/climate',
                       help='Directory for output files')
    parser.add_argument('--device-id', type=int, default=0,
                       help='CUDA device ID (negative for CPU)')
    parser.add_argument('--dem-size', type=int, default=500,
                       help='Size of the synthetic DEM (pixels)')
    parser.add_argument('--num-assets', type=int, default=30,
                       help='Number of assets to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Create directories
    data_dir = os.path.join(project_root, args.data_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Climate Risk Analysis Example")
    print("============================")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Using device ID: {args.device_id}")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate synthetic data if needed
    print("\nGenerating synthetic data...")
    dem_data, x_coords, y_coords = generate_dem(
        output_dir=data_dir,
        size=args.dem_size,
        save_plot=True,
        seed=args.seed
    )
    
    assets_df = generate_assets(
        output_dir=data_dir,
        dem_data=dem_data,
        x_coords=x_coords,
        y_coords=y_coords,
        num_assets=args.num_assets,
        save_plot=True,
        seed=args.seed
    )
    
    returns_df = generate_returns(
        output_dir=data_dir,
        assets_df=assets_df,
        days=252,
        correlation=0.7,
        save_plot=True,
        seed=args.seed
    )
    
    # Load data
    print("\nPreparing data for analysis...")
    
    # Convert returns to dictionary
    returns_dict = {}
    for asset_id in assets_df['id'].unique():
        asset_returns = returns_df[returns_df['asset_id'] == asset_id]['return'].values
        returns_dict[asset_id] = asset_returns
    
    # Create geo_transform
    if HAS_GEOSPATIAL:
        geo_transform = geospatial_dem.GeoTransform([0, 0.02, 0, 10, 0, -0.02])
    else:
        # Simple mock GeoTransform
        class MockGeoTransform:
            def geo_to_pixel(self, x, y):
                return int(x * 50), int(y * 50)
            
            def pixel_to_geo(self, x, y):
                return x / 50, y / 50
        
        geo_transform = MockGeoTransform()
    
    # Create geospatial portfolio
    print("\nCreating geospatial portfolio...")
    portfolio = GeospatialPortfolio(device_id=args.device_id)
    
    # Add assets to portfolio
    for _, row in assets_df.iterrows():
        portfolio.add_asset(
            asset_id=row['id'],
            name=row['name'],
            value=row['value'],
            x=row['x'],
            y=row['y'],
            returns=returns_dict.get(row['id'], np.array([])),
            metadata={'sector': row['sector']}
        )
    
    # Create climate risk assessor
    print("\nCreating climate risk factors...")
    assessor = ClimateRiskAssessor(
        device_id=args.device_id,
        scenario=ClimateScenario.SSP2_RCP45,
        time_horizon=TimeHorizon.MEDIUM
    )
    
    # Create flood risk factor
    # For simulation, we'll use the DEM directly
    flood_risk = create_flood_risk_factor(
        dem_path=dem_data,
        risk_weight=0.6,
        low_elevation_threshold=np.percentile(dem_data, 20),  # 20th percentile
        assessor=assessor
    )
    
    # Create heatwave risk factor
    # For simulation, we'll create synthetic temperature data
    # Higher temperatures in lower elevations and certain regions
    temperature_data = 25.0 + 15.0 * (1.0 - np.clip((dem_data - np.min(dem_data)) / 
                                              (np.max(dem_data) - np.min(dem_data)), 0, 1))
    
    # Add some regional variation
    x_idx, y_idx = np.meshgrid(np.arange(dem_data.shape[1]), np.arange(dem_data.shape[0]))
    regional_pattern = 5.0 * np.sin(x_idx / 50.0) * np.cos(y_idx / 50.0)
    temperature_data += regional_pattern
    
    heatwave_risk = create_heatwave_risk_factor(
        temperature_data=temperature_data,
        geo_transform=geo_transform,
        risk_weight=0.4,
        high_temp_threshold=35.0,
        assessor=assessor
    )
    
    # Create sea level rise risk factor
    # For simulation, we'll create synthetic coastline distance data
    # Lower part of the DEM will be the "coastline"
    coastline_y = int(dem_data.shape[0] * 0.8)
    coastline_distance = np.zeros_like(dem_data)
    
    for i in range(dem_data.shape[0]):
        for j in range(dem_data.shape[1]):
            # Distance to coastline in grid cells
            distance = abs(i - coastline_y)
            # Convert to kilometers (assuming 1 grid cell = 0.5 km)
            coastline_distance[i, j] = distance * 0.5
    
    sea_level_risk = create_sea_level_rise_factor = create_sea_level_rise_risk_factor(
        dem_path=dem_data,
        coastline_distance=coastline_distance,
        geo_transform=geo_transform,
        risk_weight=0.5,
        elevation_threshold=np.percentile(dem_data, 30),  # 30th percentile
        distance_threshold=20.0,  # 20 km
        assessor=assessor
    )
    
    # List of risk factors
    risk_factors = [flood_risk, heatwave_risk, sea_level_risk]
    
    # Perform climate risk assessment
    print("\nPerforming climate risk assessment...")
    
    # Scenarios to compare
    scenarios = [
        ClimateScenario.SSP1_RCP26,
        ClimateScenario.SSP2_RCP45,
        ClimateScenario.SSP3_RCP70,
        ClimateScenario.SSP5_RCP85
    ]
    
    # Time horizons to compare
    time_horizons = [
        TimeHorizon.SHORT,
        TimeHorizon.MEDIUM,
        TimeHorizon.LONG
    ]
    
    # Compare scenarios
    print("\nComparing climate scenarios...")
    fig_scenario, avg_risks, max_risks = plot_scenario_comparison(
        portfolio=portfolio,
        scenarios=scenarios,
        time_horizons=time_horizons,
        risk_factors=risk_factors
    )
    
    plt.savefig(os.path.join(output_dir, 'scenario_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Create transition risks
    print("\nCreating transition risks...")
    transition_risks = create_default_transition_risks()
    
    # Plot transition risks by sector
    print("\nPlotting transition risks by sector...")
    fig_transition = plot_sector_transition_risks(transition_risks)
    
    plt.savefig(os.path.join(output_dir, 'transition_risks.png'), dpi=300, bbox_inches='tight')
    
    # Calculate climate-adjusted VaR
    print("\nCalculating climate-adjusted VaR...")
    fig_var = plot_climate_var_comparison(
        portfolio=portfolio,
        scenarios=scenarios,
        time_horizons=time_horizons,
        risk_factors=risk_factors,
        confidence_level=0.95
    )
    
    plt.savefig(os.path.join(output_dir, 'climate_var_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Visualize risk factors
    print("\nVisualizing risk factors...")
    visualizer = GeoFinancialVisualizer(figsize=(12, 10))
    
    # Plot risk factors
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot flood risk
    im1 = axes[0].imshow(flood_risk.risk_data, cmap='viridis_r', origin='upper')
    axes[0].set_title(flood_risk.name)
    plt.colorbar(im1, ax=axes[0])
    
    # Plot heatwave risk
    im2 = axes[1].imshow(heatwave_risk.risk_data, cmap='viridis_r', origin='upper')
    axes[1].set_title(heatwave_risk.name)
    plt.colorbar(im2, ax=axes[1])
    
    # Plot sea level rise risk
    im3 = axes[2].imshow(sea_level_risk.risk_data, cmap='viridis_r', origin='upper')
    axes[2].set_title(sea_level_risk.name)
    plt.colorbar(im3, ax=axes[2])
    
    # Plot combined risk
    # Create combined risk factor
    combined_risk = create_combined_risk_factor(
        risk_factors=risk_factors,
        name=f"Combined Climate Risk ({assessor.scenario.value}, {assessor.time_horizon.value})",
        description=f"Combined climate risk under {assessor.scenario.value} scenario",
        method=AggregationMethod.COPULA_GAUSSIAN,
        device_id=args.device_id
    )
    
    im4 = axes[3].imshow(combined_risk.risk_data, cmap='viridis_r', origin='upper')
    axes[3].set_title(combined_risk.name)
    plt.colorbar(im4, ax=axes[3])
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'climate_risk_factors.png'), dpi=300, bbox_inches='tight')
    
    # Create risk model with combined risk
    risk_model = GeospatialRiskModel(device_id=args.device_id)
    risk_model.add_risk_factor(combined_risk)
    
    # Assess portfolio risk
    risk_scores = portfolio.assess_risk(risk_model)
    
    # Plot portfolio on risk map
    fig = visualizer.plot_portfolio_on_risk_map(
        risk_map=combined_risk.risk_data,
        geo_transform=geo_transform,
        portfolio=portfolio,
        title=f"Portfolio Assets on Climate Risk Map ({assessor.scenario.value}, {assessor.time_horizon.value})"
    )
    
    plt.savefig(os.path.join(output_dir, 'portfolio_climate_risk.png'), dpi=300, bbox_inches='tight')
    
    # Generate summary results
    print("\nGenerating summary results...")
    
    # Create summary table of scenario risks
    summary_df = pd.DataFrame(
        avg_risks,
        index=[h.value for h in time_horizons],
        columns=[s.value for s in scenarios]
    )
    
    # Save summary table
    summary_df.to_csv(os.path.join(output_dir, 'scenario_risk_summary.csv'))
    
    print("\nAnalysis complete!")
    print(f"Results saved to {output_dir}")
    
    # Print summary statistics
    print("\nScenario Risk Summary (Average Risk):")
    print(summary_df)
    
    # Get assets with highest climate risk
    high_risk_assets = sorted(
        [(asset['name'], risk_scores[asset['id']]) for asset in portfolio.assets],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    
    print("\nAssets with highest climate risk:")
    for name, risk in high_risk_assets:
        print(f"  {name}: {risk:.4f}")


if __name__ == '__main__':
    main()