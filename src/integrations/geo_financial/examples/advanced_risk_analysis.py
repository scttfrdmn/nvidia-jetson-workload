#!/usr/bin/env python3
"""
Advanced Geospatial Risk Analysis Example

This script demonstrates the advanced risk aggregation and analysis capabilities
of the Geospatial Financial Risk Analysis integration.

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
    SpatialRiskFactor,
    create_elevation_risk_factor,
    create_slope_risk_factor
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
    create_combined_risk_factor,
    interpolate_asset_risk_surface
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


def analyze_risk_sensitivity(portfolio: GeospatialPortfolio,
                            risk_model: GeospatialRiskModel,
                            risk_factors: List[SpatialRiskFactor],
                            factor_changes: List[float]) -> Dict:
    """
    Perform sensitivity analysis by adjusting risk factor weights.
    
    Args:
        portfolio: GeospatialPortfolio object
        risk_model: GeospatialRiskModel object
        risk_factors: List of risk factors in the model
        factor_changes: List of multipliers for factor weights
        
    Returns:
        Dictionary of sensitivity results
    """
    results = {}
    
    # Get baseline risk scores
    baseline_scores = portfolio.assess_risk(risk_model)
    results['baseline'] = {
        'scores': baseline_scores,
        'average': np.mean(list(baseline_scores.values())),
        'max': np.max(list(baseline_scores.values())),
        'weights': [f.risk_weight for f in risk_factors]
    }
    
    # For each factor, adjust its weight and recompute risks
    for i, factor in enumerate(risk_factors):
        factor_results = {}
        
        # Save original weight
        original_weight = factor.risk_weight
        
        for change in factor_changes:
            # Adjust factor weight
            new_weight = original_weight * change
            factor.risk_weight = new_weight
            
            # Recompute risk scores
            adjusted_scores = portfolio.assess_risk(risk_model)
            
            # Store results
            factor_results[change] = {
                'scores': adjusted_scores,
                'average': np.mean(list(adjusted_scores.values())),
                'max': np.max(list(adjusted_scores.values())),
                'weights': [f.risk_weight for f in risk_factors]
            }
        
        # Restore original weight
        factor.risk_weight = original_weight
        
        results[factor.name] = factor_results
    
    return results


def compare_aggregation_methods(risk_factors: List[SpatialRiskFactor],
                              device_id: int = 0) -> Dict:
    """
    Compare different risk aggregation methods.
    
    Args:
        risk_factors: List of risk factors to aggregate
        device_id: CUDA device ID (negative for CPU)
        
    Returns:
        Dictionary of aggregation method results
    """
    results = {}
    
    # Create risk aggregator
    aggregator = RiskAggregator(device_id=device_id)
    
    # Compute correlation matrix (used for copula methods)
    correlation_matrix = None
    if len(risk_factors) > 1:
        correlation_matrix = np.eye(len(risk_factors))
        for i in range(len(risk_factors)):
            for j in range(i+1, len(risk_factors)):
                # Flatten arrays for correlation calculation
                data_i = risk_factors[i].risk_data.flatten()
                data_j = risk_factors[j].risk_data.flatten()
                corr = np.corrcoef(data_i, data_j)[0, 1]
                correlation_matrix[i, j] = correlation_matrix[j, i] = corr
    
    # Compare different aggregation methods
    for method in AggregationMethod:
        try:
            # Aggregate risk factors
            if method in [AggregationMethod.COPULA_GAUSSIAN, AggregationMethod.COPULA_STUDENT_T]:
                risk_map, stats = aggregator.aggregate_risk_factors(
                    risk_factors=risk_factors,
                    method=method,
                    correlation_matrix=correlation_matrix
                )
            else:
                risk_map, stats = aggregator.aggregate_risk_factors(
                    risk_factors=risk_factors,
                    method=method
                )
            
            # Store results
            results[method.value] = {
                'risk_map': risk_map,
                'stats': stats
            }
        except Exception as e:
            print(f"Error with method {method.value}: {e}")
            continue
    
    return results


def plot_aggregation_comparison(results: Dict, figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Plot a comparison of different aggregation methods.
    
    Args:
        results: Dictionary of aggregation method results
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    num_methods = len(results)
    fig, axes = plt.subplots(2, (num_methods + 1) // 2, figsize=figsize)
    axes = axes.flatten()
    
    for i, (method_name, method_results) in enumerate(results.items()):
        if i < len(axes):
            # Plot risk map
            im = axes[i].imshow(method_results['risk_map'], cmap='viridis_r', origin='upper')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
            # Add title with statistics
            stats = method_results['stats']
            title = f"{method_name.replace('_', ' ').title()}\n"
            title += f"Min: {stats['min']:.3f}, Max: {stats['max']:.3f}\n"
            title += f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}"
            axes[i].set_title(title)
            
            # Remove ticks
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    # Hide any unused axes
    for i in range(len(results), len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    fig.suptitle("Comparison of Risk Aggregation Methods", fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.90)
    
    return fig


def plot_risk_sensitivity(sensitivity_results: Dict,
                         factor_changes: List[float],
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot the results of risk sensitivity analysis.
    
    Args:
        sensitivity_results: Dictionary of sensitivity analysis results
        factor_changes: List of multipliers used in the analysis
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Extract factor names (excluding baseline)
    factor_names = [name for name in sensitivity_results.keys() if name != 'baseline']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot average risk
    for factor_name in factor_names:
        avg_values = [sensitivity_results[factor_name][change]['average'] 
                     for change in factor_changes]
        axes[0].plot(factor_changes, avg_values, marker='o', label=factor_name)
    
    axes[0].axhline(y=sensitivity_results['baseline']['average'], 
                   color='black', linestyle='--', label='Baseline')
    axes[0].set_xlabel('Factor Weight Multiplier')
    axes[0].set_ylabel('Average Portfolio Risk')
    axes[0].set_title('Sensitivity of Average Risk to Factor Weights')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot maximum risk
    for factor_name in factor_names:
        max_values = [sensitivity_results[factor_name][change]['max'] 
                     for change in factor_changes]
        axes[1].plot(factor_changes, max_values, marker='o', label=factor_name)
    
    axes[1].axhline(y=sensitivity_results['baseline']['max'], 
                   color='black', linestyle='--', label='Baseline')
    axes[1].set_xlabel('Factor Weight Multiplier')
    axes[1].set_ylabel('Maximum Portfolio Risk')
    axes[1].set_title('Sensitivity of Maximum Risk to Factor Weights')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def main():
    """Main function for the advanced risk analysis example."""
    parser = argparse.ArgumentParser(description='Advanced Geospatial Risk Analysis Example')
    parser.add_argument('--data-dir', type=str, default='data/geo_financial',
                       help='Directory for data files')
    parser.add_argument('--output-dir', type=str, default='results/geo_financial',
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
    
    print("Advanced Geospatial Risk Analysis Example")
    print("========================================")
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
    
    # Create risk factors
    print("\nCreating risk factors...")
    
    # Elevation-based flood risk (lower elevation = higher risk)
    flood_risk = SpatialRiskFactor(
        name="Flood Risk",
        description="Risk based on elevation (lower elevations have higher risk)",
        risk_weight=0.5,
        spatial_data=dem_data,
        geo_transform=geo_transform,
        transform_func=lambda elev: 1.0 - np.clip((elev - np.min(elev)) / (np.max(elev) - np.min(elev)), 0, 1)
    )
    
    # Calculate slopes from DEM
    from scipy.ndimage import sobel
    dx = sobel(dem_data, axis=1)
    dy = sobel(dem_data, axis=0)
    slope_data = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Slope-based landslide risk (steeper slope = higher risk)
    landslide_risk = SpatialRiskFactor(
        name="Landslide Risk",
        description="Risk based on terrain slope (steeper slopes have higher risk)",
        risk_weight=0.3,
        spatial_data=slope_data,
        geo_transform=geo_transform,
        transform_func=lambda slope: np.clip(slope / 45.0, 0, 1)  # Normalize to 0-1 (45 degrees is max risk)
    )
    
    # Distance to river (center of DEM) as flood proximity risk
    river_y = dem_data.shape[0] // 2
    distance_to_river = np.abs(np.arange(dem_data.shape[0])[:, np.newaxis] - river_y)
    distance_to_river = np.repeat(distance_to_river, dem_data.shape[1], axis=1)
    max_distance = dem_data.shape[0] // 2
    river_risk = SpatialRiskFactor(
        name="River Proximity Risk",
        description="Risk based on proximity to river",
        risk_weight=0.2,
        spatial_data=distance_to_river,
        geo_transform=geo_transform,
        transform_func=lambda dist: 1.0 - np.clip(dist / max_distance, 0, 1)
    )
    
    # Compare aggregation methods
    print("\nComparing risk aggregation methods...")
    risk_factors = [flood_risk, landslide_risk, river_risk]
    aggregation_results = compare_aggregation_methods(
        risk_factors=risk_factors,
        device_id=args.device_id
    )
    
    # Create combined risk factor using copula method
    combined_risk = create_combined_risk_factor(
        risk_factors=risk_factors,
        name="Combined Natural Hazard Risk",
        description="Comprehensive natural hazard risk combining flood, landslide, and river proximity",
        method=AggregationMethod.COPULA_GAUSSIAN,
        device_id=args.device_id
    )
    
    # Create geospatial risk model
    print("\nCreating geospatial risk model...")
    risk_model = GeospatialRiskModel(device_id=args.device_id)
    risk_model.add_risk_factor(combined_risk)
    
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
    
    # Perform risk assessment
    print("\nPerforming risk assessment...")
    risk_scores = portfolio.assess_risk(risk_model)
    
    # Generate interpolated risk surface from asset risk scores
    print("\nGenerating interpolated risk surface...")
    interpolated_surface, interp_geo_transform = interpolate_asset_risk_surface(
        portfolio=portfolio,
        risk_scores=risk_scores,
        grid_size=(100, 100),
        x_range=(0, 10),
        y_range=(0, 10),
        method='idw',
        device_id=args.device_id
    )
    
    # Perform sensitivity analysis
    print("\nPerforming sensitivity analysis...")
    factor_changes = [0.5, 0.75, 1.0, 1.5, 2.0]
    sensitivity_results = analyze_risk_sensitivity(
        portfolio=portfolio,
        risk_model=risk_model,
        risk_factors=risk_factors,
        factor_changes=factor_changes
    )
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = GeoFinancialVisualizer(figsize=(12, 10))
    
    # Plot risk factors
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot flood risk
    im1 = axes[0].imshow(flood_risk.risk_data, cmap='viridis_r', origin='upper')
    axes[0].set_title('Flood Risk')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot landslide risk
    im2 = axes[1].imshow(landslide_risk.risk_data, cmap='viridis_r', origin='upper')
    axes[1].set_title('Landslide Risk')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot river proximity risk
    im3 = axes[2].imshow(river_risk.risk_data, cmap='viridis_r', origin='upper')
    axes[2].set_title('River Proximity Risk')
    plt.colorbar(im3, ax=axes[2])
    
    # Plot combined risk
    im4 = axes[3].imshow(combined_risk.risk_data, cmap='viridis_r', origin='upper')
    axes[3].set_title('Combined Risk')
    plt.colorbar(im4, ax=axes[3])
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'risk_factors.png'), dpi=300, bbox_inches='tight')
    
    # Plot aggregation method comparison
    fig_agg = plot_aggregation_comparison(aggregation_results)
    plt.savefig(os.path.join(output_dir, 'aggregation_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Plot interpolated risk surface with assets
    fig = visualizer.plot_portfolio_on_risk_map(
        risk_map=interpolated_surface,
        geo_transform=interp_geo_transform,
        portfolio=portfolio,
        title="Interpolated Risk Surface from Asset Risk Scores"
    )
    plt.savefig(os.path.join(output_dir, 'interpolated_risk_surface.png'), dpi=300, bbox_inches='tight')
    
    # Plot sensitivity analysis results
    fig_sens = plot_risk_sensitivity(sensitivity_results, factor_changes)
    plt.savefig(os.path.join(output_dir, 'risk_sensitivity.png'), dpi=300, bbox_inches='tight')
    
    # Plot risk heatmap
    fig = visualizer.plot_portfolio_risk_heatmap(
        portfolio=portfolio,
        risk_scores=risk_scores,
        title="Asset Risk Heatmap"
    )
    plt.savefig(os.path.join(output_dir, 'risk_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # Plot risk factors comparison
    fig = visualizer.plot_risk_factor_comparison(
        risk_factors=risk_factors,
        portfolio=portfolio
    )
    plt.savefig(os.path.join(output_dir, 'risk_factor_comparison.png'), dpi=300, bbox_inches='tight')
    
    print("\nAnalysis complete!")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()