#!/usr/bin/env python3
"""
Multi-region Geospatial Financial Analysis Example

This example demonstrates how to use the multi-region analysis capabilities
to compare financial risks across different geographic regions.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
from src.integrations.geo_financial.multiregion_analysis import (
    RegionDefinition,
    RegionalPortfolio,
    MultiRegionRiskModel,
    RegionalRiskComparator,
    MultiRegionVisualizer,
    create_region_grid,
    create_regional_portfolio_from_grid,
    perform_multi_region_analysis
)
from src.integrations.geo_financial.risk_aggregation import (
    RiskAggregator,
    AggregationMethod,
    create_combined_risk_factor
)
from src.integrations.geo_financial.climate_risk_assessment import (
    ClimateRiskAssessor,
    ClimateScenario,
    TimeHorizon,
    create_flood_risk_factor,
    create_heatwave_risk_factor
)
from src.integrations.geo_financial.generate_test_data import (
    generate_dem,
    generate_assets,
    generate_returns
)


def setup_logger(output_dir: str) -> logging.Logger:
    """
    Set up a logger for the example.
    
    Args:
        output_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("MultiRegionExample")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Also create a file handler
    log_file = os.path.join(output_dir, 'multiregion_example.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def generate_regional_data(data_dir: str, 
                          dem_size: int = 500,
                          assets_per_region: int = 20,
                          num_regions: int = 4,
                          seed: int = 42) -> tuple:
    """
    Generate synthetic data for multiple regions.
    
    Args:
        data_dir: Directory to save generated data
        dem_size: Size of DEM to generate
        assets_per_region: Number of assets per region
        num_regions: Number of regions to create
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (dem_data, dem_transform, assets_df, regions)
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate DEM
    dem_path = os.path.join(data_dir, 'synthetic_dem.npy')
    dem_data, dem_transform = generate_dem(
        size=dem_size,
        roughness=0.6,
        min_height=0.0,
        max_height=1000.0,
        output_path=dem_path
    )
    
    # Create regions
    grid_size = int(np.sqrt(num_regions))
    rows = grid_size
    cols = grid_size
    if rows * cols < num_regions:
        cols += 1
    
    # Get bounds from DEM
    bounds = {
        "min_x": dem_transform.origin_x,
        "min_y": dem_transform.origin_y,
        "max_x": dem_transform.origin_x + dem_transform.pixel_width * dem_data.shape[1],
        "max_y": dem_transform.origin_y + dem_transform.pixel_height * dem_data.shape[0]
    }
    
    # Generate region names and economic weights
    region_names = [f"Region_{i+1}" for i in range(rows * cols)]
    
    # Create regions
    regions = create_region_grid(
        base_bounds=bounds,
        grid_size=(rows, cols),
        region_names=region_names[:num_regions]
    )
    
    # Generate assets for each region
    all_assets = []
    
    # Define sectors
    sectors = ['technology', 'energy', 'financial', 'healthcare', 'consumer']
    
    # Generate assets for each region
    for i, region in enumerate(regions):
        # Generate biased sector distribution for this region
        # Some regions will have more of certain sectors
        sector_weights = np.random.dirichlet(np.ones(len(sectors)) * 0.5)
        
        # Slight value bias for each region
        value_multiplier = 0.8 + (np.random.random() * 0.5)  # 0.8 to 1.3
        
        # Generate assets slightly biased to certain regions
        region_assets = []
        for j in range(assets_per_region):
            # Generate asset
            asset_id = f"asset_{i}_{j}"
            
            # Random position within region
            x = np.random.uniform(region.bounds["min_x"], region.bounds["max_x"])
            y = np.random.uniform(region.bounds["min_y"], region.bounds["max_y"])
            
            # Random sector based on region's sector weights
            sector = np.random.choice(sectors, p=sector_weights)
            
            # Value based on sector and region
            base_value = np.random.lognormal(mean=4.0, sigma=1.0)  # Random base value
            sector_multiplier = {
                'technology': 1.2,
                'energy': 1.1,
                'financial': 1.3,
                'healthcare': 0.9,
                'consumer': 0.7
            }.get(sector, 1.0)
            
            value = base_value * sector_multiplier * value_multiplier
            
            # Create asset
            asset = {
                'id': asset_id,
                'name': f"Asset {i}.{j}",
                'value': value,
                'x': x,
                'y': y,
                'sector': sector,
                'region': region.name
            }
            
            region_assets.append(asset)
        
        all_assets.extend(region_assets)
    
    # Create assets DataFrame
    assets_df = pd.DataFrame(all_assets)
    
    # Save assets to CSV
    assets_path = os.path.join(data_dir, 'regional_assets.csv')
    assets_df.to_csv(assets_path, index=False)
    
    return dem_data, dem_transform, assets_df, regions


def create_risk_model(dem_data, dem_transform, device_id=0):
    """
    Create a multi-region risk model.
    
    Args:
        dem_data: DEM data
        dem_transform: GeoTransform for the DEM
        device_id: GPU device ID
        
    Returns:
        MultiRegionRiskModel instance
    """
    # Create multi-region risk model
    risk_model = MultiRegionRiskModel(device_id=device_id)
    
    # Create elevation risk factor
    elevation_risk = create_elevation_risk_factor(
        dem_data=dem_data,
        geo_transform=dem_transform,
        risk_weight=0.4,
        low_elevation_threshold=200.0,
        high_elevation_threshold=800.0,
        invert=True  # Lower elevation = higher risk
    )
    risk_model.add_risk_factor(elevation_risk)
    
    # Create slope risk factor
    slope_risk = create_slope_risk_factor(
        dem_data=dem_data,
        geo_transform=dem_transform,
        risk_weight=0.3,
        max_slope=30.0
    )
    risk_model.add_risk_factor(slope_risk)
    
    # Create climate risk assessor
    climate_assessor = ClimateRiskAssessor(
        scenario=ClimateScenario.SSP2_RCP45,
        time_horizon=TimeHorizon.MEDIUM,
        device_id=device_id
    )
    
    # Create flood risk factor
    flood_risk = create_flood_risk_factor(
        dem_data=dem_data,
        geo_transform=dem_transform,
        risk_weight=0.3,
        assessor=climate_assessor
    )
    risk_model.add_risk_factor(flood_risk)
    
    return risk_model


def run_example(
    data_dir: str,
    output_dir: str,
    generate_data: bool = True,
    dem_size: int = 500,
    assets_per_region: int = 20,
    num_regions: int = 4,
    device_id: int = 0,
    seed: int = 42,
    logger: logging.Logger = None
):
    """
    Run the multi-region analysis example.
    
    Args:
        data_dir: Directory for input data
        output_dir: Directory for output files
        generate_data: Whether to generate synthetic data
        dem_size: Size of DEM to generate
        assets_per_region: Number of assets per region
        num_regions: Number of regions to create
        device_id: GPU device ID
        seed: Random seed for reproducibility
        logger: Optional logger instance
    """
    # Create logger if not provided
    if logger is None:
        logger = setup_logger(output_dir)
    
    # Set random seed
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate or load data
    if generate_data:
        logger.info(f"Generating synthetic data for {num_regions} regions with {assets_per_region} assets each")
        dem_data, dem_transform, assets_df, regions = generate_regional_data(
            data_dir=data_dir,
            dem_size=dem_size,
            assets_per_region=assets_per_region,
            num_regions=num_regions,
            seed=seed
        )
    else:
        # Load existing data
        logger.info(f"Loading data from {data_dir}")
        
        # Load DEM
        dem_path = os.path.join(data_dir, 'synthetic_dem.npy')
        if not os.path.exists(dem_path):
            logger.error(f"DEM file not found: {dem_path}")
            return
        
        dem_data = np.load(dem_path)
        
        # Create transform (this would normally be loaded from a file)
        from geospatial.dem import GeoTransform
        dem_transform = GeoTransform(0.0, 0.0, 1.0, 1.0)
        
        # Load assets
        assets_path = os.path.join(data_dir, 'regional_assets.csv')
        if not os.path.exists(assets_path):
            logger.error(f"Assets file not found: {assets_path}")
            return
        
        assets_df = pd.read_csv(assets_path)
        
        # Get unique regions from assets
        region_names = assets_df['region'].unique()
        
        # We have to recreate the regions
        # This is simplified and would normally involve loading region definitions
        bounds = {
            "min_x": dem_transform.origin_x,
            "min_y": dem_transform.origin_y,
            "max_x": dem_transform.origin_x + dem_transform.pixel_width * dem_data.shape[1],
            "max_y": dem_transform.origin_y + dem_transform.pixel_height * dem_data.shape[0]
        }
        
        # Create simplified regions
        regions = []
        grid_size = int(np.sqrt(len(region_names)))
        width = bounds["max_x"] - bounds["min_x"]
        height = bounds["max_y"] - bounds["min_y"]
        region_width = width / grid_size
        region_height = height / grid_size
        
        for i, name in enumerate(region_names):
            row = i // grid_size
            col = i % grid_size
            
            region_bounds = {
                "min_x": bounds["min_x"] + col * region_width,
                "min_y": bounds["min_y"] + row * region_height,
                "max_x": bounds["min_x"] + (col + 1) * region_width,
                "max_y": bounds["min_y"] + (row + 1) * region_height
            }
            
            region = RegionDefinition(name=name, bounds=region_bounds)
            regions.append(region)
    
    # Create regional portfolio
    logger.info("Creating regional portfolio")
    regional_portfolio = RegionalPortfolio(device_id=device_id)
    
    # Add regions
    for region in regions:
        regional_portfolio.add_region(region)
    
    # Add assets
    regional_portfolio.add_assets_from_dataframe(
        assets_df=assets_df,
        id_col='id',
        name_col='name',
        value_col='value',
        x_col='x',
        y_col='y',
        metadata_cols=['sector']
    )
    
    # Create risk model
    logger.info("Creating risk model")
    risk_model = create_risk_model(
        dem_data=dem_data,
        dem_transform=dem_transform,
        device_id=device_id
    )
    
    # Create models for each region
    for region in regions:
        risk_model.create_region_risk_model(region)
    
    # Perform multi-region analysis
    logger.info("Performing multi-region analysis")
    analysis_results = perform_multi_region_analysis(
        regional_portfolio=regional_portfolio,
        risk_model=risk_model,
        output_dir=os.path.join(output_dir, 'visualizations'),
        create_visualizations=True,
        logger=logger
    )
    
    # Print summary of results
    logger.info("Analysis complete. Summary:")
    
    # Region statistics
    stats = analysis_results["analysis"]["statistics"]
    logger.info("Region statistics:")
    for region, region_stats in stats.items():
        logger.info(f"  {region}: {region_stats['count']} assets, " +
                   f"mean risk: {region_stats['mean']:.3f}, " +
                   f"weighted risk: {region_stats['value_weighted_risk']:.3f}")
    
    # Diversification benefit
    div_benefit = analysis_results["analysis"]["diversification_benefit"]
    logger.info(f"Diversification benefit: {div_benefit*100:.2f}%")
    
    # Recommended allocation
    allocation = analysis_results["analysis"]["recommended_allocation"]
    logger.info("Recommended allocation:")
    for region, alloc in allocation.items():
        logger.info(f"  {region}: {alloc*100:.1f}%")
    
    # High risk assets
    high_risk = analysis_results["analysis"]["high_risk_assets"]
    logger.info("High risk assets (top 3 per region):")
    for region, assets in high_risk.items():
        logger.info(f"  {region}:")
        for i, asset in enumerate(assets[:3]):
            logger.info(f"    {asset['asset_id']}: {asset['risk_score']:.3f}")
    
    logger.info(f"Visualizations saved to {os.path.join(output_dir, 'visualizations')}")
    
    # Save analysis results
    import json
    
    # Convert to serializable format
    serializable_results = {
        "statistics": analysis_results["analysis"]["statistics"],
        "diversification_benefit": analysis_results["analysis"]["diversification_benefit"],
        "recommended_allocation": analysis_results["analysis"]["recommended_allocation"],
        "high_risk_assets": analysis_results["analysis"]["high_risk_assets"]
    }
    
    # Save to file
    results_path = os.path.join(output_dir, 'multiregion_analysis_results.json')
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Analysis results saved to {results_path}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Multi-region Geospatial Financial Analysis Example")
    parser.add_argument("--data-dir", default="data/geo_financial/multiregion", help="Directory for input data")
    parser.add_argument("--output-dir", default="results/geo_financial/multiregion", help="Directory for output files")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data")
    parser.add_argument("--dem-size", type=int, default=500, help="Size of DEM to generate")
    parser.add_argument("--assets-per-region", type=int, default=20, help="Number of assets per region")
    parser.add_argument("--num-regions", type=int, default=4, help="Number of regions to create")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger(args.output_dir)
    
    # Run example
    run_example(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        generate_data=args.generate_data,
        dem_size=args.dem_size,
        assets_per_region=args.assets_per_region,
        num_regions=args.num_regions,
        device_id=args.device_id,
        seed=args.seed,
        logger=logger
    )


if __name__ == "__main__":
    main()