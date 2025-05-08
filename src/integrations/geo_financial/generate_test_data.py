#!/usr/bin/env python3
"""
Generate Test Data for Geospatial Financial Integration

This script generates synthetic test data for the geospatial financial integration,
including digital elevation models (DEMs), asset locations, and financial returns.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(project_root)

try:
    from src.geospatial.python.geospatial import dem as geospatial_dem
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False
    print("Warning: Geospatial module not available, some features will be limited")


def generate_dem(
    output_dir: str,
    size: int = 500,
    save_plot: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic digital elevation model (DEM).
    
    Args:
        output_dir: Directory to save the DEM data
        size: Size of the DEM (pixels)
        save_plot: Whether to save a visualization of the DEM
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (elevation_data, x_coordinates, y_coordinates)
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create coordinate grid
    x = np.linspace(0, 10, size)
    y = np.linspace(0, 10, size)
    X, Y = np.meshgrid(x, y)
    
    # Create a synthetic terrain with hills and valleys
    Z = 100 + 50 * np.sin(X * 0.5) * np.cos(Y * 0.5) + 20 * np.sin(X * 2) * np.sin(Y * 2)
    
    # Add a river valley
    river = 80 * np.exp(-((X - 5) ** 2) / 0.5)
    Z -= river
    
    # Save as NumPy array
    np.save(os.path.join(output_dir, 'synthetic_dem.npy'), Z)
    
    # Save as TIFF if geospatial module is available
    if HAS_GEOSPATIAL:
        try:
            # Create GeoTransform
            geo_transform = geospatial_dem.GeoTransform([0, 0.02, 0, 10, 0, -0.02])
            
            # Save as GeoTIFF using GDAL (if available)
            try:
                import gdal
                driver = gdal.GetDriverByName('GTiff')
                outfile = os.path.join(output_dir, 'synthetic_dem.tif')
                dataset = driver.Create(outfile, size, size, 1, gdal.GDT_Float32)
                dataset.SetGeoTransform(geo_transform.to_gdal_geotransform())
                band = dataset.GetRasterBand(1)
                band.WriteArray(Z)
                dataset = None  # Close the file
                print(f"Saved DEM as GeoTIFF: {outfile}")
            except (ImportError, ModuleNotFoundError):
                print("GDAL not available, skipping GeoTIFF export")
        except Exception as e:
            print(f"Error saving GeoTIFF: {e}")
    
    # Save visualization
    if save_plot:
        plt.figure(figsize=(10, 8))
        plt.imshow(Z, cmap='terrain')
        plt.colorbar(label='Elevation (m)')
        plt.title('Synthetic Digital Elevation Model')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, 'synthetic_dem.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return Z, x, y


def generate_assets(
    output_dir: str,
    dem_data: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    num_assets: int = 20,
    save_plot: bool = True,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic asset location data.
    
    Args:
        output_dir: Directory to save the asset data
        dem_data: DEM elevation data
        x_coords: X coordinates for the DEM
        y_coords: Y coordinates for the DEM
        num_assets: Number of assets to generate
        save_plot: Whether to save a visualization of the assets
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing asset data
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Define sectors with different characteristics
    sectors = [
        {"name": "Real Estate", "min_value": 5000, "max_value": 20000, "elevation_preference": "high"},
        {"name": "Infrastructure", "min_value": 10000, "max_value": 50000, "elevation_preference": "low"},
        {"name": "Agriculture", "min_value": 2000, "max_value": 10000, "elevation_preference": "low"},
        {"name": "Energy", "min_value": 8000, "max_value": 30000, "elevation_preference": "mixed"},
        {"name": "Industrial", "min_value": 5000, "max_value": 15000, "elevation_preference": "mixed"}
    ]
    
    # Create list to hold asset data
    assets = []
    
    # Generate assets for each sector
    assets_per_sector = num_assets // len(sectors)
    extra_assets = num_assets % len(sectors)
    
    asset_count = 0
    for i, sector in enumerate(sectors):
        # Determine number of assets for this sector
        sector_assets = assets_per_sector + (1 if i < extra_assets else 0)
        
        # Generate assets
        for j in range(sector_assets):
            # Select location based on elevation preference
            if sector["elevation_preference"] == "high":
                # Prefer higher elevations
                x_idx = int(np.random.triangular(0, dem_data.shape[1] * 0.8, dem_data.shape[1] - 1))
                y_idx = int(np.random.triangular(0, dem_data.shape[0] * 0.8, dem_data.shape[0] - 1))
            elif sector["elevation_preference"] == "low":
                # Prefer lower elevations
                x_idx = int(np.random.triangular(0, dem_data.shape[1] * 0.2, dem_data.shape[1] - 1))
                y_idx = int(np.random.triangular(0, dem_data.shape[0] * 0.2, dem_data.shape[0] - 1))
            else:
                # Mixed preference
                x_idx = int(np.random.uniform(0, dem_data.shape[1] - 1))
                y_idx = int(np.random.uniform(0, dem_data.shape[0] - 1))
            
            # Convert indices to coordinates
            x = x_coords[x_idx]
            y = y_coords[y_idx]
            
            # Generate random value within sector range
            value = np.random.uniform(sector["min_value"], sector["max_value"])
            
            # Add asset to list
            assets.append({
                "id": f"ASSET_{asset_count:03d}",
                "name": f"{sector['name']} Asset {j+1}",
                "value": round(value, 2),
                "x": x,
                "y": y,
                "sector": sector["name"]
            })
            
            asset_count += 1
    
    # Convert to DataFrame
    assets_df = pd.DataFrame(assets)
    
    # Save to CSV
    assets_df.to_csv(os.path.join(output_dir, 'synthetic_assets.csv'), index=False)
    
    # Save visualization
    if save_plot:
        plt.figure(figsize=(12, 10))
        
        # Plot DEM
        plt.imshow(dem_data, cmap='terrain', extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
        
        # Plot assets
        sector_colors = {
            "Real Estate": "red",
            "Infrastructure": "blue",
            "Agriculture": "green",
            "Energy": "orange",
            "Industrial": "purple"
        }
        
        for sector_name, sector_df in assets_df.groupby('sector'):
            plt.scatter(
                sector_df['x'], 
                sector_df['y'], 
                s=sector_df['value'] / 500,  # Scale marker size by value
                c=sector_colors.get(sector_name, "gray"),
                label=sector_name,
                alpha=0.7,
                edgecolors='white',
                linewidths=1
            )
        
        plt.colorbar(label='Elevation (m)')
        plt.title('Asset Locations on Terrain')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'synthetic_assets.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return assets_df


def generate_returns(
    output_dir: str,
    assets_df: pd.DataFrame,
    days: int = 252,
    start_date: Optional[str] = None,
    correlation: float = 0.7,
    save_plot: bool = True,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic financial returns data.
    
    Args:
        output_dir: Directory to save the returns data
        assets_df: DataFrame containing asset data
        days: Number of days to generate returns for
        start_date: Start date for returns (YYYY-MM-DD)
        correlation: Base correlation between asset returns
        save_plot: Whether to save a visualization of the returns
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing returns data
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Get asset IDs
    asset_ids = assets_df['id'].tolist()
    
    # Set start date
    if start_date:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    else:
        start_dt = datetime.now() - timedelta(days=days)
    
    # Create sector correlation matrix
    sectors = assets_df['sector'].unique()
    num_sectors = len(sectors)
    sector_corr = np.eye(num_sectors)
    
    # Fill correlation matrix
    for i in range(num_sectors):
        for j in range(i+1, num_sectors):
            # Generate random correlation between 0.3 and 0.9
            sector_corr[i, j] = sector_corr[j, i] = np.random.uniform(0.3, 0.9)
    
    # Map sectors to indices
    sector_to_idx = {sector: i for i, sector in enumerate(sectors)}
    
    # Create correlated returns
    # Base parameters for returns
    base_mean = 0.0005  # 0.05% daily return
    base_std = 0.01     # 1% standard deviation
    
    # Generate correlated returns
    num_assets = len(asset_ids)
    corr_matrix = np.eye(num_assets)
    
    # Fill correlation matrix based on sector correlations
    for i in range(num_assets):
        for j in range(i+1, num_assets):
            # Get sector indices
            sector_i = sector_to_idx[assets_df.iloc[i]['sector']]
            sector_j = sector_to_idx[assets_df.iloc[j]['sector']]
            
            # Use sector correlation
            corr_matrix[i, j] = corr_matrix[j, i] = sector_corr[sector_i, sector_j]
    
    # Ensure matrix is positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals = np.maximum(eigvals, 1e-10)
    corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Compute Cholesky decomposition
    chol = np.linalg.cholesky(corr_matrix)
    
    # Generate uncorrelated returns
    uncorrelated = np.random.normal(0, 1, (days, num_assets))
    
    # Apply correlation
    correlated = uncorrelated @ chol.T
    
    # Apply mean and std
    means = np.array([base_mean + np.random.uniform(-0.0002, 0.0002) for _ in range(num_assets)])
    stds = np.array([base_std + np.random.uniform(-0.002, 0.002) for _ in range(num_assets)])
    
    asset_returns = correlated * stds + means
    
    # Create DataFrame
    returns_data = []
    for day in range(days):
        date = start_dt + timedelta(days=day)
        for i, asset_id in enumerate(asset_ids):
            returns_data.append({
                'asset_id': asset_id,
                'date': date.strftime('%Y-%m-%d'),
                'return': asset_returns[day, i]
            })
    
    returns_df = pd.DataFrame(returns_data)
    
    # Save to CSV
    returns_df.to_csv(os.path.join(output_dir, 'synthetic_returns.csv'), index=False)
    
    # Save visualization
    if save_plot:
        # Create a pivot table of returns
        pivot_returns = returns_df.pivot(index='date', columns='asset_id', values='return')
        
        # Calculate cumulative returns
        cumulative_returns = (1 + pivot_returns).cumprod()
        
        # Plot cumulative returns by sector
        plt.figure(figsize=(12, 8))
        
        # Get sector for each asset
        asset_sectors = {row['id']: row['sector'] for _, row in assets_df.iterrows()}
        
        # Group by sector
        for sector in sectors:
            sector_assets = [asset_id for asset_id in asset_ids if asset_sectors[asset_id] == sector]
            for asset_id in sector_assets:
                plt.plot(cumulative_returns.index, cumulative_returns[asset_id], 
                        alpha=0.3, linewidth=1)
        
        # Plot sector averages
        for sector in sectors:
            sector_assets = [asset_id for asset_id in asset_ids if asset_sectors[asset_id] == sector]
            sector_avg = cumulative_returns[sector_assets].mean(axis=1)
            plt.plot(cumulative_returns.index, sector_avg, 
                    label=sector, linewidth=2)
        
        plt.title('Cumulative Returns by Sector')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'synthetic_returns.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot return correlation matrix
        plt.figure(figsize=(10, 8))
        corr = pivot_returns.corr()
        
        # Sort by sector
        sector_order = []
        for sector in sectors:
            sector_assets = [asset_id for asset_id in asset_ids if asset_sectors[asset_id] == sector]
            sector_order.extend(sector_assets)
        
        corr = corr.loc[sector_order, sector_order]
        
        plt.imshow(corr, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title('Return Correlation Matrix')
        plt.xticks([])
        plt.yticks([])
        
        # Add sector labels
        sector_positions = []
        current_pos = 0
        for sector in sectors:
            sector_assets = [asset_id for asset_id in asset_ids if asset_sectors[asset_id] == sector]
            sector_size = len(sector_assets)
            sector_positions.append((sector, current_pos + sector_size / 2))
            current_pos += sector_size
        
        # Add sector labels to the left
        for sector, pos in sector_positions:
            plt.text(-0.5, pos, sector, ha='right', va='center', fontsize=10)
        
        plt.savefig(os.path.join(output_dir, 'return_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    return returns_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate synthetic test data for geospatial financial integration')
    parser.add_argument('--output-dir', type=str, default='data/geo_financial',
                      help='Output directory for generated data')
    parser.add_argument('--dem-size', type=int, default=500,
                      help='Size of the synthetic DEM (pixels)')
    parser.add_argument('--num-assets', type=int, default=20,
                      help='Number of synthetic assets to generate')
    parser.add_argument('--days', type=int, default=252,
                      help='Number of days of returns data to generate')
    parser.add_argument('--start-date', type=str, default=None,
                      help='Start date for returns data (YYYY-MM-DD)')
    parser.add_argument('--correlation', type=float, default=0.7,
                      help='Base correlation between asset returns')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--no-plots', action='store_true',
                      help='Disable saving visualization plots')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating synthetic data in {output_dir}...")
    
    # Generate DEM
    print("Generating synthetic DEM...")
    dem_data, x_coords, y_coords = generate_dem(
        output_dir=output_dir,
        size=args.dem_size,
        save_plot=not args.no_plots,
        seed=args.seed
    )
    
    # Generate asset locations
    print(f"Generating {args.num_assets} synthetic assets...")
    assets_df = generate_assets(
        output_dir=output_dir,
        dem_data=dem_data,
        x_coords=x_coords,
        y_coords=y_coords,
        num_assets=args.num_assets,
        save_plot=not args.no_plots,
        seed=args.seed + 1 if args.seed is not None else None
    )
    
    # Generate returns
    print(f"Generating {args.days} days of returns data...")
    returns_df = generate_returns(
        output_dir=output_dir,
        assets_df=assets_df,
        days=args.days,
        start_date=args.start_date,
        correlation=args.correlation,
        save_plot=not args.no_plots,
        seed=args.seed + 2 if args.seed is not None else None
    )
    
    print("Data generation complete.")
    print(f"Generated files:")
    print(f"  - DEM data: {os.path.join(output_dir, 'synthetic_dem.npy')}")
    print(f"  - Asset locations: {os.path.join(output_dir, 'synthetic_assets.csv')}")
    print(f"  - Returns data: {os.path.join(output_dir, 'synthetic_returns.csv')}")
    
    if not args.no_plots:
        print(f"  - Visualizations:")
        print(f"    - {os.path.join(output_dir, 'synthetic_dem.png')}")
        print(f"    - {os.path.join(output_dir, 'synthetic_assets.png')}")
        print(f"    - {os.path.join(output_dir, 'synthetic_returns.png')}")
        print(f"    - {os.path.join(output_dir, 'return_correlation.png')}")


if __name__ == '__main__':
    main()