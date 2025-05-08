#!/usr/bin/env python3
"""
Real-time Geospatial Risk Monitoring Example

This example demonstrates how to use the real-time data streaming and risk
analysis capabilities for geospatial financial risk monitoring.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import argparse
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
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
from src.integrations.geo_financial.realtime_data import (
    RealTimeStreamManager,
    RealTimeRiskAnalyzer,
    create_market_data_stream,
    create_geospatial_event_stream
)


def setup_logger(output_dir: str, name: str) -> logging.Logger:
    """
    Configure and return a logger.
    
    Args:
        output_dir: Directory for log files
        name: Logger name
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
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
    log_file = os.path.join(output_dir, f'{name.lower()}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def generate_test_data(data_dir: str, num_assets: int, dem_size: int) -> Tuple[np.ndarray, Any, pd.DataFrame]:
    """
    Generate synthetic test data.
    
    Args:
        data_dir: Directory for output files
        num_assets: Number of assets to generate
        dem_size: Size of DEM to generate
    
    Returns:
        Tuple of (dem_data, dem_transform, assets_df)
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate DEM
    dem_path = os.path.join(data_dir, 'synthetic_dem.npy')
    dem_data, dem_transform = generate_dem(
        size=dem_size,
        roughness=0.5,
        min_height=0.0,
        max_height=1000.0,
        output_path=dem_path
    )
    
    # Generate assets
    assets_path = os.path.join(data_dir, 'synthetic_assets.csv')
    assets_df = generate_assets(
        num_assets=num_assets,
        dem_data=dem_data,
        dem_transform=dem_transform,
        sectors=['technology', 'energy', 'financial', 'healthcare', 'consumer'],
        output_path=assets_path
    )
    
    # Generate returns
    returns_path = os.path.join(data_dir, 'synthetic_returns.csv')
    returns_df = generate_returns(
        assets_df=assets_df,
        days=252,
        daily_volatility=0.015,
        sector_correlation=0.7,
        output_path=returns_path
    )
    
    # Save DEM transform
    transform_path = os.path.join(data_dir, 'dem_transform.json')
    with open(transform_path, 'w') as f:
        json.dump({
            'origin_x': dem_transform.origin_x,
            'origin_y': dem_transform.origin_y,
            'pixel_width': dem_transform.pixel_width,
            'pixel_height': dem_transform.pixel_height
        }, f)
    
    return dem_data, dem_transform, assets_df


def create_portfolio(assets_df: pd.DataFrame, device_id: int) -> GeospatialPortfolio:
    """
    Create a geospatial portfolio from asset data.
    
    Args:
        assets_df: DataFrame with asset data
        device_id: GPU device ID
    
    Returns:
        GeospatialPortfolio instance
    """
    portfolio = GeospatialPortfolio(device_id=device_id)
    
    metadata_cols = [col for col in assets_df.columns if col not in ['id', 'name', 'value', 'x', 'y']]
    
    portfolio.add_assets_from_dataframe(
        assets_df, 'id', 'name', 'value', 'x', 'y',
        metadata_cols=metadata_cols if metadata_cols else None
    )
    
    return portfolio


def create_risk_model(dem_data: np.ndarray, dem_transform: Any, device_id: int) -> GeospatialRiskModel:
    """
    Create a geospatial risk model.
    
    Args:
        dem_data: Digital Elevation Model data
        dem_transform: GeoTransform for the DEM
        device_id: GPU device ID
    
    Returns:
        GeospatialRiskModel instance
    """
    # Create base risk model
    risk_model = GeospatialRiskModel(device_id=device_id)
    
    # Create elevation risk factor
    elevation_risk = create_elevation_risk_factor(
        dem_data=dem_data,
        geo_transform=dem_transform,
        risk_weight=0.6,
        low_elevation_threshold=200.0,
        high_elevation_threshold=800.0,
        invert=True
    )
    risk_model.add_risk_factor(elevation_risk)
    
    # Create slope risk factor
    slope_risk = create_slope_risk_factor(
        dem_data=dem_data,
        geo_transform=dem_transform,
        risk_weight=0.4,
        max_slope=30.0
    )
    risk_model.add_risk_factor(slope_risk)
    
    # Create climate risk assessor
    climate_assessor = ClimateRiskAssessor(
        scenario=ClimateScenario.SSP2_RCP45,
        time_horizon=TimeHorizon.MEDIUM,
        device_id=device_id
    )
    
    # Create climate risk factors
    flood_risk = create_flood_risk_factor(
        dem_data=dem_data,
        geo_transform=dem_transform,
        risk_weight=0.4,
        assessor=climate_assessor
    )
    risk_model.add_risk_factor(flood_risk)
    
    return risk_model


def risk_update_callback(data: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Callback function for risk updates.
    
    Args:
        data: Risk update data
        logger: Logger instance
    """
    timestamp = data.get("timestamp")
    risk_scores = data.get("risk_scores", {})
    
    if not risk_scores:
        return
    
    # Get timestamp as string
    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    # Calculate average risk
    avg_risk = sum(risk_scores.values()) / len(risk_scores) if risk_scores else 0
    
    # Log the update
    logger.info(f"Risk update at {timestamp_str}: Avg risk = {avg_risk:.3f}, {len(risk_scores)} assets")
    
    # Identify high-risk assets (risk > 0.7)
    high_risk_assets = {asset_id: score for asset_id, score in risk_scores.items() if score > 0.7}
    
    if high_risk_assets:
        logger.warning(f"High risk assets: {len(high_risk_assets)} assets with risk > 0.7")
        for asset_id, score in sorted(high_risk_assets.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.warning(f"  Asset {asset_id}: Risk = {score:.3f}")


def market_data_callback(data: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Callback function for market data updates.
    
    Args:
        data: Market data
        logger: Logger instance
    """
    if 'timestamp' not in data or 'assets' not in data:
        return
    
    # Get timestamp as string
    timestamp_str = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    
    # Log basic update
    logger.info(f"Market update at {timestamp_str}: {len(data['assets'])} assets")
    
    # Check for large price changes (> 2%)
    large_changes = {}
    for symbol, asset_data in data['assets'].items():
        if 'percent_change' in asset_data and abs(asset_data['percent_change']) > 0.02:
            large_changes[symbol] = asset_data['percent_change']
    
    # Log large changes
    if large_changes:
        logger.warning(f"Large market movements detected ({len(large_changes)} assets)")
        for symbol, change in sorted(large_changes.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "up" if change > 0 else "down"
            logger.warning(f"  {symbol}: {direction} {abs(change)*100:.2f}%")


def geospatial_event_callback(data: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Callback function for geospatial event updates.
    
    Args:
        data: Geospatial event data
        logger: Logger instance
    """
    if 'timestamp' not in data or 'events' not in data:
        return
    
    # Get timestamp as string
    timestamp_str = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    
    # Get events
    events = data['events']
    
    # Log basic update
    logger.info(f"Geospatial update at {timestamp_str}: {len(events)} active events")
    
    # Check for new event
    new_event = data.get('new_event')
    if new_event:
        event_type = new_event.get('type', 'unknown')
        severity = new_event.get('severity', 0)
        location = new_event.get('location', {})
        x, y = location.get('x', 0), location.get('y', 0)
        
        logger.warning(f"New {event_type} event detected at ({x:.1f}, {y:.1f}) with severity {severity:.2f}")


def save_visualizations(
    portfolio: GeospatialPortfolio,
    risk_analyzer: RealTimeRiskAnalyzer,
    dem_data: np.ndarray,
    dem_transform: Any,
    output_dir: str,
    logger: logging.Logger
) -> None:
    """
    Save risk visualizations to disk.
    
    Args:
        portfolio: GeospatialPortfolio instance
        risk_analyzer: RealTimeRiskAnalyzer instance
        dem_data: Digital Elevation Model data
        dem_transform: GeoTransform for the DEM
        output_dir: Output directory
        logger: Logger instance
    """
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create visualizer
    visualizer = GeoFinancialVisualizer()
    
    try:
        # Get latest risk scores
        risk_scores = risk_analyzer.latest_risk_scores
        
        if not risk_scores:
            logger.warning("No risk scores available for visualization")
            return
        
        # Create risk map
        if dem_data is not None:
            # Create risk map with assets
            risk_map_fig = visualizer.plot_portfolio_on_risk_map(
                risk_map=dem_data,
                geo_transform=dem_transform,
                portfolio=portfolio,
                title=f"Geospatial Risk Map ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
            )
            
            # Save risk map
            risk_map_path = os.path.join(output_dir, f"risk_map_{timestamp_str}.png")
            risk_map_fig.savefig(risk_map_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved risk map to {risk_map_path}")
        
        # Create risk history plot
        plt.figure(figsize=(12, 8))
        
        # Plot risk trend for each asset
        for asset_id, history in risk_analyzer.risk_score_history.items():
            if not history:
                continue
            
            # Get asset name
            asset_name = next(
                (a['name'] for a in portfolio.assets if a['id'] == asset_id),
                asset_id
            )
            
            # Extract data
            timestamps = [datetime.fromtimestamp(ts) for ts, _ in history]
            scores = [score for _, score in history]
            
            # Plot line
            plt.plot(timestamps, scores, label=asset_name)
        
        # Customize plot
        plt.title(f"Risk Score Trend ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        plt.xlabel("Time")
        plt.ylabel("Risk Score")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add legend if we have many assets
        if len(risk_analyzer.risk_score_history) > 10:
            # Create a legend for only the highest risk assets
            avg_risks = {}
            for asset_id, history in risk_analyzer.risk_score_history.items():
                if history:
                    avg_risks[asset_id] = sum(score for _, score in history) / len(history)
            
            # Get top 5 highest risk assets
            top_assets = sorted(avg_risks.items(), key=lambda x: x[1], reverse=True)[:5]
            top_asset_ids = [asset_id for asset_id, _ in top_assets]
            
            # Filter handles and labels
            handles, labels = plt.gca().get_legend_handles_labels()
            filtered_handles = []
            filtered_labels = []
            
            for i, label in enumerate(labels):
                asset_id = next((a['id'] for a in portfolio.assets if a['name'] == label), label)
                if asset_id in top_asset_ids:
                    filtered_handles.append(handles[i])
                    filtered_labels.append(label)
            
            plt.legend(filtered_handles, filtered_labels, loc='best')
        else:
            plt.legend(loc='best')
        
        # Save risk trend plot
        risk_trend_path = os.path.join(output_dir, f"risk_trends_{timestamp_str}.png")
        plt.savefig(risk_trend_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved risk trends to {risk_trend_path}")
        
        # Generate sector summary
        sector_risks = {}
        sector_values = {}
        total_value = 0
        
        for asset in portfolio.assets:
            asset_id = asset['id']
            
            # Skip if no risk score
            if asset_id not in risk_scores:
                continue
            
            # Get sector
            sector = asset.get('metadata', {}).get('sector', 'Unknown')
            
            # Get risk score and value
            risk_score = risk_scores[asset_id]
            value = asset['value']
            
            # Update sector stats
            if sector not in sector_risks:
                sector_risks[sector] = []
                sector_values[sector] = 0
            
            sector_risks[sector].append(risk_score)
            sector_values[sector] += value
            total_value += value
        
        # Create sector risk plot
        if sector_risks:
            plt.figure(figsize=(10, 6))
            
            # Prepare data
            sectors = list(sector_risks.keys())
            avg_risks = [sum(risks) / len(risks) for risks in sector_risks.values()]
            value_percentages = [
                (sector_values[sector] / total_value) * 100 if total_value > 0 else 0
                for sector in sectors
            ]
            
            # Create color map
            colors = plt.cm.RdYlGn_r(np.array(avg_risks))
            
            # Create bar chart
            bars = plt.bar(sectors, avg_risks, color=colors)
            
            # Add value percentage on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{value_percentages[i]:.1f}%",
                    ha='center',
                    rotation=0
                )
            
            # Customize plot
            plt.title(f"Sector Risk Analysis ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            plt.xlabel("Sector")
            plt.ylabel("Average Risk Score")
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = plt.colorbar(sm)
            cbar.set_label("Risk Level")
            
            # Save sector risk plot
            sector_risk_path = os.path.join(output_dir, f"sector_risks_{timestamp_str}.png")
            plt.savefig(sector_risk_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved sector risk analysis to {sector_risk_path}")
        
    except Exception as e:
        logger.error(f"Error saving visualizations: {e}")


def run_monitoring(
    data_dir: str,
    output_dir: str,
    duration: int,
    device_id: int,
    generate_data: bool,
    num_assets: int,
    dem_size: int,
    visualization_interval: int,
    market_update_interval: int,
    geo_update_interval: int,
    risk_update_interval: int
) -> None:
    """
    Run real-time risk monitoring for the specified duration.
    
    Args:
        data_dir: Directory for input data
        output_dir: Directory for output files
        duration: Monitoring duration in seconds
        device_id: GPU device ID
        generate_data: Whether to generate synthetic data
        num_assets: Number of assets to generate
        dem_size: Size of DEM to generate
        visualization_interval: Interval to save visualizations in seconds
        market_update_interval: Market data update interval in seconds
        geo_update_interval: Geospatial event update interval in seconds
        risk_update_interval: Risk update interval in seconds
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up loggers
    logger = setup_logger(output_dir, "RealTimeMonitoring")
    market_logger = setup_logger(output_dir, "MarketData")
    geo_logger = setup_logger(output_dir, "GeospatialEvents")
    risk_logger = setup_logger(output_dir, "RiskAnalysis")
    
    logger.info(f"Starting real-time monitoring for {duration} seconds")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load or generate data
        if generate_data:
            logger.info(f"Generating synthetic data: {num_assets} assets, {dem_size}x{dem_size} DEM")
            dem_data, dem_transform, assets_df = generate_test_data(data_dir, num_assets, dem_size)
        else:
            logger.info(f"Loading data from {data_dir}")
            
            # Load DEM data
            dem_path = os.path.join(data_dir, 'synthetic_dem.npy')
            if not os.path.exists(dem_path):
                logger.error(f"DEM file not found: {dem_path}")
                return
            
            dem_data = np.load(dem_path)
            
            # Load DEM transform
            transform_path = os.path.join(data_dir, 'dem_transform.json')
            if not os.path.exists(transform_path):
                logger.error(f"Transform file not found: {transform_path}")
                return
            
            with open(transform_path, 'r') as f:
                transform_data = json.load(f)
            
            from geospatial.dem import GeoTransform
            dem_transform = GeoTransform(
                transform_data['origin_x'],
                transform_data['origin_y'],
                transform_data['pixel_width'],
                transform_data['pixel_height']
            )
            
            # Load asset data
            assets_path = os.path.join(data_dir, 'synthetic_assets.csv')
            if not os.path.exists(assets_path):
                logger.error(f"Assets file not found: {assets_path}")
                return
            
            loader = AssetLocationDataLoader()
            assets_df = loader.load_asset_csv(
                file_path=assets_path,
                id_col='id',
                name_col='name',
                value_col='value',
                x_col='x',
                y_col='y'
            )
        
        # Create portfolio and risk model
        logger.info("Creating portfolio and risk model")
        portfolio = create_portfolio(assets_df, device_id)
        risk_model = create_risk_model(dem_data, dem_transform, device_id)
        
        # Set up real-time components
        logger.info("Setting up real-time components")
        
        # Create stream manager
        stream_manager = RealTimeStreamManager(device_id=device_id)
        
        # Create market data stream
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        # Extend symbols if we have more assets
        while len(symbols) < min(20, len(portfolio.assets)):
            symbols.append(f"STOCK{len(symbols)}")
        
        market_stream = create_market_data_stream(
            symbols=symbols,
            interval=market_update_interval
        )
        stream_manager.add_stream(market_stream)
        
        # Register custom callback
        market_stream.register_callback(
            lambda data: market_data_callback(data, market_logger)
        )
        
        # Create geospatial event stream
        event_stream = create_geospatial_event_stream(
            event_types=["flood", "wildfire", "earthquake", "storm", "heatwave", "drought"],
            geographic_bounds={
                "min_x": dem_transform.origin_x,
                "min_y": dem_transform.origin_y,
                "max_x": dem_transform.origin_x + dem_transform.pixel_width * dem_data.shape[1],
                "max_y": dem_transform.origin_y + dem_transform.pixel_height * dem_data.shape[0]
            },
            interval=geo_update_interval
        )
        stream_manager.add_stream(event_stream)
        
        # Register custom callback
        event_stream.register_callback(
            lambda data: geospatial_event_callback(data, geo_logger)
        )
        
        # Create real-time risk analyzer
        risk_analyzer = RealTimeRiskAnalyzer(
            portfolio=portfolio,
            risk_model=risk_model,
            stream_manager=stream_manager,
            update_interval=risk_update_interval,
            device_id=device_id
        )
        
        # Generate a mapping between portfolio assets and market symbols
        asset_market_mapping = {}
        for i, asset in enumerate(portfolio.assets):
            if i < len(symbols):
                asset_market_mapping[asset["id"]] = symbols[i]
        
        risk_analyzer.set_asset_market_mapping(asset_market_mapping)
        
        # Register custom callback
        risk_analyzer.register_risk_update_callback(
            lambda data: risk_update_callback(data, risk_logger)
        )
        
        # Start real-time components
        logger.info("Starting data streams and risk analyzer")
        stream_manager.start()
        risk_analyzer.start()
        
        # Main monitoring loop
        start_time = time.time()
        next_visualization_time = start_time + visualization_interval
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # Check if it's time to save visualizations
                if current_time >= next_visualization_time:
                    logger.info("Saving visualizations")
                    save_visualizations(
                        portfolio=portfolio,
                        risk_analyzer=risk_analyzer,
                        dem_data=dem_data,
                        dem_transform=dem_transform,
                        output_dir=output_dir,
                        logger=logger
                    )
                    next_visualization_time = current_time + visualization_interval
                
                # Sleep briefly
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        
        finally:
            # Stop real-time components
            logger.info("Stopping risk analyzer and data streams")
            risk_analyzer.stop()
            stream_manager.stop()
            
            # Save final visualizations
            logger.info("Saving final visualizations")
            save_visualizations(
                portfolio=portfolio,
                risk_analyzer=risk_analyzer,
                dem_data=dem_data,
                dem_transform=dem_transform,
                output_dir=output_dir,
                logger=logger
            )
    
    except Exception as e:
        logger.error(f"Error in monitoring: {e}", exc_info=True)
    
    logger.info("Monitoring complete")


def main():
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description="Real-time Geospatial Financial Risk Monitoring")
    parser.add_argument("--data-dir", default="data/geo_financial", help="Directory for input data")
    parser.add_argument("--output-dir", default="results/geo_financial/realtime_monitor", help="Directory for output files")
    parser.add_argument("--duration", type=int, default=900, help="Monitoring duration in seconds")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data")
    parser.add_argument("--num-assets", type=int, default=20, help="Number of assets to generate")
    parser.add_argument("--dem-size", type=int, default=500, help="Size of DEM to generate")
    parser.add_argument("--visualization-interval", type=int, default=300, help="Interval to save visualizations in seconds")
    parser.add_argument("--market-update-interval", type=int, default=60, help="Market data update interval in seconds")
    parser.add_argument("--geo-update-interval", type=int, default=300, help="Geospatial event update interval in seconds")
    parser.add_argument("--risk-update-interval", type=int, default=30, help="Risk update interval in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Run monitoring
    run_monitoring(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        duration=args.duration,
        device_id=args.device_id,
        generate_data=args.generate_data,
        num_assets=args.num_assets,
        dem_size=args.dem_size,
        visualization_interval=args.visualization_interval,
        market_update_interval=args.market_update_interval,
        geo_update_interval=args.geo_update_interval,
        risk_update_interval=args.risk_update_interval
    )


if __name__ == "__main__":
    main()