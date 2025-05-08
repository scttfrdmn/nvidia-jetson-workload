#!/usr/bin/env python3
"""
Real-time Geospatial Financial Dashboard Application

This application provides an interactive dashboard for real-time monitoring and analysis
of geospatial financial risks as they evolve over time.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import threading
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(project_root)

try:
    import dash
    from dash import dcc, html, Input, Output, State, callback, ctx
    import dash_bootstrap_components as dbc
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from dash.exceptions import PreventUpdate
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
from src.integrations.geo_financial.realtime_data import (
    RealTimeStreamManager,
    RealTimeRiskAnalyzer,
    create_market_data_stream,
    create_geospatial_event_stream
)


class RealTimeDashboardApp:
    """
    Interactive real-time dashboard application for geospatial financial risk analysis.
    
    This class extends the base dashboard with real-time data streaming and
    continuous risk monitoring capabilities.
    """
    
    def __init__(self, 
                data_dir: str = 'data/geo_financial',
                output_dir: str = 'results/geo_financial/realtime_dashboard',
                device_id: int = 0,
                update_interval: int = 5):
        """
        Initialize the real-time dashboard application.
        
        Args:
            data_dir: Directory for input data files
            output_dir: Directory for output files
            device_id: GPU device ID (-1 for CPU)
            update_interval: Dashboard update interval in seconds
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device_id = device_id
        self.update_interval = update_interval
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = self._setup_logger()
        
        # Dash app
        self.app = None
        
        # Initialize components
        self.portfolio = None
        self.dem_data = None
        self.dem_transform = None
        self.risk_model = None
        self.climate_assessor = None
        
        # Real-time components
        self.stream_manager = None
        self.risk_analyzer = None
        
        # Buffers for real-time data
        self.market_data_buffer = []
        self.event_data_buffer = []
        self.risk_score_buffer = {}
        
        # Threading and synchronization
        self._data_lock = threading.Lock()
        self._last_update = datetime.now().timestamp()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger."""
        logger = logging.getLogger("RealTimeDashboardApp")
        logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Also create a file handler
        log_file = os.path.join(self.output_dir, 'dashboard.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_or_generate_data(self, 
                             generate: bool = False,
                             num_assets: int = 20,
                             dem_size: int = 500,
                             random_seed: int = 42) -> None:
        """
        Load or generate data for the dashboard.
        
        Args:
            generate: Whether to generate synthetic data
            num_assets: Number of assets to generate
            dem_size: Size of DEM to generate
            random_seed: Random seed for reproducibility
        """
        self.logger.info("Loading or generating data")
        
        # Set random seed
        np.random.seed(random_seed)
        
        if generate:
            self._generate_data(num_assets, dem_size)
        else:
            self._load_data()
    
    def _generate_data(self, num_assets: int, dem_size: int) -> None:
        """
        Generate synthetic data for the dashboard.
        
        Args:
            num_assets: Number of assets to generate
            dem_size: Size of DEM to generate
        """
        self.logger.info(f"Generating synthetic data: {num_assets} assets, {dem_size}x{dem_size} DEM")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Generate DEM
        dem_path = os.path.join(self.data_dir, 'synthetic_dem.npy')
        self.dem_data, self.dem_transform = generate_dem(
            size=dem_size,
            roughness=0.5,
            min_height=0.0,
            max_height=1000.0,
            output_path=dem_path
        )
        
        # Generate assets
        assets_path = os.path.join(self.data_dir, 'synthetic_assets.csv')
        assets_df = generate_assets(
            num_assets=num_assets,
            dem_data=self.dem_data,
            dem_transform=self.dem_transform,
            sectors=['technology', 'energy', 'financial', 'healthcare', 'consumer'],
            output_path=assets_path
        )
        
        # Generate returns
        returns_path = os.path.join(self.data_dir, 'synthetic_returns.csv')
        returns_df = generate_returns(
            assets_df=assets_df,
            days=252,
            daily_volatility=0.015,
            sector_correlation=0.7,
            output_path=returns_path
        )
        
        # Create portfolio
        self.portfolio = GeospatialPortfolio(device_id=self.device_id)
        self.portfolio.add_assets_from_dataframe(
            assets_df, 'id', 'name', 'value', 'x', 'y',
            metadata_cols=['sector']
        )
        
        # Save DEM transform
        transform_path = os.path.join(self.data_dir, 'dem_transform.json')
        with open(transform_path, 'w') as f:
            json.dump({
                'origin_x': self.dem_transform.origin_x,
                'origin_y': self.dem_transform.origin_y,
                'pixel_width': self.dem_transform.pixel_width,
                'pixel_height': self.dem_transform.pixel_height
            }, f)
        
        self.logger.info("Synthetic data generation complete")
    
    def _load_data(self) -> None:
        """Load data from files."""
        self.logger.info(f"Loading data from {self.data_dir}")
        
        try:
            # Load DEM data
            dem_path = os.path.join(self.data_dir, 'synthetic_dem.npy')
            if os.path.exists(dem_path):
                self.dem_data = np.load(dem_path)
                
                # Load DEM transform
                transform_path = os.path.join(self.data_dir, 'dem_transform.json')
                if os.path.exists(transform_path):
                    with open(transform_path, 'r') as f:
                        transform_data = json.load(f)
                    
                    from geospatial.dem import GeoTransform
                    self.dem_transform = GeoTransform(
                        transform_data['origin_x'],
                        transform_data['origin_y'],
                        transform_data['pixel_width'],
                        transform_data['pixel_height']
                    )
                else:
                    # Create default transform
                    from geospatial.dem import GeoTransform
                    self.dem_transform = GeoTransform(0.0, 0.0, 1.0, 1.0)
            else:
                self.logger.warning(f"DEM file not found: {dem_path}")
                return
            
            # Load asset data
            assets_path = os.path.join(self.data_dir, 'synthetic_assets.csv')
            if os.path.exists(assets_path):
                loader = AssetLocationDataLoader()
                assets_df = loader.load_asset_csv(
                    file_path=assets_path,
                    id_col='id',
                    name_col='name',
                    value_col='value',
                    x_col='x',
                    y_col='y'
                )
                
                # Create portfolio
                self.portfolio = GeospatialPortfolio(device_id=self.device_id)
                self.portfolio.add_assets_from_dataframe(
                    assets_df, 'id', 'name', 'value', 'x', 'y',
                    metadata_cols=['sector'] if 'sector' in assets_df.columns else None
                )
            else:
                self.logger.warning(f"Assets file not found: {assets_path}")
                return
            
            self.logger.info(f"Loaded {len(self.portfolio.assets)} assets and {self.dem_data.shape} DEM")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def create_risk_model(self) -> None:
        """Create the geospatial risk model."""
        self.logger.info("Creating risk model")
        
        # Create basic risk model
        self.risk_model = GeospatialRiskModel(device_id=self.device_id)
        
        # Create elevation risk factor
        elevation_risk = create_elevation_risk_factor(
            dem_data=self.dem_data,
            geo_transform=self.dem_transform,
            risk_weight=0.6,
            low_elevation_threshold=200.0,
            high_elevation_threshold=800.0,
            invert=True
        )
        self.risk_model.add_risk_factor(elevation_risk)
        
        # Create slope risk factor
        slope_risk = create_slope_risk_factor(
            dem_data=self.dem_data,
            geo_transform=self.dem_transform,
            risk_weight=0.4,
            max_slope=30.0
        )
        self.risk_model.add_risk_factor(slope_risk)
        
        # Create climate risk assessor
        self.climate_assessor = ClimateRiskAssessor(
            scenario=ClimateScenario.SSP2_RCP45,
            time_horizon=TimeHorizon.MEDIUM,
            device_id=self.device_id
        )
        
        # Create climate risk factors
        flood_risk = create_flood_risk_factor(
            dem_data=self.dem_data,
            geo_transform=self.dem_transform,
            risk_weight=0.4,
            assessor=self.climate_assessor
        )
        self.risk_model.add_risk_factor(flood_risk)
        
        self.logger.info("Risk model created with elevation, slope and flood risk factors")
    
    def setup_real_time_components(self,
                                 market_update_interval: int = 60,
                                 geospatial_update_interval: int = 300,
                                 risk_update_interval: int = 30) -> None:
        """
        Set up real-time data streaming and risk analysis components.
        
        Args:
            market_update_interval: Market data update interval in seconds
            geospatial_update_interval: Geospatial event update interval in seconds
            risk_update_interval: Risk assessment update interval in seconds
        """
        self.logger.info("Setting up real-time components")
        
        # Create stream manager
        self.stream_manager = RealTimeStreamManager(device_id=self.device_id)
        
        # Create market data stream
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        market_stream = create_market_data_stream(
            symbols=symbols,
            interval=market_update_interval
        )
        self.stream_manager.add_stream(market_stream)
        
        # Create geospatial event stream
        event_stream = create_geospatial_event_stream(
            event_types=["flood", "wildfire", "earthquake", "storm"],
            geographic_bounds={
                "min_x": self.dem_transform.origin_x,
                "min_y": self.dem_transform.origin_y,
                "max_x": self.dem_transform.origin_x + self.dem_transform.pixel_width * self.dem_data.shape[1],
                "max_y": self.dem_transform.origin_y + self.dem_transform.pixel_height * self.dem_data.shape[0]
            },
            interval=geospatial_update_interval
        )
        self.stream_manager.add_stream(event_stream)
        
        # Create real-time risk analyzer
        self.risk_analyzer = RealTimeRiskAnalyzer(
            portfolio=self.portfolio,
            risk_model=self.risk_model,
            stream_manager=self.stream_manager,
            update_interval=risk_update_interval,
            device_id=self.device_id
        )
        
        # Generate a mapping between portfolio assets and market symbols
        # In a real application, this would be based on actual asset identifiers
        asset_market_mapping = {}
        for i, asset in enumerate(self.portfolio.assets):
            if i < len(symbols):
                asset_market_mapping[asset["id"]] = symbols[i]
        
        self.risk_analyzer.set_asset_market_mapping(asset_market_mapping)
        
        # Register callbacks for data updates
        market_stream.register_callback(self._market_data_callback)
        event_stream.register_callback(self._geospatial_event_callback)
        self.risk_analyzer.register_risk_update_callback(self._risk_update_callback)
        
        self.logger.info("Real-time components configured")
    
    def _market_data_callback(self, data: Dict[str, Any]) -> None:
        """Handle incoming market data."""
        with self._data_lock:
            self.market_data_buffer.append(data)
            # Keep only the last 100 data points
            if len(self.market_data_buffer) > 100:
                self.market_data_buffer = self.market_data_buffer[-100:]
    
    def _geospatial_event_callback(self, data: Dict[str, Any]) -> None:
        """Handle incoming geospatial event data."""
        with self._data_lock:
            self.event_data_buffer.append(data)
            # Keep only the last 100 data points
            if len(self.event_data_buffer) > 100:
                self.event_data_buffer = self.event_data_buffer[-100:]
    
    def _risk_update_callback(self, data: Dict[str, Any]) -> None:
        """Handle risk update data."""
        with self._data_lock:
            timestamp = data.get("timestamp", datetime.now().timestamp())
            risk_scores = data.get("risk_scores", {})
            
            # Update risk score buffer
            for asset_id, score in risk_scores.items():
                if asset_id not in self.risk_score_buffer:
                    self.risk_score_buffer[asset_id] = []
                
                self.risk_score_buffer[asset_id].append((timestamp, score))
                
                # Keep only the last 100 data points
                if len(self.risk_score_buffer[asset_id]) > 100:
                    self.risk_score_buffer[asset_id] = self.risk_score_buffer[asset_id][-100:]
    
    def start_real_time_streaming(self) -> None:
        """Start real-time data streams and risk analysis."""
        if not self.stream_manager or not self.risk_analyzer:
            self.logger.error("Real-time components not configured")
            return
        
        self.logger.info("Starting real-time data streams and risk analysis")
        
        # Start stream manager
        self.stream_manager.start()
        
        # Start risk analyzer
        self.risk_analyzer.start()
    
    def stop_real_time_streaming(self) -> None:
        """Stop real-time data streams and risk analysis."""
        if self.risk_analyzer:
            self.logger.info("Stopping risk analyzer")
            self.risk_analyzer.stop()
        
        if self.stream_manager:
            self.logger.info("Stopping stream manager")
            self.stream_manager.stop()
    
    def create_dashboard(self, debug: bool = False, port: int = 8050) -> None:
        """
        Create and launch the interactive dashboard.
        
        Args:
            debug: Whether to run Dash in debug mode
            port: Port to run the dashboard on
        """
        if not HAS_DASH:
            self.logger.error("Dash not available. Cannot create dashboard.")
            print("Dash not available. Cannot create dashboard.")
            print("Install with: pip install dash dash-bootstrap-components plotly")
            return
        
        self.logger.info("Creating real-time dashboard")
        
        # Initialize Dash app
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Store app reference
        self.app = app
        
        # Define layout
        app.layout = self._create_dashboard_layout()
        
        # Define callbacks
        self._register_callbacks()
        
        # Start real-time components
        self.start_real_time_streaming()
        
        # Run the server
        self.logger.info(f"Starting dashboard server on port {port}")
        app.run_server(debug=debug, port=port)
    
    def _create_dashboard_layout(self) -> html.Div:
        """Create the dashboard layout."""
        # Create main layout
        layout = html.Div([
            # Header
            dbc.NavbarSimple(
                brand="Real-time Geospatial Financial Risk Dashboard",
                brand_href="#",
                color="primary",
                dark=True,
            ),
            
            # Main content
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H2("Real-time Risk Monitoring", className="mt-4 mb-4"),
                        html.Div(id="last-update-info", className="text-muted small mb-3"),
                    ]),
                ]),
                
                # Control row
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Controls"),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Update Interval (seconds)"),
                                        dcc.Slider(
                                            id="update-interval-slider",
                                            min=1,
                                            max=30,
                                            step=1,
                                            value=self.update_interval,
                                            marks={i: str(i) for i in range(0, 31, 5)},
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        html.Label("Climate Scenario"),
                                        dcc.Dropdown(
                                            id="climate-scenario-dropdown",
                                            options=[
                                                {"label": s.name, "value": s.value}
                                                for s in ClimateScenario
                                            ],
                                            value=ClimateScenario.SSP2_RCP45.value,
                                            clearable=False,
                                        ),
                                    ], width=6),
                                ]),
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Button(
                                            "Refresh Data",
                                            id="refresh-button",
                                            color="primary",
                                            className="mt-3",
                                        ),
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Button(
                                            "Restart Streaming",
                                            id="restart-streaming-button",
                                            color="secondary",
                                            className="mt-3",
                                        ),
                                    ], width=6),
                                ]),
                            ]),
                        ]),
                    ], width=12),
                ], className="mb-4"),
                
                # Main dashboard content
                dbc.Row([
                    # Left column: Risk map and market data
                    dbc.Col([
                        # Risk Map panel
                        dbc.Card([
                            dbc.CardHeader("Geospatial Risk Map with Assets"),
                            dbc.CardBody([
                                dcc.Graph(id="risk-map-graph", style={"height": "500px"}),
                            ]),
                        ], className="mb-4"),
                        
                        # Active Events panel
                        dbc.Card([
                            dbc.CardHeader("Active Geospatial Events"),
                            dbc.CardBody([
                                html.Div(id="active-events-table"),
                            ]),
                        ]),
                    ], width=6),
                    
                    # Right column: Risk trends and portfolio analysis
                    dbc.Col([
                        # Risk Score Trends panel
                        dbc.Card([
                            dbc.CardHeader("Risk Score Trends"),
                            dbc.CardBody([
                                dcc.Graph(id="risk-trends-graph", style={"height": "300px"}),
                            ]),
                        ], className="mb-4"),
                        
                        # Market Data panel
                        dbc.Card([
                            dbc.CardHeader("Market Data"),
                            dbc.CardBody([
                                dcc.Graph(id="market-data-graph", style={"height": "250px"}),
                            ]),
                        ], className="mb-4"),
                        
                        # Portfolio Risk Summary panel
                        dbc.Card([
                            dbc.CardHeader("Portfolio Risk Summary"),
                            dbc.CardBody([
                                html.Div(id="portfolio-risk-summary"),
                            ]),
                        ]),
                    ], width=6),
                ]),
                
                # Hidden div for storing JSON data
                html.Div(id='streaming-data-store', style={'display': 'none'}),
                
                # Interval component for periodic updates
                dcc.Interval(
                    id='streaming-update-interval',
                    interval=self.update_interval * 1000,  # in milliseconds
                    n_intervals=0
                ),
                
            ], fluid=True, className="mt-4"),
            
            # Footer
            html.Footer(
                dbc.Container([
                    html.Hr(),
                    html.P(
                        "NVIDIA Jetson Workload Project - Geospatial Financial Integration",
                        className="text-center text-muted",
                    ),
                ]),
                className="mt-5",
            ),
        ])
        
        return layout
    
    def _register_callbacks(self) -> None:
        """Register dashboard callbacks."""
        app = self.app
        
        @app.callback(
            [Output('streaming-data-store', 'children'),
             Output('last-update-info', 'children')],
            [Input('streaming-update-interval', 'n_intervals'),
             Input('refresh-button', 'n_clicks')],
            prevent_initial_call=False
        )
        def update_streaming_data(n_intervals, n_clicks):
            """Update streaming data store with latest data."""
            now = datetime.now()
            
            with self._data_lock:
                # Get latest data
                market_data = self.market_data_buffer[-1] if self.market_data_buffer else {}
                event_data = self.event_data_buffer[-1] if self.event_data_buffer else {}
                
                # Prepare risk score data
                risk_scores = {}
                for asset_id, history in self.risk_score_buffer.items():
                    if history:
                        risk_scores[asset_id] = history[-1][1]  # Latest score
                
                # Prepare data store
                data_store = {
                    'timestamp': now.timestamp(),
                    'market_data': market_data,
                    'event_data': event_data,
                    'risk_scores': risk_scores
                }
                
                # Update text showing last update time
                last_update_text = f"Last updated: {now.strftime('%Y-%m-%d %H:%M:%S')}"
                
                return json.dumps(data_store), last_update_text
        
        @app.callback(
            Output('risk-map-graph', 'figure'),
            [Input('streaming-data-store', 'children')],
            prevent_initial_call=True
        )
        def update_risk_map(json_data):
            """Update risk map with latest data."""
            if not json_data:
                raise PreventUpdate
            
            # Parse data
            data = json.loads(json_data)
            event_data = data.get('event_data', {})
            risk_scores = data.get('risk_scores', {})
            
            # Create base figure
            fig = go.Figure()
            
            # Add risk surface as heatmap if DEM data is available
            if self.dem_data is not None:
                # Create a smoother risk surface for visualization
                risk_surface = self.dem_data.copy()
                
                # Normalize to 0-1 range
                risk_surface = (risk_surface - risk_surface.min()) / (risk_surface.max() - risk_surface.min())
                
                # Add as heatmap
                fig.add_trace(go.Heatmap(
                    z=risk_surface,
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title='Risk Level'),
                    hoverinfo='none'
                ))
            
            # Add assets as scatter points
            if self.portfolio and self.portfolio.assets:
                x_coords = []
                y_coords = []
                sizes = []
                colors = []
                texts = []
                
                for asset in self.portfolio.assets:
                    asset_id = asset['id']
                    x_coords.append(asset['x'])
                    y_coords.append(asset['y'])
                    
                    # Size based on value
                    sizes.append(np.sqrt(asset['value']) / 10 + 10)
                    
                    # Color based on risk score
                    risk_score = risk_scores.get(asset_id, 0.5)
                    colors.append(risk_score)
                    
                    # Hover text
                    sector = asset.get('metadata', {}).get('sector', 'Unknown')
                    texts.append(
                        f"ID: {asset_id}<br>"
                        f"Name: {asset['name']}<br>"
                        f"Value: ${asset['value']:,.2f}<br>"
                        f"Sector: {sector}<br>"
                        f"Risk Score: {risk_score:.3f}"
                    )
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=colors,
                        colorscale='RdYlGn_r',
                        cmin=0,
                        cmax=1,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    text=texts,
                    hoverinfo='text',
                    name='Assets'
                ))
            
            # Add active events as larger markers with different shapes
            if 'events' in event_data and event_data['events']:
                events = event_data['events']
                
                # Event-specific markers and colors
                event_markers = {
                    'flood': 'circle-open',
                    'wildfire': 'hexagram',
                    'earthquake': 'diamond',
                    'storm': 'triangle-up',
                    'heatwave': 'star',
                    'drought': 'square-open'
                }
                
                event_colors = {
                    'flood': 'blue',
                    'wildfire': 'red',
                    'earthquake': 'purple',
                    'storm': 'cyan',
                    'heatwave': 'orange',
                    'drought': 'brown'
                }
                
                for event_type in set(e['type'] for e in events):
                    # Filter events by type
                    type_events = [e for e in events if e['type'] == event_type]
                    
                    if not type_events:
                        continue
                    
                    x_coords = [e['location']['x'] for e in type_events]
                    y_coords = [e['location']['y'] for e in type_events]
                    sizes = [30 + (e['severity'] * 20) for e in type_events]
                    texts = [
                        f"Type: {e['type']}<br>"
                        f"Severity: {e['severity']:.2f}<br>"
                        f"Radius: {e['radius_km']:.1f} km<br>"
                        f"Started: {datetime.fromtimestamp(e['start_time']).strftime('%Y-%m-%d %H:%M')}"
                        for e in type_events
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers',
                        marker=dict(
                            symbol=event_markers.get(event_type, 'circle-open'),
                            size=sizes,
                            color=event_colors.get(event_type, 'grey'),
                            line=dict(width=2, color='white')
                        ),
                        text=texts,
                        hoverinfo='text',
                        name=f"{event_type.capitalize()} Events"
                    ))
            
            # Update layout
            fig.update_layout(
                title='Real-time Geospatial Risk Map',
                xaxis=dict(title='X Coordinate', zeroline=False),
                yaxis=dict(title='Y Coordinate', zeroline=False),
                height=500,
                margin=dict(t=30, b=30, l=30, r=30),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.7)"
                ),
                hovermode='closest'
            )
            
            return fig
        
        @app.callback(
            Output('risk-trends-graph', 'figure'),
            [Input('streaming-data-store', 'children')],
            prevent_initial_call=True
        )
        def update_risk_trends(json_data):
            """Update risk trends graph with latest data."""
            if not json_data:
                raise PreventUpdate
            
            # Parse data
            data = json.loads(json_data)
            
            # Create figure
            fig = go.Figure()
            
            # Add risk trend lines for each asset
            with self._data_lock:
                for asset_id, history in self.risk_score_buffer.items():
                    if not history:
                        continue
                    
                    # Find asset details
                    asset_name = next(
                        (a['name'] for a in self.portfolio.assets if a['id'] == asset_id),
                        asset_id
                    )
                    
                    # Extract timestamps and scores
                    timestamps = [datetime.fromtimestamp(h[0]) for h in history]
                    scores = [h[1] for h in history]
                    
                    # Add line
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=scores,
                        mode='lines',
                        name=asset_name
                    ))
            
            # Update layout
            fig.update_layout(
                title='Risk Score Trends Over Time',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Risk Score', range=[0, 1]),
                height=300,
                margin=dict(t=30, b=30, l=30, r=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
        
        @app.callback(
            Output('market-data-graph', 'figure'),
            [Input('streaming-data-store', 'children')],
            prevent_initial_call=True
        )
        def update_market_data(json_data):
            """Update market data graph with latest data."""
            if not json_data:
                raise PreventUpdate
            
            # Parse data
            data = json.loads(json_data)
            
            # Create figure
            fig = go.Figure()
            
            # Add price trends for each asset
            with self._data_lock:
                if not self.market_data_buffer:
                    # No market data yet
                    fig.update_layout(
                        title='Market Data (No data available)',
                        height=250
                    )
                    return fig
                
                # Collect price data for each symbol
                symbol_data = {}
                
                for data_point in self.market_data_buffer:
                    if 'timestamp' not in data_point or 'assets' not in data_point:
                        continue
                    
                    timestamp = datetime.fromtimestamp(data_point['timestamp'])
                    
                    for symbol, asset_data in data_point['assets'].items():
                        if 'price' not in asset_data:
                            continue
                        
                        if symbol not in symbol_data:
                            symbol_data[symbol] = {'timestamps': [], 'prices': []}
                        
                        symbol_data[symbol]['timestamps'].append(timestamp)
                        symbol_data[symbol]['prices'].append(asset_data['price'])
                
                # Add data for each symbol
                for symbol, data_series in symbol_data.items():
                    fig.add_trace(go.Scatter(
                        x=data_series['timestamps'],
                        y=data_series['prices'],
                        mode='lines',
                        name=symbol
                    ))
            
            # Update layout
            fig.update_layout(
                title='Market Price Trends',
                xaxis=dict(title='Time'),
                yaxis=dict(title='Price ($)'),
                height=250,
                margin=dict(t=30, b=30, l=30, r=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            return fig
        
        @app.callback(
            Output('active-events-table', 'children'),
            [Input('streaming-data-store', 'children')],
            prevent_initial_call=True
        )
        def update_active_events(json_data):
            """Update active events table with latest data."""
            if not json_data:
                raise PreventUpdate
            
            # Parse data
            data = json.loads(json_data)
            event_data = data.get('event_data', {})
            
            # Check if there are any events
            if 'events' not in event_data or not event_data['events']:
                return html.Div("No active events", className="text-center text-muted my-3")
            
            # Create event table
            events = event_data['events']
            
            # Sort events by severity (descending)
            events = sorted(events, key=lambda e: e.get('severity', 0), reverse=True)
            
            # Create table rows
            rows = []
            for event in events:
                # Format start time
                start_time = datetime.fromtimestamp(event['start_time']).strftime('%Y-%m-%d %H:%M')
                
                # Determine color based on event type
                event_colors = {
                    'flood': 'primary',
                    'wildfire': 'danger',
                    'earthquake': 'secondary',
                    'storm': 'info',
                    'heatwave': 'warning',
                    'drought': 'dark'
                }
                color = event_colors.get(event['type'], 'secondary')
                
                # Create row
                rows.append(
                    dbc.Row([
                        dbc.Col(html.Span(
                            event['type'].capitalize(),
                            className=f"badge bg-{color} me-1"
                        ), width=3),
                        dbc.Col(f"{event['severity']:.2f}", width=2),
                        dbc.Col(start_time, width=4),
                        dbc.Col(f"{event['radius_km']:.1f} km", width=3),
                    ], className="mb-1")
                )
            
            # Create header
            header = dbc.Row([
                dbc.Col(html.Strong("Type"), width=3),
                dbc.Col(html.Strong("Severity"), width=2),
                dbc.Col(html.Strong("Start Time"), width=4),
                dbc.Col(html.Strong("Radius"), width=3),
            ], className="mb-2")
            
            # Combine header and rows
            table = html.Div([header, html.Hr(className="my-1")] + rows)
            
            return table
        
        @app.callback(
            Output('portfolio-risk-summary', 'children'),
            [Input('streaming-data-store', 'children')],
            prevent_initial_call=True
        )
        def update_portfolio_summary(json_data):
            """Update portfolio risk summary with latest data."""
            if not json_data:
                raise PreventUpdate
            
            # Parse data
            data = json.loads(json_data)
            risk_scores = data.get('risk_scores', {})
            
            # If no risk scores yet
            if not risk_scores:
                return html.Div("No risk data available yet", className="text-center text-muted my-3")
            
            # Group by sector if available
            sector_risks = {}
            sector_values = {}
            total_value = 0
            weighted_risk = 0
            
            for asset in self.portfolio.assets:
                asset_id = asset['id']
                asset_value = asset['value']
                sector = asset.get('metadata', {}).get('sector', 'Unknown')
                
                # Skip if no risk score
                if asset_id not in risk_scores:
                    continue
                
                risk_score = risk_scores[asset_id]
                
                # Update sector stats
                if sector not in sector_risks:
                    sector_risks[sector] = []
                    sector_values[sector] = 0
                
                sector_risks[sector].append(risk_score)
                sector_values[sector] += asset_value
                
                # Update portfolio stats
                total_value += asset_value
                weighted_risk += risk_score * asset_value
            
            # Calculate weighted average risk
            if total_value > 0:
                weighted_risk /= total_value
            
            # Create sector summary
            sector_rows = []
            for sector, risks in sector_risks.items():
                avg_risk = sum(risks) / len(risks)
                value = sector_values[sector]
                value_pct = (value / total_value) * 100 if total_value > 0 else 0
                
                # Determine color based on risk
                if avg_risk < 0.3:
                    color = "success"
                elif avg_risk < 0.7:
                    color = "warning"
                else:
                    color = "danger"
                
                # Create row with risk badge
                sector_rows.append(
                    dbc.Row([
                        dbc.Col(sector, width=4),
                        dbc.Col(f"${value:,.2f} ({value_pct:.1f}%)", width=4),
                        dbc.Col([
                            html.Span(
                                f"{avg_risk:.2f}",
                                className=f"badge bg-{color}"
                            )
                        ], width=4),
                    ], className="mb-1")
                )
            
            # Sort sectors by risk (highest first)
            sector_rows.sort(key=lambda row: row.props['children'][2].props['children'][0].props['children'], reverse=True)
            
            # Create header
            header = dbc.Row([
                dbc.Col(html.Strong("Sector"), width=4),
                dbc.Col(html.Strong("Value"), width=4),
                dbc.Col(html.Strong("Risk"), width=4),
            ], className="mb-2")
            
            # Create overall risk card
            overall_risk_color = "success"
            if weighted_risk >= 0.3:
                overall_risk_color = "warning"
            if weighted_risk >= 0.7:
                overall_risk_color = "danger"
            
            overall_card = dbc.Card([
                dbc.CardBody([
                    html.H5("Overall Portfolio Risk", className="card-title"),
                    html.H2([
                        html.Span(
                            f"{weighted_risk:.3f}",
                            className=f"badge bg-{overall_risk_color}"
                        )
                    ], className="text-center"),
                    html.P(f"Total Value: ${total_value:,.2f}", className="card-text text-center"),
                ])
            ], className="mb-3")
            
            # Combine all elements
            summary = html.Div([
                overall_card,
                html.H5("Risk by Sector"),
                html.Div([header, html.Hr(className="my-1")] + sector_rows)
            ])
            
            return summary
        
        @app.callback(
            Output('streaming-update-interval', 'interval'),
            [Input('update-interval-slider', 'value')],
            prevent_initial_call=True
        )
        def update_interval(value):
            """Update the streaming update interval."""
            self.update_interval = value
            return value * 1000  # Convert to milliseconds
        
        @app.callback(
            Output('restart-streaming-button', 'disabled'),
            [Input('restart-streaming-button', 'n_clicks')],
            prevent_initial_call=True
        )
        def restart_streaming(n_clicks):
            """Restart real-time data streams."""
            if n_clicks:
                self.stop_real_time_streaming()
                time.sleep(1)  # Brief pause
                self.start_real_time_streaming()
            
            return False  # Don't disable the button
        
    def run_headless(self, 
                    duration: int = 3600,
                    output_interval: int = 300) -> None:
        """
        Run dashboard in headless mode for a specified duration.
        
        Args:
            duration: Duration to run in seconds
            output_interval: Interval to save outputs in seconds
        """
        self.logger.info(f"Running in headless mode for {duration} seconds")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Start real-time components
        self.start_real_time_streaming()
        
        # Create visualizer
        visualizer = GeoFinancialVisualizer()
        
        # Run for specified duration
        start_time = time.time()
        next_output_time = start_time + output_interval
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # Check if it's time to save outputs
                if current_time >= next_output_time:
                    self._save_headless_outputs(visualizer)
                    next_output_time = current_time + output_interval
                
                # Sleep briefly
                time.sleep(1)
        
        except KeyboardInterrupt:
            self.logger.info("Headless mode interrupted")
        
        finally:
            # Stop real-time components
            self.stop_real_time_streaming()
            self.logger.info("Headless mode complete")
    
    def _save_headless_outputs(self, visualizer: GeoFinancialVisualizer) -> None:
        """
        Save outputs in headless mode.
        
        Args:
            visualizer: GeoFinancialVisualizer instance
        """
        now = datetime.now()
        timestamp_str = now.strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save risk map
            with self._data_lock:
                if self.dem_data is not None and hasattr(self.portfolio, 'assets'):
                    # Create risk map with assets
                    risk_map_fig = visualizer.plot_portfolio_on_risk_map(
                        risk_map=self.dem_data,
                        geo_transform=self.dem_transform,
                        portfolio=self.portfolio,
                        title=f"Risk Map at {now.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    
                    # Save figure
                    risk_map_path = os.path.join(self.output_dir, f"risk_map_{timestamp_str}.png")
                    risk_map_fig.savefig(risk_map_path, dpi=150, bbox_inches='tight')
                    self.logger.info(f"Saved risk map to {risk_map_path}")
            
            # Save risk trends
            with self._data_lock:
                if self.risk_score_buffer:
                    # Create figure
                    plt.figure(figsize=(10, 6))
                    
                    # Plot risk trends for each asset
                    for asset_id, history in self.risk_score_buffer.items():
                        if not history:
                            continue
                        
                        # Find asset name
                        asset_name = next(
                            (a['name'] for a in self.portfolio.assets if a['id'] == asset_id),
                            asset_id
                        )
                        
                        # Extract data
                        timestamps = [datetime.fromtimestamp(h[0]) for h in history]
                        scores = [h[1] for h in history]
                        
                        # Plot line
                        plt.plot(timestamps, scores, label=asset_name)
                    
                    # Add labels and title
                    plt.title(f"Risk Score Trends at {now.strftime('%Y-%m-%d %H:%M:%S')}")
                    plt.xlabel("Time")
                    plt.ylabel("Risk Score")
                    plt.ylim(0, 1)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Save figure
                    trends_path = os.path.join(self.output_dir, f"risk_trends_{timestamp_str}.png")
                    plt.savefig(trends_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    self.logger.info(f"Saved risk trends to {trends_path}")
            
            # Save event data
            with self._data_lock:
                if self.event_data_buffer:
                    latest_events = self.event_data_buffer[-1]
                    if 'events' in latest_events and latest_events['events']:
                        # Save to JSON
                        events_path = os.path.join(self.output_dir, f"events_{timestamp_str}.json")
                        with open(events_path, 'w') as f:
                            json.dump(latest_events, f, indent=2)
                        self.logger.info(f"Saved event data to {events_path}")
            
            # Save risk score summary
            with self._data_lock:
                if self.risk_score_buffer:
                    # Calculate latest risk scores
                    latest_scores = {}
                    for asset_id, history in self.risk_score_buffer.items():
                        if history:
                            latest_scores[asset_id] = history[-1][1]
                    
                    # Group by sector
                    sector_risks = {}
                    sector_values = {}
                    total_value = 0
                    weighted_risk = 0
                    
                    for asset in self.portfolio.assets:
                        asset_id = asset['id']
                        asset_value = asset['value']
                        sector = asset.get('metadata', {}).get('sector', 'Unknown')
                        
                        # Skip if no risk score
                        if asset_id not in latest_scores:
                            continue
                        
                        risk_score = latest_scores[asset_id]
                        
                        # Update sector stats
                        if sector not in sector_risks:
                            sector_risks[sector] = []
                            sector_values[sector] = 0
                        
                        sector_risks[sector].append(risk_score)
                        sector_values[sector] += asset_value
                        
                        # Update portfolio stats
                        total_value += asset_value
                        weighted_risk += risk_score * asset_value
                    
                    # Calculate weighted average risk
                    if total_value > 0:
                        weighted_risk /= total_value
                    
                    # Prepare summary data
                    summary_data = {
                        'timestamp': now.timestamp(),
                        'formatted_time': now.strftime('%Y-%m-%d %H:%M:%S'),
                        'overall_risk': weighted_risk,
                        'total_value': total_value,
                        'asset_count': len(self.portfolio.assets),
                        'sectors': []
                    }
                    
                    # Add sector data
                    for sector, risks in sector_risks.items():
                        avg_risk = sum(risks) / len(risks)
                        value = sector_values[sector]
                        value_pct = (value / total_value) * 100 if total_value > 0 else 0
                        
                        summary_data['sectors'].append({
                            'name': sector,
                            'average_risk': avg_risk,
                            'value': value,
                            'value_percentage': value_pct,
                            'asset_count': len(risks)
                        })
                    
                    # Save to JSON
                    summary_path = os.path.join(self.output_dir, f"risk_summary_{timestamp_str}.json")
                    with open(summary_path, 'w') as f:
                        json.dump(summary_data, f, indent=2)
                    self.logger.info(f"Saved risk summary to {summary_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving outputs: {e}")


def main():
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description="Real-time Geospatial Financial Dashboard")
    parser.add_argument("--data-dir", default="data/geo_financial", help="Directory for input data")
    parser.add_argument("--output-dir", default="results/geo_financial/realtime_dashboard", help="Directory for output files")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic data")
    parser.add_argument("--num-assets", type=int, default=20, help="Number of assets to generate")
    parser.add_argument("--dem-size", type=int, default=500, help="Size of DEM to generate")
    parser.add_argument("--update-interval", type=int, default=5, help="Dashboard update interval in seconds")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--headless-duration", type=int, default=3600, help="Duration to run headless mode in seconds")
    parser.add_argument("--headless-output-interval", type=int, default=300, help="Interval to save outputs in headless mode")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create and configure the dashboard app
    app = RealTimeDashboardApp(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device_id=args.device_id,
        update_interval=args.update_interval
    )
    
    # Load or generate data
    app.load_or_generate_data(
        generate=args.generate_data,
        num_assets=args.num_assets,
        dem_size=args.dem_size,
        random_seed=args.seed
    )
    
    # Create risk model
    app.create_risk_model()
    
    # Set up real-time components
    app.setup_real_time_components()
    
    # Run in appropriate mode
    if args.headless:
        app.run_headless(
            duration=args.headless_duration,
            output_interval=args.headless_output_interval
        )
    else:
        app.create_dashboard(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()