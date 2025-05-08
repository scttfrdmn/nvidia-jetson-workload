#!/usr/bin/env python3
"""
Real-time Data Streaming Module for Geospatial Financial Analysis

This module provides real-time data streaming capabilities for monitoring
and analyzing geospatial financial risks as they evolve.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import time
import json
import numpy as np
import pandas as pd
import threading
import queue
import logging
import websocket
import requests
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime
from urllib.parse import urlparse
import multiprocessing as mp
from functools import partial

# Add try/except for optional dependencies
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Import project modules
from src.integrations.geo_financial.geo_risk import (
    GeospatialRiskModel,
    GeospatialPortfolio,
    SpatialRiskFactor
)
from src.integrations.geo_financial.data_connectors import (
    AssetLocationDataLoader,
    FinancialDataLoader
)
from src.integrations.geo_financial.risk_aggregation import RiskAggregator


class DataStreamSource:
    """Base class for data stream sources."""
    
    def __init__(self, name: str):
        """
        Initialize a data stream source.
        
        Args:
            name: Identifier for this data source
        """
        self.name = name
        self.callbacks = []
        self._running = False
        self._thread = None
        self.logger = logging.getLogger(f"DataStreamSource.{name}")
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to be called when new data arrives.
        
        Args:
            callback: Function that takes a data dictionary as input
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable) -> None:
        """Remove a callback function."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _notify_callbacks(self, data: Dict[str, Any]) -> None:
        """Notify all registered callbacks with new data."""
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Error in callback: {e}")
    
    def start(self) -> None:
        """Start the data stream."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop)
        self._thread.daemon = True
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the data stream."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
    
    def _stream_loop(self) -> None:
        """Main stream processing loop. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _stream_loop")


class MarketDataStream(DataStreamSource):
    """Stream real-time market data for financial assets."""
    
    def __init__(self, 
                 name: str = "market_data",
                 api_key: Optional[str] = None,
                 symbols: List[str] = None,
                 interval: int = 60):
        """
        Initialize a market data stream.
        
        Args:
            name: Identifier for this data source
            api_key: API key for market data provider
            symbols: List of asset symbols to stream
            interval: Update interval in seconds
        """
        super().__init__(name)
        self.api_key = api_key
        self.symbols = symbols or []
        self.interval = interval
        self.last_data = {}
        
        # Default to simulated data if no API key provided
        self.use_simulation = api_key is None
        
        # Initialize simulated data state if needed
        if self.use_simulation:
            self._init_simulation()
    
    def _init_simulation(self) -> None:
        """Initialize simulation data."""
        if not self.symbols:
            self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
        self.sim_data = {}
        for symbol in self.symbols:
            # Initialize with random starting price between $50 and $500
            base_price = np.random.uniform(50, 500)
            
            # Assign random volatility between 0.5% and 5%
            volatility = np.random.uniform(0.005, 0.05)
            
            self.sim_data[symbol] = {
                "price": base_price,
                "volatility": volatility,
                "trend": np.random.uniform(-0.0002, 0.0002),  # Slight trend bias
                "last_update": datetime.now().timestamp()
            }
    
    def _generate_simulated_data(self) -> Dict[str, Any]:
        """Generate simulated market data."""
        now = datetime.now()
        timestamp = now.timestamp()
        data = {"timestamp": timestamp, "assets": {}}
        
        for symbol in self.symbols:
            sim = self.sim_data[symbol]
            
            # Calculate seconds since last update
            time_diff = timestamp - sim["last_update"]
            
            # Generate random price movement based on volatility
            # Scale by time difference to handle variable intervals
            price_change = np.random.normal(
                sim["trend"] * time_diff,
                sim["volatility"] * np.sqrt(time_diff / 86400)  # Scale by sqrt of days
            ) * sim["price"]
            
            # Update price
            new_price = max(0.01, sim["price"] + price_change)
            sim["price"] = new_price
            sim["last_update"] = timestamp
            
            # Add to data
            data["assets"][symbol] = {
                "price": new_price,
                "change": price_change,
                "percent_change": price_change / (new_price - price_change),
                "volume": int(np.random.lognormal(10, 1))
            }
        
        return data
    
    def _fetch_real_data(self) -> Dict[str, Any]:
        """Fetch real market data using API."""
        # This would normally fetch from a market data API
        # Implement the specific API integration as needed
        now = datetime.now()
        data = {"timestamp": now.timestamp(), "assets": {}}
        
        try:
            # Mock API call
            self.logger.info(f"Fetching real data for {len(self.symbols)} symbols")
            
            # In a real implementation, this would make API calls
            # For now, we'll return empty data to be safe
            for symbol in self.symbols:
                data["assets"][symbol] = {
                    "price": 0.0,
                    "change": 0.0,
                    "percent_change": 0.0,
                    "volume": 0
                }
                
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
        
        return data
    
    def _stream_loop(self) -> None:
        """Main streaming loop."""
        while self._running:
            try:
                # Generate data (either simulated or real)
                if self.use_simulation:
                    data = self._generate_simulated_data()
                else:
                    data = self._fetch_real_data()
                
                # Store last data
                self.last_data = data
                
                # Notify callbacks
                self._notify_callbacks(data)
                
                # Sleep until next update
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.error(f"Error in market data stream: {e}")
                time.sleep(5)  # Sleep briefly before retry


class GeospatialEventStream(DataStreamSource):
    """Stream real-time geospatial events that may affect risk."""
    
    def __init__(self, 
                name: str = "geospatial_events",
                api_url: Optional[str] = None,
                event_types: List[str] = None,
                geographic_bounds: Optional[Dict[str, float]] = None,
                interval: int = 300):
        """
        Initialize a geospatial event stream.
        
        Args:
            name: Identifier for this data source
            api_url: URL for geospatial event API
            event_types: List of event types to monitor
            geographic_bounds: Geographic bounds {min_x, min_y, max_x, max_y}
            interval: Update interval in seconds
        """
        super().__init__(name)
        self.api_url = api_url
        self.event_types = event_types or ["flood", "wildfire", "earthquake", "storm"]
        self.geographic_bounds = geographic_bounds or {
            "min_x": -180.0, "min_y": -90.0, "max_x": 180.0, "max_y": 90.0
        }
        self.interval = interval
        self.last_data = {}
        
        # Default to simulated data if no API URL provided
        self.use_simulation = api_url is None
        
        # Initialize simulated data state if needed
        if self.use_simulation:
            self._init_simulation()
    
    def _init_simulation(self) -> None:
        """Initialize simulation parameters."""
        # Event probabilities (events per day, globally)
        self.event_probabilities = {
            "flood": 0.2,
            "wildfire": 0.15,
            "earthquake": 0.05,
            "storm": 0.25,
            "heatwave": 0.1,
            "drought": 0.05
        }
        
        # Keep track of active events
        self.active_events = []
        
        # Event durations in hours
        self.event_durations = {
            "flood": np.random.uniform(24, 72),
            "wildfire": np.random.uniform(24, 168),
            "earthquake": 0.25,  # 15 minutes
            "storm": np.random.uniform(12, 48),
            "heatwave": np.random.uniform(72, 240),
            "drought": np.random.uniform(720, 2160)  # 30-90 days
        }
    
    def _generate_simulated_event(self) -> Optional[Dict[str, Any]]:
        """Generate a simulated geospatial event."""
        # Determine if a new event should be generated
        daily_probability = sum(
            self.event_probabilities[et] for et in self.event_types
        )
        
        # Convert daily probability to the current interval
        interval_probability = daily_probability * (self.interval / 86400)
        
        if np.random.random() > interval_probability:
            return None  # No new event
        
        # Select an event type
        event_probs = [
            self.event_probabilities[et] for et in self.event_types
        ]
        event_probs = [p / sum(event_probs) for p in event_probs]
        event_type = np.random.choice(self.event_types, p=event_probs)
        
        # Generate random location within bounds
        bounds = self.geographic_bounds
        x = np.random.uniform(bounds["min_x"], bounds["max_x"])
        y = np.random.uniform(bounds["min_y"], bounds["max_y"])
        
        # Generate event parameters
        severity = np.random.uniform(0.1, 1.0)  # 0.1 to 1.0
        
        # Calculate duration in hours
        duration = self.event_durations[event_type] * (0.5 + severity)
        
        # Generate event
        now = datetime.now()
        event = {
            "id": f"{event_type}_{now.strftime('%Y%m%d%H%M%S')}",
            "type": event_type,
            "location": {"x": x, "y": y},
            "severity": severity,
            "start_time": now.timestamp(),
            "predicted_end_time": now.timestamp() + (duration * 3600),
            "description": f"Simulated {event_type} event",
            "radius_km": 10 + (severity * 40)  # 10km to 50km radius
        }
        
        # Add to active events
        self.active_events.append(event)
        
        return event
    
    def _update_active_events(self) -> List[Dict[str, Any]]:
        """Update the list of active events."""
        now = datetime.now().timestamp()
        
        # Filter out ended events
        self.active_events = [
            event for event in self.active_events
            if event["predicted_end_time"] > now
        ]
        
        return self.active_events
    
    def _fetch_real_events(self) -> Dict[str, Any]:
        """Fetch real geospatial events from API."""
        now = datetime.now()
        data = {
            "timestamp": now.timestamp(),
            "events": []
        }
        
        try:
            # Mock API call
            self.logger.info("Fetching real geospatial events")
            
            # In a real implementation, this would make API calls
            # to services like USGS, NOAA, or specialized geospatial data providers
            
        except Exception as e:
            self.logger.error(f"Error fetching geospatial events: {e}")
        
        return data
    
    def _stream_loop(self) -> None:
        """Main streaming loop."""
        while self._running:
            try:
                if self.use_simulation:
                    # Generate potential new event
                    new_event = self._generate_simulated_event()
                    
                    # Update active events
                    active_events = self._update_active_events()
                    
                    # Create data packet
                    now = datetime.now()
                    data = {
                        "timestamp": now.timestamp(),
                        "events": active_events,
                        "new_event": new_event
                    }
                else:
                    data = self._fetch_real_events()
                
                # Store last data
                self.last_data = data
                
                # Notify callbacks
                self._notify_callbacks(data)
                
            except Exception as e:
                self.logger.error(f"Error in geospatial event stream: {e}")
            
            # Sleep until next update
            time.sleep(self.interval)


class RealTimeStreamManager:
    """
    Manager for multiple real-time data streams.
    
    This class coordinates data streams, processes incoming data,
    and updates risk models in real time.
    """
    
    def __init__(self, 
                device_id: int = 0,
                max_buffer_size: int = 1000):
        """
        Initialize the stream manager.
        
        Args:
            device_id: GPU device ID (-1 for CPU only)
            max_buffer_size: Maximum size of data buffers
        """
        self.device_id = device_id
        self.max_buffer_size = max_buffer_size
        self.streams = {}
        self.data_buffers = {}
        self.logger = logging.getLogger("RealTimeStreamManager")
        
        # Keep track of last update time for risk models
        self.last_risk_update = datetime.now().timestamp()
        
        # Initialize processing queue
        self.processing_queue = queue.Queue(maxsize=max_buffer_size)
        
        # Initialize GPU resources if available
        self.use_gpu = device_id >= 0 and HAS_CUPY
        if self.use_gpu:
            try:
                if cp:
                    cp.cuda.Device(device_id).use()
                    self.logger.info(f"Using GPU device {device_id}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GPU: {e}")
                self.use_gpu = False
        
        # Start processing thread
        self._processing_thread = None
        self._running = False
    
    def add_stream(self, stream: DataStreamSource) -> None:
        """
        Add a data stream to the manager.
        
        Args:
            stream: DataStreamSource instance
        """
        if stream.name in self.streams:
            self.logger.warning(f"Stream '{stream.name}' already exists, replacing")
        
        self.streams[stream.name] = stream
        self.data_buffers[stream.name] = []
        
        # Register callback
        stream.register_callback(self._create_data_handler(stream.name))
    
    def remove_stream(self, stream_name: str) -> None:
        """
        Remove a data stream from the manager.
        
        Args:
            stream_name: Name of the stream to remove
        """
        if stream_name not in self.streams:
            self.logger.warning(f"Stream '{stream_name}' not found")
            return
        
        # Stop the stream and clean up
        stream = self.streams[stream_name]
        stream.stop()
        del self.streams[stream_name]
        del self.data_buffers[stream_name]
    
    def _create_data_handler(self, stream_name: str) -> Callable[[Dict[str, Any]], None]:
        """Create a handler function for a specific stream."""
        def handle_data(data: Dict[str, Any]) -> None:
            # Add stream name to the data
            data["stream_name"] = stream_name
            
            # Add timestamp if not present
            if "timestamp" not in data:
                data["timestamp"] = datetime.now().timestamp()
            
            # Add to buffer
            buffer = self.data_buffers[stream_name]
            buffer.append(data)
            
            # Trim buffer if needed
            if len(buffer) > self.max_buffer_size:
                buffer.pop(0)
            
            # Add to processing queue
            try:
                self.processing_queue.put(data, block=False)
            except queue.Full:
                self.logger.warning("Processing queue full, dropping data")
        
        return handle_data
    
    def start(self) -> None:
        """Start all data streams and processing."""
        if self._running:
            return
        
        self._running = True
        
        # Start processing thread
        self._processing_thread = threading.Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        # Start all streams
        for stream in self.streams.values():
            stream.start()
    
    def stop(self) -> None:
        """Stop all data streams and processing."""
        self._running = False
        
        # Stop all streams
        for stream in self.streams.values():
            stream.stop()
        
        # Stop processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
    
    def _processing_loop(self) -> None:
        """Main data processing loop."""
        while self._running:
            try:
                # Get data from queue with timeout
                try:
                    data = self.processing_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process the data
                self._process_data(data)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
    
    def _process_data(self, data: Dict[str, Any]) -> None:
        """
        Process incoming data.
        
        This method can be extended to implement specific data processing logic.
        """
        # Basic processing - just log the data type
        stream_name = data.get("stream_name", "unknown")
        self.logger.debug(f"Processing data from {stream_name}")
        
        # Implement specific processing logic here
        # This could include updating risk models, triggering alerts, etc.


class RealTimeRiskAnalyzer:
    """
    Real-time risk analyzer for geospatial financial data.
    
    This class uses streaming data to continuously update risk models
    and provide real-time risk assessments.
    """
    
    def __init__(self,
                portfolio: 'GeospatialPortfolio',
                risk_model: 'GeospatialRiskModel',
                stream_manager: RealTimeStreamManager,
                update_interval: int = 60,
                device_id: int = 0):
        """
        Initialize the real-time risk analyzer.
        
        Args:
            portfolio: GeospatialPortfolio to analyze
            risk_model: Base GeospatialRiskModel to update
            stream_manager: RealTimeStreamManager for data streams
            update_interval: How often to update risk assessments (seconds)
            device_id: GPU device ID (-1 for CPU only)
        """
        self.portfolio = portfolio
        self.base_risk_model = risk_model
        self.working_risk_model = self._clone_risk_model(risk_model)
        self.stream_manager = stream_manager
        self.update_interval = update_interval
        self.device_id = device_id
        
        self.logger = logging.getLogger("RealTimeRiskAnalyzer")
        
        # Risk assessment results
        self.latest_risk_scores = {}
        self.risk_score_history = {}
        self.last_update_time = datetime.now().timestamp()
        
        # Callbacks for risk updates
        self.risk_update_callbacks = []
        
        # Market data mapping
        self.asset_market_mapping = {}  # Map portfolio assets to market symbols
        
        # Start update thread
        self._running = False
        self._update_thread = None
    
    def _clone_risk_model(self, model: 'GeospatialRiskModel') -> 'GeospatialRiskModel':
        """Create a working copy of the risk model."""
        # This is a simplified clone - in practice, you'd need to
        # properly copy all risk factors and their data
        new_model = GeospatialRiskModel(device_id=self.device_id)
        
        # Copy risk factors
        for rf in model.risk_factors:
            new_model.add_risk_factor(rf)
        
        return new_model
    
    def set_asset_market_mapping(self, mapping: Dict[str, str]) -> None:
        """
        Set mapping between portfolio assets and market symbols.
        
        Args:
            mapping: Dictionary mapping asset IDs to market symbols
        """
        self.asset_market_mapping = mapping
    
    def register_risk_update_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback for risk updates.
        
        Args:
            callback: Function to call with risk update data
        """
        if callback not in self.risk_update_callbacks:
            self.risk_update_callbacks.append(callback)
    
    def unregister_risk_update_callback(self, callback: Callable) -> None:
        """Remove a risk update callback."""
        if callback in self.risk_update_callbacks:
            self.risk_update_callbacks.remove(callback)
    
    def start(self) -> None:
        """Start real-time risk analysis."""
        if self._running:
            return
        
        self._running = True
        
        # Initialize risk score history for all assets
        for asset in self.portfolio.assets:
            asset_id = asset["id"]
            self.risk_score_history[asset_id] = []
        
        # Start update thread
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
        
        self.logger.info("Real-time risk analyzer started")
    
    def stop(self) -> None:
        """Stop real-time risk analysis."""
        self._running = False
        
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=2.0)
        
        self.logger.info("Real-time risk analyzer stopped")
    
    def _update_loop(self) -> None:
        """Main risk update loop."""
        while self._running:
            try:
                # Check if it's time to update
                now = datetime.now().timestamp()
                if now - self.last_update_time >= self.update_interval:
                    self._update_risk_assessment()
                    self.last_update_time = now
                
                # Sleep briefly
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in risk update loop: {e}")
                time.sleep(5)  # Sleep longer after an error
    
    def _update_risk_assessment(self) -> None:
        """Update the risk assessment based on latest data."""
        self.logger.debug("Updating risk assessment")
        
        # Update risk factors based on market data
        self._update_market_based_risk()
        
        # Update risk factors based on geospatial events
        self._update_geospatial_event_risk()
        
        # Perform risk assessment
        try:
            risk_scores = self.portfolio.assess_risk(self.working_risk_model)
            self.latest_risk_scores = risk_scores
            
            # Update history
            for asset_id, score in risk_scores.items():
                if asset_id in self.risk_score_history:
                    history = self.risk_score_history[asset_id]
                    history.append((datetime.now().timestamp(), score))
                    
                    # Trim history if needed (keep last 1000 entries)
                    if len(history) > 1000:
                        history = history[-1000:]
                        self.risk_score_history[asset_id] = history
            
            # Notify callbacks
            self._notify_risk_update_callbacks()
            
            self.logger.debug(f"Risk assessment updated with {len(risk_scores)} scores")
            
        except Exception as e:
            self.logger.error(f"Error assessing risk: {e}")
    
    def _update_market_based_risk(self) -> None:
        """Update risk factors based on market data."""
        # Check if market data stream exists
        if "market_data" not in self.stream_manager.streams:
            return
        
        # Get latest market data
        market_stream = self.stream_manager.streams["market_data"]
        if not market_stream.last_data:
            return
        
        market_data = market_stream.last_data
        if "assets" not in market_data:
            return
        
        # Update asset values in portfolio based on market data
        for asset in self.portfolio.assets:
            asset_id = asset["id"]
            
            # Skip if no market mapping
            if asset_id not in self.asset_market_mapping:
                continue
            
            market_symbol = self.asset_market_mapping[asset_id]
            
            # Skip if symbol not in market data
            if market_symbol not in market_data["assets"]:
                continue
            
            # Update asset value
            asset_market_data = market_data["assets"][market_symbol]
            
            # In a real application, you would apply more sophisticated
            # logic to update asset values based on market movements
            
            # For now, we'll just adjust based on percent change
            if "percent_change" in asset_market_data:
                pct_change = asset_market_data["percent_change"]
                asset["value"] *= (1 + pct_change)
    
    def _update_geospatial_event_risk(self) -> None:
        """Update risk factors based on geospatial events."""
        # Check if geospatial event stream exists
        if "geospatial_events" not in self.stream_manager.streams:
            return
        
        # Get latest geospatial event data
        event_stream = self.stream_manager.streams["geospatial_events"]
        if not event_stream.last_data:
            return
        
        event_data = event_stream.last_data
        if "events" not in event_data or not event_data["events"]:
            return
        
        # Process each active event
        for event in event_data["events"]:
            event_type = event.get("type")
            
            # Skip unknown event types
            if not event_type:
                continue
            
            # Adjust risk factors based on event type
            self._adjust_risk_for_event(event)
    
    def _adjust_risk_for_event(self, event: Dict[str, Any]) -> None:
        """
        Adjust risk factors for a specific geospatial event.
        
        Args:
            event: Event data dictionary
        """
        event_type = event.get("type")
        location = event.get("location", {})
        severity = event.get("severity", 0.5)
        radius_km = event.get("radius_km", 50.0)
        
        # Skip if missing critical data
        if not event_type or not location:
            return
        
        x, y = location.get("x"), location.get("y")
        if x is None or y is None:
            return
        
        # Calculate distance to event for each asset
        for asset in self.portfolio.assets:
            asset_x = asset["x"]
            asset_y = asset["y"]
            
            # Calculate distance (simple Euclidean for now)
            # In a real application, you would use proper geographic distance
            distance = np.sqrt((asset_x - x)**2 + (asset_y - y)**2)
            
            # Convert to approximate km (assuming degrees)
            distance_km = distance * 111.0  # rough conversion at equator
            
            # Skip if outside event radius
            if distance_km > radius_km:
                continue
            
            # Calculate impact based on distance and severity
            impact = severity * (1 - (distance_km / radius_km))
            
            # In a real application, you would update specific risk factors
            # based on the event type and impact
            self.logger.debug(f"Asset {asset['id']} has impact {impact:.3f} from {event_type} event")
    
    def _notify_risk_update_callbacks(self) -> None:
        """Notify all risk update callbacks with latest data."""
        update_data = {
            "timestamp": datetime.now().timestamp(),
            "risk_scores": self.latest_risk_scores,
            "portfolio": self.portfolio
        }
        
        for callback in self.risk_update_callbacks:
            try:
                callback(update_data)
            except Exception as e:
                self.logger.error(f"Error in risk update callback: {e}")


# Utility functions for creating real-time data streams

def create_market_data_stream(symbols: List[str] = None,
                           api_key: Optional[str] = None,
                           interval: int = 60) -> MarketDataStream:
    """
    Create a market data stream.
    
    Args:
        symbols: List of financial symbols to track
        api_key: API key for market data provider
        interval: Update interval in seconds
    
    Returns:
        Configured MarketDataStream
    """
    return MarketDataStream(
        name="market_data",
        api_key=api_key,
        symbols=symbols,
        interval=interval
    )


def create_geospatial_event_stream(
    event_types: List[str] = None,
    geographic_bounds: Optional[Dict[str, float]] = None,
    api_url: Optional[str] = None,
    interval: int = 300
) -> GeospatialEventStream:
    """
    Create a geospatial event stream.
    
    Args:
        event_types: List of event types to monitor
        geographic_bounds: Geographic bounds {min_x, min_y, max_x, max_y}
        api_url: URL for geospatial event API
        interval: Update interval in seconds
    
    Returns:
        Configured GeospatialEventStream
    """
    return GeospatialEventStream(
        name="geospatial_events",
        api_url=api_url,
        event_types=event_types,
        geographic_bounds=geographic_bounds,
        interval=interval
    )