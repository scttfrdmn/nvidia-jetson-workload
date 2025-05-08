#!/usr/bin/env python3
"""
Jetson Orin NX Optimization for Geospatial Financial Integration

This module provides optimizations specific to NVIDIA Jetson Orin NX devices
for the geospatial financial integration. It includes specialized kernels,
memory management strategies, and performance tuning for the constrained
resources of Jetson devices.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from enum import Enum

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

# Import common utilities
from src.integrations.common.gpu_memory_manager import (
    get_memory_manager, ManagedGPUArray, to_gpu, zeros
)
from src.integrations.common.shared_memory import (
    DataTransferOptimizer, get_data_transfer_optimizer
)

# Import geo_financial modules
from src.integrations.geo_financial.geo_risk import (
    GeospatialRiskModel, GeospatialPortfolio, SpatialRiskFactor
)
from src.integrations.geo_financial.risk_aggregation import (
    RiskAggregator, AggregationMethod, RiskSurfaceGenerator
)
from src.integrations.geo_financial.climate_risk_assessment import (
    ClimateRiskAssessor
)
from src.integrations.geo_financial.scenario_analysis import (
    ScenarioAnalyzer, ScenarioVisualizer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("JetsonOptimization")


class MemoryProfile(Enum):
    """Memory usage profile for optimization."""
    
    HIGH_PERFORMANCE = "high_performance"  # Use maximum available memory for performance
    BALANCED = "balanced"                  # Balance memory usage and performance
    MEMORY_CONSTRAINED = "memory_constrained"  # Minimize memory usage, even at cost of performance


class JetsonDeviceInfo:
    """Information about the Jetson device for optimization."""
    
    def __init__(self):
        """Initialize device information."""
        self.is_jetson = False
        self.model = "unknown"
        self.compute_capability = "0.0"
        self.total_memory_mb = 0
        self.cuda_cores = 0
        self.sm_count = 0
        self.detect_jetson()
    
    def detect_jetson(self) -> None:
        """Detect if running on a Jetson device and gather device information."""
        try:
            # Check for Jetson-specific files
            if os.path.exists("/etc/nv_tegra_release"):
                self.is_jetson = True
                
                # Try to get Jetson model
                try:
                    import subprocess
                    output = subprocess.check_output(["cat", "/proc/device-tree/model"]).decode("utf-8")
                    if "Orin" in output:
                        if "Nano" in output:
                            self.model = "Jetson Orin Nano"
                            self.cuda_cores = 128
                            self.sm_count = 1
                        else:
                            self.model = "Jetson Orin"
                            self.cuda_cores = 1024
                            self.sm_count = 8
                    elif "Xavier" in output:
                        self.model = "Jetson Xavier"
                        self.cuda_cores = 512
                        self.sm_count = 4
                    else:
                        self.model = "Unknown Jetson"
                except:
                    pass
                
                # Try to detect compute capability
                if HAS_CUPY:
                    try:
                        self.compute_capability = cp.cuda.runtime.getDeviceProperties(0)["computeCapability"]
                    except:
                        # For Jetson Orin Nano, compute capability is 8.7
                        if "Orin Nano" in self.model:
                            self.compute_capability = "8.7"
                        elif "Orin" in self.model:
                            self.compute_capability = "8.7"
                        elif "Xavier" in self.model:
                            self.compute_capability = "7.2"
                
                # Get total memory
                try:
                    if HAS_CUPY:
                        self.total_memory_mb = cp.cuda.runtime.memGetInfo()[1] / (1024 * 1024)
                    else:
                        # Check using nvidia-smi
                        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"]).decode("utf-8")
                        self.total_memory_mb = float(output.strip())
                except:
                    # Jetson Orin Nano typically has 8GB memory (shared)
                    if "Orin Nano" in self.model:
                        self.total_memory_mb = 8 * 1024
                    elif "Orin" in self.model:
                        self.total_memory_mb = 16 * 1024
                    elif "Xavier" in self.model:
                        self.total_memory_mb = 16 * 1024
            
            # Not a Jetson, check if we can get device info from CUDA
            elif HAS_CUPY:
                try:
                    props = cp.cuda.runtime.getDeviceProperties(0)
                    self.compute_capability = props["computeCapability"]
                    self.total_memory_mb = cp.cuda.runtime.memGetInfo()[1] / (1024 * 1024)
                    self.cuda_cores = props["multiProcessorCount"] * 128  # Approximate
                    self.sm_count = props["multiProcessorCount"]
                    self.model = "CUDA-compatible device"
                except:
                    pass
            
        except Exception as e:
            logger.warning(f"Error detecting Jetson device: {e}")
            self.is_jetson = False
    
    def __str__(self) -> str:
        """String representation of device info."""
        if not self.is_jetson and self.model == "unknown":
            return "Non-Jetson device with unknown specifications"
        
        return (f"{self.model} (SM {self.compute_capability}) with "
                f"{self.cuda_cores} CUDA cores, {self.sm_count} SMs, "
                f"{self.total_memory_mb:.0f}MB memory")
    
    def is_memory_constrained(self) -> bool:
        """Check if device has limited memory."""
        return self.total_memory_mb < 8 * 1024  # Less than 8GB is considered constrained
    
    def is_compute_constrained(self) -> bool:
        """Check if device has limited compute capability."""
        return self.cuda_cores < 256  # Fewer than 256 CUDA cores is considered constrained
    
    def is_orin_nano(self) -> bool:
        """Check if device is a Jetson Orin Nano."""
        return self.is_jetson and "Orin Nano" in self.model


class JetsonOptimizer:
    """
    Optimization for geospatial financial integration on Jetson devices.
    
    This class provides optimization strategies specific to Jetson Orin NX
    devices, including memory management, compute kernel selection, and
    workload partitioning.
    """
    
    def __init__(self, 
                device_id: int = 0, 
                memory_profile: MemoryProfile = MemoryProfile.BALANCED,
                enable_tiling: bool = True,
                enable_half_precision: bool = True,
                enable_mixed_precision: bool = True,
                batch_size: Optional[int] = None,
                logger: Optional[logging.Logger] = None):
        """
        Initialize the Jetson optimizer.
        
        Args:
            device_id: GPU device ID
            memory_profile: Memory usage profile
            enable_tiling: Whether to enable tiling for large datasets
            enable_half_precision: Whether to use half precision where possible
            enable_mixed_precision: Whether to use mixed precision
            batch_size: Custom batch size (auto-determined if None)
            logger: Optional logger instance
        """
        self.device_id = device_id
        self.memory_profile = memory_profile
        self.enable_tiling = enable_tiling
        self.enable_half_precision = enable_half_precision
        self.enable_mixed_precision = enable_mixed_precision
        self.custom_batch_size = batch_size
        self.logger = logger or logging.getLogger("JetsonOptimizer")
        
        # Detect device
        self.device_info = JetsonDeviceInfo()
        
        # Initialize memory management
        self.memory_manager = None
        if HAS_CUPY:
            try:
                self.memory_manager = get_memory_manager(device_id)
            except:
                pass
        
        # Determine optimal batch size if not specified
        self.batch_size = self._determine_optimal_batch_size() if batch_size is None else batch_size
        
        # Detect optimal tile size for tiling
        self.tile_size = self._determine_optimal_tile_size()
        
        # Log initialization
        self.logger.info(f"Initialized Jetson optimizer for {self.device_info}")
        self.logger.info(f"Memory profile: {memory_profile.value}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Tile size: {self.tile_size}")
        self.logger.info(f"Half precision: {enable_half_precision}")
        self.logger.info(f"Mixed precision: {enable_mixed_precision}")
        self.logger.info(f"Tiling: {enable_tiling}")
    
    def _determine_optimal_batch_size(self) -> int:
        """
        Determine the optimal batch size based on device capabilities.
        
        Returns:
            Optimal batch size
        """
        if self.device_info.is_orin_nano():
            # Jetson Orin Nano with 8GB memory
            if self.memory_profile == MemoryProfile.HIGH_PERFORMANCE:
                return 256
            elif self.memory_profile == MemoryProfile.BALANCED:
                return 128
            else:  # MEMORY_CONSTRAINED
                return 64
        elif self.device_info.is_jetson:
            # Other Jetson devices
            if self.memory_profile == MemoryProfile.HIGH_PERFORMANCE:
                return 512
            elif self.memory_profile == MemoryProfile.BALANCED:
                return 256
            else:  # MEMORY_CONSTRAINED
                return 128
        else:
            # Non-Jetson devices
            return 1024  # Default for non-Jetson devices
    
    def _determine_optimal_tile_size(self) -> int:
        """
        Determine the optimal tile size for tiled operations.
        
        Returns:
            Optimal tile size
        """
        if self.device_info.is_orin_nano():
            # Jetson Orin Nano with limited CUDA cores
            return 128  # Small tile size to maximize SM utilization
        elif self.device_info.is_jetson:
            # Other Jetson devices
            return 256
        else:
            # Non-Jetson devices
            return 512  # Default for non-Jetson devices
    
    def optimize_portfolio(self, portfolio: GeospatialPortfolio) -> GeospatialPortfolio:
        """
        Optimize a geospatial portfolio for Jetson devices.
        
        Args:
            portfolio: Portfolio to optimize
            
        Returns:
            Optimized portfolio
        """
        # Create a copy of the portfolio
        optimized = GeospatialPortfolio(
            assets=portfolio.assets,
            name=portfolio.name,
            description=f"{portfolio.description} (Jetson optimized)",
            metadata=portfolio.metadata.copy()
        )
        
        # Set device ID
        optimized.device_id = self.device_id
        
        # Set batch size for batched operations
        optimized.batch_size = self.batch_size
        
        # Add optimization metadata
        optimized.metadata["jetson_optimized"] = True
        optimized.metadata["memory_profile"] = self.memory_profile.value
        optimized.metadata["batch_size"] = self.batch_size
        optimized.metadata["enable_half_precision"] = self.enable_half_precision
        optimized.metadata["enable_mixed_precision"] = self.enable_mixed_precision
        
        # Override assess_risk method to use optimized version
        original_assess_risk = optimized.assess_risk
        
        def optimized_assess_risk(risk_model, *args, **kwargs):
            return self.optimized_risk_assessment(optimized, risk_model, *args, **kwargs)
        
        # Replace the method
        optimized.assess_risk = optimized_assess_risk
        
        return optimized
    
    def optimize_risk_model(self, risk_model: GeospatialRiskModel) -> GeospatialRiskModel:
        """
        Optimize a geospatial risk model for Jetson devices.
        
        Args:
            risk_model: Risk model to optimize
            
        Returns:
            Optimized risk model
        """
        # Create a copy of the risk model
        optimized = GeospatialRiskModel(
            name=risk_model.name,
            description=f"{risk_model.description} (Jetson optimized)",
            risk_factors=risk_model.risk_factors.copy(),
            aggregation_method=risk_model.aggregation_method,
            metadata=risk_model.metadata.copy() if risk_model.metadata else {}
        )
        
        # Set device ID
        optimized.device_id = self.device_id
        
        # Set batch size for batched operations
        optimized.batch_size = self.batch_size
        
        # Add optimization metadata
        optimized.metadata["jetson_optimized"] = True
        optimized.metadata["memory_profile"] = self.memory_profile.value
        optimized.metadata["batch_size"] = self.batch_size
        optimized.metadata["tile_size"] = self.tile_size
        optimized.metadata["enable_half_precision"] = self.enable_half_precision
        optimized.metadata["enable_mixed_precision"] = self.enable_mixed_precision
        
        # Optimize risk factors if needed
        for i, rf in enumerate(optimized.risk_factors):
            optimized.risk_factors[i] = self.optimize_risk_factor(rf)
        
        return optimized
    
    def optimize_risk_factor(self, risk_factor: SpatialRiskFactor) -> SpatialRiskFactor:
        """
        Optimize a spatial risk factor for Jetson devices.
        
        Args:
            risk_factor: Risk factor to optimize
            
        Returns:
            Optimized risk factor
        """
        # Create a copy of the risk factor
        optimized = SpatialRiskFactor(
            name=risk_factor.name,
            risk_data=risk_factor.risk_data,
            x_coords=risk_factor.x_coords,
            y_coords=risk_factor.y_coords,
            risk_weight=risk_factor.risk_weight,
            description=f"{risk_factor.description} (Jetson optimized)",
            metadata=risk_factor.metadata.copy() if risk_factor.metadata else {}
        )
        
        # Convert risk data to half precision if enabled and appropriate
        if self.enable_half_precision and hasattr(optimized, 'risk_data') and optimized.risk_data is not None:
            if isinstance(optimized.risk_data, np.ndarray) and optimized.risk_data.dtype == np.float32:
                # Check if values are in range for float16
                if np.min(optimized.risk_data) >= -65504 and np.max(optimized.risk_data) <= 65504:
                    optimized.risk_data = optimized.risk_data.astype(np.float16)
                    optimized.metadata["half_precision"] = True
        
        return optimized
    
    def optimize_scenario_analyzer(self, analyzer: ScenarioAnalyzer) -> ScenarioAnalyzer:
        """
        Optimize a scenario analyzer for Jetson devices.
        
        Args:
            analyzer: Scenario analyzer to optimize
            
        Returns:
            Optimized scenario analyzer
        """
        # Optimize the base portfolio and risk model
        optimized_portfolio = self.optimize_portfolio(analyzer.base_portfolio)
        optimized_risk_model = self.optimize_risk_model(analyzer.base_risk_model)
        
        # Create a new analyzer with optimized components
        optimized = ScenarioAnalyzer(
            portfolio=optimized_portfolio,
            risk_model=optimized_risk_model,
            device_id=self.device_id,
            logger=analyzer.logger
        )
        
        # Copy existing results if any
        optimized.scenario_results = analyzer.scenario_results.copy()
        
        return optimized
    
    def optimized_risk_assessment(self, 
                                portfolio: GeospatialPortfolio, 
                                risk_model: GeospatialRiskModel, 
                                *args, **kwargs) -> Dict[str, float]:
        """
        Optimized risk assessment for Jetson devices.
        
        This implementation uses tiling, batching, and precision optimizations
        to efficiently assess risk on Jetson devices with limited memory and compute.
        
        Args:
            portfolio: Portfolio to assess
            risk_model: Risk model to use
            
        Returns:
            Dictionary mapping asset IDs to risk scores
        """
        # Check if CUDA/CuPy is available
        if not HAS_CUPY:
            # Fall back to original implementation
            if hasattr(portfolio, '_original_assess_risk'):
                return portfolio._original_assess_risk(risk_model, *args, **kwargs)
            else:
                # Use the original method from GeospatialPortfolio
                return portfolio.__class__.assess_risk(portfolio, risk_model, *args, **kwargs)
        
        # Get assets and risk factors
        assets = portfolio.assets
        risk_factors = risk_model.risk_factors
        
        if not assets or not risk_factors:
            return {}
        
        # Prepare result dictionary
        risk_scores = {}
        
        # Determine whether to use tiling based on memory constraints
        use_tiling = self.enable_tiling and len(assets) > self.batch_size
        
        # Process assets in batches
        if use_tiling:
            # Tiled implementation for memory efficiency
            for i in range(0, len(assets), self.batch_size):
                batch_assets = assets[i:i+self.batch_size]
                batch_scores = self._process_asset_batch(batch_assets, risk_factors, risk_model.aggregation_method)
                risk_scores.update(batch_scores)
        else:
            # Process all assets at once
            risk_scores = self._process_asset_batch(assets, risk_factors, risk_model.aggregation_method)
        
        return risk_scores
    
    def _process_asset_batch(self, 
                           assets: List[Dict[str, Any]], 
                           risk_factors: List[SpatialRiskFactor],
                           aggregation_method: AggregationMethod) -> Dict[str, float]:
        """
        Process a batch of assets for risk assessment.
        
        Args:
            assets: List of assets to process
            risk_factors: List of risk factors to apply
            aggregation_method: Method for aggregating risk factors
            
        Returns:
            Dictionary mapping asset IDs to risk scores
        """
        # Extract asset locations and IDs
        asset_ids = [asset["id"] for asset in assets]
        asset_locations = np.array([[asset["x"], asset["y"]] for asset in assets])
        
        # Dictionary to store risk scores for each asset and factor
        all_factor_scores = {}
        
        # Process each risk factor
        for rf in risk_factors:
            # Skip if risk factor has no risk_data or risk_function
            if not hasattr(rf, 'risk_data') and not hasattr(rf, 'risk_function'):
                continue
            
            # Apply risk factor to get scores
            if hasattr(rf, 'risk_function') and rf.risk_function is not None:
                # Use risk function for custom factors
                try:
                    factor_scores = rf.risk_function({
                        "assets": assets,
                        "locations": asset_locations
                    })
                    if isinstance(factor_scores, dict):
                        for asset_id, score in factor_scores.items():
                            if asset_id not in all_factor_scores:
                                all_factor_scores[asset_id] = {}
                            all_factor_scores[asset_id][rf.name] = score
                except Exception as e:
                    self.logger.error(f"Error applying risk function for {rf.name}: {e}")
            
            elif hasattr(rf, 'risk_data') and rf.risk_data is not None:
                # Interpolate risk scores from risk data grid
                try:
                    # Use GPU for interpolation if available
                    if HAS_CUPY:
                        # Convert risk data to GPU
                        risk_data_device = cp.asarray(rf.risk_data)
                        
                        # Use half precision if enabled and type is compatible
                        if self.enable_half_precision and risk_data_device.dtype == cp.float32:
                            risk_data_device = risk_data_device.astype(cp.float16)
                        
                        # Convert coordinates to GPU
                        x_coords_device = cp.asarray(rf.x_coords)
                        y_coords_device = cp.asarray(rf.y_coords)
                        
                        # Convert asset locations to GPU
                        locations_device = cp.asarray(asset_locations)
                        
                        # Create output array
                        interpolated_scores = cp.zeros(len(assets), dtype=cp.float32)
                        
                        # Launch custom CUDA kernel for efficient interpolation
                        # This is a high-performance implementation optimized for Jetson devices
                        if HAS_CUPY and hasattr(cp, 'ElementwiseKernel'):
                            # Define the interpolation kernel
                            interpolate_kernel = cp.ElementwiseKernel(
                                'float32 x, float32 y, raw float32 grid, raw float32 x_coords, raw float32 y_coords',
                                'float32 z',
                                '''
                                // Find indices for interpolation
                                int nx = x_coords.size();
                                int ny = y_coords.size();
                                
                                // Clip the input coordinates to the valid range
                                float x_clipped = max(min(x, x_coords[nx-1]), x_coords[0]);
                                float y_clipped = max(min(y, y_coords[ny-1]), y_coords[0]);
                                
                                // Find the indices of the grid cell containing the point
                                int x_idx = 0;
                                int y_idx = 0;
                                
                                // Binary search for x index
                                int left = 0;
                                int right = nx - 1;
                                while (left <= right) {
                                    int mid = (left + right) / 2;
                                    if (x_coords[mid] <= x_clipped && (mid == nx-1 || x_clipped < x_coords[mid+1])) {
                                        x_idx = mid;
                                        break;
                                    } else if (x_coords[mid] > x_clipped) {
                                        right = mid - 1;
                                    } else {
                                        left = mid + 1;
                                    }
                                }
                                
                                // Binary search for y index
                                left = 0;
                                right = ny - 1;
                                while (left <= right) {
                                    int mid = (left + right) / 2;
                                    if (y_coords[mid] <= y_clipped && (mid == ny-1 || y_clipped < y_coords[mid+1])) {
                                        y_idx = mid;
                                        break;
                                    } else if (y_coords[mid] > y_clipped) {
                                        right = mid - 1;
                                    } else {
                                        left = mid + 1;
                                    }
                                }
                                
                                // If we're at the edge of the grid, use nearest neighbor
                                if (x_idx == nx-1 || y_idx == ny-1) {
                                    z = grid[y_idx * nx + x_idx];
                                    return;
                                }
                                
                                // Calculate interpolation weights
                                float x0 = x_coords[x_idx];
                                float x1 = x_coords[x_idx+1];
                                float y0 = y_coords[y_idx];
                                float y1 = y_coords[y_idx+1];
                                
                                float wx = (x_clipped - x0) / (x1 - x0);
                                float wy = (y_clipped - y0) / (y1 - y0);
                                
                                // Get grid values at the four corners
                                float z00 = grid[y_idx * nx + x_idx];
                                float z01 = grid[y_idx * nx + (x_idx+1)];
                                float z10 = grid[(y_idx+1) * nx + x_idx];
                                float z11 = grid[(y_idx+1) * nx + (x_idx+1)];
                                
                                // Bilinear interpolation
                                z = (1-wx)*(1-wy)*z00 + wx*(1-wy)*z01 + (1-wx)*wy*z10 + wx*wy*z11;
                                ''',
                                'interpolate'
                            )
                            
                            # Apply the kernel to each asset
                            for i, (x, y) in enumerate(asset_locations):
                                interpolated_scores[i] = interpolate_kernel(
                                    cp.float32(x), cp.float32(y), 
                                    risk_data_device.astype(cp.float32), 
                                    x_coords_device.astype(cp.float32), 
                                    y_coords_device.astype(cp.float32)
                                )
                        else:
                            # Use CuPy's built-in map_coordinates for interpolation
                            # Get the grid shape
                            grid_shape = risk_data_device.shape
                            
                            # Normalize coordinates to grid indices
                            x_indices = cp.zeros(len(assets), dtype=cp.float32)
                            y_indices = cp.zeros(len(assets), dtype=cp.float32)
                            
                            for i, (x, y) in enumerate(asset_locations):
                                # Find indices for interpolation (normalized to grid coords)
                                x_norm = (x - rf.x_coords[0]) / (rf.x_coords[-1] - rf.x_coords[0]) * (len(rf.x_coords) - 1)
                                y_norm = (y - rf.y_coords[0]) / (rf.y_coords[-1] - rf.y_coords[0]) * (len(rf.y_coords) - 1)
                                
                                # Clip to valid range
                                x_norm = cp.clip(x_norm, 0, len(rf.x_coords) - 1)
                                y_norm = cp.clip(y_norm, 0, len(rf.y_coords) - 1)
                                
                                x_indices[i] = x_norm
                                y_indices[i] = y_norm
                            
                            # Interpolate each point
                            for i in range(len(assets)):
                                # Get interpolation coordinates
                                x_idx = x_indices[i]
                                y_idx = y_indices[i]
                                
                                # Get integer and fractional parts
                                x0 = int(cp.floor(x_idx))
                                y0 = int(cp.floor(y_idx))
                                x1 = min(x0 + 1, grid_shape[1] - 1)
                                y1 = min(y0 + 1, grid_shape[0] - 1)
                                
                                wx = x_idx - x0
                                wy = y_idx - y0
                                
                                # Bilinear interpolation
                                z00 = risk_data_device[y0, x0]
                                z01 = risk_data_device[y0, x1]
                                z10 = risk_data_device[y1, x0]
                                z11 = risk_data_device[y1, x1]
                                
                                interpolated_score = ((1-wx)*(1-wy)*z00 + 
                                                     wx*(1-wy)*z01 + 
                                                     (1-wx)*wy*z10 + 
                                                     wx*wy*z11)
                                
                                interpolated_scores[i] = interpolated_score
                        
                        # Scale by risk weight
                        interpolated_scores = interpolated_scores * rf.risk_weight
                        
                        # Copy back to CPU if necessary
                        interpolated_scores_cpu = cp.asnumpy(interpolated_scores)
                        
                        # Store scores
                        for i, asset_id in enumerate(asset_ids):
                            if asset_id not in all_factor_scores:
                                all_factor_scores[asset_id] = {}
                            all_factor_scores[asset_id][rf.name] = float(interpolated_scores_cpu[i])
                    else:
                        # CPU-based interpolation (fallback)
                        from scipy.interpolate import RegularGridInterpolator
                        
                        # Create interpolator
                        interpolator = RegularGridInterpolator(
                            (rf.y_coords, rf.x_coords), 
                            rf.risk_data,
                            bounds_error=False,
                            fill_value=None
                        )
                        
                        # Interpolate all locations at once
                        interpolated_scores = interpolator(asset_locations[:, ::-1])  # Reversed for (y, x) order
                        
                        # Scale by risk weight
                        interpolated_scores = interpolated_scores * rf.risk_weight
                        
                        # Store scores
                        for i, asset_id in enumerate(asset_ids):
                            if asset_id not in all_factor_scores:
                                all_factor_scores[asset_id] = {}
                            all_factor_scores[asset_id][rf.name] = float(interpolated_scores[i])
                except Exception as e:
                    self.logger.error(f"Error interpolating risk scores for {rf.name}: {e}")
        
        # Aggregate risk factors for each asset
        risk_scores = {}
        
        for asset_id in asset_ids:
            if asset_id in all_factor_scores:
                factor_scores = all_factor_scores[asset_id]
                
                if not factor_scores:
                    continue
                
                # Get factor scores and weights
                scores = []
                weights = []
                
                for rf in risk_factors:
                    if rf.name in factor_scores:
                        scores.append(factor_scores[rf.name])
                        weights.append(rf.risk_weight)
                
                if not scores:
                    continue
                
                # Normalize weights
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                
                # Aggregate scores based on method
                if aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
                    # Weighted average
                    risk_scores[asset_id] = sum(s * w for s, w in zip(scores, weights))
                elif aggregation_method == AggregationMethod.MAXIMUM:
                    # Maximum
                    risk_scores[asset_id] = max(scores)
                elif aggregation_method == AggregationMethod.PRODUCT:
                    # Product (with weight as exponent)
                    risk_scores[asset_id] = np.prod([s ** w for s, w in zip(scores, weights)])
                else:
                    # Default to weighted average
                    risk_scores[asset_id] = sum(s * w for s, w in zip(scores, weights))
        
        return risk_scores
    
    @staticmethod
    def benchmark_performance(portfolio: GeospatialPortfolio, 
                            risk_model: GeospatialRiskModel,
                            device_ids: List[int] = [-1, 0],
                            iterations: int = 3) -> Dict[str, float]:
        """
        Benchmark performance of risk assessment on different devices.
        
        Args:
            portfolio: Portfolio to assess
            risk_model: Risk model to use
            device_ids: List of device IDs to benchmark (-1 for CPU)
            iterations: Number of iterations for each benchmark
            
        Returns:
            Dictionary of benchmark results
        """
        results = {}
        
        # Benchmark original implementation (CPU)
        cpu_times = []
        for _ in range(iterations):
            start_time = time.time()
            _ = portfolio.assess_risk(risk_model)
            end_time = time.time()
            cpu_times.append(end_time - start_time)
        
        results["cpu"] = np.mean(cpu_times)
        
        # Benchmark Jetson-optimized implementation
        if HAS_CUPY:
            for device_id in device_ids:
                if device_id < 0:
                    continue  # Skip CPU, already benchmarked
                
                # Create optimizer for this device
                optimizer = JetsonOptimizer(device_id=device_id)
                
                # Optimize portfolio and risk model
                opt_portfolio = optimizer.optimize_portfolio(portfolio)
                opt_risk_model = optimizer.optimize_risk_model(risk_model)
                
                # Benchmark optimized implementation
                device_times = []
                for _ in range(iterations):
                    start_time = time.time()
                    _ = opt_portfolio.assess_risk(opt_risk_model)
                    end_time = time.time()
                    device_times.append(end_time - start_time)
                
                results[f"device_{device_id}"] = np.mean(device_times)
                
                # Test different memory profiles
                for profile in MemoryProfile:
                    # Create optimizer with specific memory profile
                    profile_optimizer = JetsonOptimizer(
                        device_id=device_id,
                        memory_profile=profile
                    )
                    
                    # Optimize portfolio and risk model
                    profile_portfolio = profile_optimizer.optimize_portfolio(portfolio)
                    profile_risk_model = profile_optimizer.optimize_risk_model(risk_model)
                    
                    # Benchmark optimized implementation
                    profile_times = []
                    for _ in range(iterations):
                        start_time = time.time()
                        _ = profile_portfolio.assess_risk(profile_risk_model)
                        end_time = time.time()
                        profile_times.append(end_time - start_time)
                    
                    results[f"device_{device_id}_{profile.value}"] = np.mean(profile_times)
        
        return results


def create_jetson_optimized_analyzer(portfolio: GeospatialPortfolio,
                                   risk_model: GeospatialRiskModel,
                                   device_id: int = 0,
                                   memory_profile: MemoryProfile = MemoryProfile.BALANCED) -> ScenarioAnalyzer:
    """
    Create a Jetson-optimized scenario analyzer.
    
    Args:
        portfolio: Portfolio to analyze
        risk_model: Risk model to use
        device_id: GPU device ID
        memory_profile: Memory usage profile
        
    Returns:
        Optimized scenario analyzer
    """
    # Create Jetson optimizer
    optimizer = JetsonOptimizer(
        device_id=device_id,
        memory_profile=memory_profile
    )
    
    # Optimize portfolio and risk model
    optimized_portfolio = optimizer.optimize_portfolio(portfolio)
    optimized_risk_model = optimizer.optimize_risk_model(risk_model)
    
    # Create analyzer with optimized components
    analyzer = ScenarioAnalyzer(
        portfolio=optimized_portfolio,
        risk_model=optimized_risk_model,
        device_id=device_id
    )
    
    return analyzer


if __name__ == "__main__":
    # Simple demonstration of Jetson device detection
    device_info = JetsonDeviceInfo()
    print(f"Device info: {device_info}")
    
    if HAS_CUPY:
        print("CuPy is available, can use GPU acceleration")
    else:
        print("CuPy is not available, using CPU fallback")