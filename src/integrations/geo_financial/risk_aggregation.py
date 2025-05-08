"""
Risk Aggregation Module

This module provides advanced methods for aggregating multiple risk factors in the
geospatial financial integration, including weighted aggregation, copula-based
aggregation, and GPU-accelerated risk surface generation.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class AggregationMethod(Enum):
    """Enumeration of risk aggregation methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    WEIGHTED_MAX = "weighted_max"
    WEIGHTED_PRODUCT = "weighted_product"
    COPULA_GAUSSIAN = "copula_gaussian"
    COPULA_STUDENT_T = "copula_student_t"


class RiskAggregator:
    """
    Class for aggregating multiple spatial risk factors.
    
    This class provides various methods for combining multiple risk factors
    into a single comprehensive risk measure, using different aggregation
    strategies appropriate for different types of risks.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize a risk aggregator.
        
        Args:
            device_id: CUDA device ID (negative for CPU)
        """
        self.device_id = device_id
        self.use_gpu = device_id >= 0 and HAS_CUPY
    
    def aggregate_risk_factors(self,
                              risk_factors: List['SpatialRiskFactor'],
                              method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
                              correlation_matrix: Optional[np.ndarray] = None,
                              degrees_of_freedom: int = 5) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aggregate multiple risk factors into a single risk map.
        
        Args:
            risk_factors: List of SpatialRiskFactor objects
            method: Aggregation method to use
            correlation_matrix: Optional correlation matrix for copula methods
            degrees_of_freedom: Degrees of freedom for Student's t-copula
            
        Returns:
            Tuple of (aggregated_risk_map, statistics)
        """
        if not risk_factors:
            raise ValueError("No risk factors provided")
        
        # Extract risk data and weights
        risk_data_list = []
        weights = []
        
        for factor in risk_factors:
            risk_data_list.append(factor.risk_data)
            weights.append(factor.risk_weight)
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(weights)
        if weight_sum > 0:
            normalized_weights = [w / weight_sum for w in weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)
        
        # Check that all risk data arrays have the same shape
        shape = risk_data_list[0].shape
        for i, data in enumerate(risk_data_list):
            if data.shape != shape:
                raise ValueError(f"Risk factor {i} has shape {data.shape}, expected {shape}")
        
        # Choose aggregation method
        if method == AggregationMethod.WEIGHTED_AVERAGE:
            aggregated, stats = self._weighted_average(risk_data_list, normalized_weights)
        elif method == AggregationMethod.WEIGHTED_MAX:
            aggregated, stats = self._weighted_max(risk_data_list, normalized_weights)
        elif method == AggregationMethod.WEIGHTED_PRODUCT:
            aggregated, stats = self._weighted_product(risk_data_list, normalized_weights)
        elif method == AggregationMethod.COPULA_GAUSSIAN:
            if correlation_matrix is None:
                correlation_matrix = self._estimate_correlation(risk_data_list)
            aggregated, stats = self._copula_gaussian(risk_data_list, normalized_weights, correlation_matrix)
        elif method == AggregationMethod.COPULA_STUDENT_T:
            if correlation_matrix is None:
                correlation_matrix = self._estimate_correlation(risk_data_list)
            aggregated, stats = self._copula_student_t(risk_data_list, normalized_weights, correlation_matrix, degrees_of_freedom)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return aggregated, stats
    
    def _weighted_average(self, 
                         risk_data_list: List[np.ndarray], 
                         weights: List[float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aggregate risk factors using a weighted average.
        
        Args:
            risk_data_list: List of risk data arrays
            weights: List of normalized weights
            
        Returns:
            Tuple of (aggregated_risk_map, statistics)
        """
        if self.use_gpu:
            try:
                # Move data to GPU
                gpu_data = [cp.array(data) for data in risk_data_list]
                gpu_weights = cp.array(weights).reshape(-1, 1, 1)
                
                # Calculate weighted average
                weighted_sum = cp.sum(gpu_data[i] * gpu_weights[i] for i in range(len(gpu_data)))
                
                # Move result back to CPU
                result = cp.asnumpy(weighted_sum)
                
                # Calculate statistics
                stats = {
                    "min": float(cp.min(weighted_sum)),
                    "max": float(cp.max(weighted_sum)),
                    "mean": float(cp.mean(weighted_sum)),
                    "std": float(cp.std(weighted_sum))
                }
                
                return result, stats
            except Exception as e:
                print(f"GPU calculation failed: {e}")
                print("Falling back to CPU implementation")
        
        # CPU implementation
        result = np.zeros_like(risk_data_list[0])
        for i, data in enumerate(risk_data_list):
            result += data * weights[i]
        
        # Calculate statistics
        stats = {
            "min": float(np.min(result)),
            "max": float(np.max(result)),
            "mean": float(np.mean(result)),
            "std": float(np.std(result))
        }
        
        return result, stats
    
    def _weighted_max(self,
                     risk_data_list: List[np.ndarray],
                     weights: List[float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aggregate risk factors using a weighted maximum.
        
        This method is useful when you want to prioritize the highest risk
        factor at each location. The weight determines the contribution of
        each risk factor to the maximum.
        
        Args:
            risk_data_list: List of risk data arrays
            weights: List of normalized weights
            
        Returns:
            Tuple of (aggregated_risk_map, statistics)
        """
        if self.use_gpu:
            try:
                # Move data to GPU
                gpu_data = [cp.array(data) * w for data, w in zip(risk_data_list, weights)]
                
                # Calculate weighted maximum
                stacked = cp.stack(gpu_data, axis=0)
                weighted_max = cp.max(stacked, axis=0)
                
                # Move result back to CPU
                result = cp.asnumpy(weighted_max)
                
                # Calculate statistics
                stats = {
                    "min": float(cp.min(weighted_max)),
                    "max": float(cp.max(weighted_max)),
                    "mean": float(cp.mean(weighted_max)),
                    "std": float(cp.std(weighted_max))
                }
                
                return result, stats
            except Exception as e:
                print(f"GPU calculation failed: {e}")
                print("Falling back to CPU implementation")
        
        # CPU implementation
        weighted_data = [data * w for data, w in zip(risk_data_list, weights)]
        stacked = np.stack(weighted_data, axis=0)
        result = np.max(stacked, axis=0)
        
        # Calculate statistics
        stats = {
            "min": float(np.min(result)),
            "max": float(np.max(result)),
            "mean": float(np.mean(result)),
            "std": float(np.std(result))
        }
        
        return result, stats
    
    def _weighted_product(self,
                         risk_data_list: List[np.ndarray],
                         weights: List[float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aggregate risk factors using a weighted product.
        
        This method is useful for combining independent risk factors where
        the combined probability is the product of individual probabilities.
        
        Args:
            risk_data_list: List of risk data arrays
            weights: List of normalized weights
            
        Returns:
            Tuple of (aggregated_risk_map, statistics)
        """
        if self.use_gpu:
            try:
                # Move data to GPU
                gpu_data = [cp.array(data) ** w for data, w in zip(risk_data_list, weights)]
                
                # Calculate weighted product
                result_gpu = gpu_data[0].copy()
                for i in range(1, len(gpu_data)):
                    result_gpu *= gpu_data[i]
                
                # Move result back to CPU
                result = cp.asnumpy(result_gpu)
                
                # Calculate statistics
                stats = {
                    "min": float(cp.min(result_gpu)),
                    "max": float(cp.max(result_gpu)),
                    "mean": float(cp.mean(result_gpu)),
                    "std": float(cp.std(result_gpu))
                }
                
                return result, stats
            except Exception as e:
                print(f"GPU calculation failed: {e}")
                print("Falling back to CPU implementation")
        
        # CPU implementation
        result = np.ones_like(risk_data_list[0])
        for i, data in enumerate(risk_data_list):
            result *= data ** weights[i]
        
        # Calculate statistics
        stats = {
            "min": float(np.min(result)),
            "max": float(np.max(result)),
            "mean": float(np.mean(result)),
            "std": float(np.std(result))
        }
        
        return result, stats
    
    def _estimate_correlation(self, risk_data_list: List[np.ndarray]) -> np.ndarray:
        """
        Estimate the correlation matrix between risk factors.
        
        Args:
            risk_data_list: List of risk data arrays
            
        Returns:
            Correlation matrix
        """
        n_factors = len(risk_data_list)
        corr_matrix = np.eye(n_factors)
        
        # Flatten arrays for correlation calculation
        flat_data = [data.flatten() for data in risk_data_list]
        
        # Calculate correlation matrix
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                # Calculate correlation coefficient
                corr = np.corrcoef(flat_data[i], flat_data[j])[0, 1]
                corr_matrix[i, j] = corr_matrix[j, i] = corr
        
        return corr_matrix
    
    def _copula_gaussian(self,
                        risk_data_list: List[np.ndarray],
                        weights: List[float],
                        correlation_matrix: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aggregate risk factors using a Gaussian copula.
        
        This method uses a Gaussian copula to model the dependence structure
        between risk factors, which is more sophisticated than simple weighted
        averaging.
        
        Args:
            risk_data_list: List of risk data arrays
            weights: List of normalized weights
            correlation_matrix: Correlation matrix between risk factors
            
        Returns:
            Tuple of (aggregated_risk_map, statistics)
        """
        from scipy import stats
        
        # Get dimensions
        n_factors = len(risk_data_list)
        shape = risk_data_list[0].shape
        flattened_shape = risk_data_list[0].size
        
        # Flatten arrays
        flat_data = [data.flatten() for data in risk_data_list]
        
        # Convert risk scores to standard normal quantiles (via probability integral transform)
        normal_quantiles = []
        for data in flat_data:
            # Convert to empirical CDF values (between 0 and 1)
            ecdf = stats.rankdata(data) / (flattened_shape + 1)
            # Convert to standard normal quantiles
            quantiles = stats.norm.ppf(ecdf)
            normal_quantiles.append(quantiles)
        
        # Create multivariate normal random variables
        mv_normal = np.vstack(normal_quantiles).T  # Shape: (flattened_shape, n_factors)
        
        # Apply weights to the correlation matrix
        weighted_corr = correlation_matrix.copy()
        for i in range(n_factors):
            for j in range(n_factors):
                weighted_corr[i, j] *= np.sqrt(weights[i] * weights[j])
        
        # Compute the weighted sum of the multivariate normal variables
        weighted_sum = np.zeros(flattened_shape)
        for i in range(n_factors):
            weighted_sum += mv_normal[:, i] * weights[i]
        
        # Convert back to uniform(0,1) via standard normal CDF
        uniform_scores = stats.norm.cdf(weighted_sum)
        
        # Scale to 0-1 risk scores
        result = uniform_scores.reshape(shape)
        
        # Calculate statistics
        stats_dict = {
            "min": float(np.min(result)),
            "max": float(np.max(result)),
            "mean": float(np.mean(result)),
            "std": float(np.std(result)),
            "correlation_matrix": correlation_matrix.tolist()
        }
        
        return result, stats_dict
    
    def _copula_student_t(self,
                         risk_data_list: List[np.ndarray],
                         weights: List[float],
                         correlation_matrix: np.ndarray,
                         df: int = 5) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Aggregate risk factors using a Student's t-copula.
        
        This method uses a Student's t-copula to model the dependence structure
        between risk factors, which is more sophisticated than simple weighted
        averaging and allows for more tail dependence than a Gaussian copula.
        
        Args:
            risk_data_list: List of risk data arrays
            weights: List of normalized weights
            correlation_matrix: Correlation matrix between risk factors
            df: Degrees of freedom for the Student's t distribution
            
        Returns:
            Tuple of (aggregated_risk_map, statistics)
        """
        from scipy import stats
        
        # Get dimensions
        n_factors = len(risk_data_list)
        shape = risk_data_list[0].shape
        flattened_shape = risk_data_list[0].size
        
        # Flatten arrays
        flat_data = [data.flatten() for data in risk_data_list]
        
        # Convert risk scores to Student's t quantiles (via probability integral transform)
        t_quantiles = []
        for data in flat_data:
            # Convert to empirical CDF values (between 0 and 1)
            ecdf = stats.rankdata(data) / (flattened_shape + 1)
            # Convert to Student's t quantiles
            quantiles = stats.t.ppf(ecdf, df)
            t_quantiles.append(quantiles)
        
        # Create multivariate t random variables
        mv_t = np.vstack(t_quantiles).T  # Shape: (flattened_shape, n_factors)
        
        # Apply weights to the correlation matrix
        weighted_corr = correlation_matrix.copy()
        for i in range(n_factors):
            for j in range(n_factors):
                weighted_corr[i, j] *= np.sqrt(weights[i] * weights[j])
        
        # Compute the weighted sum of the multivariate t variables
        weighted_sum = np.zeros(flattened_shape)
        for i in range(n_factors):
            weighted_sum += mv_t[:, i] * weights[i]
        
        # Convert back to uniform(0,1) via Student's t CDF
        uniform_scores = stats.t.cdf(weighted_sum, df)
        
        # Scale to 0-1 risk scores
        result = uniform_scores.reshape(shape)
        
        # Calculate statistics
        stats_dict = {
            "min": float(np.min(result)),
            "max": float(np.max(result)),
            "mean": float(np.mean(result)),
            "std": float(np.std(result)),
            "correlation_matrix": correlation_matrix.tolist(),
            "degrees_of_freedom": df
        }
        
        return result, stats_dict


class RiskSurfaceGenerator:
    """
    Class for generating risk surfaces from point data.
    
    This class provides methods for interpolating risk values from point data
    to create continuous risk surfaces, which can be useful when risk factors
    are only known at specific locations.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize a risk surface generator.
        
        Args:
            device_id: CUDA device ID (negative for CPU)
        """
        self.device_id = device_id
        self.use_gpu = device_id >= 0 and HAS_CUPY
    
    def interpolate_risk_surface(self,
                                points_x: np.ndarray,
                                points_y: np.ndarray,
                                risk_values: np.ndarray,
                                grid_size: Tuple[int, int],
                                x_range: Tuple[float, float],
                                y_range: Tuple[float, float],
                                method: str = 'idw',
                                power: float = 2.0,
                                smoothing: float = 0.0) -> np.ndarray:
        """
        Interpolate a risk surface from point data.
        
        Args:
            points_x: X coordinates of points
            points_y: Y coordinates of points
            risk_values: Risk values at points
            grid_size: Size of output grid (height, width)
            x_range: Range of X coordinates (min, max)
            y_range: Range of Y coordinates (min, max)
            method: Interpolation method ('idw', 'rbf', or 'kriging')
            power: Power parameter for IDW method
            smoothing: Smoothing parameter for RBF method
            
        Returns:
            Interpolated risk surface as 2D array
        """
        if method.lower() == 'idw':
            return self._idw_interpolation(
                points_x, points_y, risk_values, grid_size, x_range, y_range, power
            )
        elif method.lower() == 'rbf':
            return self._rbf_interpolation(
                points_x, points_y, risk_values, grid_size, x_range, y_range, smoothing
            )
        elif method.lower() == 'kriging':
            return self._kriging_interpolation(
                points_x, points_y, risk_values, grid_size, x_range, y_range
            )
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
    
    def _idw_interpolation(self,
                          points_x: np.ndarray,
                          points_y: np.ndarray,
                          risk_values: np.ndarray,
                          grid_size: Tuple[int, int],
                          x_range: Tuple[float, float],
                          y_range: Tuple[float, float],
                          power: float = 2.0) -> np.ndarray:
        """
        Interpolate using Inverse Distance Weighting (IDW).
        
        This method interpolates values based on the weighted average of nearby points,
        where weights decrease with distance according to a power function.
        
        Args:
            points_x: X coordinates of points
            points_y: Y coordinates of points
            risk_values: Risk values at points
            grid_size: Size of output grid (height, width)
            x_range: Range of X coordinates (min, max)
            y_range: Range of Y coordinates (min, max)
            power: Power parameter (higher values increase the influence of nearby points)
            
        Returns:
            Interpolated risk surface as 2D array
        """
        height, width = grid_size
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Create output grid
        result = np.zeros((height, width), dtype=np.float32)
        
        # Create coordinate meshgrid
        x_coords = np.linspace(x_min, x_max, width)
        y_coords = np.linspace(y_max, y_min, height)  # Reversed for image coordinates
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        if self.use_gpu:
            try:
                # Move data to GPU
                gpu_points_x = cp.array(points_x)
                gpu_points_y = cp.array(points_y)
                gpu_risk_values = cp.array(risk_values)
                gpu_xx = cp.array(xx)
                gpu_yy = cp.array(yy)
                
                # Compute distances from each grid point to each data point
                # This is done in a vectorized way to avoid loops
                n_points = len(points_x)
                
                # Reshape grid coordinates to iterate over them
                grid_shape = gpu_xx.shape
                gpu_xx_flat = gpu_xx.flatten()
                gpu_yy_flat = gpu_yy.flatten()
                
                # Initialize result arrays
                gpu_result_flat = cp.zeros_like(gpu_xx_flat)
                gpu_weight_sum_flat = cp.zeros_like(gpu_xx_flat)
                
                # Process in batches to avoid memory issues
                batch_size = 10000
                n_batches = (len(gpu_xx_flat) + batch_size - 1) // batch_size
                
                for b in range(n_batches):
                    start_idx = b * batch_size
                    end_idx = min((b + 1) * batch_size, len(gpu_xx_flat))
                    
                    batch_xx = gpu_xx_flat[start_idx:end_idx, cp.newaxis]
                    batch_yy = gpu_yy_flat[start_idx:end_idx, cp.newaxis]
                    
                    # Calculate distances
                    batch_dist = cp.sqrt(
                        (batch_xx - gpu_points_x) ** 2 + 
                        (batch_yy - gpu_points_y) ** 2
                    )
                    
                    # Apply small epsilon to avoid division by zero
                    batch_dist = cp.maximum(batch_dist, 1e-10)
                    
                    # Calculate weights
                    weights = 1.0 / (batch_dist ** power)
                    
                    # Calculate weighted sum
                    weighted_values = weights * gpu_risk_values
                    gpu_result_flat[start_idx:end_idx] = cp.sum(weighted_values, axis=1)
                    gpu_weight_sum_flat[start_idx:end_idx] = cp.sum(weights, axis=1)
                
                # Normalize by weight sum
                gpu_result_flat /= gpu_weight_sum_flat
                
                # Reshape result back to grid
                gpu_result = gpu_result_flat.reshape(grid_shape)
                
                # Move result back to CPU
                result = cp.asnumpy(gpu_result)
                
                return result
            except Exception as e:
                print(f"GPU calculation failed: {e}")
                print("Falling back to CPU implementation")
        
        # CPU implementation
        for i in range(height):
            for j in range(width):
                x, y = xx[i, j], yy[i, j]
                
                # Calculate distances to all points
                distances = np.sqrt((x - points_x) ** 2 + (y - points_y) ** 2)
                
                # Apply small epsilon to avoid division by zero
                distances = np.maximum(distances, 1e-10)
                
                # Calculate weights
                weights = 1.0 / (distances ** power)
                
                # Calculate weighted average
                result[i, j] = np.sum(weights * risk_values) / np.sum(weights)
        
        return result
    
    def _rbf_interpolation(self,
                          points_x: np.ndarray,
                          points_y: np.ndarray,
                          risk_values: np.ndarray,
                          grid_size: Tuple[int, int],
                          x_range: Tuple[float, float],
                          y_range: Tuple[float, float],
                          smoothing: float = 0.0) -> np.ndarray:
        """
        Interpolate using Radial Basis Function (RBF).
        
        This method interpolates values using radial basis functions, which can provide
        smoother interpolation than IDW.
        
        Args:
            points_x: X coordinates of points
            points_y: Y coordinates of points
            risk_values: Risk values at points
            grid_size: Size of output grid (height, width)
            x_range: Range of X coordinates (min, max)
            y_range: Range of Y coordinates (min, max)
            smoothing: Smoothing parameter
            
        Returns:
            Interpolated risk surface as 2D array
        """
        try:
            from scipy.interpolate import Rbf
        except ImportError:
            raise ImportError("SciPy is required for RBF interpolation")
        
        height, width = grid_size
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Create coordinate meshgrid
        x_coords = np.linspace(x_min, x_max, width)
        y_coords = np.linspace(y_max, y_min, height)  # Reversed for image coordinates
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Create RBF interpolator
        rbf = Rbf(points_x, points_y, risk_values, function='multiquadric', smooth=smoothing)
        
        # Interpolate values
        result = rbf(xx, yy)
        
        # Clip to valid range
        result = np.clip(result, 0.0, 1.0)
        
        return result
    
    def _kriging_interpolation(self,
                              points_x: np.ndarray,
                              points_y: np.ndarray,
                              risk_values: np.ndarray,
                              grid_size: Tuple[int, int],
                              x_range: Tuple[float, float],
                              y_range: Tuple[float, float]) -> np.ndarray:
        """
        Interpolate using Kriging (Gaussian Process Regression).
        
        This method interpolates values using Kriging, which provides a statistical
        approach to interpolation and can estimate uncertainty.
        
        Args:
            points_x: X coordinates of points
            points_y: Y coordinates of points
            risk_values: Risk values at points
            grid_size: Size of output grid (height, width)
            x_range: Range of X coordinates (min, max)
            y_range: Range of Y coordinates (min, max)
            
        Returns:
            Interpolated risk surface as 2D array
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        except ImportError:
            raise ImportError("scikit-learn is required for Kriging interpolation")
        
        height, width = grid_size
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Create coordinate meshgrid
        x_coords = np.linspace(x_min, x_max, width)
        y_coords = np.linspace(y_max, y_min, height)  # Reversed for image coordinates
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Stack coordinates as features
        X = np.column_stack((points_x, points_y))
        y = risk_values
        
        # Create and fit Gaussian Process model
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp.fit(X, y)
        
        # Predict on grid
        X_grid = np.column_stack((xx.ravel(), yy.ravel()))
        y_pred = gp.predict(X_grid)
        
        # Reshape to grid
        result = y_pred.reshape(height, width)
        
        # Clip to valid range
        result = np.clip(result, 0.0, 1.0)
        
        return result


# Utility functions

def compute_correlation_matrix(risk_factors: List['SpatialRiskFactor']) -> np.ndarray:
    """
    Compute the correlation matrix between risk factors.
    
    Args:
        risk_factors: List of SpatialRiskFactor objects
        
    Returns:
        Correlation matrix
    """
    n_factors = len(risk_factors)
    corr_matrix = np.eye(n_factors)
    
    # Flatten arrays for correlation calculation
    flat_data = [factor.risk_data.flatten() for factor in risk_factors]
    
    # Calculate correlation matrix
    for i in range(n_factors):
        for j in range(i+1, n_factors):
            # Calculate correlation coefficient
            corr = np.corrcoef(flat_data[i], flat_data[j])[0, 1]
            corr_matrix[i, j] = corr_matrix[j, i] = corr
    
    return corr_matrix


def create_combined_risk_factor(risk_factors: List['SpatialRiskFactor'],
                                name: str,
                                description: str,
                                method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
                                device_id: int = 0) -> 'SpatialRiskFactor':
    """
    Create a combined risk factor from multiple individual risk factors.
    
    Args:
        risk_factors: List of SpatialRiskFactor objects to combine
        name: Name for the combined risk factor
        description: Description of the combined risk factor
        method: Aggregation method to use
        device_id: CUDA device ID (negative for CPU)
        
    Returns:
        A new SpatialRiskFactor representing the combined risk
    """
    # Import locally to avoid circular imports
    from .geo_risk import SpatialRiskFactor
    
    # Create risk aggregator
    aggregator = RiskAggregator(device_id=device_id)
    
    # Compute correlation matrix if needed
    if method in [AggregationMethod.COPULA_GAUSSIAN, AggregationMethod.COPULA_STUDENT_T]:
        correlation_matrix = compute_correlation_matrix(risk_factors)
    else:
        correlation_matrix = None
    
    # Aggregate risk factors
    risk_data, _ = aggregator.aggregate_risk_factors(
        risk_factors=risk_factors,
        method=method,
        correlation_matrix=correlation_matrix
    )
    
    # Get geo_transform from first risk factor
    geo_transform = risk_factors[0].geo_transform
    
    # Create new risk factor
    return SpatialRiskFactor(
        name=name,
        description=description,
        risk_weight=1.0,  # This is already a weighted combination
        spatial_data=risk_data,
        geo_transform=geo_transform
    )


def interpolate_asset_risk_surface(portfolio: 'GeospatialPortfolio',
                                  risk_scores: Dict[str, float],
                                  grid_size: Tuple[int, int],
                                  x_range: Tuple[float, float],
                                  y_range: Tuple[float, float],
                                  method: str = 'idw',
                                  device_id: int = 0) -> Tuple[np.ndarray, 'GeoTransform']:
    """
    Interpolate a risk surface from asset risk scores.
    
    Args:
        portfolio: GeospatialPortfolio object
        risk_scores: Dictionary mapping asset IDs to risk scores
        grid_size: Size of output grid (height, width)
        x_range: Range of X coordinates (min, max)
        y_range: Range of Y coordinates (min, max)
        method: Interpolation method ('idw', 'rbf', or 'kriging')
        device_id: CUDA device ID (negative for CPU)
        
    Returns:
        Tuple of (risk_surface, geo_transform)
    """
    # Import locally to avoid circular imports
    from geospatial.dem import GeoTransform
    
    # Extract asset locations and risk scores
    points_x = []
    points_y = []
    values = []
    
    for asset in portfolio.assets:
        if asset['id'] in risk_scores:
            points_x.append(asset['x'])
            points_y.append(asset['y'])
            values.append(risk_scores[asset['id']])
    
    # Convert to numpy arrays
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    values = np.array(values)
    
    # Create risk surface generator
    generator = RiskSurfaceGenerator(device_id=device_id)
    
    # Interpolate risk surface
    risk_surface = generator.interpolate_risk_surface(
        points_x=points_x,
        points_y=points_y,
        risk_values=values,
        grid_size=grid_size,
        x_range=x_range,
        y_range=y_range,
        method=method
    )
    
    # Create geo transform
    geo_transform = GeoTransform([
        x_range[0],                                       # top-left x
        (x_range[1] - x_range[0]) / (grid_size[1] - 1),  # w-e pixel resolution
        0,                                                # row rotation
        y_range[1],                                       # top-left y
        0,                                                # column rotation
        (y_range[0] - y_range[1]) / (grid_size[0] - 1)   # n-s pixel resolution (negative)
    ])
    
    return risk_surface, geo_transform