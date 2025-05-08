"""
Visualization Module

This module provides visualization utilities for geospatial financial risk analysis,
combining visualization features from both the Financial Modeling and Geospatial Analysis workloads.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import seaborn as sns

from .geo_risk import GeospatialRiskModel, GeospatialPortfolio, SpatialRiskFactor


class GeoFinancialVisualizer:
    """
    Visualizer for geospatial financial data and analysis results.
    
    This class provides methods for creating various visualizations that combine
    geospatial and financial data, including risk maps, portfolio distributions,
    and optimization results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches
        """
        self.figsize = figsize
    
    def plot_risk_map(self, 
                     risk_map: np.ndarray, 
                     title: str = "Geospatial Risk Map",
                     colormap: str = "viridis_r",
                     add_colorbar: bool = True) -> plt.Figure:
        """
        Plot a geospatial risk map.
        
        Args:
            risk_map: 2D array of risk values
            title: Title for the plot
            colormap: Matplotlib colormap name
            add_colorbar: Whether to add a colorbar
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot risk map
        im = ax.imshow(risk_map, cmap=colormap, origin='upper')
        
        # Add title
        ax.set_title(title, fontsize=14)
        
        # Add colorbar if requested
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Risk Level', fontsize=12)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def plot_portfolio_on_risk_map(self,
                                 risk_map: np.ndarray,
                                 geo_transform: 'GeoTransform',
                                 portfolio: 'GeospatialPortfolio',
                                 title: str = "Portfolio Assets on Risk Map",
                                 colormap: str = "viridis_r",
                                 marker_size_scale: float = 30.0) -> plt.Figure:
        """
        Plot portfolio assets on a geospatial risk map.
        
        Args:
            risk_map: 2D array of risk values
            geo_transform: GeoTransform object for the risk map
            portfolio: GeospatialPortfolio object
            title: Title for the plot
            colormap: Matplotlib colormap name
            marker_size_scale: Scaling factor for marker sizes
            
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot risk map
        im = ax.imshow(risk_map, cmap=colormap, origin='upper', alpha=0.7)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Risk Level', fontsize=12)
        
        # Get asset locations
        x_coords, y_coords, names, values = portfolio.get_asset_locations()
        
        # Convert geographic coordinates to pixel coordinates for the risk map
        pixel_coords = []
        for x, y in zip(x_coords, y_coords):
            pixel_x, pixel_y = geo_transform.geo_to_pixel(x, y)
            pixel_coords.append((pixel_x, pixel_y))
        
        pixel_x_coords, pixel_y_coords = zip(*pixel_coords)
        
        # Calculate marker sizes based on asset values
        marker_sizes = np.array(values) / max(values) * marker_size_scale
        
        # Plot assets
        sc = ax.scatter(pixel_x_coords, pixel_y_coords, s=marker_sizes, c='red',
                        marker='o', alpha=0.7, edgecolors='white', linewidths=1)
        
        # Add title
        ax.set_title(title, fontsize=14)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def plot_portfolio_risk_heatmap(self,
                                  portfolio: 'GeospatialPortfolio',
                                  risk_scores: Dict[str, float],
                                  title: str = "Asset Risk Heatmap",
                                  colormap: str = "viridis") -> plt.Figure:
        """
        Plot a heatmap of asset values vs. risk scores.
        
        Args:
            portfolio: GeospatialPortfolio object
            risk_scores: Dictionary mapping asset IDs to risk scores
            title: Title for the plot
            colormap: Matplotlib colormap name
            
        Returns:
            Matplotlib Figure object
        """
        # Extract asset data
        data = []
        for asset in portfolio.assets:
            if asset['id'] in risk_scores:
                data.append({
                    'id': asset['id'],
                    'name': asset['name'],
                    'value': asset['value'],
                    'risk_score': risk_scores[asset['id']]
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Sort by value (descending)
        df = df.sort_values('value', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(
            df[['risk_score']].T,
            cmap=colormap,
            linewidths=1,
            linecolor='white',
            square=True,
            cbar_kws={"shrink": 0.5},
            ax=ax
        )
        
        # Set custom y-ticks
        ax.set_yticks([0.5])
        ax.set_yticklabels(['Risk Score'], rotation=0)
        
        # Set custom x-ticks
        ax.set_xticks(np.arange(len(df)) + 0.5)
        ax.set_xticklabels(df['name'], rotation=45, ha='right')
        
        # Add asset values as text
        for i, value in enumerate(df['value']):
            ax.text(i + 0.5, 0.85, f"${value/1000:.1f}K", 
                   ha='center', va='center', color='white', fontsize=10)
        
        # Add title
        ax.set_title(title, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_risk_factor_comparison(self,
                                   risk_factors: List['SpatialRiskFactor'],
                                   portfolio: 'GeospatialPortfolio') -> plt.Figure:
        """
        Plot a comparison of how different risk factors affect the portfolio.
        
        Args:
            risk_factors: List of SpatialRiskFactor objects
            portfolio: GeospatialPortfolio object
            
        Returns:
            Matplotlib Figure object
        """
        # Calculate risk scores for each factor
        factor_risk_scores = []
        for factor in risk_factors:
            scores = {}
            for asset in portfolio.assets:
                try:
                    scores[asset['id']] = factor.get_risk_at_point(asset['x'], asset['y'])
                except ValueError:
                    scores[asset['id']] = 0.0
            factor_risk_scores.append(scores)
        
        # Extract asset data
        asset_names = [asset['name'] for asset in portfolio.assets]
        asset_ids = [asset['id'] for asset in portfolio.assets]
        
        # Create matrix of risk scores
        risk_matrix = np.zeros((len(risk_factors), len(asset_ids)))
        for i, scores in enumerate(factor_risk_scores):
            for j, asset_id in enumerate(asset_ids):
                risk_matrix[i, j] = scores.get(asset_id, 0.0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(
            risk_matrix,
            cmap='viridis',
            linewidths=1,
            linecolor='white',
            square=True,
            cbar_kws={"shrink": 0.5},
            ax=ax
        )
        
        # Set custom y-ticks
        ax.set_yticks(np.arange(len(risk_factors)) + 0.5)
        ax.set_yticklabels([factor.name for factor in risk_factors], rotation=0)
        
        # Set custom x-ticks
        ax.set_xticks(np.arange(len(asset_names)) + 0.5)
        ax.set_xticklabels(asset_names, rotation=45, ha='right')
        
        # Add title
        ax.set_title("Risk Factor Comparison by Asset", fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_portfolio_optimization_results(self,
                                          portfolio: 'GeospatialPortfolio',
                                          original_weights: Dict[str, float],
                                          optimized_weights: Dict[str, float],
                                          risk_scores: Dict[str, float],
                                          title: str = "Portfolio Optimization Results") -> plt.Figure:
        """
        Plot the results of portfolio optimization.
        
        Args:
            portfolio: GeospatialPortfolio object
            original_weights: Dictionary mapping asset IDs to original weights
            optimized_weights: Dictionary mapping asset IDs to optimized weights
            risk_scores: Dictionary mapping asset IDs to risk scores
            title: Title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Extract asset data
        data = []
        for asset in portfolio.assets:
            if asset['id'] in original_weights and asset['id'] in optimized_weights:
                data.append({
                    'id': asset['id'],
                    'name': asset['name'],
                    'original_weight': original_weights[asset['id']],
                    'optimized_weight': optimized_weights[asset['id']],
                    'risk_score': risk_scores.get(asset['id'], 0.0)
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Sort by risk score (descending)
        df = df.sort_values('risk_score', ascending=False)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Plot original weights
        df_melted = pd.melt(df, id_vars=['name', 'risk_score'], 
                            value_vars=['original_weight', 'optimized_weight'],
                            var_name='Weight Type', value_name='Weight')
        
        # Use custom colormap based on risk scores
        norm = Normalize(vmin=0, vmax=1)
        risk_colors = plt.cm.viridis_r(norm(df['risk_score']))
        
        # Bar chart of weights
        g = sns.barplot(x='name', y='Weight', hue='Weight Type', data=df_melted, ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_title('Original vs. Optimized Weights')
        ax1.set_ylabel('Portfolio Weight')
        ax1.set_xlabel('')
        
        # Add risk scores as text
        for i, row in enumerate(df.itertuples()):
            ax1.text(i, -0.02, f"Risk: {row.risk_score:.2f}", 
                    ha='center', va='top', rotation=45, fontsize=8)
        
        # Plot risk-weight scatter plot
        scatter = ax2.scatter(df['risk_score'], df['original_weight'], 
                             label='Original', alpha=0.7, s=100, c='blue')
        scatter = ax2.scatter(df['risk_score'], df['optimized_weight'], 
                             label='Optimized', alpha=0.7, s=100, c='orange')
        
        # Add arrow to show weight changes
        for i, row in enumerate(df.itertuples()):
            ax2.annotate('', 
                        xy=(row.risk_score, row.optimized_weight),
                        xytext=(row.risk_score, row.original_weight),
                        arrowprops=dict(arrowstyle='->', linewidth=1, color='red', alpha=0.5))
            ax2.text(row.risk_score+0.01, (row.original_weight+row.optimized_weight)/2, 
                    row.name, fontsize=8)
        
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Portfolio Weight')
        ax2.set_title('Weight vs. Risk Adjustment')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_efficient_frontier_with_risk_overlay(self,
                                                returns: np.ndarray,
                                                covariance: np.ndarray,
                                                risk_scores: np.ndarray,
                                                num_portfolios: int = 1000,
                                                title: str = "Efficient Frontier with Risk Overlay") -> plt.Figure:
        """
        Plot the efficient frontier with a risk overlay.
        
        Args:
            returns: Array of expected returns for each asset
            covariance: Covariance matrix of returns
            risk_scores: Array of geospatial risk scores for each asset
            num_portfolios: Number of portfolios to generate
            title: Title for the plot
            
        Returns:
            Matplotlib Figure object
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Generate random portfolios
        num_assets = len(returns)
        results = []
        
        for _ in range(num_portfolios):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            # Calculate portfolio return and risk
            portfolio_return = np.sum(weights * returns)
            portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
            
            # Calculate geospatial risk
            geo_risk = np.sum(weights * risk_scores)
            
            results.append({
                'weights': weights,
                'return': portfolio_return,
                'risk': portfolio_stddev,
                'geo_risk': geo_risk
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create scatter plot with geospatial risk as color
        scatter = ax.scatter(df['risk'], df['return'], 
                           c=df['geo_risk'], cmap='viridis_r',
                           alpha=0.7, s=50, edgecolors='white', linewidths=0.5)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Geospatial Risk Exposure')
        
        # Set labels
        ax.set_xlabel('Portfolio Risk (Standard Deviation)')
        ax.set_ylabel('Portfolio Return')
        ax.set_title(title, fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def create_dashboard(self,
                        risk_model: 'GeospatialRiskModel',
                        portfolio: 'GeospatialPortfolio',
                        risk_map: np.ndarray,
                        geo_transform: 'GeoTransform',
                        optimized_weights: Optional[Dict[str, float]] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard of geospatial financial analysis.
        
        Args:
            risk_model: GeospatialRiskModel object
            portfolio: GeospatialPortfolio object
            risk_map: 2D array of risk values
            geo_transform: GeoTransform object for the risk map
            optimized_weights: Optional dictionary mapping asset IDs to optimized weights
            
        Returns:
            Matplotlib Figure object
        """
        # Calculate risk scores
        risk_scores = portfolio.assess_risk(risk_model)
        
        # Calculate original weights
        total_value = sum(asset['value'] for asset in portfolio.assets)
        original_weights = {asset['id']: asset['value'] / total_value for asset in portfolio.assets}
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define grid spec
        gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
        
        # Plot risk map with assets
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(risk_map, cmap='viridis_r', origin='upper', alpha=0.7)
        cbar = fig.colorbar(im, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Risk Level', fontsize=12)
        
        # Get asset locations
        x_coords, y_coords, names, values = portfolio.get_asset_locations()
        
        # Convert geographic coordinates to pixel coordinates for the risk map
        pixel_coords = []
        for x, y in zip(x_coords, y_coords):
            pixel_x, pixel_y = geo_transform.geo_to_pixel(x, y)
            pixel_coords.append((pixel_x, pixel_y))
        
        pixel_x_coords, pixel_y_coords = zip(*pixel_coords)
        
        # Calculate marker sizes based on asset values
        marker_sizes = np.array(values) / max(values) * 100
        
        # Plot assets
        sc = ax1.scatter(pixel_x_coords, pixel_y_coords, s=marker_sizes, c='red',
                        marker='o', alpha=0.7, edgecolors='white', linewidths=1)
        
        # Add asset labels
        for x, y, name, value in zip(pixel_x_coords, pixel_y_coords, names, values):
            ax1.annotate(f"{name} (${value/1000:.1f}K)", 
                        xy=(x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='white', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
        
        ax1.set_title('Geospatial Risk Map with Portfolio Assets', fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Plot asset risk bar chart
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Extract asset data
        data = []
        for asset in portfolio.assets:
            if asset['id'] in risk_scores:
                data.append({
                    'id': asset['id'],
                    'name': asset['name'],
                    'value': asset['value'],
                    'risk_score': risk_scores[asset['id']]
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Sort by risk score (descending)
        df = df.sort_values('risk_score', ascending=False)
        
        # Create bar chart
        colors = plt.cm.viridis_r(df['risk_score'])
        bars = ax2.bar(df['name'], df['risk_score'], color=colors)
        
        # Add labels
        for bar, value in zip(bars, df['value']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f"${value/1000:.1f}K", ha='center', va='bottom', fontsize=10)
        
        ax2.set_xlabel('Assets')
        ax2.set_ylabel('Risk Score')
        ax2.set_title('Asset Risk Scores', fontsize=14)
        ax2.set_ylim(0, 1.1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Plot portfolio optimization results if available
        ax3 = fig.add_subplot(gs[1, 0])
        
        if optimized_weights:
            # Bar chart of weights
            df['original_weight'] = df['id'].map(original_weights)
            df['optimized_weight'] = df['id'].map(optimized_weights)
            
            df_melted = pd.melt(df, id_vars=['name', 'risk_score'], 
                              value_vars=['original_weight', 'optimized_weight'],
                              var_name='Weight Type', value_name='Weight')
            
            sns.barplot(x='name', y='Weight', hue='Weight Type', data=df_melted, ax=ax3)
        else:
            # Just show original weights
            sns.barplot(x='name', y='value', data=df, ax=ax3)
            ax3.set_ylabel('Asset Value ($)')
        
        ax3.set_title('Portfolio Allocation', fontsize=14)
        ax3.set_xlabel('')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Plot risk distribution
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Create histogram of risk values
        valid_mask = ~np.isnan(risk_map) & ~np.isinf(risk_map)
        valid_risk = risk_map[valid_mask].flatten()
        
        sns.histplot(valid_risk, kde=True, ax=ax4, bins=50, color='skyblue')
        
        # Add vertical lines for asset risk scores
        for i, row in enumerate(df.itertuples()):
            ax4.axvline(x=row.risk_score, color=plt.cm.viridis_r(row.risk_score), 
                       linestyle='--', alpha=0.8, label=row.name)
        
        ax4.set_xlabel('Risk Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Risk Distribution with Asset Positions', fontsize=14)
        ax4.legend(fontsize=8)
        
        # Add overall title
        fig.suptitle('Geospatial Financial Risk Analysis Dashboard', fontsize=20, y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.94)
        
        return fig