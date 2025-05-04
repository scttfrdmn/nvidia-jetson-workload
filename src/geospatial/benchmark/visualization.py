#!/usr/bin/env python3
"""
Visualization utilities for Geospatial Analysis benchmarking.

This module provides specialized visualization tools for geospatial
benchmark results, including interactive maps, terrain rendering,
and geospatial-specific performance metrics.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from matplotlib.cm import ScalarMappable
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime

# Set Seaborn style for better visualizations
sns.set_style('whitegrid')
sns.set_context('paper')

# Constants for visualization
TERRAIN_COLORS = plt.cm.terrain
POINT_CLOUD_COLORS = {
    0: (0.7, 0.7, 0.7),  # Created - gray
    1: (0.4, 0.4, 0.4),  # Unclassified - dark gray
    2: (0.8, 0.7, 0.6),  # Ground - tan
    3: (0.4, 0.8, 0.4),  # Low Vegetation - light green
    4: (0.2, 0.6, 0.2),  # Medium Vegetation - medium green
    5: (0.0, 0.4, 0.0),  # High Vegetation - dark green
    6: (1.0, 0.0, 0.0)   # Building - red
}

class GeospatialVisualizer:
    """Visualization tools for geospatial benchmark results."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create visualizations directory
        self.viz_dir = self.output_dir / "visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def load_results(self, results_file: str) -> Dict[str, Any]:
        """
        Load benchmark results from file.
        
        Args:
            results_file: Path to results file
            
        Returns:
            Dictionary with benchmark results
        """
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def plot_dem_operations_performance(self, 
                                       results: Dict[str, Any], 
                                       output_file: Optional[str] = None) -> str:
        """
        Plot DEM operations performance comparison.
        
        Args:
            results: Benchmark results dictionary
            output_file: Output file path (default: auto-generated)
            
        Returns:
            Path to the generated plot
        """
        # Extract DEM benchmark results
        dem_results = {}
        for key, benchmark in results.get("benchmarks", {}).items():
            if key.startswith("dem_") and benchmark.get("name") == "dem_processing":
                dem_results[key] = benchmark
        
        if not dem_results:
            print("No DEM benchmark results found.")
            return ""
        
        # Set up figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract operations and organize by DEM type
        operations = set()
        dem_types = {}
        
        for key, result in dem_results.items():
            # Extract DEM size and type from key
            parts = key.split("_")
            if len(parts) >= 3:
                dem_size = parts[1]
                dem_type = "_".join(parts[2:])
                
                dem_key = f"{dem_size}_{dem_type}"
                if dem_key not in dem_types:
                    dem_types[dem_key] = {}
                
                # Add operations
                for op_name, op_time in result.get("operations", {}).items():
                    operations.add(op_name)
                    dem_types[dem_key][op_name] = op_time
        
        # Sort operations by average execution time
        op_avg_times = {}
        for op in operations:
            times = [result.get(op, 0) for dem_key, result in dem_types.items() if op in result]
            if times:
                op_avg_times[op] = sum(times) / len(times)
        
        operations = sorted(operations, key=lambda op: op_avg_times.get(op, 0), reverse=True)
        
        # Prepare data for plotting
        dem_labels = list(dem_types.keys())
        x = np.arange(len(dem_labels))
        width = 0.8 / len(operations)
        
        # Plot bars for each operation
        for i, op in enumerate(operations):
            op_times = [dem_types[dem_key].get(op, 0) for dem_key in dem_labels]
            ax.bar(x + i * width - 0.4 + width/2, op_times, width, label=op.replace("_", " ").title())
        
        # Add labels and title
        ax.set_xlabel("DEM Size and Type")
        ax.set_ylabel("Execution Time (seconds)")
        ax.set_title("DEM Processing Operations Performance")
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace("_", " ").title() for label in dem_labels], rotation=45, ha="right")
        ax.legend()
        
        # Use log scale if there's a large range of values
        if any(v > 10 * min([v for v in op_avg_times.values() if v > 0]) for v in op_avg_times.values()):
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
        
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.viz_dir / f"dem_operations_{timestamp}.png"
        
        plt.savefig(output_file)
        plt.close(fig)
        
        print(f"DEM operations performance plot saved to {output_file}")
        return str(output_file)
    
    def plot_point_cloud_operations_performance(self, 
                                              results: Dict[str, Any], 
                                              output_file: Optional[str] = None) -> str:
        """
        Plot point cloud operations performance comparison.
        
        Args:
            results: Benchmark results dictionary
            output_file: Output file path (default: auto-generated)
            
        Returns:
            Path to the generated plot
        """
        # Extract point cloud benchmark results
        pc_results = {}
        for key, benchmark in results.get("benchmarks", {}).items():
            if key.startswith("point_cloud_") and benchmark.get("name") == "point_cloud_processing":
                pc_results[key] = benchmark
        
        if not pc_results:
            print("No point cloud benchmark results found.")
            return ""
        
        # Set up figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract operations and organize by point cloud type
        operations = set()
        pc_types = {}
        
        for key, result in pc_results.items():
            # Extract point cloud size and type from key
            parts = key.split("_")
            if len(parts) >= 4:
                pc_size = parts[2]
                pc_type = parts[3]
                
                pc_key = f"{pc_size}_{pc_type}"
                if pc_key not in pc_types:
                    pc_types[pc_key] = {}
                
                # Add operations
                for op_name, op_time in result.get("operations", {}).items():
                    operations.add(op_name)
                    pc_types[pc_key][op_name] = op_time
        
        # Sort operations by average execution time
        op_avg_times = {}
        for op in operations:
            times = [result.get(op, 0) for pc_key, result in pc_types.items() if op in result]
            if times:
                op_avg_times[op] = sum(times) / len(times)
        
        operations = sorted(operations, key=lambda op: op_avg_times.get(op, 0), reverse=True)
        
        # Prepare data for plotting
        pc_labels = list(pc_types.keys())
        x = np.arange(len(pc_labels))
        width = 0.8 / len(operations)
        
        # Plot bars for each operation
        for i, op in enumerate(operations):
            op_times = [pc_types[pc_key].get(op, 0) for pc_key in pc_labels]
            ax.bar(x + i * width - 0.4 + width/2, op_times, width, label=op.replace("_", " ").title())
        
        # Add labels and title
        ax.set_xlabel("Point Cloud Size and Type")
        ax.set_ylabel("Execution Time (seconds)")
        ax.set_title("Point Cloud Processing Operations Performance")
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace("_", " ").title() for label in pc_labels], rotation=45, ha="right")
        ax.legend()
        
        # Use log scale if there's a large range of values
        if any(v > 10 * min([v for v in op_avg_times.values() if v > 0]) for v in op_avg_times.values()):
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
        
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.viz_dir / f"point_cloud_operations_{timestamp}.png"
        
        plt.savefig(output_file)
        plt.close(fig)
        
        print(f"Point cloud operations performance plot saved to {output_file}")
        return str(output_file)
    
    def plot_throughput_comparison(self, 
                                 results: Dict[str, Any], 
                                 output_file: Optional[str] = None) -> str:
        """
        Plot throughput comparison for all benchmarks.
        
        Args:
            results: Benchmark results dictionary
            output_file: Output file path (default: auto-generated)
            
        Returns:
            Path to the generated plot
        """
        # Extract benchmark results
        benchmarks = results.get("benchmarks", {})
        
        if not benchmarks:
            print("No benchmark results found.")
            return ""
        
        # Set up figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by benchmark type
        benchmark_types = {}
        
        for key, result in benchmarks.items():
            # Determine benchmark type and subtype
            if key.startswith("dem_"):
                benchmark_type = "DEM"
                parts = key.split("_")
                if len(parts) >= 3:
                    subtype = f"{parts[1]}_{parts[2]}"
                else:
                    subtype = "unknown"
            elif key.startswith("point_cloud_"):
                benchmark_type = "Point Cloud"
                parts = key.split("_")
                if len(parts) >= 4:
                    subtype = f"{parts[2]}_{parts[3]}"
                else:
                    subtype = "unknown"
            else:
                benchmark_type = "Other"
                subtype = key
            
            # Create type group if needed
            if benchmark_type not in benchmark_types:
                benchmark_types[benchmark_type] = {}
            
            # Add throughput data
            throughput = result.get("total_throughput", 0)
            benchmark_types[benchmark_type][subtype] = throughput
        
        # Prepare data for plotting
        types = list(benchmark_types.keys())
        
        for i, benchmark_type in enumerate(types):
            subtypes = list(benchmark_types[benchmark_type].keys())
            throughputs = list(benchmark_types[benchmark_type].values())
            
            # Plot as grouped bars
            x = np.arange(len(subtypes))
            ax.bar(x + i * 0.3 - 0.3, throughputs, 0.25, label=benchmark_type)
            
            # Add subtype labels
            if i == 0:
                ax.set_xticks(x)
                ax.set_xticklabels([subtype.replace("_", " ").title() for subtype in subtypes], 
                                  rotation=45, ha="right")
        
        # Add labels and title
        ax.set_xlabel("Dataset Type")
        ax.set_ylabel("Throughput (operations/second)")
        ax.set_title("Processing Throughput Comparison")
        ax.legend()
        
        # Use log scale if there's a large range of values
        throughputs = [t for bt in benchmark_types.values() for t in bt.values()]
        if throughputs and any(t > 10 * min([t for t in throughputs if t > 0]) for t in throughputs):
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
        
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.viz_dir / f"throughput_comparison_{timestamp}.png"
        
        plt.savefig(output_file)
        plt.close(fig)
        
        print(f"Throughput comparison plot saved to {output_file}")
        return str(output_file)
    
    def plot_cost_efficiency(self, 
                           results: Dict[str, Any], 
                           output_file: Optional[str] = None) -> str:
        """
        Plot cost efficiency comparison.
        
        Args:
            results: Benchmark results dictionary
            output_file: Output file path (default: auto-generated)
            
        Returns:
            Path to the generated plot
        """
        # Extract benchmark results with cost metrics
        benchmarks_with_cost = {}
        for key, benchmark in results.get("benchmarks", {}).items():
            if "cost_metrics" in benchmark:
                benchmarks_with_cost[key] = benchmark
        
        if not benchmarks_with_cost:
            print("No benchmark results with cost metrics found.")
            return ""
        
        # Set up figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Organize data by benchmark and cloud provider
        benchmark_names = []
        jetson_costs = []
        cloud_costs = {
            "AWS": [],
            "Azure": [],
            "GCP": []
        }
        
        for key, benchmark in benchmarks_with_cost.items():
            # Add benchmark name
            benchmark_names.append(key)
            
            # Add Jetson cost
            jetson_cost = benchmark.get("cost_metrics", {}).get("jetson", {}).get("total_cost", 0)
            jetson_costs.append(jetson_cost)
            
            # Add cloud costs
            cloud_metrics = benchmark.get("cost_metrics", {}).get("cloud", {})
            for provider in cloud_costs:
                provider_key = provider.lower()
                if provider_key in cloud_metrics:
                    cloud_costs[provider].append(cloud_metrics[provider_key].get("total_cost", 0))
                else:
                    cloud_costs[provider].append(0)
        
        # Calculate throughput per dollar
        throughputs = [benchmark.get("total_throughput", 0) for benchmark in benchmarks_with_cost.values()]
        
        jetson_efficiency = []
        for t, c in zip(throughputs, jetson_costs):
            if c > 0:
                jetson_efficiency.append(t / c)
            else:
                jetson_efficiency.append(0)
        
        cloud_efficiency = {}
        for provider in cloud_costs:
            cloud_efficiency[provider] = []
            for t, c in zip(throughputs, cloud_costs[provider]):
                if c > 0:
                    cloud_efficiency[provider].append(t / c)
                else:
                    cloud_efficiency[provider].append(0)
        
        # Set up plot
        x = np.arange(len(benchmark_names))
        width = 0.15
        
        # Plot bars
        ax.bar(x - 2*width, jetson_efficiency, width, label="Jetson", color="#76b900")  # NVIDIA green
        
        for i, (provider, efficiency) in enumerate(cloud_efficiency.items()):
            color = plt.cm.tab10(i)
            ax.bar(x + (i-1)*width, efficiency, width, label=provider, color=color)
        
        # Add labels and title
        ax.set_xlabel("Benchmark")
        ax.set_ylabel("Throughput per Dollar (operations/$)")
        ax.set_title("Cost Efficiency Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace("_", " ").title() for name in benchmark_names], 
                          rotation=45, ha="right")
        ax.legend()
        
        # Use log scale if there's a large range of values
        all_efficiencies = (
            jetson_efficiency + 
            [e for provider_eff in cloud_efficiency.values() for e in provider_eff]
        )
        
        if all_efficiencies and any(e > 10 * min([e for e in all_efficiencies if e > 0]) for e in all_efficiencies):
            ax.set_yscale('log')
            ax.yaxis.set_major_formatter(ScalarFormatter())
        
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.viz_dir / f"cost_efficiency_{timestamp}.png"
        
        plt.savefig(output_file)
        plt.close(fig)
        
        print(f"Cost efficiency plot saved to {output_file}")
        return str(output_file)
    
    def plot_breakeven_analysis(self, 
                              results: Dict[str, Any], 
                              output_file: Optional[str] = None) -> str:
        """
        Plot break-even analysis for cloud vs. local computing.
        
        Args:
            results: Benchmark results dictionary
            output_file: Output file path (default: auto-generated)
            
        Returns:
            Path to the generated plot
        """
        # Extract benchmark results with cost metrics
        benchmarks_with_cost = {}
        for key, benchmark in results.get("benchmarks", {}).items():
            if (
                "cost_metrics" in benchmark and 
                "comparison" in benchmark["cost_metrics"] and
                "break_even_hours" in benchmark["cost_metrics"]["comparison"]
            ):
                benchmarks_with_cost[key] = benchmark
        
        if not benchmarks_with_cost:
            print("No benchmark results with break-even metrics found.")
            return ""
        
        # Set up figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Organize data by benchmark and cloud provider
        benchmark_names = []
        breakeven_hours = {
            "AWS": [],
            "Azure": [],
            "GCP": []
        }
        
        for key, benchmark in benchmarks_with_cost.items():
            # Add benchmark name
            benchmark_names.append(key)
            
            # Add break-even hours
            be_hours = benchmark["cost_metrics"]["comparison"]["break_even_hours"]
            for provider in breakeven_hours:
                provider_key = provider.lower()
                if provider_key in be_hours:
                    hours = be_hours[provider_key]
                    # Cap at 10000 hours for visualization
                    if hours != float('inf'):
                        hours = min(hours, 10000)
                    else:
                        hours = 10000  # Use 10000 as proxy for infinity
                    breakeven_hours[provider].append(hours)
                else:
                    breakeven_hours[provider].append(0)
        
        # Set up horizontal bar plot
        y_pos = np.arange(len(benchmark_names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(breakeven_hours)))
        
        # Plot bars
        for i, (provider, hours) in enumerate(breakeven_hours.items()):
            ax.barh(y_pos + i*0.3 - 0.3, hours, 0.25, label=provider, color=colors[i])
        
        # Add labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name.replace("_", " ").title() for name in benchmark_names])
        ax.set_xlabel("Break-Even Time (Hours)")
        ax.set_title("Cloud vs. Local Computing Break-Even Analysis")
        
        # Add reference lines
        day_line = 24
        week_line = 24 * 7
        month_line = 24 * 30
        year_line = 24 * 365
        
        ax.axvline(x=day_line, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=week_line, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=month_line, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=year_line, color='gray', linestyle='--', alpha=0.7)
        
        ax.text(day_line, len(benchmark_names) + 0.5, "1 Day", ha='center', va='bottom')
        ax.text(week_line, len(benchmark_names) + 0.5, "1 Week", ha='center', va='bottom')
        ax.text(month_line, len(benchmark_names) + 0.5, "1 Month", ha='center', va='bottom')
        ax.text(year_line, len(benchmark_names) + 0.5, "1 Year", ha='center', va='bottom')
        
        # Use log scale for x-axis
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        
        # Add infinity marker for values at the cap
        for i, (provider, hours) in enumerate(breakeven_hours.items()):
            for j, h in enumerate(hours):
                if h >= 9999:
                    ax.text(h, y_pos[j] + i*0.3 - 0.3, "∞", va='center')
        
        ax.legend()
        plt.tight_layout()
        
        # Save plot
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.viz_dir / f"breakeven_analysis_{timestamp}.png"
        
        plt.savefig(output_file)
        plt.close(fig)
        
        print(f"Break-even analysis plot saved to {output_file}")
        return str(output_file)
    
    def render_terrain_3d(self, 
                         dem_path: str, 
                         output_file: Optional[str] = None) -> str:
        """
        Render a 3D visualization of a DEM.
        
        Args:
            dem_path: Path to DEM file
            output_file: Output file path (default: auto-generated)
            
        Returns:
            Path to the generated plot
        """
        try:
            from osgeo import gdal
            # Open DEM file
            dem = gdal.Open(dem_path)
            if dem is None:
                print(f"Error: Could not open DEM file {dem_path}")
                return ""
            
            # Read data
            band = dem.GetRasterBand(1)
            dem_data = band.ReadAsArray()
            
            # Get geotransform
            gt = dem.GetGeoTransform()
            pixel_width = gt[1]
            pixel_height = gt[5]
            
            # Generate X and Y coordinates
            rows, cols = dem_data.shape
            x = np.arange(0, cols) * pixel_width
            y = np.arange(0, rows) * abs(pixel_height)
            X, Y = np.meshgrid(x, y)
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Downsample for faster rendering if DEM is large
            downsample = max(1, min(rows, cols) // 200)
            X_d = X[::downsample, ::downsample]
            Y_d = Y[::downsample, ::downsample]
            Z_d = dem_data[::downsample, ::downsample]
            
            # Plot surface
            surf = ax.plot_surface(X_d, Y_d, Z_d, cmap=TERRAIN_COLORS, 
                                  linewidth=0, antialiased=True, alpha=0.8)
            
            # Add colorbar
            cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            cbar.set_label('Elevation (m)')
            
            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Elevation (m)')
            ax.set_title(f'3D Terrain Visualization: {os.path.basename(dem_path)}')
            
            # Save plot
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.viz_dir / f"terrain_3d_{timestamp}.png"
            
            plt.savefig(output_file, dpi=300)
            plt.close(fig)
            
            print(f"3D terrain visualization saved to {output_file}")
            return str(output_file)
        
        except (ImportError, Exception) as e:
            print(f"Error rendering 3D terrain: {e}")
            return ""
    
    def generate_report(self, 
                      results_file: str, 
                      output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive HTML report for geospatial benchmarks.
        
        Args:
            results_file: Path to results file
            output_file: Output file path (default: auto-generated)
            
        Returns:
            Path to the generated report
        """
        # Load results
        results = self.load_results(results_file)
        
        # Generate plots
        dem_plot = self.plot_dem_operations_performance(results)
        pc_plot = self.plot_point_cloud_operations_performance(results)
        throughput_plot = self.plot_throughput_comparison(results)
        
        # Check if cost metrics are available
        has_cost_metrics = False
        for benchmark in results.get("benchmarks", {}).values():
            if "cost_metrics" in benchmark:
                has_cost_metrics = True
                break
        
        if has_cost_metrics:
            cost_plot = self.plot_cost_efficiency(results)
            breakeven_plot = self.plot_breakeven_analysis(results)
        
        # Create output file path
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"geospatial_report_{timestamp}.html"
        
        # Generate HTML report
        html = self._generate_html_report(results, 
                                        dem_plot=dem_plot, 
                                        pc_plot=pc_plot,
                                        throughput_plot=throughput_plot,
                                        cost_plot=cost_plot if has_cost_metrics else None,
                                        breakeven_plot=breakeven_plot if has_cost_metrics else None)
        
        # Write report
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"Report generated: {output_file}")
        return str(output_file)
    
    def _generate_html_report(self, 
                            results: Dict[str, Any],
                            dem_plot: Optional[str] = None,
                            pc_plot: Optional[str] = None,
                            throughput_plot: Optional[str] = None,
                            cost_plot: Optional[str] = None,
                            breakeven_plot: Optional[str] = None) -> str:
        """
        Generate HTML report content.
        
        Args:
            results: Benchmark results
            dem_plot: Path to DEM operations plot
            pc_plot: Path to point cloud operations plot
            throughput_plot: Path to throughput comparison plot
            cost_plot: Path to cost efficiency plot
            breakeven_plot: Path to break-even analysis plot
            
        Returns:
            HTML report content
        """
        system_info = results.get("system_info", {})
        timestamp = results.get("timestamp", datetime.now().isoformat())
        
        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp
        
        # Generate benchmark summary table
        benchmark_summary = self._generate_benchmark_summary_table(results)
        
        # Check for cost metrics
        has_cost_metrics = cost_plot is not None and breakeven_plot is not None
        
        # Generate HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Geospatial Analysis Benchmark Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            max-width: 1200px;
            margin: 0 auto;
            color: #333;
        }}
        h1, h2, h3 {{
            color: #0066cc;
        }}
        .section {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border: 1px solid #ddd;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .tabs {{
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            margin-top: 20px;
        }}
        .tab-button {{
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 16px;
        }}
        .tab-button:hover {{
            background-color: #ddd;
        }}
        .tab-button.active {{
            background-color: #ccc;
        }}
        .tab-content {{
            display: none;
            padding: 6px 12px;
            border: 1px solid #ccc;
            border-top: none;
        }}
        .highlight {{
            background-color: #f9f3e0;
            padding: 10px;
            border-left: 4px solid #f0ad4e;
            margin-bottom: 15px;
        }}
    </style>
    <script>
        function openTab(evt, tabName) {{
            var i, tabcontent, tabbuttons;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{
                tabcontent[i].style.display = "none";
            }}
            tabbuttons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabbuttons.length; i++) {{
                tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
            }}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}
        
        // Set the first tab as active when the page loads
        window.onload = function() {{
            document.getElementsByClassName("tab-button")[0].click();
        }}
    </script>
</head>
<body>
    <h1>Geospatial Analysis Benchmark Report</h1>
    <p>Generated on: {formatted_time}</p>
    
    <div class="tabs">
        <button class="tab-button" onclick="openTab(event, 'summary')">Summary</button>
        <button class="tab-button" onclick="openTab(event, 'dem')">DEM Performance</button>
        <button class="tab-button" onclick="openTab(event, 'point-cloud')">Point Cloud Performance</button>
        {'<button class="tab-button" onclick="openTab(event, \'cost\')">Cost Analysis</button>' if has_cost_metrics else ''}
        <button class="tab-button" onclick="openTab(event, 'system')">System Information</button>
    </div>
    
    <div id="summary" class="tab-content">
        <div class="section">
            <h2>Benchmark Summary</h2>
            {benchmark_summary}
        </div>
        
        <div class="section">
            <h2>Throughput Comparison</h2>
            {'<div class="image-container"><img src="' + os.path.relpath(throughput_plot, str(self.output_dir)) + '" alt="Throughput Comparison"></div>' if throughput_plot else '<p>No throughput comparison available.</p>'}
        </div>
    </div>
    
    <div id="dem" class="tab-content">
        <div class="section">
            <h2>DEM Processing Performance</h2>
            {'<div class="image-container"><img src="' + os.path.relpath(dem_plot, str(self.output_dir)) + '" alt="DEM Operations Performance"></div>' if dem_plot else '<p>No DEM benchmark results available.</p>'}
            
            <h3>DEM Processing Operations</h3>
            <p>Digital Elevation Model (DEM) processing includes operations such as:</p>
            <ul>
                <li><strong>Viewshed Analysis:</strong> Determining visible areas from an observer point</li>
                <li><strong>Terrain Derivatives:</strong> Computing slope, aspect, and curvature</li>
                <li><strong>Hydrological Features:</strong> Generating flow direction and accumulation</li>
                <li><strong>Least Cost Path:</strong> Finding optimal paths between points</li>
            </ul>
        </div>
    </div>
    
    <div id="point-cloud" class="tab-content">
        <div class="section">
            <h2>Point Cloud Processing Performance</h2>
            {'<div class="image-container"><img src="' + os.path.relpath(pc_plot, str(self.output_dir)) + '" alt="Point Cloud Operations Performance"></div>' if pc_plot else '<p>No point cloud benchmark results available.</p>'}
            
            <h3>Point Cloud Processing Operations</h3>
            <p>Point cloud processing includes operations such as:</p>
            <ul>
                <li><strong>Classification:</strong> Categorizing points as ground, vegetation, buildings, etc.</li>
                <li><strong>DEM/DSM Creation:</strong> Generating elevation models from point clouds</li>
                <li><strong>Feature Extraction:</strong> Identifying buildings, vegetation, and other features</li>
                <li><strong>Segmentation:</strong> Grouping points into coherent objects</li>
            </ul>
        </div>
    </div>
    """
        
        # Add cost analysis tab if available
        if has_cost_metrics:
            html += f"""
    <div id="cost" class="tab-content">
        <div class="section">
            <h2>Cost Efficiency</h2>
            <p>This section compares the cost efficiency of running geospatial workloads on Jetson versus cloud providers.</p>
            {'<div class="image-container"><img src="' + os.path.relpath(cost_plot, str(self.output_dir)) + '" alt="Cost Efficiency"></div>' if cost_plot else '<p>No cost efficiency data available.</p>'}
        </div>
        
        <div class="section">
            <h2>Break-Even Analysis</h2>
            <p>This analysis shows how many hours of operation are required for Jetson to become more cost-effective than cloud options.</p>
            {'<div class="image-container"><img src="' + os.path.relpath(breakeven_plot, str(self.output_dir)) + '" alt="Break-Even Analysis"></div>' if breakeven_plot else '<p>No break-even analysis data available.</p>'}
            
            <div class="highlight">
                <p><strong>Note:</strong> Break-even analysis considers hardware amortization, electricity costs, and cloud computing rates to determine the point at which local computing becomes more economical than cloud-based alternatives.</p>
            </div>
        </div>
    </div>
    """
        
        # Add system information tab
        html += f"""
    <div id="system" class="tab-content">
        <div class="section">
            <h2>System Information</h2>
            <table>
                <tr><th>Platform</th><td>{system_info.get('platform', 'Unknown')}</td></tr>
                <tr><th>Processor</th><td>{system_info.get('processor', 'Unknown')}</td></tr>
                <tr><th>Python Version</th><td>{system_info.get('python_version', 'Unknown')}</td></tr>
                <tr><th>CPU Count</th><td>{system_info.get('cpu_count', 'Unknown')}</td></tr>
            </table>
            
            <h3>Memory</h3>
            <table>
                <tr><th>Total Memory (MB)</th><td>{system_info.get('memory', {}).get('total', 'Unknown')}</td></tr>
                <tr><th>Available Memory (MB)</th><td>{system_info.get('memory', {}).get('available', 'Unknown')}</td></tr>
                <tr><th>Memory Usage (%)</th><td>{system_info.get('memory', {}).get('percent', 'Unknown')}</td></tr>
            </table>
            
            <h3>GPU Information</h3>
            <table>
                <tr><th>Device Count</th><td>{system_info.get('gpu', {}).get('device_count', 'Unknown')}</td></tr>
            """
        
        # Add GPU device information
        for i, device in enumerate(system_info.get('gpu', {}).get('devices', [])):
            html += f"""
                <tr><th colspan="2">Device {i}</th></tr>
                <tr><th>Name</th><td>{device.get('name', 'Unknown')}</td></tr>
                <tr><th>Compute Capability</th><td>{device.get('compute_capability', 'Unknown')}</td></tr>
                <tr><th>Memory (MB)</th><td>{device.get('total_memory', 'Unknown')}</td></tr>
                <tr><th>Clock Rate (MHz)</th><td>{device.get('clock_rate', 'Unknown')}</td></tr>
            """
        
        # Complete the HTML
        html += """
            </table>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_benchmark_summary_table(self, results: Dict[str, Any]) -> str:
        """
        Generate HTML table with benchmark summary.
        
        Args:
            results: Benchmark results
            
        Returns:
            HTML table with benchmark summary
        """
        benchmarks = results.get("benchmarks", {})
        
        if not benchmarks:
            return "<p>No benchmark results available.</p>"
        
        # Create table
        html = """
<table>
    <tr>
        <th>Benchmark</th>
        <th>Execution Time (s)</th>
        <th>Throughput</th>
        <th>Memory Usage (MB)</th>
        <th>GPU Utilization (%)</th>
"""
        
        # Add cost column if available
        has_cost = False
        for benchmark in benchmarks.values():
            if "cost_metrics" in benchmark:
                has_cost = True
                break
        
        if has_cost:
            html += "        <th>Cost (USD)</th>\n"
        
        html += "    </tr>\n"
        
        # Add benchmark rows
        for key, benchmark in sorted(benchmarks.items()):
            name = key.replace("_", " ").title()
            execution_time = f"{benchmark.get('total_execution_time', 0):.4f}"
            throughput = f"{benchmark.get('total_throughput', 0):.4f}"
            
            memory = benchmark.get("memory", {})
            memory_usage = f"Host: {memory.get('host', 0):.2f}, Device: {memory.get('device', 0):.2f}"
            
            gpu_utilization = "N/A"
            if "gpu_utilization" in benchmark and benchmark["gpu_utilization"] is not None:
                gpu_utilization = f"{benchmark['gpu_utilization']:.2f}"
            
            html += f"""    <tr>
        <td>{name}</td>
        <td>{execution_time}</td>
        <td>{throughput}</td>
        <td>{memory_usage}</td>
        <td>{gpu_utilization}</td>
"""
            
            # Add cost information if available
            if has_cost:
                cost = "N/A"
                if "cost_metrics" in benchmark and "jetson" in benchmark["cost_metrics"]:
                    jetson_cost = benchmark["cost_metrics"]["jetson"].get("total_cost", 0)
                    cost = f"${jetson_cost:.4f}"
                html += f"        <td>{cost}</td>\n"
            
            html += "    </tr>\n"
        
        html += "</table>\n"
        return html

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize geospatial benchmark results")
    parser.add_argument("results_file", type=str, help="Path to results file")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save visualizations")
    parser.add_argument("--dem-file", type=str, default=None,
                        help="Path to DEM file for 3D visualization")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = GeospatialVisualizer(args.output_dir)
    
    # Generate report
    report_file = visualizer.generate_report(args.results_file)
    
    # Render 3D terrain if DEM file provided
    if args.dem_file:
        visualizer.render_terrain_3d(args.dem_file)