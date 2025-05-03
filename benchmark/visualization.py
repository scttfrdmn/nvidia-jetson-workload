#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Visualization utilities for the benchmarking suite.
Provides functions for plotting benchmark results and generating reports.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

def plot_execution_time_comparison(results: Dict[str, List[Any]], 
                                   output_file: Optional[str] = None,
                                   title: str = "Execution Time Comparison",
                                   figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot execution time comparison for different workloads.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the plot
        title: Title of the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Group results by device and workload
    devices = set()
    workloads = set()
    data = {}
    
    for workload_name, workload_results in results.items():
        workloads.add(workload_name)
        for result in workload_results:
            device_name = result.device_name
            devices.add(device_name)
            
            if device_name not in data:
                data[device_name] = {}
            
            if workload_name not in data[device_name]:
                data[device_name][workload_name] = []
            
            data[device_name][workload_name].append(result.execution_time)
    
    # Sort devices and workloads
    devices = sorted(list(devices))
    workloads = sorted(list(workloads))
    
    # Calculate positions for bars
    bar_width = 0.8 / len(devices)
    indices = np.arange(len(workloads))
    
    # Plot bars
    for i, device in enumerate(devices):
        execution_times = []
        for workload in workloads:
            if device in data and workload in data[device]:
                # Average execution time if multiple results
                execution_times.append(np.mean(data[device][workload]))
            else:
                execution_times.append(0)
        
        plt.bar(indices + i * bar_width, execution_times, bar_width, label=device)
    
    # Add labels and title
    plt.xlabel("Workload")
    plt.ylabel("Execution Time (seconds)")
    plt.title(title)
    plt.xticks(indices + bar_width * len(devices) / 2, workloads, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_memory_usage(results: Dict[str, List[Any]], 
                      output_file: Optional[str] = None,
                      title: str = "Memory Usage",
                      figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot memory usage for different workloads.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the plot
        title: Title of the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Group results by device and workload
    devices = set()
    workloads = set()
    host_data = {}
    device_data = {}
    
    for workload_name, workload_results in results.items():
        workloads.add(workload_name)
        for result in workload_results:
            device_name = result.device_name
            devices.add(device_name)
            
            # Host memory
            if device_name not in host_data:
                host_data[device_name] = {}
            
            if workload_name not in host_data[device_name]:
                host_data[device_name][workload_name] = []
            
            host_data[device_name][workload_name].append(result.memory_usage.get("host", 0))
            
            # Device memory
            if device_name not in device_data:
                device_data[device_name] = {}
            
            if workload_name not in device_data[device_name]:
                device_data[device_name][workload_name] = []
            
            device_data[device_name][workload_name].append(result.memory_usage.get("device", 0))
    
    # Sort devices and workloads
    devices = sorted(list(devices))
    workloads = sorted(list(workloads))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate positions for bars
    bar_width = 0.8 / len(devices)
    indices = np.arange(len(workloads))
    
    # Plot host memory usage
    for i, device in enumerate(devices):
        memory_usage = []
        for workload in workloads:
            if device in host_data and workload in host_data[device]:
                # Average memory usage if multiple results
                memory_usage.append(np.mean(host_data[device][workload]))
            else:
                memory_usage.append(0)
        
        ax1.bar(indices + i * bar_width, memory_usage, bar_width, label=device)
    
    # Add labels and title for host memory
    ax1.set_xlabel("Workload")
    ax1.set_ylabel("Host Memory Usage (MB)")
    ax1.set_title("Host Memory Usage")
    ax1.set_xticks(indices + bar_width * len(devices) / 2)
    ax1.set_xticklabels(workloads, rotation=45, ha="right")
    ax1.legend()
    
    # Plot device memory usage
    for i, device in enumerate(devices):
        memory_usage = []
        for workload in workloads:
            if device in device_data and workload in device_data[device]:
                # Average memory usage if multiple results
                memory_usage.append(np.mean(device_data[device][workload]))
            else:
                memory_usage.append(0)
        
        ax2.bar(indices + i * bar_width, memory_usage, bar_width, label=device)
    
    # Add labels and title for device memory
    ax2.set_xlabel("Workload")
    ax2.set_ylabel("Device Memory Usage (MB)")
    ax2.set_title("Device Memory Usage")
    ax2.set_xticks(indices + bar_width * len(devices) / 2)
    ax2.set_xticklabels(workloads, rotation=45, ha="right")
    ax2.legend()
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_gpu_utilization(results: Dict[str, List[Any]], 
                         output_file: Optional[str] = None,
                         title: str = "GPU Utilization",
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot GPU utilization for different workloads.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the plot
        title: Title of the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Group results by device and workload
    devices = set()
    workloads = set()
    data = {}
    
    for workload_name, workload_results in results.items():
        workloads.add(workload_name)
        for result in workload_results:
            if result.gpu_utilization is None:
                continue
                
            device_name = result.device_name
            devices.add(device_name)
            
            if device_name not in data:
                data[device_name] = {}
            
            if workload_name not in data[device_name]:
                data[device_name][workload_name] = []
            
            data[device_name][workload_name].append(result.gpu_utilization)
    
    # Check if we have any GPU utilization data
    if not data:
        print("No GPU utilization data available.")
        return
    
    # Sort devices and workloads
    devices = sorted(list(devices))
    workloads = sorted(list(workloads))
    
    # Calculate positions for bars
    bar_width = 0.8 / len(devices)
    indices = np.arange(len(workloads))
    
    # Plot bars
    for i, device in enumerate(devices):
        utilization = []
        for workload in workloads:
            if device in data and workload in data[device]:
                # Average utilization if multiple results
                utilization.append(np.mean(data[device][workload]))
            else:
                utilization.append(0)
        
        plt.bar(indices + i * bar_width, utilization, bar_width, label=device)
    
    # Add labels and title
    plt.xlabel("Workload")
    plt.ylabel("GPU Utilization (%)")
    plt.title(title)
    plt.xticks(indices + bar_width * len(devices) / 2, workloads, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_energy_consumption(results: Dict[str, List[Any]], 
                            output_file: Optional[str] = None,
                            title: str = "Energy Consumption",
                            figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot energy consumption for different workloads.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the plot
        title: Title of the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Group results by device and workload
    devices = set()
    workloads = set()
    data = {}
    
    for workload_name, workload_results in results.items():
        workloads.add(workload_name)
        for result in workload_results:
            if result.energy_consumption is None:
                continue
                
            device_name = result.device_name
            devices.add(device_name)
            
            if device_name not in data:
                data[device_name] = {}
            
            if workload_name not in data[device_name]:
                data[device_name][workload_name] = []
            
            data[device_name][workload_name].append(result.energy_consumption)
    
    # Check if we have any energy consumption data
    if not data:
        print("No energy consumption data available.")
        return
    
    # Sort devices and workloads
    devices = sorted(list(devices))
    workloads = sorted(list(workloads))
    
    # Calculate positions for bars
    bar_width = 0.8 / len(devices)
    indices = np.arange(len(workloads))
    
    # Plot bars
    for i, device in enumerate(devices):
        energy = []
        for workload in workloads:
            if device in data and workload in data[device]:
                # Average energy if multiple results
                energy.append(np.mean(data[device][workload]))
            else:
                energy.append(0)
        
        plt.bar(indices + i * bar_width, energy, bar_width, label=device)
    
    # Add labels and title
    plt.xlabel("Workload")
    plt.ylabel("Energy Consumption (Joules)")
    plt.title(title)
    plt.xticks(indices + bar_width * len(devices) / 2, workloads, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_throughput_comparison(results: Dict[str, List[Any]], 
                               output_file: Optional[str] = None,
                               title: str = "Throughput Comparison",
                               figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot throughput comparison for different workloads.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the plot
        title: Title of the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Group results by device and workload
    devices = set()
    workloads = set()
    data = {}
    
    for workload_name, workload_results in results.items():
        workloads.add(workload_name)
        for result in workload_results:
            if result.throughput is None:
                continue
                
            device_name = result.device_name
            devices.add(device_name)
            
            if device_name not in data:
                data[device_name] = {}
            
            if workload_name not in data[device_name]:
                data[device_name][workload_name] = []
            
            data[device_name][workload_name].append(result.throughput)
    
    # Check if we have any throughput data
    if not data:
        print("No throughput data available.")
        return
    
    # Sort devices and workloads
    devices = sorted(list(devices))
    workloads = sorted(list(workloads))
    
    # Calculate positions for bars
    bar_width = 0.8 / len(devices)
    indices = np.arange(len(workloads))
    
    # Plot bars
    for i, device in enumerate(devices):
        throughput = []
        for workload in workloads:
            if device in data and workload in data[device]:
                # Average throughput if multiple results
                throughput.append(np.mean(data[device][workload]))
            else:
                throughput.append(0)
        
        plt.bar(indices + i * bar_width, throughput, bar_width, label=device)
    
    # Add labels and title
    plt.xlabel("Workload")
    plt.ylabel("Throughput (iterations/s)")
    plt.title(title)
    plt.xticks(indices + bar_width * len(devices) / 2, workloads, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def generate_html_table(results: Dict[str, List[Any]]) -> str:
    """
    Generate HTML table for benchmark results.
    
    Args:
        results: Dictionary of benchmark results
    
    Returns:
        str: HTML table
    """
    html = "<table border='1' cellpadding='5' cellspacing='0'>\n"
    
    # Header row
    html += "<tr><th>Workload</th><th>Device</th><th>Execution Time (s)</th><th>Host Memory (MB)</th><th>Device Memory (MB)</th><th>GPU Utilization (%)</th><th>Throughput</th></tr>\n"
    
    # Data rows
    for workload_name, workload_results in sorted(results.items()):
        for result in workload_results:
            device_name = result.device_name
            execution_time = f"{result.execution_time:.4f}"
            host_memory = f"{result.memory_usage.get('host', 0):.2f}"
            device_memory = f"{result.memory_usage.get('device', 0):.2f}"
            gpu_utilization = f"{result.gpu_utilization:.2f}" if result.gpu_utilization is not None else "N/A"
            throughput = f"{result.throughput:.4f}" if result.throughput is not None else "N/A"
            
            html += f"<tr><td>{workload_name}</td><td>{device_name}</td><td>{execution_time}</td><td>{host_memory}</td><td>{device_memory}</td><td>{gpu_utilization}</td><td>{throughput}</td></tr>\n"
    
    html += "</table>\n"
    return html

def generate_summary_report(results: Dict[str, List[Any]], output_file: str) -> None:
    """
    Generate summary report for benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the report
    """
    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Generate plots
    plot_execution_time_comparison(results, os.path.join(images_dir, "execution_time.png"))
    plot_memory_usage(results, os.path.join(images_dir, "memory_usage.png"))
    plot_gpu_utilization(results, os.path.join(images_dir, "gpu_utilization.png"))
    plot_energy_consumption(results, os.path.join(images_dir, "energy_consumption.png"))
    plot_throughput_comparison(results, os.path.join(images_dir, "throughput.png"))
    
    # Generate HTML report
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1, h2 {
            color: #333;
        }
        .section {
            margin-bottom: 30px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
        }
        th, td {
            padding: 8px;
            text-align: left;
            border: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <h1>GPU-Accelerated Scientific Workloads Benchmark Results</h1>
    <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    
    <div class="section">
        <h2>Results Summary</h2>
        """ + generate_html_table(results) + """
    </div>
    
    <div class="section">
        <h2>Execution Time Comparison</h2>
        <div class="image-container">
            <img src="images/execution_time.png" alt="Execution Time Comparison">
        </div>
    </div>
    
    <div class="section">
        <h2>Memory Usage</h2>
        <div class="image-container">
            <img src="images/memory_usage.png" alt="Memory Usage">
        </div>
    </div>
    
    <div class="section">
        <h2>GPU Utilization</h2>
        <div class="image-container">
            <img src="images/gpu_utilization.png" alt="GPU Utilization">
        </div>
    </div>
    
    <div class="section">
        <h2>Throughput Comparison</h2>
        <div class="image-container">
            <img src="images/throughput.png" alt="Throughput Comparison">
        </div>
    </div>
    
    <div class="section">
        <h2>Energy Consumption</h2>
        <div class="image-container">
            <img src="images/energy_consumption.png" alt="Energy Consumption">
        </div>
    </div>
    
    <div class="section">
        <h2>System Information</h2>
        <table>
            <tr><th>Device</th><th>Compute Capability</th><th>Total Memory (MB)</th><th>Clock Rate (MHz)</th><th>Multiprocessors</th></tr>
"""
    
    # Add system information
    devices_info = {}
    for workload_results in results.values():
        for result in workload_results:
            device_name = result.device_name
            if device_name in devices_info:
                continue
            
            devices_info[device_name] = result.device_capabilities
            
            compute_capability = result.device_capabilities.get("compute_capability", "N/A")
            total_memory = f"{result.device_capabilities.get('total_memory', 0):.2f}"
            clock_rate = f"{result.device_capabilities.get('clock_rate', 0):.2f}"
            num_multiprocessors = result.device_capabilities.get("num_multiprocessors", "N/A")
            
            html += f"<tr><td>{device_name}</td><td>{compute_capability}</td><td>{total_memory}</td><td>{clock_rate}</td><td>{num_multiprocessors}</td></tr>\n"
    
    html += """        </table>
    </div>
</body>
</html>
"""
    
    # Write HTML report
    with open(output_file, "w") as f:
        f.write(html)
    
    print(f"Generated report: {output_file}")

if __name__ == "__main__":
    # Example usage
    pass