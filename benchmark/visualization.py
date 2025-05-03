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

# Import cost_modeling utilities
from benchmark.cost_modeling import format_cost

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

def plot_cost_comparison(results: Dict[str, List[Any]], 
                        output_file: Optional[str] = None,
                        title: str = "Cost Comparison",
                        figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Plot cost comparison for different workloads and computing environments.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the plot
        title: Title of the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Group results by workload
    workloads = set()
    cost_data = {}
    
    for workload_name, workload_results in results.items():
        workloads.add(workload_name)
        
        for result in workload_results:
            if not hasattr(result, 'cost_metrics') or not result.cost_metrics:
                continue
                
            if workload_name not in cost_data:
                cost_data[workload_name] = {
                    "jetson": [],
                    "aws": [],
                    "azure": [],
                    "gcp": [],
                    "dgx_spark": [],
                    "slurm_cluster": []
                }
            
            # Jetson cost
            if "jetson" in result.cost_metrics:
                cost_data[workload_name]["jetson"].append(result.cost_metrics["jetson"]["total_cost"])
            
            # Cloud costs
            if "cloud" in result.cost_metrics:
                cloud_costs = result.cost_metrics["cloud"]
                
                if "aws" in cloud_costs:
                    cost_data[workload_name]["aws"].append(cloud_costs["aws"]["total_cost"])
                
                if "azure" in cloud_costs:
                    cost_data[workload_name]["azure"].append(cloud_costs["azure"]["total_cost"])
                
                if "gcp" in cloud_costs:
                    cost_data[workload_name]["gcp"].append(cloud_costs["gcp"]["total_cost"])
                
                if "dgx_spark" in cloud_costs:
                    cost_data[workload_name]["dgx_spark"].append(cloud_costs["dgx_spark"]["total_cost"])
                
                if "slurm_cluster" in cloud_costs:
                    cost_data[workload_name]["slurm_cluster"].append(cloud_costs["slurm_cluster"]["total_cost"])
    
    # Check if we have any cost data
    if not cost_data:
        print("No cost metrics available.")
        return
    
    # Sort workloads
    workloads = sorted(list(workloads))
    
    # Define environments to plot
    environments = ["jetson", "aws", "azure", "gcp", "dgx_spark", "slurm_cluster"]
    env_labels = ["Jetson", "AWS", "Azure", "GCP", "DGX Spark", "Slurm Cluster"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    # Calculate positions for grouped bars
    bar_width = 0.8 / len(environments)
    indices = np.arange(len(workloads))
    
    # Plot grouped bars for each environment
    for i, (env, label, color) in enumerate(zip(environments, env_labels, colors)):
        costs = []
        for workload in workloads:
            if workload in cost_data and cost_data[workload][env]:
                # Average cost if multiple results
                costs.append(np.mean(cost_data[workload][env]))
            else:
                costs.append(0)
        
        plt.bar(indices + i * bar_width, costs, bar_width, label=label, color=color)
    
    # Add labels and title
    plt.xlabel("Workload")
    plt.ylabel("Cost (USD)")
    plt.title(title)
    plt.xticks(indices + bar_width * len(environments) / 2, workloads, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_slurm_dgx_comparison(results: Dict[str, List[Any]],
                             output_file: Optional[str] = None,
                             title: str = "Enterprise Computing Cost Comparison",
                             figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    Plot a specialized cost comparison between Jetson, DGX systems, and Slurm clusters.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the plot
        title: Title of the plot
        figsize: Figure size
    """
    # Group results by workload
    workloads = set()
    cost_data = {}
    system_info = {
        "dgx_spark": {},
        "slurm_cluster": {}
    }
    
    for workload_name, workload_results in results.items():
        workloads.add(workload_name)
        
        for result in workload_results:
            if not hasattr(result, 'cost_metrics') or not result.cost_metrics:
                continue
                
            if workload_name not in cost_data:
                cost_data[workload_name] = {
                    "jetson": [],
                    "dgx_spark": [],
                    "slurm_cluster": []
                }
            
            # Jetson cost
            if "jetson" in result.cost_metrics:
                cost_data[workload_name]["jetson"].append(result.cost_metrics["jetson"]["total_cost"])
            
            # Enterprise systems
            if "cloud" in result.cost_metrics:
                cloud_costs = result.cost_metrics["cloud"]
                
                if "dgx_spark" in cloud_costs:
                    cost_data[workload_name]["dgx_spark"].append(cloud_costs["dgx_spark"]["total_cost"])
                    # Capture system info for labeling
                    if "system_info" in cloud_costs["dgx_spark"]:
                        system_info["dgx_spark"] = cloud_costs["dgx_spark"]["system_info"]
                
                if "slurm_cluster" in cloud_costs:
                    cost_data[workload_name]["slurm_cluster"].append(cloud_costs["slurm_cluster"]["total_cost"])
                    # Capture node info for labeling
                    if "node_info" in cloud_costs["slurm_cluster"]:
                        system_info["slurm_cluster"] = cloud_costs["slurm_cluster"]["node_info"]
    
    # Check if we have any relevant data
    has_dgx = any(len(cost_data.get(w, {}).get("dgx_spark", [])) > 0 for w in workloads)
    has_slurm = any(len(cost_data.get(w, {}).get("slurm_cluster", [])) > 0 for w in workloads)
    
    if not has_dgx and not has_slurm:
        print("No DGX or Slurm cluster data available.")
        return
    
    # Create plot with multiple facets
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Sort workloads
    workloads = sorted(list(workloads))
    indices = np.arange(len(workloads))
    
    # Create cost comparison subplot
    ax1 = axs[0]
    
    # Define environments to plot
    environments = ["jetson"]
    env_labels = ["Jetson"]
    colors = ["#1f77b4"]
    
    if has_dgx:
        environments.append("dgx_spark")
        # Create label with DGX system info
        dgx_label = "DGX"
        if system_info.get("dgx_spark", {}).get("type"):
            dgx_type = system_info["dgx_spark"]["type"]
            if dgx_type == "dgx_superpod":
                nodes = system_info["dgx_spark"].get("nodes", "")
                dgx_label = f"DGX SuperPOD ({nodes} nodes)"
            elif dgx_type == "dgx_a100":
                quantity = system_info["dgx_spark"].get("quantity", 1)
                dgx_label = f"DGX A100 (x{quantity})" if quantity > 1 else "DGX A100"
            elif dgx_type == "dgx_h100":
                quantity = system_info["dgx_spark"].get("quantity", 1)
                dgx_label = f"DGX H100 (x{quantity})" if quantity > 1 else "DGX H100"
            elif dgx_type == "dgx_station_a100":
                dgx_label = "DGX Station A100"
            elif dgx_type == "dgx_station_h100":
                dgx_label = "DGX Station H100"
        env_labels.append(dgx_label)
        colors.append("#9467bd")
    
    if has_slurm:
        environments.append("slurm_cluster")
        # Create label with Slurm cluster info
        slurm_label = "Slurm Cluster"
        if system_info.get("slurm_cluster", {}).get("type"):
            slurm_type = system_info["slurm_cluster"]["type"]
            nodes = system_info["slurm_cluster"].get("count", "")
            if slurm_type == "basic_cpu":
                slurm_label = f"CPU Cluster ({nodes} nodes)"
            elif slurm_type == "basic_gpu":
                slurm_label = f"Basic GPU Cluster ({nodes} nodes)"
            elif slurm_type == "highend_gpu":
                slurm_label = f"High-end GPU Cluster ({nodes} nodes)"
            elif slurm_type == "jetson_cluster":
                slurm_label = f"Jetson Cluster ({nodes} nodes)"
            elif slurm_type == "custom":
                gpu_type = system_info["slurm_cluster"].get("gpu_type", "GPU")
                slurm_label = f"Custom Cluster with {gpu_type} ({nodes} nodes)"
        env_labels.append(slurm_label)
        colors.append("#8c564b")
    
    # Calculate positions for grouped bars
    bar_width = 0.8 / len(environments)
    
    # Plot cost comparison (absolute costs)
    for i, (env, label, color) in enumerate(zip(environments, env_labels, colors)):
        costs = []
        for workload in workloads:
            if workload in cost_data and cost_data[workload][env]:
                # Average cost if multiple results
                costs.append(np.mean(cost_data[workload][env]))
            else:
                costs.append(0)
        
        ax1.bar(indices + i * bar_width, costs, bar_width, label=label, color=color)
    
    ax1.set_ylabel("Cost (USD)")
    ax1.set_title("Absolute Cost")
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create cost ratio subplot
    ax2 = axs[1]
    
    # For each system, calculate cost ratio to Jetson
    for i, (env, label, color) in enumerate(zip(environments[1:], env_labels[1:], colors[1:])):
        ratios = []
        for workload in workloads:
            if (workload in cost_data and 
                cost_data[workload][env] and 
                cost_data[workload]["jetson"]):
                # Average costs if multiple results
                env_cost = np.mean(cost_data[workload][env])
                jetson_cost = np.mean(cost_data[workload]["jetson"])
                # Calculate ratio (enterprise system / Jetson)
                if jetson_cost > 0:
                    ratio = env_cost / jetson_cost
                else:
                    ratio = float('nan')
                ratios.append(ratio)
            else:
                ratios.append(float('nan'))
        
        # Plot cost ratio bars
        ax2.bar(indices + i * bar_width, ratios, bar_width, label=label, color=color)
    
    # Add reference line for equal cost
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.8, label="Equal Cost")
    
    # Add labels for cost ratio subplot
    ax2.set_xlabel("Workload")
    ax2.set_ylabel("Cost Ratio (vs Jetson)")
    ax2.set_title("Cost Ratio (Higher = More Expensive than Jetson)")
    ax2.set_xticks(indices)
    ax2.set_xticklabels(workloads, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_system_scaling_analysis(results: Dict[str, List[Any]],
                                output_file: Optional[str] = None,
                                title: str = "System Scaling Analysis",
                                figsize: Tuple[int, int] = (16, 10)) -> None:
    """
    Plot analysis of how costs scale with system size for DGX and Slurm clusters.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the plot
        title: Title of the plot
        figsize: Figure size
    """
    # Create data structure for system scaling
    scaling_data = {
        "dgx_spark": {},
        "slurm_cluster": {}
    }
    
    # Extract system info and costs
    for workload_name, workload_results in results.items():
        for result in workload_results:
            if not hasattr(result, 'cost_metrics') or not result.cost_metrics or "cloud" not in result.cost_metrics:
                continue
            
            cloud_costs = result.cost_metrics["cloud"]
            
            # DGX scaling
            if "dgx_spark" in cloud_costs:
                dgx_info = cloud_costs["dgx_spark"].get("system_info", {})
                dgx_type = dgx_info.get("type", "unknown")
                
                # Determine system scale metric
                if dgx_type == "dgx_superpod":
                    scale_metric = dgx_info.get("nodes", 1)
                    scale_name = "Nodes"
                else:
                    scale_metric = dgx_info.get("quantity", 1) * dgx_info.get("gpus", 8)
                    scale_name = "Total GPUs"
                
                # Store data
                if dgx_type not in scaling_data["dgx_spark"]:
                    scaling_data["dgx_spark"][dgx_type] = {
                        "scale_metrics": [],
                        "costs": [],
                        "scale_name": scale_name,
                        "workloads": []
                    }
                
                scaling_data["dgx_spark"][dgx_type]["scale_metrics"].append(scale_metric)
                scaling_data["dgx_spark"][dgx_type]["costs"].append(cloud_costs["dgx_spark"]["total_cost"])
                scaling_data["dgx_spark"][dgx_type]["workloads"].append(workload_name)
            
            # Slurm scaling
            if "slurm_cluster" in cloud_costs:
                slurm_info = cloud_costs["slurm_cluster"].get("node_info", {})
                slurm_type = slurm_info.get("type", "unknown")
                nodes = slurm_info.get("count", 1)
                
                # Store data
                if slurm_type not in scaling_data["slurm_cluster"]:
                    scaling_data["slurm_cluster"][slurm_type] = {
                        "scale_metrics": [],
                        "costs": [],
                        "scale_name": "Nodes",
                        "workloads": []
                    }
                
                scaling_data["slurm_cluster"][slurm_type]["scale_metrics"].append(nodes)
                scaling_data["slurm_cluster"][slurm_type]["costs"].append(cloud_costs["slurm_cluster"]["total_cost"])
                scaling_data["slurm_cluster"][slurm_type]["workloads"].append(workload_name)
    
    # Check if we have any scaling data
    has_dgx_scaling = any(len(data["scale_metrics"]) > 0 for data in scaling_data["dgx_spark"].values())
    has_slurm_scaling = any(len(data["scale_metrics"]) > 0 for data in scaling_data["slurm_cluster"].values())
    
    if not has_dgx_scaling and not has_slurm_scaling:
        print("No system scaling data available.")
        return
    
    # Create plot with multiple facets
    fig = plt.figure(figsize=figsize)
    
    # Determine grid layout based on available data
    dgx_types = list(scaling_data["dgx_spark"].keys())
    slurm_types = list(scaling_data["slurm_cluster"].keys())
    
    total_plots = len(dgx_types) + len(slurm_types)
    cols = min(3, total_plots)
    rows = (total_plots + cols - 1) // cols
    
    # Create subplots
    plot_idx = 1
    
    # DGX system scaling plots
    dgx_colors = {
        "dgx_a100": "#9467bd",
        "dgx_h100": "#c44e52",
        "dgx_station_a100": "#8c564b",
        "dgx_station_h100": "#e377c2",
        "dgx_superpod": "#7f7f7f"
    }
    
    for dgx_type in dgx_types:
        if not scaling_data["dgx_spark"][dgx_type]["scale_metrics"]:
            continue
            
        # Create subplot
        ax = fig.add_subplot(rows, cols, plot_idx)
        plot_idx += 1
        
        # Get data
        data = scaling_data["dgx_spark"][dgx_type]
        x = np.array(data["scale_metrics"])
        y = np.array(data["costs"])
        
        # Sort data by scale
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        workloads = [data["workloads"][i] for i in sort_idx]
        
        # Create scatter plot with labels
        scatter = ax.scatter(x, y, c=dgx_colors.get(dgx_type, "#9467bd"), s=100, alpha=0.7)
        
        # Add text labels for workloads
        for i, workload in enumerate(workloads):
            ax.annotate(workload, (x[i], y[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8)
        
        # Add trendline if we have enough points
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.7)
            
            # Add slope annotation
            if z[0] != 0:
                ax.text(0.05, 0.95, f"Slope: {z[0]:.2f} USD/{data['scale_name'].lower()}", 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top')
        
        # Format subplot
        dgx_type_nice = dgx_type.replace('_', ' ').upper()
        ax.set_title(f"{dgx_type_nice} Scaling")
        ax.set_xlabel(data["scale_name"])
        ax.set_ylabel("Cost (USD)")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Force integer ticks for scale metric
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Slurm cluster scaling plots
    slurm_colors = {
        "basic_cpu": "#1f77b4",
        "basic_gpu": "#ff7f0e",
        "highend_gpu": "#2ca02c",
        "jetson_cluster": "#d62728",
        "custom": "#8c564b"
    }
    
    for slurm_type in slurm_types:
        if not scaling_data["slurm_cluster"][slurm_type]["scale_metrics"]:
            continue
            
        # Create subplot
        ax = fig.add_subplot(rows, cols, plot_idx)
        plot_idx += 1
        
        # Get data
        data = scaling_data["slurm_cluster"][slurm_type]
        x = np.array(data["scale_metrics"])
        y = np.array(data["costs"])
        
        # Sort data by scale
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        workloads = [data["workloads"][i] for i in sort_idx]
        
        # Create scatter plot with labels
        scatter = ax.scatter(x, y, c=slurm_colors.get(slurm_type, "#8c564b"), s=100, alpha=0.7)
        
        # Add text labels for workloads
        for i, workload in enumerate(workloads):
            ax.annotate(workload, (x[i], y[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8)
        
        # Add trendline if we have enough points
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.7)
            
            # Add slope annotation
            if z[0] != 0:
                ax.text(0.05, 0.95, f"Slope: {z[0]:.2f} USD/node", 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top')
        
        # Format subplot
        slurm_type_nice = slurm_type.replace('_', ' ').title()
        ax.set_title(f"{slurm_type_nice} Cluster Scaling")
        ax.set_xlabel(data["scale_name"])
        ax.set_ylabel("Cost (USD)")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Force integer ticks for scale metric
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_cost_per_operation(results: Dict[str, List[Any]], 
                           output_file: Optional[str] = None,
                           title: str = "Cost per Operation",
                           figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Plot cost per operation for different workloads and computing environments.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the plot
        title: Title of the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Group results by workload
    workloads = set()
    cost_op_data = {}
    
    for workload_name, workload_results in results.items():
        workloads.add(workload_name)
        
        for result in workload_results:
            if not hasattr(result, 'cost_metrics') or not result.cost_metrics:
                continue
                
            # Check if cost per operation data is available
            if ("comparison" not in result.cost_metrics or 
                "cost_per_operation" not in result.cost_metrics["comparison"]):
                continue
                
            cost_per_op = result.cost_metrics["comparison"]["cost_per_operation"]
            
            if workload_name not in cost_op_data:
                cost_op_data[workload_name] = {
                    "Jetson": [],
                    "AWS": [],
                    "Azure": [],
                    "GCP": [],
                    "DGX Spark": [],
                    "Slurm Cluster": []
                }
            
            # Add cost per operation data
            for env, cost in cost_per_op.items():
                if env in cost_op_data[workload_name]:
                    cost_op_data[workload_name][env].append(cost)
    
    # Check if we have any cost per operation data
    if not cost_op_data:
        print("No cost per operation metrics available.")
        return
    
    # Sort workloads
    workloads = sorted(list(workloads))
    
    # Define environments to plot
    environments = ["Jetson", "AWS", "Azure", "GCP", "DGX Spark", "Slurm Cluster"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    
    # Calculate positions for grouped bars
    bar_width = 0.8 / len(environments)
    indices = np.arange(len(workloads))
    
    # Plot grouped bars for each environment
    for i, (env, color) in enumerate(zip(environments, colors)):
        costs = []
        for workload in workloads:
            if workload in cost_op_data and cost_op_data[workload][env]:
                # Average cost if multiple results
                costs.append(np.mean(cost_op_data[workload][env]))
            else:
                costs.append(0)
        
        plt.bar(indices + i * bar_width, costs, bar_width, label=env, color=color)
    
    # Add labels and title
    plt.xlabel("Workload")
    plt.ylabel("Cost per Operation (USD)")
    plt.title(title)
    plt.xticks(indices + bar_width * len(environments) / 2, workloads, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

def plot_break_even_analysis(results: Dict[str, List[Any]], 
                            output_file: Optional[str] = None,
                            title: str = "Break-Even Analysis",
                            figsize: Tuple[int, int] = (14, 8)) -> None:
    """
    Plot break-even analysis for different workloads and cloud providers.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Output file for the plot
        title: Title of the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Group results by cloud provider
    providers = ["AWS", "Azure", "GCP"]
    provider_keys = ["aws", "azure", "gcp"]
    break_even_data = {provider: [] for provider in providers}
    workload_names = []
    
    # Collect break-even data
    for workload_name, workload_results in results.items():
        for result in workload_results:
            if not hasattr(result, 'cost_metrics') or not result.cost_metrics:
                continue
                
            # Check if break-even data is available
            if ("comparison" not in result.cost_metrics or 
                "break_even_hours" not in result.cost_metrics["comparison"]):
                continue
                
            break_even_hours = result.cost_metrics["comparison"]["break_even_hours"]
            workload_names.append(workload_name)
            
            # Add break-even data for each provider
            for provider, key in zip(providers, provider_keys):
                if key in break_even_hours:
                    hours = break_even_hours[key]
                    # Cap at 10000 hours (about 1 year) for visualization
                    if hours != float('inf'):
                        hours = min(hours, 10000)
                    else:
                        hours = 10000  # Use 10000 as proxy for infinity
                    break_even_data[provider].append(hours)
                else:
                    break_even_data[provider].append(0)
    
    # Check if we have any break-even data
    if not workload_names:
        print("No break-even metrics available.")
        return
    
    # Set up subplots - one for each provider
    fig, axs = plt.subplots(len(providers), 1, figsize=figsize, sharex=True)
    colors = ["#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, (provider, color) in enumerate(zip(providers, colors)):
        ax = axs[i] if len(providers) > 1 else axs
        
        # Sort workload names and break-even hours together
        sorted_data = sorted(zip(workload_names, break_even_data[provider]), 
                             key=lambda x: x[1])
        sorted_names = [x[0] for x in sorted_data]
        sorted_hours = [x[1] for x in sorted_data]
        
        # Create horizontal bar chart
        ax.barh(range(len(sorted_names)), sorted_hours, color=color)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel(f"Break-Even Time (Hours) - {provider}")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add hour values as text
        for j, hours in enumerate(sorted_hours):
            if hours >= 9999:  # Close to our infinity proxy
                ax.text(hours * 0.95, j, "∞", va='center')
            elif hours > 0:
                ax.text(hours * 1.02, j, f"{hours:.1f}", va='center')
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate suptitle
    
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

def generate_cost_comparison_table(results: Dict[str, List[Any]]) -> str:
    """
    Generate HTML table for cost comparison results.
    
    Args:
        results: Dictionary of benchmark results
    
    Returns:
        str: HTML table
    """
    # Check if any results have cost metrics
    has_cost_metrics = False
    for workload_results in results.values():
        for result in workload_results:
            if hasattr(result, 'cost_metrics') and result.cost_metrics:
                has_cost_metrics = True
                break
        if has_cost_metrics:
            break
    
    if not has_cost_metrics:
        return "<p>No cost metrics available.</p>"
    
    html = "<table border='1' cellpadding='5' cellspacing='0'>\n"
    
    # Header row
    html += "<tr><th>Workload</th><th>Device</th><th>Execution Time (s)</th>"
    html += "<th>Jetson Cost</th><th>AWS Cost</th><th>Azure Cost</th><th>GCP Cost</th>"
    html += "<th>Cost Ratio (AWS/Jetson)</th><th>Cost Ratio (Azure/Jetson)</th><th>Cost Ratio (GCP/Jetson)</th>"
    html += "</tr>\n"
    
    # Data rows
    for workload_name, workload_results in sorted(results.items()):
        for result in workload_results:
            if not hasattr(result, 'cost_metrics') or not result.cost_metrics:
                continue
                
            device_name = result.device_name
            execution_time = f"{result.execution_time:.4f}"
            
            # Jetson cost
            jetson_cost = "N/A"
            if "jetson" in result.cost_metrics:
                jetson_cost = format_cost(result.cost_metrics["jetson"]["total_cost"])
            
            # Cloud costs
            aws_cost = "N/A"
            azure_cost = "N/A"
            gcp_cost = "N/A"
            
            if "cloud" in result.cost_metrics:
                cloud_costs = result.cost_metrics["cloud"]
                
                if "aws" in cloud_costs:
                    aws_cost = format_cost(cloud_costs["aws"]["total_cost"])
                
                if "azure" in cloud_costs:
                    azure_cost = format_cost(cloud_costs["azure"]["total_cost"])
                
                if "gcp" in cloud_costs:
                    gcp_cost = format_cost(cloud_costs["gcp"]["total_cost"])
            
            # Cost ratios
            aws_ratio = "N/A"
            azure_ratio = "N/A"
            gcp_ratio = "N/A"
            
            if "comparison" in result.cost_metrics and "cost_ratios" in result.cost_metrics["comparison"]:
                cost_ratios = result.cost_metrics["comparison"]["cost_ratios"]
                
                if "aws" in cost_ratios:
                    aws_ratio = f"{cost_ratios['aws']:.2f}x"
                
                if "azure" in cost_ratios:
                    azure_ratio = f"{cost_ratios['azure']:.2f}x"
                
                if "gcp" in cost_ratios:
                    gcp_ratio = f"{cost_ratios['gcp']:.2f}x"
            
            html += f"<tr><td>{workload_name}</td><td>{device_name}</td><td>{execution_time}</td>"
            html += f"<td>{jetson_cost}</td><td>{aws_cost}</td><td>{azure_cost}</td><td>{gcp_cost}</td>"
            html += f"<td>{aws_ratio}</td><td>{azure_ratio}</td><td>{gcp_ratio}</td></tr>\n"
    
    html += "</table>\n"
    return html

def generate_break_even_table(results: Dict[str, List[Any]]) -> str:
    """
    Generate HTML table for break-even analysis.
    
    Args:
        results: Dictionary of benchmark results
    
    Returns:
        str: HTML table
    """
    # Check if any results have break-even metrics
    has_break_even = False
    for workload_results in results.values():
        for result in workload_results:
            if (hasattr(result, 'cost_metrics') and 
                result.cost_metrics and 
                "comparison" in result.cost_metrics and 
                "break_even_hours" in result.cost_metrics["comparison"]):
                has_break_even = True
                break
        if has_break_even:
            break
    
    if not has_break_even:
        return "<p>No break-even analysis available.</p>"
    
    html = "<table border='1' cellpadding='5' cellspacing='0'>\n"
    
    # Header row
    html += "<tr><th>Workload</th><th>AWS Break-Even (hours)</th>"
    html += "<th>Azure Break-Even (hours)</th><th>GCP Break-Even (hours)</th>"
    html += "<th>DGX Spark Break-Even (hours)</th><th>Slurm Cluster Break-Even (hours)</th></tr>\n"
    
    # Data rows
    for workload_name, workload_results in sorted(results.items()):
        for result in workload_results:
            if (not hasattr(result, 'cost_metrics') or 
                not result.cost_metrics or 
                "comparison" not in result.cost_metrics or 
                "break_even_hours" not in result.cost_metrics["comparison"]):
                continue
                
            break_even_hours = result.cost_metrics["comparison"]["break_even_hours"]
            
            # Get break-even hours for each provider
            aws_hours = "N/A"
            azure_hours = "N/A"
            gcp_hours = "N/A"
            dgx_hours = "N/A"
            slurm_hours = "N/A"
            
            if "aws" in break_even_hours:
                hours = break_even_hours["aws"]
                aws_hours = "∞" if hours == float('inf') else f"{hours:.1f}"
            
            if "azure" in break_even_hours:
                hours = break_even_hours["azure"]
                azure_hours = "∞" if hours == float('inf') else f"{hours:.1f}"
            
            if "gcp" in break_even_hours:
                hours = break_even_hours["gcp"]
                gcp_hours = "∞" if hours == float('inf') else f"{hours:.1f}"
            
            if "dgx_spark" in break_even_hours:
                hours = break_even_hours["dgx_spark"]
                dgx_hours = "∞" if hours == float('inf') else f"{hours:.1f}"
            
            if "slurm_cluster" in break_even_hours:
                hours = break_even_hours["slurm_cluster"]
                slurm_hours = "∞" if hours == float('inf') else f"{hours:.1f}"
            
            html += f"<tr><td>{workload_name}</td><td>{aws_hours}</td>"
            html += f"<td>{azure_hours}</td><td>{gcp_hours}</td>"
            html += f"<td>{dgx_hours}</td><td>{slurm_hours}</td></tr>\n"
    
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
    
    # Check if any results have cost metrics
    has_cost_metrics = False
    has_dgx_or_slurm = False
    for workload_results in results.values():
        for result in workload_results:
            if hasattr(result, 'cost_metrics') and result.cost_metrics:
                has_cost_metrics = True
                # Check for DGX or Slurm data
                if "cloud" in result.cost_metrics:
                    if "dgx_spark" in result.cost_metrics["cloud"] or "slurm_cluster" in result.cost_metrics["cloud"]:
                        has_dgx_or_slurm = True
                        break
        if has_dgx_or_slurm:
            break
    
    # Generate performance plots
    plot_execution_time_comparison(results, os.path.join(images_dir, "execution_time.png"))
    plot_memory_usage(results, os.path.join(images_dir, "memory_usage.png"))
    plot_gpu_utilization(results, os.path.join(images_dir, "gpu_utilization.png"))
    plot_energy_consumption(results, os.path.join(images_dir, "energy_consumption.png"))
    plot_throughput_comparison(results, os.path.join(images_dir, "throughput.png"))
    
    # Generate cost comparison plots if cost metrics are available
    if has_cost_metrics:
        plot_cost_comparison(results, os.path.join(images_dir, "cost_comparison.png"))
        plot_cost_per_operation(results, os.path.join(images_dir, "cost_per_operation.png"))
        plot_break_even_analysis(results, os.path.join(images_dir, "break_even_analysis.png"))
        
        # Generate enterprise system plots if DGX or Slurm data is available
        if has_dgx_or_slurm:
            plot_slurm_dgx_comparison(results, os.path.join(images_dir, "enterprise_comparison.png"))
            plot_system_scaling_analysis(results, os.path.join(images_dir, "system_scaling.png"))
    
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
        h1, h2, h3 {
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
        .tabs {
            overflow: hidden;
            border: 1px solid #ccc;
            background-color: #f1f1f1;
            margin-top: 20px;
        }
        .tab-button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 14px 16px;
            transition: 0.3s;
            font-size: 16px;
        }
        .tab-button:hover {
            background-color: #ddd;
        }
        .tab-button.active {
            background-color: #ccc;
        }
        .tab-content {
            display: none;
            padding: 6px 12px;
            border: 1px solid #ccc;
            border-top: none;
        }
    </style>
    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tabbuttons;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tabbuttons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabbuttons.length; i++) {
                tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        
        // Set the first tab as active when the page loads
        window.onload = function() {
            document.getElementsByClassName("tab-button")[0].click();
        }
    </script>
</head>
<body>
    <h1>GPU-Accelerated Scientific Workloads Benchmark Results</h1>
    <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    
    <div class="section">
        <h2>Results Summary</h2>
        """ + generate_html_table(results) + """
    </div>
    
    <div class="tabs">
        <button class="tab-button" onclick="openTab(event, 'performance')">Performance Metrics</button>
        """ + ("""<button class="tab-button" onclick="openTab(event, 'cost')">Cost Analysis</button>""" if has_cost_metrics else "") + """
        <button class="tab-button" onclick="openTab(event, 'system')">System Information</button>
    </div>
    
    <div id="performance" class="tab-content">
        <div class="section">
            <h3>Execution Time Comparison</h3>
            <div class="image-container">
                <img src="images/execution_time.png" alt="Execution Time Comparison">
            </div>
        </div>
        
        <div class="section">
            <h3>Memory Usage</h3>
            <div class="image-container">
                <img src="images/memory_usage.png" alt="Memory Usage">
            </div>
        </div>
        
        <div class="section">
            <h3>GPU Utilization</h3>
            <div class="image-container">
                <img src="images/gpu_utilization.png" alt="GPU Utilization">
            </div>
        </div>
        
        <div class="section">
            <h3>Throughput Comparison</h3>
            <div class="image-container">
                <img src="images/throughput.png" alt="Throughput Comparison">
            </div>
        </div>
        
        <div class="section">
            <h3>Energy Consumption</h3>
            <div class="image-container">
                <img src="images/energy_consumption.png" alt="Energy Consumption">
            </div>
        </div>
    </div>
    """ + ("""
    <div id="cost" class="tab-content">
        <div class="section">
            <h3>Cost Comparison</h3>
            <p>This section compares the cost of running workloads on Jetson devices versus cloud providers.</p>
            """ + generate_cost_comparison_table(results) + """
            <div class="image-container">
                <img src="images/cost_comparison.png" alt="Cost Comparison">
            </div>
        </div>
        
        <div class="section">
            <h3>Cost per Operation</h3>
            <p>This metric shows the cost per operation for each computing environment, providing insight into cost efficiency.</p>
            <div class="image-container">
                <img src="images/cost_per_operation.png" alt="Cost per Operation">
            </div>
        </div>
        
        <div class="section">
            <h3>Break-Even Analysis</h3>
            <p>This analysis shows how many hours of operation are required for Jetson to become more cost-effective than cloud options.</p>
            """ + generate_break_even_table(results) + """
            <div class="image-container">
                <img src="images/break_even_analysis.png" alt="Break-Even Analysis">
            </div>
        </div>
        """ + ("""
        <div class="section">
            <h3>Enterprise Computing Comparison</h3>
            <p>This comparison shows how Jetson systems stack up against enterprise options like DGX systems and Slurm clusters.</p>
            <div class="image-container">
                <img src="images/enterprise_comparison.png" alt="Enterprise Computing Comparison">
            </div>
        </div>
        
        <div class="section">
            <h3>System Scaling Analysis</h3>
            <p>This analysis shows how costs scale with system size for different types of computing infrastructure.</p>
            <div class="image-container">
                <img src="images/system_scaling.png" alt="System Scaling Analysis">
            </div>
        </div>
        """ if has_dgx_or_slurm else "") + """
    </div>
    """ if has_cost_metrics else "") + """
    
    <div id="system" class="tab-content">
        <div class="section">
            <h3>System Information</h3>
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
    
    html += """            </table>
        </div>
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