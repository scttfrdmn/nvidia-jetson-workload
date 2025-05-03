#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Cost modeling for benchmarking suite.
Provides functionality to estimate and compare costs between local computing
and cloud providers based on workload performance.
"""

import os
import json
import yaml
import time
import math
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from enum import Enum, auto

class ComputeEnvironment(Enum):
    """Types of computing environments."""
    LOCAL_JETSON = auto()
    AWS_GPU = auto()
    AZURE_GPU = auto()
    GCP_GPU = auto()
    DGX_SPARK = auto()
    SLURM_CLUSTER = auto()

class CostModel:
    """Base class for cost models."""
    
    def __init__(self, name: str, currency: str = "USD"):
        """
        Initialize cost model.
        
        Args:
            name: Name of the cost model
            currency: Currency code (default: USD)
        """
        self.name = name
        self.currency = currency
    
    def estimate_cost(self, 
                      execution_time: float, 
                      memory_usage: Dict[str, float], 
                      gpu_utilization: Optional[float] = None,
                      energy_consumption: Optional[float] = None,
                      additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Estimate cost based on execution metrics.
        
        Args:
            execution_time: Execution time in seconds
            memory_usage: Dictionary with keys 'host' and 'device' (in MB)
            gpu_utilization: GPU utilization percentage (0-100)
            energy_consumption: Energy consumption in joules
            additional_metrics: Additional workload-specific metrics
        
        Returns:
            Dictionary with cost estimates
        """
        raise NotImplementedError("Subclasses must implement estimate_cost()")

class JetsonCostModel(CostModel):
    """Cost model for Jetson devices."""
    
    def __init__(self, 
                hardware_cost: float = 599.0,  # Cost of Jetson Orin NX Developer Kit
                power_cost: float = 0.12,  # Cost per kWh in USD
                amortization_period: int = 1095,  # 3 years in days
                maintenance_factor: float = 0.1,  # 10% of hardware cost per year for maintenance
                currency: str = "USD"):
        """
        Initialize Jetson cost model.
        
        Args:
            hardware_cost: Cost of Jetson hardware in USD
            power_cost: Cost per kWh in USD
            amortization_period: Hardware amortization period in days
            maintenance_factor: Annual maintenance cost as fraction of hardware cost
            currency: Currency code
        """
        super().__init__("Jetson", currency)
        self.hardware_cost = hardware_cost
        self.power_cost = power_cost
        self.amortization_period = amortization_period
        self.maintenance_factor = maintenance_factor
        
        # Compute daily amortized hardware cost
        self.daily_hardware_cost = hardware_cost / amortization_period
        
        # Daily maintenance cost
        self.daily_maintenance_cost = (hardware_cost * maintenance_factor) / 365
        
        # Jetson Orin NX specs
        self.max_power_watts = 25.0  # Maximum power consumption in watts
    
    def estimate_cost(self, 
                      execution_time: float, 
                      memory_usage: Dict[str, float], 
                      gpu_utilization: Optional[float] = None,
                      energy_consumption: Optional[float] = None,
                      additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Estimate cost for running workload on Jetson.
        
        Args:
            execution_time: Execution time in seconds
            memory_usage: Dictionary with keys 'host' and 'device' (in MB)
            gpu_utilization: GPU utilization percentage (0-100)
            energy_consumption: Energy consumption in joules
            additional_metrics: Additional workload-specific metrics
        
        Returns:
            Dictionary with cost components
        """
        # Calculate hardware cost (amortized for execution time)
        execution_days = execution_time / (24 * 3600)  # Convert seconds to days
        hardware_cost = self.daily_hardware_cost * execution_days
        maintenance_cost = self.daily_maintenance_cost * execution_days
        
        # Calculate power cost
        if energy_consumption is not None:
            # Convert joules to kWh
            kwh = energy_consumption / 3600000
            power_cost = kwh * self.power_cost
        else:
            # Estimate power based on GPU utilization
            if gpu_utilization is not None:
                utilization_factor = gpu_utilization / 100.0
            else:
                utilization_factor = 0.5  # Default to 50% if not provided
            
            # Calculate power consumption (25W at full utilization)
            power_watts = self.max_power_watts * (0.3 + 0.7 * utilization_factor)
            energy_kwh = (power_watts * execution_time) / 3600000
            power_cost = energy_kwh * self.power_cost
        
        # Total cost
        total_cost = hardware_cost + maintenance_cost + power_cost
        
        return {
            "total_cost": total_cost,
            "hardware_cost": hardware_cost,
            "maintenance_cost": maintenance_cost,
            "power_cost": power_cost,
            "currency": self.currency,
            "per_hour_cost": (total_cost / execution_time) * 3600 if execution_time > 0 else 0
        }

class CloudCostModel(CostModel):
    """Base class for cloud provider cost models."""
    
    def __init__(self, 
                name: str, 
                instance_type: str,
                instance_cost: float,  # Cost per hour
                currency: str = "USD",
                storage_cost: float = 0.08,  # Cost per GB-month
                data_transfer_cost: float = 0.09,  # Cost per GB out
                minimum_billing_time: float = 60.0):  # Minimum billing time in seconds
        """
        Initialize cloud provider cost model.
        
        Args:
            name: Name of the cloud provider
            instance_type: Type of instance
            instance_cost: Cost per hour for the instance
            currency: Currency code
            storage_cost: Cost per GB-month for storage
            data_transfer_cost: Cost per GB for data transfer out
            minimum_billing_time: Minimum billing time in seconds
        """
        super().__init__(name, currency)
        self.instance_type = instance_type
        self.instance_cost = instance_cost
        self.storage_cost = storage_cost
        self.data_transfer_cost = data_transfer_cost
        self.minimum_billing_time = minimum_billing_time
    
    def estimate_cost(self, 
                      execution_time: float, 
                      memory_usage: Dict[str, float], 
                      gpu_utilization: Optional[float] = None,
                      energy_consumption: Optional[float] = None,
                      additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Estimate cost for running workload on cloud provider.
        
        Args:
            execution_time: Execution time in seconds
            memory_usage: Dictionary with keys 'host' and 'device' (in MB)
            gpu_utilization: GPU utilization percentage (0-100)
            energy_consumption: Energy consumption in joules
            additional_metrics: Additional workload-specific metrics
        
        Returns:
            Dictionary with cost components
        """
        # Calculate billable time
        billable_time = max(execution_time, self.minimum_billing_time)
        
        # Calculate instance cost
        instance_cost = self.instance_cost * (billable_time / 3600)  # Convert seconds to hours
        
        # Calculate storage cost (assuming data is stored for 1 hour)
        # Convert MB to GB
        storage_gb = (memory_usage.get("host", 0) + memory_usage.get("device", 0)) / 1024
        storage_cost = storage_gb * self.storage_cost * (1 / (30 * 24))  # Convert GB-month to GB-hour
        
        # Calculate data transfer cost
        data_transfer_gb = storage_gb  # Assume all data is transferred out
        data_transfer_cost = data_transfer_gb * self.data_transfer_cost
        
        # Total cost
        total_cost = instance_cost + storage_cost + data_transfer_cost
        
        return {
            "total_cost": total_cost,
            "instance_cost": instance_cost,
            "storage_cost": storage_cost,
            "data_transfer_cost": data_transfer_cost,
            "billable_time": billable_time,
            "currency": self.currency,
            "per_hour_cost": self.instance_cost
        }

class AWSCostModel(CloudCostModel):
    """Cost model for AWS EC2 instances."""
    
    # AWS GPU instance types and pricing (on-demand, us-east-1)
    INSTANCE_TYPES = {
        "g4dn.xlarge": {
            "gpus": 1,
            "gpu_type": "NVIDIA T4",
            "vcpus": 4,
            "memory_gb": 16,
            "hourly_cost": 0.526
        },
        "g4dn.2xlarge": {
            "gpus": 1,
            "gpu_type": "NVIDIA T4",
            "vcpus": 8,
            "memory_gb": 32,
            "hourly_cost": 0.752
        },
        "g5.xlarge": {
            "gpus": 1,
            "gpu_type": "NVIDIA A10G",
            "vcpus": 4,
            "memory_gb": 16,
            "hourly_cost": 1.006
        },
        "p3.2xlarge": {
            "gpus": 1,
            "gpu_type": "NVIDIA V100",
            "vcpus": 8,
            "memory_gb": 61,
            "hourly_cost": 3.06
        },
        "g3s.xlarge": {
            "gpus": 1,
            "gpu_type": "NVIDIA Tesla M60",
            "vcpus": 4,
            "memory_gb": 30.5,
            "hourly_cost": 0.75
        }
    }
    
    def __init__(self, 
                instance_type: str = "g4dn.xlarge",
                currency: str = "USD",
                region: str = "us-east-1"):
        """
        Initialize AWS cost model.
        
        Args:
            instance_type: Type of AWS EC2 instance
            currency: Currency code
            region: AWS region
        """
        if instance_type not in self.INSTANCE_TYPES:
            raise ValueError(f"Unknown instance type: {instance_type}. Available types: {list(self.INSTANCE_TYPES.keys())}")
        
        self.region = region
        self.instance_specs = self.INSTANCE_TYPES[instance_type]
        
        super().__init__(
            name="AWS",
            instance_type=instance_type,
            instance_cost=self.instance_specs["hourly_cost"],
            currency=currency,
            storage_cost=0.08,  # $0.08 per GB-month for general purpose SSD (gp3)
            data_transfer_cost=0.09,  # $0.09 per GB for data transfer out
            minimum_billing_time=60.0  # AWS bills by the second with 1-minute minimum
        )

class AzureCostModel(CloudCostModel):
    """Cost model for Azure instances."""
    
    # Azure GPU instance types and pricing (pay-as-you-go)
    INSTANCE_TYPES = {
        "Standard_NC6s_v3": {
            "gpus": 1,
            "gpu_type": "NVIDIA Tesla V100",
            "vcpus": 6,
            "memory_gb": 112,
            "hourly_cost": 3.06
        },
        "Standard_NC4as_T4_v3": {
            "gpus": 1,
            "gpu_type": "NVIDIA Tesla T4",
            "vcpus": 4,
            "memory_gb": 28,
            "hourly_cost": 0.526
        },
        "Standard_ND96asr_A100_v4": {
            "gpus": 8,
            "gpu_type": "NVIDIA A100 80GB",
            "vcpus": 96,
            "memory_gb": 900,
            "hourly_cost": 32.77
        }
    }
    
    def __init__(self, 
                instance_type: str = "Standard_NC4as_T4_v3",
                currency: str = "USD",
                region: str = "eastus"):
        """
        Initialize Azure cost model.
        
        Args:
            instance_type: Type of Azure VM instance
            currency: Currency code
            region: Azure region
        """
        if instance_type not in self.INSTANCE_TYPES:
            raise ValueError(f"Unknown instance type: {instance_type}. Available types: {list(self.INSTANCE_TYPES.keys())}")
        
        self.region = region
        self.instance_specs = self.INSTANCE_TYPES[instance_type]
        
        super().__init__(
            name="Azure",
            instance_type=instance_type,
            instance_cost=self.instance_specs["hourly_cost"],
            currency=currency,
            storage_cost=0.095,  # $0.095 per GB-month for Premium SSD (P10)
            data_transfer_cost=0.087,  # $0.087 per GB for data transfer out
            minimum_billing_time=60.0  # Azure bills per minute
        )

class GCPCostModel(CloudCostModel):
    """Cost model for Google Cloud Platform instances."""
    
    # GCP GPU instance types and pricing (on-demand)
    INSTANCE_TYPES = {
        "n1-standard-4-t4": {
            "gpus": 1,
            "gpu_type": "NVIDIA T4",
            "vcpus": 4,
            "memory_gb": 15,
            "hourly_cost": 0.571  # $0.35 for n1-standard-4 + $0.35 for T4 GPU
        },
        "n1-standard-8-v100": {
            "gpus": 1,
            "gpu_type": "NVIDIA V100",
            "vcpus": 8,
            "memory_gb": 30,
            "hourly_cost": 2.98  # $0.38 for n1-standard-8 + $2.48 for V100 GPU
        },
        "a2-highgpu-1g": {
            "gpus": 1,
            "gpu_type": "NVIDIA A100",
            "vcpus": 12,
            "memory_gb": 85,
            "hourly_cost": 4.10
        }
    }
    
    def __init__(self, 
                instance_type: str = "n1-standard-4-t4",
                currency: str = "USD",
                region: str = "us-central1"):
        """
        Initialize GCP cost model.
        
        Args:
            instance_type: Type of GCP instance
            currency: Currency code
            region: GCP region
        """
        if instance_type not in self.INSTANCE_TYPES:
            raise ValueError(f"Unknown instance type: {instance_type}. Available types: {list(self.INSTANCE_TYPES.keys())}")
        
        self.region = region
        self.instance_specs = self.INSTANCE_TYPES[instance_type]
        
        super().__init__(
            name="GCP",
            instance_type=instance_type,
            instance_cost=self.instance_specs["hourly_cost"],
            currency=currency,
            storage_cost=0.17,  # $0.17 per GB-month for SSD Persistent Disk
            data_transfer_cost=0.08,  # $0.08 per GB for data transfer out (after 1st GB free)
            minimum_billing_time=60.0  # GCP bills by the second with 1-minute minimum
        )

class DGXSparkCostModel(CostModel):
    """Cost model for NVIDIA DGX Spark system."""
    
    # DGX system presets
    DGX_SYSTEMS = {
        "dgx_a100": {
            "name": "DGX A100",
            "base_cost": 199000.0,
            "gpus": 8,
            "gpu_type": "NVIDIA A100 80GB",
            "gpu_vram_gb": 80,
            "cpu_cores": 128,
            "system_memory_gb": 1024,
            "max_power_watts": 6500,
            "interconnect": "NVLink & NVSwitch"
        },
        "dgx_h100": {
            "name": "DGX H100",
            "base_cost": 300000.0,
            "gpus": 8,
            "gpu_type": "NVIDIA H100 80GB",
            "gpu_vram_gb": 80,
            "cpu_cores": 144,
            "system_memory_gb": 2048,
            "max_power_watts": 10800,
            "interconnect": "NVLink & NVSwitch"
        },
        "dgx_station_a100": {
            "name": "DGX Station A100",
            "base_cost": 99000.0,
            "gpus": 4,
            "gpu_type": "NVIDIA A100 80GB",
            "gpu_vram_gb": 80,
            "cpu_cores": 64,
            "system_memory_gb": 512,
            "max_power_watts": 1500,
            "interconnect": "NVLink"
        },
        "dgx_station_h100": {
            "name": "DGX Station H100",
            "base_cost": 150000.0,
            "gpus": 4,
            "gpu_type": "NVIDIA H100 80GB",
            "gpu_vram_gb": 80,
            "cpu_cores": 72,
            "system_memory_gb": 1024,
            "max_power_watts": 2000,
            "interconnect": "NVLink"
        },
        "dgx_superpod": {
            "name": "DGX SuperPOD",
            "base_cost": 5000000.0, # Estimated base cost for a small configuration
            "nodes": 20,  # Configurable number of DGX systems
            "gpus_per_node": 8,
            "total_gpus": 160,
            "gpu_type": "NVIDIA H100 80GB",
            "gpu_vram_gb": 80,
            "cpu_cores_per_node": 144,
            "system_memory_gb_per_node": 2048,
            "max_power_watts_per_node": 10800,
            "interconnect": "NVIDIA Quantum InfiniBand",
            "storage_tb": 1000
        }
    }
    
    def __init__(self, 
                system_type: str = "dgx_a100",
                quantity: int = 1,
                nodes_for_superpod: int = 20,
                power_cost: float = 0.12,  # Cost per kWh in USD
                amortization_period: int = 1095,  # 3 years in days
                maintenance_factor: float = 0.15,  # 15% of hardware cost per year for maintenance
                utilization_factor: float = 0.5,  # Average system utilization
                networking_infrastructure_cost: float = 20000.0,  # Additional networking infrastructure cost
                datacenter_overhead_factor: float = 0.2,  # Datacenter overhead (cooling, space, etc.) as fraction of power
                admin_cost_per_year: float = 120000.0,  # System administrator cost per year
                config_file: Optional[str] = None,  # Path to DGX system config file
                currency: str = "USD"):
        """
        Initialize DGX Spark cost model.
        
        Args:
            system_type: Type of DGX system ("dgx_a100", "dgx_h100", "dgx_station_a100", "dgx_station_h100", "dgx_superpod")
            quantity: Number of DGX systems (for non-SuperPOD systems)
            nodes_for_superpod: Number of DGX nodes in a SuperPOD configuration (only used when system_type is "dgx_superpod")
            power_cost: Cost per kWh in USD
            amortization_period: Hardware amortization period in days
            maintenance_factor: Annual maintenance cost as fraction of hardware cost
            utilization_factor: Average system utilization
            networking_infrastructure_cost: Additional networking infrastructure cost
            datacenter_overhead_factor: Datacenter overhead factor (cooling, space, etc.)
            admin_cost_per_year: System administrator cost per year
            config_file: Path to DGX system configuration file (overrides other parameters if provided)
            currency: Currency code
        """
        super().__init__("DGX Spark", currency)
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.json'):
                        config = json.load(f)
                    elif config_file.endswith(('.yaml', '.yml')):
                        import yaml
                        config = yaml.safe_load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {config_file}")
                
                # Override parameters with config file values
                system_type = config.get('system_type', system_type)
                quantity = config.get('quantity', quantity)
                nodes_for_superpod = config.get('nodes_for_superpod', nodes_for_superpod)
                power_cost = config.get('power_cost', power_cost)
                amortization_period = config.get('amortization_period', amortization_period)
                maintenance_factor = config.get('maintenance_factor', maintenance_factor)
                utilization_factor = config.get('utilization_factor', utilization_factor)
                networking_infrastructure_cost = config.get('networking_infrastructure_cost', networking_infrastructure_cost)
                datacenter_overhead_factor = config.get('datacenter_overhead_factor', datacenter_overhead_factor)
                admin_cost_per_year = config.get('admin_cost_per_year', admin_cost_per_year)
                currency = config.get('currency', currency)
            except Exception as e:
                print(f"Error loading DGX configuration from {config_file}: {e}")
                print("Using default values instead")
        
        # Get system configuration
        if system_type not in self.DGX_SYSTEMS:
            print(f"Unknown DGX system type: {system_type}, using dgx_a100 instead")
            system_type = "dgx_a100"
        
        self.system_type = system_type
        self.system_config = self.DGX_SYSTEMS[system_type].copy()
        self.quantity = quantity
        self.power_cost = power_cost
        self.amortization_period = amortization_period
        self.maintenance_factor = maintenance_factor
        self.utilization_factor = utilization_factor
        self.networking_infrastructure_cost = networking_infrastructure_cost
        self.datacenter_overhead_factor = datacenter_overhead_factor
        self.admin_cost_per_year = admin_cost_per_year
        
        # Adjust SuperPOD configuration if selected
        if system_type == "dgx_superpod":
            self.system_config["nodes"] = nodes_for_superpod
            self.system_config["total_gpus"] = nodes_for_superpod * self.system_config["gpus_per_node"]
            # Adjust cost based on number of nodes (non-linear scaling due to infrastructure sharing)
            self.system_config["base_cost"] = 200000.0 * nodes_for_superpod + 1000000.0  # Base cost plus per-node cost
        
        # Calculate total hardware cost
        if system_type == "dgx_superpod":
            self.hardware_cost = self.system_config["base_cost"] + networking_infrastructure_cost
        else:
            self.hardware_cost = (self.system_config["base_cost"] * quantity) + networking_infrastructure_cost
        
        # Compute hourly amortized hardware cost
        self.hourly_hardware_cost = (self.hardware_cost / amortization_period) / 24
        
        # Hourly maintenance cost
        self.hourly_maintenance_cost = (self.hardware_cost * maintenance_factor) / (365 * 24)
        
        # Hourly admin cost
        self.hourly_admin_cost = admin_cost_per_year / (365 * 24)
        
        # Calculate max power consumption
        if system_type == "dgx_superpod":
            self.max_power_watts = self.system_config["max_power_watts_per_node"] * nodes_for_superpod
        else:
            self.max_power_watts = self.system_config["max_power_watts"] * quantity
    
    def estimate_cost(self, 
                      execution_time: float, 
                      memory_usage: Dict[str, float], 
                      gpu_utilization: Optional[float] = None,
                      energy_consumption: Optional[float] = None,
                      additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Estimate cost for running workload on DGX Spark.
        
        Args:
            execution_time: Execution time in seconds
            memory_usage: Dictionary with keys 'host' and 'device' (in MB)
            gpu_utilization: GPU utilization percentage (0-100)
            energy_consumption: Energy consumption in joules
            additional_metrics: Additional workload-specific metrics
        
        Returns:
            Dictionary with cost components
        """
        # Calculate hardware cost (amortized for execution time)
        execution_hours = execution_time / 3600  # Convert seconds to hours
        hardware_cost = self.hourly_hardware_cost * execution_hours / self.utilization_factor
        maintenance_cost = self.hourly_maintenance_cost * execution_hours / self.utilization_factor
        admin_cost = self.hourly_admin_cost * execution_hours / self.utilization_factor
        
        # Calculate power cost
        if energy_consumption is not None:
            # Convert joules to kWh
            kwh = energy_consumption / 3600000
            power_cost = kwh * self.power_cost
            # Add datacenter overhead for cooling, etc.
            power_cost = power_cost * (1 + self.datacenter_overhead_factor)
        else:
            # Estimate power based on GPU utilization
            if gpu_utilization is not None:
                utilization_factor = gpu_utilization / 100.0
            else:
                utilization_factor = 0.7  # Default to 70% if not provided
            
            # Calculate power consumption with a base load plus utilization
            # Base load is 30% of max power, remaining 70% scales with utilization
            power_watts = self.max_power_watts * (0.3 + 0.7 * utilization_factor)
            
            # Convert to kWh and calculate cost
            energy_kwh = (power_watts * execution_time) / 3600000
            power_cost = energy_kwh * self.power_cost
            
            # Add datacenter overhead for cooling, etc.
            power_cost = power_cost * (1 + self.datacenter_overhead_factor)
        
        # Total cost
        total_cost = hardware_cost + maintenance_cost + admin_cost + power_cost
        
        # Prepare detailed result
        result = {
            "total_cost": total_cost,
            "hardware_cost": hardware_cost,
            "maintenance_cost": maintenance_cost,
            "admin_cost": admin_cost,
            "power_cost": power_cost,
            "currency": self.currency,
            "per_hour_cost": (total_cost / execution_time) * 3600 if execution_time > 0 else 0
        }
        
        # Add system information
        result["system_info"] = {
            "type": self.system_type,
            "name": self.system_config["name"],
        }
        
        if self.system_type == "dgx_superpod":
            result["system_info"]["nodes"] = self.system_config["nodes"]
            result["system_info"]["gpus_per_node"] = self.system_config["gpus_per_node"]
            result["system_info"]["total_gpus"] = self.system_config["total_gpus"]
            result["system_info"]["gpu_type"] = self.system_config["gpu_type"]
            result["system_info"]["interconnect"] = self.system_config["interconnect"]
        else:
            result["system_info"]["quantity"] = self.quantity
            result["system_info"]["gpus"] = self.system_config["gpus"] * self.quantity
            result["system_info"]["gpu_type"] = self.system_config["gpu_type"]
            result["system_info"]["system_memory_gb"] = self.system_config["system_memory_gb"]
            result["system_info"]["interconnect"] = self.system_config["interconnect"]
        
        return result

class SlurmClusterCostModel(CostModel):
    """Cost model for Slurm clusters."""
    
    # Node type presets
    NODE_TYPES = {
        "basic_cpu": {
            "cost": 1500.0,
            "power_watts": 200.0,
            "has_gpu": False,
            "memory_gb": 32,
            "cores": 16
        },
        "basic_gpu": {
            "cost": 3000.0,
            "power_watts": 300.0,
            "has_gpu": True,
            "memory_gb": 64,
            "cores": 16,
            "gpu_type": "NVIDIA T4",
            "gpus_per_node": 1
        },
        "highend_gpu": {
            "cost": 6000.0,
            "power_watts": 500.0,
            "has_gpu": True,
            "memory_gb": 128,
            "cores": 32,
            "gpu_type": "NVIDIA A100",
            "gpus_per_node": 4
        },
        "jetson_cluster": {
            "cost": 800.0,
            "power_watts": 30.0,
            "has_gpu": True,
            "memory_gb": 16,
            "cores": 8,
            "gpu_type": "NVIDIA Orin",
            "gpus_per_node": 1
        },
        "custom": {
            "cost": 2000.0,
            "power_watts": 300.0,
            "has_gpu": True,
            "memory_gb": 64,
            "cores": 16,
            "gpu_type": "Custom",
            "gpus_per_node": 1
        }
    }
    
    def __init__(self, 
                nodes: int = 4,
                node_type: str = "basic_gpu",
                custom_node_config: Optional[Dict[str, Any]] = None,
                power_cost: float = 0.12,  # Cost per kWh in USD
                amortization_period: int = 1095,  # 3 years in days
                maintenance_factor: float = 0.1,  # 10% of hardware cost per year for maintenance
                utilization_factor: float = 0.7,  # Average cluster utilization
                network_cost: float = 5000.0,  # Network infrastructure cost
                admin_cost_per_year: float = 10000.0,  # System administrator cost per year
                config_file: Optional[str] = None,  # Path to cluster config file
                currency: str = "USD"):
        """
        Initialize Slurm cluster cost model.
        
        Args:
            nodes: Number of nodes in the cluster
            node_type: Type of nodes ("basic_cpu", "basic_gpu", "highend_gpu", "jetson_cluster", "custom")
            custom_node_config: Custom node configuration (required if node_type is "custom")
            power_cost: Cost per kWh in USD
            amortization_period: Hardware amortization period in days
            maintenance_factor: Annual maintenance cost as fraction of hardware cost
            utilization_factor: Average cluster utilization
            network_cost: Network infrastructure cost in USD
            admin_cost_per_year: System administrator cost per year in USD
            config_file: Path to cluster configuration file (overrides other parameters if provided)
            currency: Currency code
        """
        super().__init__("Slurm Cluster", currency)
        
        # Load configuration from file if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    if config_file.endswith('.json'):
                        config = json.load(f)
                    elif config_file.endswith(('.yaml', '.yml')):
                        import yaml
                        config = yaml.safe_load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {config_file}")
                
                # Override parameters with config file values
                nodes = config.get('nodes', nodes)
                node_type = config.get('node_type', node_type)
                custom_node_config = config.get('custom_node_config', custom_node_config)
                power_cost = config.get('power_cost', power_cost)
                amortization_period = config.get('amortization_period', amortization_period)
                maintenance_factor = config.get('maintenance_factor', maintenance_factor)
                utilization_factor = config.get('utilization_factor', utilization_factor)
                network_cost = config.get('network_cost', network_cost)
                admin_cost_per_year = config.get('admin_cost_per_year', admin_cost_per_year)
                currency = config.get('currency', currency)
            except Exception as e:
                print(f"Error loading cluster configuration from {config_file}: {e}")
                print("Using default values instead")
        
        self.nodes = nodes
        self.node_type = node_type
        self.power_cost = power_cost
        self.amortization_period = amortization_period
        self.maintenance_factor = maintenance_factor
        self.utilization_factor = utilization_factor
        self.network_cost = network_cost
        self.admin_cost_per_year = admin_cost_per_year
        
        # Get node configuration
        if node_type == "custom" and custom_node_config:
            self.node_config = custom_node_config
        elif node_type in self.NODE_TYPES:
            self.node_config = self.NODE_TYPES[node_type].copy()
        else:
            print(f"Unknown node type: {node_type}, using basic_gpu instead")
            self.node_config = self.NODE_TYPES["basic_gpu"].copy()
        
        # Set cost per node
        self.cost_per_node = self.node_config["cost"]
        self.power_per_node = self.node_config["power_watts"]
        
        # Total hardware cost including network infrastructure
        self.hardware_cost = (nodes * self.cost_per_node) + network_cost
        
        # Compute hourly amortized hardware cost
        self.hourly_hardware_cost = (self.hardware_cost / amortization_period) / 24
        
        # Hourly maintenance cost
        self.hourly_maintenance_cost = (self.hardware_cost * maintenance_factor) / (365 * 24)
        
        # Hourly admin cost
        self.hourly_admin_cost = admin_cost_per_year / (365 * 24)
    
    def estimate_cost(self, 
                      execution_time: float, 
                      memory_usage: Dict[str, float], 
                      gpu_utilization: Optional[float] = None,
                      energy_consumption: Optional[float] = None,
                      additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Estimate cost for running workload on Slurm cluster.
        
        Args:
            execution_time: Execution time in seconds
            memory_usage: Dictionary with keys 'host' and 'device' (in MB)
            gpu_utilization: GPU utilization percentage (0-100)
            energy_consumption: Energy consumption in joules
            additional_metrics: Additional workload-specific metrics
        
        Returns:
            Dictionary with cost components
        """
        # Determine number of nodes used for this workload
        if additional_metrics and "nodes_used" in additional_metrics:
            nodes_used = additional_metrics["nodes_used"]
        else:
            # Estimate nodes needed based on memory requirements
            if "host" in memory_usage and memory_usage["host"] > 0:
                mem_gb_needed = memory_usage["host"] / 1024  # Convert MB to GB
                mem_gb_per_node = self.node_config.get("memory_gb", 64)
                nodes_for_memory = math.ceil(mem_gb_needed / mem_gb_per_node)
                nodes_used = max(1, min(nodes_for_memory, self.nodes))
            else:
                # Assume 1 node if no better estimate is available
                nodes_used = 1
        
        # Apply allocation factor (ratio of nodes used to total nodes)
        allocation_factor = nodes_used / self.nodes
        
        # Calculate hardware cost (amortized for execution time)
        execution_hours = execution_time / 3600  # Convert seconds to hours
        hardware_cost = self.hourly_hardware_cost * execution_hours * allocation_factor / self.utilization_factor
        maintenance_cost = self.hourly_maintenance_cost * execution_hours * allocation_factor / self.utilization_factor
        admin_cost = self.hourly_admin_cost * execution_hours * allocation_factor / self.utilization_factor
        
        # Calculate power cost
        if energy_consumption is not None:
            # Convert joules to kWh
            kwh = energy_consumption / 3600000
            power_cost = kwh * self.power_cost
        else:
            # Estimate power based on GPU utilization and nodes used
            if gpu_utilization is not None and self.node_config.get("has_gpu", False):
                utilization_factor = gpu_utilization / 100.0
            else:
                utilization_factor = 0.6  # Default to 60% if not provided
            
            # Calculate power consumption with a base load plus utilization
            power_watts = self.power_per_node * nodes_used * (0.4 + 0.6 * utilization_factor)
            energy_kwh = (power_watts * execution_time) / 3600000
            power_cost = energy_kwh * self.power_cost
        
        # Total cost
        total_cost = hardware_cost + maintenance_cost + admin_cost + power_cost
        
        # Prepare result
        result = {
            "total_cost": total_cost,
            "hardware_cost": hardware_cost,
            "maintenance_cost": maintenance_cost,
            "admin_cost": admin_cost,
            "power_cost": power_cost,
            "nodes_used": nodes_used,
            "currency": self.currency,
            "per_hour_cost": (total_cost / execution_time) * 3600 if execution_time > 0 else 0
        }
        
        # Add node configuration information
        result["node_info"] = {
            "type": self.node_type,
            "count": self.nodes,
            "cost_per_node": self.cost_per_node,
            "power_per_node": self.power_per_node
        }
        
        if self.node_config.get("has_gpu", False):
            result["node_info"]["gpu_type"] = self.node_config.get("gpu_type", "Unknown")
            result["node_info"]["gpus_per_node"] = self.node_config.get("gpus_per_node", 1)
        
        return result

class CostModelFactory:
    """Factory for creating cost models."""
    
    @staticmethod
    def create_model(environment: ComputeEnvironment, **kwargs) -> CostModel:
        """
        Create a cost model for the specified environment.
        
        Args:
            environment: Type of compute environment
            **kwargs: Additional parameters for the cost model
        
        Returns:
            CostModel: Cost model for the environment
        """
        if environment == ComputeEnvironment.LOCAL_JETSON:
            return JetsonCostModel(**kwargs)
        elif environment == ComputeEnvironment.AWS_GPU:
            return AWSCostModel(**kwargs)
        elif environment == ComputeEnvironment.AZURE_GPU:
            return AzureCostModel(**kwargs)
        elif environment == ComputeEnvironment.GCP_GPU:
            return GCPCostModel(**kwargs)
        elif environment == ComputeEnvironment.DGX_SPARK:
            return DGXSparkCostModel(**kwargs)
        elif environment == ComputeEnvironment.SLURM_CLUSTER:
            return SlurmClusterCostModel(**kwargs)
        else:
            raise ValueError(f"Unknown compute environment: {environment}")

def calculate_cost_comparison(jetson_result: Dict[str, float], 
                             cloud_results: Dict[str, Dict[str, float]],
                             workload_name: str,
                             execution_time: float,
                             throughput: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculate cost comparison metrics between Jetson and cloud providers.
    
    Args:
        jetson_result: Jetson cost estimate
        cloud_results: Dictionary of cloud provider cost estimates
        workload_name: Name of the workload
        execution_time: Execution time in seconds
        throughput: Throughput in operations per second
    
    Returns:
        Dictionary with cost comparison metrics
    """
    # Calculate total cost ratio (cloud / Jetson)
    cost_ratios = {}
    for provider, result in cloud_results.items():
        cost_ratios[provider] = result["total_cost"] / jetson_result["total_cost"] if jetson_result["total_cost"] > 0 else float('inf')
    
    # Calculate hourly cost ratio
    hourly_cost_ratios = {}
    for provider, result in cloud_results.items():
        hourly_cost_ratios[provider] = result["per_hour_cost"] / jetson_result["per_hour_cost"] if jetson_result["per_hour_cost"] > 0 else float('inf')
    
    # Calculate cost per operation
    cost_per_op = {}
    if throughput is not None and throughput > 0:
        jetson_cost_per_op = jetson_result["total_cost"] / (execution_time * throughput)
        cost_per_op["Jetson"] = jetson_cost_per_op
        
        for provider, result in cloud_results.items():
            cost_per_op[provider] = result["total_cost"] / (execution_time * throughput)
    
    # Calculate break-even time (hours)
    # Time at which Jetson becomes more cost-effective than cloud
    break_even_hours = {}
    for provider, result in cloud_results.items():
        if result["per_hour_cost"] > jetson_result["per_hour_cost"]:
            # If cloud hourly cost > Jetson hourly cost, Jetson is always more cost-effective
            break_even_hours[provider] = 0
        else:
            # Break-even point: Jetson hardware cost / (cloud hourly cost - Jetson hourly cost)
            hourly_cost_diff = jetson_result["per_hour_cost"] - result["per_hour_cost"]
            if hourly_cost_diff > 0:
                break_even_hours[provider] = jetson_result["hardware_cost"] / hourly_cost_diff
            else:
                # Cloud is always cheaper
                break_even_hours[provider] = float('inf')
    
    return {
        "workload_name": workload_name,
        "jetson_cost": jetson_result,
        "cloud_costs": cloud_results,
        "cost_ratios": cost_ratios,
        "hourly_cost_ratios": hourly_cost_ratios,
        "cost_per_operation": cost_per_op,
        "break_even_hours": break_even_hours
    }

def format_cost(cost: float, currency: str = "USD") -> str:
    """
    Format cost as a string with currency symbol.
    
    Args:
        cost: Cost value
        currency: Currency code
    
    Returns:
        Formatted cost string
    """
    if currency == "USD":
        return f"${cost:.4f}"
    elif currency == "EUR":
        return f"€{cost:.4f}"
    elif currency == "GBP":
        return f"£{cost:.4f}"
    else:
        return f"{cost:.4f} {currency}"

# Example usage
if __name__ == "__main__":
    # Create cost models
    jetson_model = JetsonCostModel()
    aws_model = AWSCostModel(instance_type="g4dn.xlarge")
    azure_model = AzureCostModel(instance_type="Standard_NC4as_T4_v3")
    
    # Sample execution metrics
    execution_time = 300  # 5 minutes
    memory_usage = {"host": 4096, "device": 2048}  # MB
    gpu_utilization = 80  # %
    
    # Estimate costs
    jetson_cost = jetson_model.estimate_cost(execution_time, memory_usage, gpu_utilization)
    aws_cost = aws_model.estimate_cost(execution_time, memory_usage, gpu_utilization)
    azure_cost = azure_model.estimate_cost(execution_time, memory_usage, gpu_utilization)
    
    # Compare costs
    comparison = calculate_cost_comparison(
        jetson_cost,
        {"AWS": aws_cost, "Azure": azure_cost},
        workload_name="nbody_sim",
        execution_time=execution_time,
        throughput=1000  # operations per second
    )
    
    # Print results
    print(f"Jetson cost: {format_cost(jetson_cost['total_cost'])}")
    print(f"AWS cost: {format_cost(aws_cost['total_cost'])}")
    print(f"Azure cost: {format_cost(azure_cost['total_cost'])}")
    print(f"Cost ratio (AWS/Jetson): {comparison['cost_ratios']['AWS']:.2f}x")
    print(f"Break-even time (AWS): {comparison['break_even_hours']['AWS']:.2f} hours")