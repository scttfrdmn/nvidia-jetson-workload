# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Tests for the cost modeling functionality.
"""

import os
import sys
import pytest
import yaml
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmark.cost_modeling import (
    ComputeEnvironment,
    CostModel,
    JetsonCostModel,
    CloudCostModel,
    AWSCostModel,
    AzureCostModel,
    GCPCostModel,
    DGXSparkCostModel,
    SlurmClusterCostModel,
    CostModelFactory,
    calculate_cost_comparison
)


class TestCostModelBase:
    """Base test case for cost models."""
    
    @pytest.fixture
    def standard_metrics(self):
        """Standard metrics for testing."""
        return {
            "execution_time": 300.0,  # 5 minutes
            "memory_usage": {"host": 4096, "device": 2048},  # MB
            "gpu_utilization": 80.0,  # %
            "energy_consumption": None,
            "additional_metrics": {
                "throughput": 1000.0,  # operations per second
                "num_steps": 1000
            }
        }

    def validate_cost_result(self, result):
        """Validate a cost estimation result."""
        assert "total_cost" in result
        assert result["total_cost"] >= 0
        assert "currency" in result
        assert "per_hour_cost" in result
        assert result["per_hour_cost"] >= 0


class TestJetsonCostModel(TestCostModelBase):
    """Tests for the Jetson cost model."""
    
    def test_initialization(self):
        """Test initialization with default values."""
        model = JetsonCostModel()
        assert model.name == "Jetson"
        assert model.hardware_cost == 599.0
        assert model.power_cost == 0.12
        assert model.amortization_period == 1095
        assert model.maintenance_factor == 0.1
    
    def test_initialization_custom(self):
        """Test initialization with custom values."""
        model = JetsonCostModel(
            hardware_cost=800.0,
            power_cost=0.15,
            amortization_period=730,  # 2 years
            maintenance_factor=0.05,
            currency="EUR"
        )
        assert model.name == "Jetson"
        assert model.hardware_cost == 800.0
        assert model.power_cost == 0.15
        assert model.amortization_period == 730
        assert model.maintenance_factor == 0.05
        assert model.currency == "EUR"
    
    def test_estimate_cost(self, standard_metrics):
        """Test cost estimation."""
        model = JetsonCostModel()
        result = model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        # Validate result structure
        self.validate_cost_result(result)
        
        # Check specific fields
        assert "hardware_cost" in result
        assert "maintenance_cost" in result
        assert "power_cost" in result
        
        # Verify calculations
        assert result["hardware_cost"] > 0
        assert result["maintenance_cost"] > 0
        assert result["power_cost"] > 0
        assert result["total_cost"] == result["hardware_cost"] + result["maintenance_cost"] + result["power_cost"]
    
    def test_estimate_cost_with_energy(self, standard_metrics):
        """Test cost estimation with energy consumption data."""
        model = JetsonCostModel()
        metrics = standard_metrics.copy()
        metrics["energy_consumption"] = 5000.0  # joules
        
        result = model.estimate_cost(
            metrics["execution_time"],
            metrics["memory_usage"],
            metrics["gpu_utilization"],
            metrics["energy_consumption"],
            metrics["additional_metrics"]
        )
        
        # Validate result
        self.validate_cost_result(result)
        assert result["power_cost"] > 0
        
        # Power cost should be lower from energy than from estimation
        metrics_no_energy = standard_metrics.copy()
        result_no_energy = model.estimate_cost(
            metrics_no_energy["execution_time"],
            metrics_no_energy["memory_usage"],
            metrics_no_energy["gpu_utilization"],
            None,
            metrics_no_energy["additional_metrics"]
        )
        
        # Compare power costs
        # Energy consumption is usually more accurate and often lower than estimation
        assert result["power_cost"] != result_no_energy["power_cost"]


class TestCloudCostModels(TestCostModelBase):
    """Tests for cloud provider cost models."""
    
    def test_aws_model(self, standard_metrics):
        """Test AWS cost model."""
        model = AWSCostModel(instance_type="g4dn.xlarge")
        result = model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        # Validate result
        self.validate_cost_result(result)
        assert "instance_cost" in result
        assert "storage_cost" in result
        assert "data_transfer_cost" in result
        assert "billable_time" in result
        
        # Check specific AWS details
        assert model.instance_type == "g4dn.xlarge"
        assert model.instance_specs["gpus"] == 1
        assert model.instance_specs["gpu_type"] == "NVIDIA T4"
    
    def test_azure_model(self, standard_metrics):
        """Test Azure cost model."""
        model = AzureCostModel(instance_type="Standard_NC4as_T4_v3")
        result = model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        # Validate result
        self.validate_cost_result(result)
        assert model.instance_type == "Standard_NC4as_T4_v3"
        assert model.instance_specs["gpus"] == 1
        assert model.instance_specs["gpu_type"] == "NVIDIA T4"
    
    def test_gcp_model(self, standard_metrics):
        """Test GCP cost model."""
        model = GCPCostModel(instance_type="n1-standard-4-t4")
        result = model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        # Validate result
        self.validate_cost_result(result)
        assert model.instance_type == "n1-standard-4-t4"
        assert model.instance_specs["gpus"] == 1
        assert model.instance_specs["gpu_type"] == "NVIDIA T4"


class TestDGXSparkCostModel(TestCostModelBase):
    """Tests for the DGX Spark cost model."""
    
    def test_initialization(self):
        """Test initialization with default values."""
        model = DGXSparkCostModel()
        assert model.name == "DGX Spark"
        assert model.system_type == "dgx_a100"
        assert model.quantity == 1
        assert model.power_cost == 0.12
    
    def test_initialization_custom(self):
        """Test initialization with custom values."""
        model = DGXSparkCostModel(
            system_type="dgx_h100",
            quantity=2,
            power_cost=0.15,
            amortization_period=1460,  # 4 years
            maintenance_factor=0.2,
            currency="EUR"
        )
        assert model.name == "DGX Spark"
        assert model.system_type == "dgx_h100"
        assert model.quantity == 2
        assert model.power_cost == 0.15
        assert model.amortization_period == 1460
        assert model.maintenance_factor == 0.2
        assert model.currency == "EUR"
    
    def test_estimate_cost(self, standard_metrics):
        """Test cost estimation."""
        model = DGXSparkCostModel()
        result = model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        # Validate result structure
        self.validate_cost_result(result)
        
        # Check specific fields
        assert "hardware_cost" in result
        assert "maintenance_cost" in result
        assert "admin_cost" in result
        assert "power_cost" in result
        assert "system_info" in result
        
        # Check system info
        assert result["system_info"]["type"] == "dgx_a100"
        assert result["system_info"]["name"] == "DGX A100"
    
    def test_superpod_config(self, standard_metrics):
        """Test DGX SuperPOD configuration."""
        model = DGXSparkCostModel(system_type="dgx_superpod", nodes_for_superpod=20)
        result = model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        # Validate result
        self.validate_cost_result(result)
        
        # Check SuperPOD specifics
        assert result["system_info"]["type"] == "dgx_superpod"
        assert result["system_info"]["nodes"] == 20
        assert result["system_info"]["total_gpus"] == 160  # 20 nodes * 8 GPUs
        assert "interconnect" in result["system_info"]
    
    def test_config_file(self, standard_metrics, temp_output_dir):
        """Test loading from config file."""
        # Create a temp config file
        config_file = Path(temp_output_dir) / "dgx_config.yaml"
        config = {
            "system_type": "dgx_h100",
            "quantity": 3,
            "power_cost": 0.10,
            "datacenter_overhead_factor": 0.25,
            "admin_cost_per_year": 150000.0
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        # Create model from config
        model = DGXSparkCostModel(config_file=str(config_file))
        
        # Verify config was loaded correctly
        assert model.system_type == "dgx_h100"
        assert model.quantity == 3
        assert model.power_cost == 0.10
        assert model.datacenter_overhead_factor == 0.25
        assert model.admin_cost_per_year == 150000.0
        
        # Test cost estimation
        result = model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        self.validate_cost_result(result)
        assert result["system_info"]["type"] == "dgx_h100"


class TestSlurmClusterCostModel(TestCostModelBase):
    """Tests for the Slurm cluster cost model."""
    
    def test_initialization(self):
        """Test initialization with default values."""
        model = SlurmClusterCostModel()
        assert model.name == "Slurm Cluster"
        assert model.nodes == 4
        assert model.node_type == "basic_gpu"
        assert model.power_cost == 0.12
    
    def test_initialization_custom(self):
        """Test initialization with custom values."""
        model = SlurmClusterCostModel(
            nodes=16,
            node_type="highend_gpu",
            power_cost=0.15,
            amortization_period=1460,  # 4 years
            maintenance_factor=0.12,
            network_cost=10000.0,
            admin_cost_per_year=80000.0,
            currency="EUR"
        )
        assert model.name == "Slurm Cluster"
        assert model.nodes == 16
        assert model.node_type == "highend_gpu"
        assert model.power_cost == 0.15
        assert model.amortization_period == 1460
        assert model.maintenance_factor == 0.12
        assert model.network_cost == 10000.0
        assert model.admin_cost_per_year == 80000.0
        assert model.currency == "EUR"
    
    def test_estimate_cost(self, standard_metrics):
        """Test cost estimation."""
        model = SlurmClusterCostModel()
        result = model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        # Validate result structure
        self.validate_cost_result(result)
        
        # Check specific fields
        assert "hardware_cost" in result
        assert "maintenance_cost" in result
        assert "admin_cost" in result
        assert "power_cost" in result
        assert "nodes_used" in result
        assert "node_info" in result
        
        # Check node info
        assert result["node_info"]["type"] == "basic_gpu"
        assert result["node_info"]["count"] == 4
        assert "power_per_node" in result["node_info"]
    
    def test_custom_node_config(self, standard_metrics):
        """Test custom node configuration."""
        custom_node_config = {
            "cost": 5000.0,
            "power_watts": 450.0,
            "has_gpu": True,
            "memory_gb": 128,
            "cores": 32,
            "gpu_type": "NVIDIA RTX A6000",
            "gpus_per_node": 2
        }
        
        model = SlurmClusterCostModel(
            nodes=8,
            node_type="custom",
            custom_node_config=custom_node_config
        )
        
        result = model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        # Validate result
        self.validate_cost_result(result)
        
        # Check custom node specifics
        assert result["node_info"]["type"] == "custom"
        assert result["node_info"]["gpu_type"] == "NVIDIA RTX A6000"
        assert result["node_info"]["gpus_per_node"] == 2
    
    def test_config_file(self, standard_metrics, temp_output_dir):
        """Test loading from config file."""
        # Create a temp config file
        config_file = Path(temp_output_dir) / "slurm_config.yaml"
        config = {
            "nodes": 32,
            "node_type": "jetson_cluster",
            "power_cost": 0.11,
            "utilization_factor": 0.8,
            "admin_cost_per_year": 50000.0
        }
        
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        # Create model from config
        model = SlurmClusterCostModel(config_file=str(config_file))
        
        # Verify config was loaded correctly
        assert model.nodes == 32
        assert model.node_type == "jetson_cluster"
        assert model.power_cost == 0.11
        assert model.utilization_factor == 0.8
        assert model.admin_cost_per_year == 50000.0
        
        # Test cost estimation
        result = model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        self.validate_cost_result(result)
        assert result["node_info"]["type"] == "jetson_cluster"


class TestCostModelFactory:
    """Tests for the cost model factory."""
    
    def test_create_jetson_model(self):
        """Test creating a Jetson cost model."""
        model = CostModelFactory.create_model(ComputeEnvironment.LOCAL_JETSON)
        assert isinstance(model, JetsonCostModel)
    
    def test_create_aws_model(self):
        """Test creating an AWS cost model."""
        model = CostModelFactory.create_model(
            ComputeEnvironment.AWS_GPU,
            instance_type="g4dn.xlarge"
        )
        assert isinstance(model, AWSCostModel)
        assert model.instance_type == "g4dn.xlarge"
    
    def test_create_azure_model(self):
        """Test creating an Azure cost model."""
        model = CostModelFactory.create_model(
            ComputeEnvironment.AZURE_GPU,
            instance_type="Standard_NC4as_T4_v3"
        )
        assert isinstance(model, AzureCostModel)
        assert model.instance_type == "Standard_NC4as_T4_v3"
    
    def test_create_gcp_model(self):
        """Test creating a GCP cost model."""
        model = CostModelFactory.create_model(
            ComputeEnvironment.GCP_GPU,
            instance_type="n1-standard-4-t4"
        )
        assert isinstance(model, GCPCostModel)
        assert model.instance_type == "n1-standard-4-t4"
    
    def test_create_dgx_model(self):
        """Test creating a DGX Spark cost model."""
        model = CostModelFactory.create_model(
            ComputeEnvironment.DGX_SPARK,
            system_type="dgx_h100",
            quantity=2
        )
        assert isinstance(model, DGXSparkCostModel)
        assert model.system_type == "dgx_h100"
        assert model.quantity == 2
    
    def test_create_slurm_model(self):
        """Test creating a Slurm cluster cost model."""
        model = CostModelFactory.create_model(
            ComputeEnvironment.SLURM_CLUSTER,
            nodes=16,
            node_type="highend_gpu"
        )
        assert isinstance(model, SlurmClusterCostModel)
        assert model.nodes == 16
        assert model.node_type == "highend_gpu"
    
    def test_invalid_environment(self):
        """Test error handling for invalid environment."""
        with pytest.raises(ValueError):
            CostModelFactory.create_model("invalid_environment")


class TestCostComparison(TestCostModelBase):
    """Tests for cost comparison calculations."""
    
    def test_calculate_cost_comparison(self, standard_metrics):
        """Test cost comparison calculation."""
        # Create cost results
        jetson_model = JetsonCostModel()
        jetson_cost = jetson_model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        aws_model = AWSCostModel()
        aws_cost = aws_model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        azure_model = AzureCostModel()
        azure_cost = azure_model.estimate_cost(
            standard_metrics["execution_time"],
            standard_metrics["memory_usage"],
            standard_metrics["gpu_utilization"],
            standard_metrics["energy_consumption"],
            standard_metrics["additional_metrics"]
        )
        
        cloud_costs = {
            "aws": aws_cost,
            "azure": azure_cost
        }
        
        # Calculate comparison
        comparison = calculate_cost_comparison(
            jetson_cost,
            cloud_costs,
            "test_workload",
            standard_metrics["execution_time"],
            standard_metrics["additional_metrics"]["throughput"]
        )
        
        # Validate comparison
        assert "workload_name" in comparison
        assert comparison["workload_name"] == "test_workload"
        assert "jetson_cost" in comparison
        assert "cloud_costs" in comparison
        assert "cost_ratios" in comparison
        assert "hourly_cost_ratios" in comparison
        assert "cost_per_operation" in comparison
        assert "break_even_hours" in comparison
        
        # Check cost ratios
        assert "aws" in comparison["cost_ratios"]
        assert "azure" in comparison["cost_ratios"]
        assert comparison["cost_ratios"]["aws"] > 0
        assert comparison["cost_ratios"]["azure"] > 0
        
        # Check break-even hours
        assert "aws" in comparison["break_even_hours"]
        assert "azure" in comparison["break_even_hours"]