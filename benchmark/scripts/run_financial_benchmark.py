#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Runner script for Financial Modeling benchmarks.
Provides a convenient way to run benchmarks with different configurations.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import benchmark suite
from benchmark.benchmark_suite import BenchmarkSuite

def load_config(config_file):
    """Load configuration from YAML file."""
    try:
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration from {config_file}: {e}")
        sys.exit(1)

def run_financial_benchmark(config, test_size=None, test_type=None, device_id=None):
    """
    Run financial modeling benchmark with specified configuration.
    
    Args:
        config: Configuration dictionary
        test_size: Size of test to run (small, medium, large)
        test_type: Type of test to run (risk_metrics, options_pricing, portfolio_optimization, all)
        device_id: GPU device ID to use
    """
    # Use device ID from arguments or config
    device_id = device_id if device_id is not None else config.get('device_id', 0)
    
    # Use output directory from config
    output_dir = config.get('output_dir', 'results/financial_modeling')
    
    # Enable cost modeling from config
    enable_cost_modeling = config.get('enable_cost_modeling', False)
    
    # Get financial parameters from config
    financial_config = config.get('financial', {})
    
    # Get test types to run
    if test_type:
        test_types = [test_type]
    else:
        test_types = financial_config.get('test_types', ['all'])
    
    # Get financial parameters based on test size
    if test_size and test_size in financial_config.get('performance_tests', {}):
        size_config = financial_config['performance_tests'][test_size]
        iterations = size_config.get('iterations', 5)
        data_size = size_config.get('data_size', 5000)
        monte_carlo_paths = size_config.get('monte_carlo_paths', 10000)
        portfolio_size = size_config.get('portfolio_size', 100)
    else:
        iterations = financial_config.get('iterations', 5)
        data_size = financial_config.get('risk_metrics', {}).get('data_size', 5000)
        monte_carlo_paths = financial_config.get('options_pricing', {}).get('monte_carlo_paths', 10000)
        portfolio_size = financial_config.get('portfolio_optimization', {}).get('portfolio_size', 100)
    
    # Get cost modeling parameters
    cost_config = config.get('cost_modeling', {})
    aws_instance = cost_config.get('aws_instance_type', 'g4dn.xlarge')
    azure_instance = cost_config.get('azure_instance_type', 'Standard_NC4as_T4_v3')
    gcp_instance = cost_config.get('gcp_instance_type', 'n1-standard-4-t4')
    include_dgx = cost_config.get('include_dgx_spark', True)
    dgx_system = cost_config.get('dgx_system_type', 'dgx_a100')
    include_slurm = cost_config.get('include_slurm_cluster', True)
    slurm_node_type = cost_config.get('slurm_node_type', 'basic_gpu')
    slurm_nodes = cost_config.get('slurm_nodes', 4)
    
    # Create benchmark suite
    suite = BenchmarkSuite(
        device_id=device_id,
        output_dir=output_dir,
        enable_cost_modeling=enable_cost_modeling,
        aws_instance_type=aws_instance,
        azure_instance_type=azure_instance,
        gcp_instance_type=gcp_instance,
        include_dgx_spark=include_dgx,
        dgx_system_type=dgx_system,
        include_slurm_cluster=include_slurm,
        slurm_node_type=slurm_node_type,
        slurm_nodes=slurm_nodes
    )
    
    # Run benchmarks for each test type
    for test in test_types:
        print(f"\nRunning Financial Modeling benchmark for {test}...")
        
        try:
            # Prepare parameters
            params = {
                "test_type": test,
                "num_iterations": iterations,
                "data_size": data_size,
                "monte_carlo_paths": monte_carlo_paths,
                "portfolio_size": portfolio_size
            }
            
            # Run benchmark
            result = suite.run_benchmark("financial_modeling", **params)
            
            # Print summary
            print(f"\nResults for {test}:")
            print(f"  Execution Time: {result.execution_time:.4f} seconds")
            print(f"  Throughput: {result.throughput:.2f} iterations/second")
            if result.gpu_utilization is not None:
                print(f"  GPU Utilization: {result.gpu_utilization:.2f}%")
            print(f"  Host Memory: {result.memory_usage['host']:.2f} MB")
            print(f"  Device Memory: {result.memory_usage['device']:.2f} MB")
        
        except Exception as e:
            print(f"Error running benchmark for {test}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate report
    suite.generate_reports()
    print(f"\nBenchmark report generated in {output_dir}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Financial Modeling Benchmark Runner")
    parser.add_argument("--config", type=str, default=None, 
                      help="Path to configuration file")
    parser.add_argument("--size", type=str, choices=["small", "medium", "large"], default=None,
                      help="Size of test to run (small, medium, large)")
    parser.add_argument("--test", type=str, 
                      choices=["all", "risk_metrics", "options_pricing", "portfolio_optimization"], 
                      default=None,
                      help="Type of test to run")
    parser.add_argument("--device", type=int, default=None, 
                      help="GPU device ID to use")
    
    args = parser.parse_args()
    
    # Determine config file path
    if args.config:
        config_file = args.config
    else:
        default_config = Path(project_root) / "benchmark" / "configs" / "financial_modeling.yaml"
        if default_config.exists():
            config_file = str(default_config)
        else:
            print("Error: No configuration file specified and default configuration not found.")
            sys.exit(1)
    
    # Load config
    config = load_config(config_file)
    
    # Run benchmark
    run_financial_benchmark(config, args.size, args.test, args.device)

if __name__ == "__main__":
    main()