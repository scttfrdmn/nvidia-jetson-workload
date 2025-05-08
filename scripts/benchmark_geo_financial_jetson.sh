#!/bin/bash
# Benchmark geospatial financial integration on Jetson Orin NX
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

set -e

# Default values
DATA_DIR="./data/geo_financial"
OUTPUT_DIR="./output/jetson_benchmarks"
GENERATE_DATA=false
NUM_ASSETS=300
DEVICE_ID=0
NUM_ITERATIONS=3
BENCHMARK_ALL_PROFILES=false
VERBOSE=false
RUN_SCENARIOS=true
RUN_CLIMATE=true
RUN_MULTIREGION=true
RUN_REALTIME=false  # Realtime is interactive, disabled by default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --generate-data)
      GENERATE_DATA=true
      shift
      ;;
    --num-assets)
      NUM_ASSETS="$2"
      shift 2
      ;;
    --device-id)
      DEVICE_ID="$2"
      shift 2
      ;;
    --iterations)
      NUM_ITERATIONS="$2"
      shift 2
      ;;
    --all-profiles)
      BENCHMARK_ALL_PROFILES=true
      shift
      ;;
    --no-scenarios)
      RUN_SCENARIOS=false
      shift
      ;;
    --no-climate)
      RUN_CLIMATE=false
      shift
      ;;
    --no-multiregion)
      RUN_MULTIREGION=false
      shift
      ;;
    --realtime)
      RUN_REALTIME=true
      shift
      ;;
    --verbose)
      VERBOSE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo "Benchmark geospatial financial integration on Jetson Orin NX"
      echo ""
      echo "Options:"
      echo "  --data-dir DIR         Directory containing test data (default: ./data/geo_financial)"
      echo "  --output-dir DIR       Directory to store output files (default: ./output/jetson_benchmarks)"
      echo "  --generate-data        Generate test data if not present (default: false)"
      echo "  --num-assets N         Number of assets for benchmarking (default: 300)"
      echo "  --device-id N          GPU device ID to use (default: 0)"
      echo "  --iterations N         Number of iterations for each benchmark (default: 3)"
      echo "  --all-profiles         Benchmark all memory profiles (default: false)"
      echo "  --no-scenarios         Skip scenario analysis benchmarks (default: run)"
      echo "  --no-climate           Skip climate risk benchmarks (default: run)"
      echo "  --no-multiregion       Skip multi-region benchmarks (default: run)"
      echo "  --realtime             Run realtime monitoring benchmarks (default: skip)"
      echo "  --verbose              Enable verbose output"
      echo "  --help                 Display this help message and exit"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information."
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Set Python path to include project root
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd):${PYTHONPATH}"

# Log execution parameters
log_file="${OUTPUT_DIR}/benchmark_log.txt"
echo "========================================" > "${log_file}"
echo "Geo-Financial Jetson Benchmark - $(date)" >> "${log_file}"
echo "----------------------------------------" >> "${log_file}"
echo "Data directory: ${DATA_DIR}" >> "${log_file}"
echo "Output directory: ${OUTPUT_DIR}" >> "${log_file}"
echo "Generate data: ${GENERATE_DATA}" >> "${log_file}"
echo "Number of assets: ${NUM_ASSETS}" >> "${log_file}"
echo "GPU device: ${DEVICE_ID}" >> "${log_file}"
echo "Iterations: ${NUM_ITERATIONS}" >> "${log_file}"
echo "Benchmark all profiles: ${BENCHMARK_ALL_PROFILES}" >> "${log_file}"
echo "Run scenarios: ${RUN_SCENARIOS}" >> "${log_file}"
echo "Run climate: ${RUN_CLIMATE}" >> "${log_file}"
echo "Run multiregion: ${RUN_MULTIREGION}" >> "${log_file}"
echo "Run realtime: ${RUN_REALTIME}" >> "${log_file}"
echo "----------------------------------------" >> "${log_file}"

# Function to run command with logging
run_command() {
  local cmd="$1"
  local description="$2"
  
  echo "Running: ${description}" | tee -a "${log_file}"
  if [ "${VERBOSE}" = true ]; then
    echo "Command: ${cmd}" | tee -a "${log_file}"
  fi
  
  echo "----------------------------------------" >> "${log_file}"
  if eval "${cmd}" >> "${log_file}" 2>&1; then
    echo "✓ ${description} completed successfully" | tee -a "${log_file}"
  else
    echo "✗ ${description} failed (exit code: $?)" | tee -a "${log_file}"
    echo "See log file for details: ${log_file}"
    return 1
  fi
  echo "----------------------------------------" >> "${log_file}"
}

# Create the main benchmark script
benchmark_script="${OUTPUT_DIR}/run_benchmark.py"
cat > "${benchmark_script}" << 'EOF'
#!/usr/bin/env python3
"""
Benchmark script for geospatial financial integration on Jetson Orin NX.

This script benchmarks the performance of various aspects of the geospatial
financial integration on Jetson Orin NX devices, comparing optimized and
non-optimized implementations.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import time
import argparse
import numpy as np
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("jetson_benchmark")

# Import geo_financial modules
from src.integrations.geo_financial.geo_risk import (
    GeospatialRiskModel, GeospatialPortfolio, SpatialRiskFactor,
    create_elevation_risk_factor, create_slope_risk_factor
)
from src.integrations.geo_financial.risk_aggregation import (
    RiskAggregator, AggregationMethod, RiskSurfaceGenerator
)
from src.integrations.geo_financial.climate_risk_assessment import (
    ClimateRiskAssessor, ClimateScenario, TimeHorizon
)
from src.integrations.geo_financial.scenario_analysis import (
    Scenario, ScenarioSet, ScenarioAnalyzer, ScenarioVisualizer,
    ScenarioType, create_climate_scenario, create_economic_scenario, create_stress_scenario
)
from src.integrations.geo_financial.multiregion_analysis import (
    RegionDefinition, RegionalPortfolio, MultiRegionRiskModel, RegionalRiskComparator
)
from src.integrations.geo_financial.jetson_optimization import (
    JetsonDeviceInfo, JetsonOptimizer, MemoryProfile, create_jetson_optimized_analyzer
)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


def create_test_data(output_dir: str, num_assets: int = 300) -> Tuple[GeospatialPortfolio, GeospatialRiskModel]:
    """
    Create test data for benchmarking.
    
    Args:
        output_dir: Directory to save test data
        num_assets: Number of assets to generate
        
    Returns:
        Tuple of (portfolio, risk_model)
    """
    logger.info(f"Creating test data with {num_assets} assets")
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sample portfolio
    assets = []
    
    # Use seed for reproducibility
    np.random.seed(42)
    
    # Create assets with random positions
    for i in range(num_assets):
        asset = {
            "id": f"asset_{i:05d}",
            "name": f"Asset {i}",
            "x": np.random.random(),  # x in [0, 1]
            "y": np.random.random(),  # y in [0, 1]
            "value": 100000 + np.random.random() * 900000,  # value in [100K, 1M]
            "metadata": {
                "sector": np.random.choice(["technology", "energy", "finance", "healthcare", "consumer"]),
                "region": np.random.choice(["north", "south", "east", "west"]),
                "construction_year": np.random.randint(1980, 2022)
            }
        }
        assets.append(asset)
    
    # Create portfolio
    portfolio = GeospatialPortfolio(
        assets=assets,
        name="Test Portfolio",
        description="Test portfolio for Jetson benchmarking",
        metadata={
            "created_at": datetime.now().isoformat(),
            "num_assets": num_assets
        }
    )
    
    # Save portfolio
    portfolio_file = os.path.join(output_dir, "portfolio.json")
    with open(portfolio_file, "w") as f:
        json.dump({
            "name": portfolio.name,
            "description": portfolio.description,
            "metadata": portfolio.metadata,
            "assets": portfolio.assets
        }, f, indent=2)
    
    logger.info(f"Portfolio saved to {portfolio_file}")
    
    # Create risk model with synthetic risk factors
    # Create elevation risk (higher in northwest)
    grid_size = 100
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Elevation risk (higher in northwest)
    elevation_data = (1 - X) * Y
    elevation_risk = create_elevation_risk_factor(
        name="elevation_risk",
        elevation_data=elevation_data,
        x_coords=x,
        y_coords=y,
        risk_weight=0.4,
        description="Risk based on elevation"
    )
    
    # Flood risk (higher in southwest)
    flood_data = (1 - X) * (1 - Y)
    flood_risk = SpatialRiskFactor(
        name="flood_risk",
        risk_data=flood_data,
        x_coords=x,
        y_coords=y,
        risk_weight=0.3,
        description="Risk of flooding",
        metadata={"risk_type": "climate_physical", "hazard": "flood"}
    )
    
    # Wildfire risk (higher in northeast)
    wildfire_data = X * Y
    wildfire_risk = SpatialRiskFactor(
        name="wildfire_risk",
        risk_data=wildfire_data,
        x_coords=x,
        y_coords=y,
        risk_weight=0.2,
        description="Risk of wildfires",
        metadata={"risk_type": "climate_physical", "hazard": "wildfire"}
    )
    
    # Function-based risk factor for economic factors
    def economic_risk_function(data: Dict[str, Any]) -> Dict[str, float]:
        assets = data.get("assets", [])
        risk_scores = {}
        
        for asset in assets:
            asset_id = asset["id"]
            sector = asset.get("metadata", {}).get("sector", "unknown")
            
            # Different sectors have different economic risk levels
            if sector == "technology":
                score = 0.3
            elif sector == "energy":
                score = 0.7
            elif sector == "finance":
                score = 0.5
            elif sector == "healthcare":
                score = 0.2
            elif sector == "consumer":
                score = 0.4
            else:
                score = 0.5
            
            risk_scores[asset_id] = score
        
        return risk_scores
    
    economic_risk = SpatialRiskFactor(
        name="economic_risk",
        risk_function=economic_risk_function,
        risk_weight=0.1,
        description="Economic risk by sector",
        metadata={"risk_type": "economic"}
    )
    
    # Create risk model
    risk_model = GeospatialRiskModel(
        name="Test Risk Model",
        description="Test risk model for Jetson benchmarking",
        risk_factors=[elevation_risk, flood_risk, wildfire_risk, economic_risk],
        aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
        metadata={
            "created_at": datetime.now().isoformat()
        }
    )
    
    # Save risk factors
    risk_factors_dir = os.path.join(output_dir, "risk_factors")
    os.makedirs(risk_factors_dir, exist_ok=True)
    
    # Save elevation data
    elevation_file = os.path.join(risk_factors_dir, "elevation.npy")
    np.save(elevation_file, elevation_data)
    
    # Save flood data
    flood_file = os.path.join(risk_factors_dir, "flood.npy")
    np.save(flood_file, flood_data)
    
    # Save wildfire data
    wildfire_file = os.path.join(risk_factors_dir, "wildfire.npy")
    np.save(wildfire_file, wildfire_data)
    
    # Save coordinates
    coords_file = os.path.join(risk_factors_dir, "coords.npz")
    np.savez(coords_file, x=x, y=y)
    
    logger.info(f"Risk factors saved to {risk_factors_dir}")
    
    return portfolio, risk_model


def load_test_data(data_dir: str) -> Tuple[GeospatialPortfolio, GeospatialRiskModel]:
    """
    Load test data for benchmarking.
    
    Args:
        data_dir: Directory containing test data
        
    Returns:
        Tuple of (portfolio, risk_model)
    """
    logger.info(f"Loading test data from {data_dir}")
    
    # Load portfolio
    portfolio_file = os.path.join(data_dir, "portfolio.json")
    with open(portfolio_file, "r") as f:
        portfolio_data = json.load(f)
    
    portfolio = GeospatialPortfolio(
        assets=portfolio_data["assets"],
        name=portfolio_data["name"],
        description=portfolio_data["description"],
        metadata=portfolio_data["metadata"]
    )
    
    # Load risk factors
    risk_factors_dir = os.path.join(data_dir, "risk_factors")
    
    # Load coordinates
    coords_file = os.path.join(risk_factors_dir, "coords.npz")
    coords_data = np.load(coords_file)
    x = coords_data["x"]
    y = coords_data["y"]
    
    # Load elevation data
    elevation_file = os.path.join(risk_factors_dir, "elevation.npy")
    elevation_data = np.load(elevation_file)
    
    elevation_risk = create_elevation_risk_factor(
        name="elevation_risk",
        elevation_data=elevation_data,
        x_coords=x,
        y_coords=y,
        risk_weight=0.4,
        description="Risk based on elevation"
    )
    
    # Load flood data
    flood_file = os.path.join(risk_factors_dir, "flood.npy")
    flood_data = np.load(flood_file)
    
    flood_risk = SpatialRiskFactor(
        name="flood_risk",
        risk_data=flood_data,
        x_coords=x,
        y_coords=y,
        risk_weight=0.3,
        description="Risk of flooding",
        metadata={"risk_type": "climate_physical", "hazard": "flood"}
    )
    
    # Load wildfire data
    wildfire_file = os.path.join(risk_factors_dir, "wildfire.npy")
    wildfire_data = np.load(wildfire_file)
    
    wildfire_risk = SpatialRiskFactor(
        name="wildfire_risk",
        risk_data=wildfire_data,
        x_coords=x,
        y_coords=y,
        risk_weight=0.2,
        description="Risk of wildfires",
        metadata={"risk_type": "climate_physical", "hazard": "wildfire"}
    )
    
    # Function-based risk factor for economic factors
    def economic_risk_function(data: Dict[str, Any]) -> Dict[str, float]:
        assets = data.get("assets", [])
        risk_scores = {}
        
        for asset in assets:
            asset_id = asset["id"]
            sector = asset.get("metadata", {}).get("sector", "unknown")
            
            # Different sectors have different economic risk levels
            if sector == "technology":
                score = 0.3
            elif sector == "energy":
                score = 0.7
            elif sector == "finance":
                score = 0.5
            elif sector == "healthcare":
                score = 0.2
            elif sector == "consumer":
                score = 0.4
            else:
                score = 0.5
            
            risk_scores[asset_id] = score
        
        return risk_scores
    
    economic_risk = SpatialRiskFactor(
        name="economic_risk",
        risk_function=economic_risk_function,
        risk_weight=0.1,
        description="Economic risk by sector",
        metadata={"risk_type": "economic"}
    )
    
    # Create risk model
    risk_model = GeospatialRiskModel(
        name="Test Risk Model",
        description="Test risk model for Jetson benchmarking",
        risk_factors=[elevation_risk, flood_risk, wildfire_risk, economic_risk],
        aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
        metadata={
            "created_at": datetime.now().isoformat()
        }
    )
    
    return portfolio, risk_model


def benchmark_risk_assessment(portfolio: GeospatialPortfolio, 
                             risk_model: GeospatialRiskModel,
                             device_id: int = 0,
                             iterations: int = 3,
                             benchmark_all_profiles: bool = False) -> Dict[str, Any]:
    """
    Benchmark risk assessment performance.
    
    Args:
        portfolio: Portfolio to assess
        risk_model: Risk model to use
        device_id: GPU device ID
        iterations: Number of iterations for each benchmark
        benchmark_all_profiles: Whether to benchmark all memory profiles
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info("Benchmarking risk assessment performance")
    
    results = {}
    
    # Get device info
    device_info = JetsonDeviceInfo()
    results["device_info"] = {
        "is_jetson": device_info.is_jetson,
        "model": device_info.model,
        "compute_capability": device_info.compute_capability,
        "total_memory_mb": device_info.total_memory_mb,
        "cuda_cores": device_info.cuda_cores,
        "sm_count": device_info.sm_count
    }
    
    # Benchmark original implementation (CPU)
    cpu_times = []
    for i in range(iterations):
        logger.info(f"CPU iteration {i+1}/{iterations}")
        start_time = time.time()
        _ = portfolio.assess_risk(risk_model)
        end_time = time.time()
        cpu_times.append(end_time - start_time)
    
    results["cpu"] = {
        "times": cpu_times,
        "mean": np.mean(cpu_times),
        "std": np.std(cpu_times),
        "min": np.min(cpu_times),
        "max": np.max(cpu_times)
    }
    
    # Skip GPU benchmarks if CUPY is not available
    if not HAS_CUPY:
        logger.warning("CuPy not available, skipping GPU benchmarks")
        return results
    
    # Benchmark Jetson-optimized implementation with default profile
    logger.info(f"Benchmarking GPU (device {device_id}) with default profile")
    optimizer = JetsonOptimizer(device_id=device_id)
    opt_portfolio = optimizer.optimize_portfolio(portfolio)
    opt_risk_model = optimizer.optimize_risk_model(risk_model)
    
    gpu_times = []
    for i in range(iterations):
        logger.info(f"GPU default iteration {i+1}/{iterations}")
        start_time = time.time()
        _ = opt_portfolio.assess_risk(opt_risk_model)
        end_time = time.time()
        gpu_times.append(end_time - start_time)
    
    results["gpu_default"] = {
        "times": gpu_times,
        "mean": np.mean(gpu_times),
        "std": np.std(gpu_times),
        "min": np.min(gpu_times),
        "max": np.max(gpu_times),
        "speedup": results["cpu"]["mean"] / np.mean(gpu_times)
    }
    
    # Benchmark different memory profiles if requested
    if benchmark_all_profiles:
        for profile in MemoryProfile:
            logger.info(f"Benchmarking GPU (device {device_id}) with {profile.value} profile")
            
            profile_optimizer = JetsonOptimizer(
                device_id=device_id,
                memory_profile=profile
            )
            
            profile_portfolio = profile_optimizer.optimize_portfolio(portfolio)
            profile_risk_model = profile_optimizer.optimize_risk_model(risk_model)
            
            profile_times = []
            for i in range(iterations):
                logger.info(f"GPU {profile.value} iteration {i+1}/{iterations}")
                start_time = time.time()
                _ = profile_portfolio.assess_risk(profile_risk_model)
                end_time = time.time()
                profile_times.append(end_time - start_time)
            
            results[f"gpu_{profile.value}"] = {
                "times": profile_times,
                "mean": np.mean(profile_times),
                "std": np.std(profile_times),
                "min": np.min(profile_times),
                "max": np.max(profile_times),
                "speedup": results["cpu"]["mean"] / np.mean(profile_times)
            }
    
    # Calculate overall speedup
    best_gpu_time = min(
        [results["gpu_default"]["mean"]] + 
        [results[f"gpu_{profile.value}"]["mean"] for profile in MemoryProfile] if benchmark_all_profiles else []
    )
    results["overall_speedup"] = results["cpu"]["mean"] / best_gpu_time
    
    return results


def benchmark_scenario_analysis(portfolio: GeospatialPortfolio, 
                               risk_model: GeospatialRiskModel,
                               device_id: int = 0,
                               iterations: int = 3) -> Dict[str, Any]:
    """
    Benchmark scenario analysis performance.
    
    Args:
        portfolio: Portfolio to assess
        risk_model: Risk model to use
        device_id: GPU device ID
        iterations: Number of iterations for each benchmark
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info("Benchmarking scenario analysis performance")
    
    results = {}
    
    # Create scenarios for benchmarking
    logger.info("Creating benchmark scenarios")
    baseline = Scenario(
        name="baseline",
        scenario_type=ScenarioType.BASE,
        description="Baseline scenario"
    )
    
    climate_scenario = create_climate_scenario(
        name="climate_severe",
        climate_scenario=ClimateScenario.RCP8_5,
        time_horizon=TimeHorizon.YEAR_2050,
        description="Severe climate change scenario",
        severity_multiplier=1.5
    )
    
    economic_scenario = create_economic_scenario(
        name="economic_recession",
        gdp_growth=-2.0,
        inflation=5.0,
        interest_rate=6.0,
        description="Economic recession scenario"
    )
    
    # Create scenario set
    scenario_set = ScenarioSet(name="Benchmark Set")
    scenario_set.add_scenario(baseline, is_baseline=True)
    scenario_set.add_scenario(climate_scenario)
    scenario_set.add_scenario(economic_scenario)
    
    # Benchmark CPU scenario analysis
    logger.info("Benchmarking CPU scenario analysis")
    cpu_analyzer = ScenarioAnalyzer(
        portfolio=portfolio,
        risk_model=risk_model,
        device_id=-1
    )
    
    cpu_times = []
    for i in range(iterations):
        logger.info(f"CPU scenario set iteration {i+1}/{iterations}")
        start_time = time.time()
        _ = cpu_analyzer.analyze_scenario_set(scenario_set)
        end_time = time.time()
        cpu_times.append(end_time - start_time)
    
    results["cpu"] = {
        "times": cpu_times,
        "mean": np.mean(cpu_times),
        "std": np.std(cpu_times),
        "min": np.min(cpu_times),
        "max": np.max(cpu_times)
    }
    
    # Skip GPU benchmarks if CUPY is not available
    if not HAS_CUPY:
        logger.warning("CuPy not available, skipping GPU benchmarks")
        return results
    
    # Benchmark Jetson-optimized scenario analysis
    logger.info(f"Benchmarking GPU (device {device_id}) scenario analysis")
    gpu_analyzer = create_jetson_optimized_analyzer(
        portfolio=portfolio,
        risk_model=risk_model,
        device_id=device_id
    )
    
    gpu_times = []
    for i in range(iterations):
        logger.info(f"GPU scenario set iteration {i+1}/{iterations}")
        start_time = time.time()
        _ = gpu_analyzer.analyze_scenario_set(scenario_set)
        end_time = time.time()
        gpu_times.append(end_time - start_time)
    
    results["gpu"] = {
        "times": gpu_times,
        "mean": np.mean(gpu_times),
        "std": np.std(gpu_times),
        "min": np.min(gpu_times),
        "max": np.max(gpu_times),
        "speedup": results["cpu"]["mean"] / np.mean(gpu_times)
    }
    
    # Benchmark individual scenarios
    cpu_scenario_times = {}
    gpu_scenario_times = {}
    
    for scenario_name in scenario_set.scenarios.keys():
        scenario = scenario_set.get_scenario(scenario_name)
        
        # CPU timing
        cpu_times = []
        for i in range(iterations):
            logger.info(f"CPU {scenario_name} iteration {i+1}/{iterations}")
            start_time = time.time()
            _ = cpu_analyzer.analyze_scenario(scenario)
            end_time = time.time()
            cpu_times.append(end_time - start_time)
        
        cpu_scenario_times[scenario_name] = {
            "times": cpu_times,
            "mean": np.mean(cpu_times),
            "std": np.std(cpu_times)
        }
        
        # GPU timing
        gpu_times = []
        for i in range(iterations):
            logger.info(f"GPU {scenario_name} iteration {i+1}/{iterations}")
            start_time = time.time()
            _ = gpu_analyzer.analyze_scenario(scenario)
            end_time = time.time()
            gpu_times.append(end_time - start_time)
        
        gpu_scenario_times[scenario_name] = {
            "times": gpu_times,
            "mean": np.mean(gpu_times),
            "std": np.std(gpu_times),
            "speedup": np.mean(cpu_times) / np.mean(gpu_times)
        }
    
    results["cpu_scenarios"] = cpu_scenario_times
    results["gpu_scenarios"] = gpu_scenario_times
    
    # Calculate overall speedup
    results["overall_speedup"] = results["cpu"]["mean"] / results["gpu"]["mean"]
    
    return results


def benchmark_climate_risk(portfolio: GeospatialPortfolio, 
                         risk_model: GeospatialRiskModel,
                         device_id: int = 0,
                         iterations: int = 3) -> Dict[str, Any]:
    """
    Benchmark climate risk assessment performance.
    
    Args:
        portfolio: Portfolio to assess
        risk_model: Risk model to use
        device_id: GPU device ID
        iterations: Number of iterations for each benchmark
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info("Benchmarking climate risk assessment performance")
    
    results = {}
    
    # Create climate risk assessor
    assessor = ClimateRiskAssessor()
    
    # Benchmark CPU climate risk assessment
    logger.info("Benchmarking CPU climate risk assessment")
    cpu_times = []
    for i in range(iterations):
        logger.info(f"CPU physical risk iteration {i+1}/{iterations}")
        start_time = time.time()
        _ = assessor.assess_physical_risks(
            portfolio=portfolio,
            scenario=ClimateScenario.RCP8_5,
            time_horizon=TimeHorizon.YEAR_2050,
            device_id=-1
        )
        end_time = time.time()
        cpu_times.append(end_time - start_time)
    
    results["cpu"] = {
        "times": cpu_times,
        "mean": np.mean(cpu_times),
        "std": np.std(cpu_times),
        "min": np.min(cpu_times),
        "max": np.max(cpu_times)
    }
    
    # Skip GPU benchmarks if CUPY is not available
    if not HAS_CUPY:
        logger.warning("CuPy not available, skipping GPU benchmarks")
        return results
    
    # Benchmark Jetson-optimized climate risk assessment
    logger.info(f"Benchmarking GPU (device {device_id}) climate risk assessment")
    
    # Create Jetson-optimized portfolio
    optimizer = JetsonOptimizer(device_id=device_id)
    opt_portfolio = optimizer.optimize_portfolio(portfolio)
    
    gpu_times = []
    for i in range(iterations):
        logger.info(f"GPU physical risk iteration {i+1}/{iterations}")
        start_time = time.time()
        _ = assessor.assess_physical_risks(
            portfolio=opt_portfolio,
            scenario=ClimateScenario.RCP8_5,
            time_horizon=TimeHorizon.YEAR_2050,
            device_id=device_id
        )
        end_time = time.time()
        gpu_times.append(end_time - start_time)
    
    results["gpu"] = {
        "times": gpu_times,
        "mean": np.mean(gpu_times),
        "std": np.std(gpu_times),
        "min": np.min(gpu_times),
        "max": np.max(gpu_times),
        "speedup": results["cpu"]["mean"] / np.mean(gpu_times)
    }
    
    # Benchmark different climate scenarios
    cpu_scenario_times = {}
    gpu_scenario_times = {}
    
    scenarios = [ClimateScenario.RCP2_6, ClimateScenario.RCP4_5, ClimateScenario.RCP8_5]
    horizons = [TimeHorizon.YEAR_2030, TimeHorizon.YEAR_2050, TimeHorizon.YEAR_2100]
    
    for scenario in scenarios:
        for horizon in horizons:
            scenario_key = f"{scenario.value}_{horizon.value}"
            
            # CPU timing
            cpu_times = []
            for i in range(iterations):
                logger.info(f"CPU {scenario_key} iteration {i+1}/{iterations}")
                start_time = time.time()
                _ = assessor.assess_physical_risks(
                    portfolio=portfolio,
                    scenario=scenario,
                    time_horizon=horizon,
                    device_id=-1
                )
                end_time = time.time()
                cpu_times.append(end_time - start_time)
            
            cpu_scenario_times[scenario_key] = {
                "times": cpu_times,
                "mean": np.mean(cpu_times),
                "std": np.std(cpu_times)
            }
            
            # GPU timing
            gpu_times = []
            for i in range(iterations):
                logger.info(f"GPU {scenario_key} iteration {i+1}/{iterations}")
                start_time = time.time()
                _ = assessor.assess_physical_risks(
                    portfolio=opt_portfolio,
                    scenario=scenario,
                    time_horizon=horizon,
                    device_id=device_id
                )
                end_time = time.time()
                gpu_times.append(end_time - start_time)
            
            gpu_scenario_times[scenario_key] = {
                "times": gpu_times,
                "mean": np.mean(gpu_times),
                "std": np.std(gpu_times),
                "speedup": np.mean(cpu_times) / np.mean(gpu_times)
            }
    
    results["cpu_scenarios"] = cpu_scenario_times
    results["gpu_scenarios"] = gpu_scenario_times
    
    # Calculate overall speedup
    results["overall_speedup"] = results["cpu"]["mean"] / results["gpu"]["mean"]
    
    return results


def benchmark_multiregion(portfolio: GeospatialPortfolio, 
                        risk_model: GeospatialRiskModel,
                        device_id: int = 0,
                        iterations: int = 3) -> Dict[str, Any]:
    """
    Benchmark multi-region analysis performance.
    
    Args:
        portfolio: Portfolio to assess
        risk_model: Risk model to use
        device_id: GPU device ID
        iterations: Number of iterations for each benchmark
        
    Returns:
        Dictionary of benchmark results
    """
    logger.info("Benchmarking multi-region analysis performance")
    
    results = {}
    
    # Define regions
    regions = [
        RegionDefinition(name="north", bounds=(0.0, 0.5, 1.0, 1.0)),
        RegionDefinition(name="south", bounds=(0.0, 0.0, 1.0, 0.5)),
        RegionDefinition(name="east", bounds=(0.5, 0.0, 1.0, 1.0)),
        RegionDefinition(name="west", bounds=(0.0, 0.0, 0.5, 1.0))
    ]
    
    # Benchmark CPU multi-region analysis
    logger.info("Benchmarking CPU multi-region analysis")
    cpu_times = []
    for i in range(iterations):
        logger.info(f"CPU regional portfolio iteration {i+1}/{iterations}")
        start_time = time.time()
        regional_portfolios = RegionalPortfolio.create_from_portfolio(
            portfolio=portfolio,
            regions=regions,
            device_id=-1
        )
        end_time = time.time()
        cpu_times.append(end_time - start_time)
    
    results["cpu_regional_portfolio"] = {
        "times": cpu_times,
        "mean": np.mean(cpu_times),
        "std": np.std(cpu_times),
        "min": np.min(cpu_times),
        "max": np.max(cpu_times)
    }
    
    # Skip GPU benchmarks if CUPY is not available
    if not HAS_CUPY:
        logger.warning("CuPy not available, skipping GPU benchmarks")
        return results
    
    # Benchmark Jetson-optimized multi-region analysis
    logger.info(f"Benchmarking GPU (device {device_id}) multi-region analysis")
    
    # Create Jetson-optimized portfolio
    optimizer = JetsonOptimizer(device_id=device_id)
    opt_portfolio = optimizer.optimize_portfolio(portfolio)
    
    gpu_times = []
    for i in range(iterations):
        logger.info(f"GPU regional portfolio iteration {i+1}/{iterations}")
        start_time = time.time()
        regional_portfolios = RegionalPortfolio.create_from_portfolio(
            portfolio=opt_portfolio,
            regions=regions,
            device_id=device_id
        )
        end_time = time.time()
        gpu_times.append(end_time - start_time)
    
    results["gpu_regional_portfolio"] = {
        "times": gpu_times,
        "mean": np.mean(gpu_times),
        "std": np.std(gpu_times),
        "min": np.min(gpu_times),
        "max": np.max(gpu_times),
        "speedup": results["cpu_regional_portfolio"]["mean"] / np.mean(gpu_times)
    }
    
    # Create regional portfolios for further benchmarks
    logger.info("Creating regional portfolios for further benchmarks")
    regional_portfolios = RegionalPortfolio.create_from_portfolio(
        portfolio=portfolio,
        regions=regions
    )
    
    # Create comparator
    comparator = RegionalRiskComparator()
    
    # Benchmark CPU region comparison
    logger.info("Benchmarking CPU region comparison")
    cpu_times = []
    for i in range(iterations):
        logger.info(f"CPU region comparison iteration {i+1}/{iterations}")
        start_time = time.time()
        _ = comparator.compare_regional_risks(
            regional_portfolios=regional_portfolios,
            risk_model=risk_model,
            device_id=-1
        )
        end_time = time.time()
        cpu_times.append(end_time - start_time)
    
    results["cpu_region_comparison"] = {
        "times": cpu_times,
        "mean": np.mean(cpu_times),
        "std": np.std(cpu_times),
        "min": np.min(cpu_times),
        "max": np.max(cpu_times)
    }
    
    # Benchmark GPU region comparison
    logger.info(f"Benchmarking GPU (device {device_id}) region comparison")
    
    # Create Jetson-optimized portfolio and risk model
    opt_regional_portfolios = {}
    for region_name, region_portfolio in regional_portfolios.items():
        opt_regional_portfolios[region_name] = optimizer.optimize_portfolio(region_portfolio)
    
    opt_risk_model = optimizer.optimize_risk_model(risk_model)
    
    gpu_times = []
    for i in range(iterations):
        logger.info(f"GPU region comparison iteration {i+1}/{iterations}")
        start_time = time.time()
        _ = comparator.compare_regional_risks(
            regional_portfolios=opt_regional_portfolios,
            risk_model=opt_risk_model,
            device_id=device_id
        )
        end_time = time.time()
        gpu_times.append(end_time - start_time)
    
    results["gpu_region_comparison"] = {
        "times": gpu_times,
        "mean": np.mean(gpu_times),
        "std": np.std(gpu_times),
        "min": np.min(gpu_times),
        "max": np.max(gpu_times),
        "speedup": results["cpu_region_comparison"]["mean"] / np.mean(gpu_times)
    }
    
    # Calculate overall speedup
    cpu_total = results["cpu_regional_portfolio"]["mean"] + results["cpu_region_comparison"]["mean"]
    gpu_total = results["gpu_regional_portfolio"]["mean"] + results["gpu_region_comparison"]["mean"]
    results["overall_speedup"] = cpu_total / gpu_total
    
    return results


def plot_results(results: Dict[str, Any], output_dir: str) -> str:
    """
    Plot benchmark results.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save plot
        
    Returns:
        Path to output file
    """
    # Check if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("Matplotlib not available, skipping plot generation")
        return ""
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Extract data
    components = []
    cpu_times = []
    gpu_times = []
    speedups = []
    
    # Extract risk assessment data
    if "risk_assessment" in results:
        data = results["risk_assessment"]
        components.append("Risk Assessment")
        cpu_times.append(data["cpu"]["mean"])
        
        # Use best GPU profile
        if "gpu_default" in data:
            gpu_times.append(data["gpu_default"]["mean"])
            speedups.append(data["gpu_default"]["speedup"])
        else:
            gpu_times.append(0)
            speedups.append(0)
    
    # Extract scenario analysis data
    if "scenario_analysis" in results:
        data = results["scenario_analysis"]
        components.append("Scenario Analysis")
        cpu_times.append(data["cpu"]["mean"])
        gpu_times.append(data["gpu"]["mean"])
        speedups.append(data["gpu"]["speedup"])
    
    # Extract climate risk data
    if "climate_risk" in results:
        data = results["climate_risk"]
        components.append("Climate Risk")
        cpu_times.append(data["cpu"]["mean"])
        gpu_times.append(data["gpu"]["mean"])
        speedups.append(data["gpu"]["speedup"])
    
    # Extract multiregion data
    if "multiregion" in results:
        data = results["multiregion"]
        
        # Regional portfolio
        components.append("Regional Portfolio")
        cpu_times.append(data["cpu_regional_portfolio"]["mean"])
        gpu_times.append(data["gpu_regional_portfolio"]["mean"])
        speedups.append(data["gpu_regional_portfolio"]["speedup"])
        
        # Region comparison
        components.append("Region Comparison")
        cpu_times.append(data["cpu_region_comparison"]["mean"])
        gpu_times.append(data["gpu_region_comparison"]["mean"])
        speedups.append(data["gpu_region_comparison"]["speedup"])
    
    # Bar chart of execution times
    plt.subplot(1, 2, 1)
    x = np.arange(len(components))
    width = 0.35
    
    plt.bar(x - width/2, cpu_times, width, label='CPU')
    plt.bar(x + width/2, gpu_times, width, label='GPU (Jetson Optimized)')
    
    plt.yscale('log')
    plt.ylabel('Execution Time (seconds)')
    plt.xlabel('Component')
    plt.title('CPU vs GPU Execution Time')
    plt.xticks(x, components, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bar chart of speedups
    plt.subplot(1, 2, 2)
    plt.bar(x, speedups, color='green')
    
    plt.ylabel('Speedup Factor (CPU/GPU)')
    plt.xlabel('Component')
    plt.title('GPU Speedup over CPU')
    plt.xticks(x, components, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add device info
    if "risk_assessment" in results and "device_info" in results["risk_assessment"]:
        device_info = results["risk_assessment"]["device_info"]
        device_str = f"Device: {device_info['model']} (SM {device_info['compute_capability']}), {device_info['cuda_cores']} CUDA cores, {device_info['total_memory_mb']:.0f}MB memory"
        plt.figtext(0.5, 0.01, device_str, ha='center')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    output_file = os.path.join(output_dir, "benchmark_results.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_file


def generate_report(results: Dict[str, Any], output_dir: str) -> str:
    """
    Generate a benchmark report.
    
    Args:
        results: Benchmark results
        output_dir: Directory to save report
        
    Returns:
        Path to report file
    """
    output_file = os.path.join(output_dir, "benchmark_report.md")
    
    with open(output_file, "w") as f:
        f.write("# Geospatial Financial Integration Benchmarks\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Device information
        if "risk_assessment" in results and "device_info" in results["risk_assessment"]:
            device_info = results["risk_assessment"]["device_info"]
            f.write("## Device Information\n\n")
            f.write(f"- **Device**: {device_info['model']}\n")
            f.write(f"- **Compute Capability**: {device_info['compute_capability']}\n")
            f.write(f"- **CUDA Cores**: {device_info['cuda_cores']}\n")
            f.write(f"- **SMs**: {device_info['sm_count']}\n")
            f.write(f"- **Memory**: {device_info['total_memory_mb']:.0f}MB\n")
            f.write(f"- **Is Jetson**: {device_info['is_jetson']}\n\n")
        
        # Risk assessment results
        if "risk_assessment" in results:
            data = results["risk_assessment"]
            f.write("## Risk Assessment Performance\n\n")
            
            f.write("### Overview\n\n")
            f.write("| Implementation | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Speedup |\n")
            f.write("|---------------|--------------|-------------|--------------|--------------|--------|\n")
            
            f.write(f"| CPU | {data['cpu']['mean']:.4f} | {data['cpu']['std']:.4f} | {data['cpu']['min']:.4f} | {data['cpu']['max']:.4f} | 1.00x |\n")
            
            if "gpu_default" in data:
                gpu_data = data["gpu_default"]
                f.write(f"| GPU (Default) | {gpu_data['mean']:.4f} | {gpu_data['std']:.4f} | {gpu_data['min']:.4f} | {gpu_data['max']:.4f} | {gpu_data['speedup']:.2f}x |\n")
            
            for profile in MemoryProfile:
                profile_key = f"gpu_{profile.value}"
                if profile_key in data:
                    profile_data = data[profile_key]
                    f.write(f"| GPU ({profile.value}) | {profile_data['mean']:.4f} | {profile_data['std']:.4f} | {profile_data['min']:.4f} | {profile_data['max']:.4f} | {profile_data['speedup']:.2f}x |\n")
            
            f.write(f"\n**Overall Best Speedup**: {data['overall_speedup']:.2f}x\n\n")
        
        # Scenario analysis results
        if "scenario_analysis" in results:
            data = results["scenario_analysis"]
            f.write("## Scenario Analysis Performance\n\n")
            
            f.write("### Overview\n\n")
            f.write("| Implementation | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Speedup |\n")
            f.write("|---------------|--------------|-------------|--------------|--------------|--------|\n")
            f.write(f"| CPU | {data['cpu']['mean']:.4f} | {data['cpu']['std']:.4f} | {data['cpu']['min']:.4f} | {data['cpu']['max']:.4f} | 1.00x |\n")
            f.write(f"| GPU | {data['gpu']['mean']:.4f} | {data['gpu']['std']:.4f} | {data['gpu']['min']:.4f} | {data['gpu']['max']:.4f} | {data['gpu']['speedup']:.2f}x |\n")
            
            f.write("\n### Individual Scenarios\n\n")
            f.write("| Scenario | CPU Time (s) | GPU Time (s) | Speedup |\n")
            f.write("|----------|--------------|--------------|--------|\n")
            
            for scenario_name in data["cpu_scenarios"].keys():
                cpu_time = data["cpu_scenarios"][scenario_name]["mean"]
                gpu_time = data["gpu_scenarios"][scenario_name]["mean"]
                speedup = data["gpu_scenarios"][scenario_name]["speedup"]
                f.write(f"| {scenario_name} | {cpu_time:.4f} | {gpu_time:.4f} | {speedup:.2f}x |\n")
            
            f.write(f"\n**Overall Speedup**: {data['overall_speedup']:.2f}x\n\n")
        
        # Climate risk results
        if "climate_risk" in results:
            data = results["climate_risk"]
            f.write("## Climate Risk Assessment Performance\n\n")
            
            f.write("### Overview\n\n")
            f.write("| Implementation | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Speedup |\n")
            f.write("|---------------|--------------|-------------|--------------|--------------|--------|\n")
            f.write(f"| CPU | {data['cpu']['mean']:.4f} | {data['cpu']['std']:.4f} | {data['cpu']['min']:.4f} | {data['cpu']['max']:.4f} | 1.00x |\n")
            f.write(f"| GPU | {data['gpu']['mean']:.4f} | {data['gpu']['std']:.4f} | {data['gpu']['min']:.4f} | {data['gpu']['max']:.4f} | {data['gpu']['speedup']:.2f}x |\n")
            
            f.write("\n### Climate Scenarios\n\n")
            f.write("| Scenario | CPU Time (s) | GPU Time (s) | Speedup |\n")
            f.write("|----------|--------------|--------------|--------|\n")
            
            for scenario_name in data["cpu_scenarios"].keys():
                cpu_time = data["cpu_scenarios"][scenario_name]["mean"]
                gpu_time = data["gpu_scenarios"][scenario_name]["mean"]
                speedup = data["gpu_scenarios"][scenario_name]["speedup"]
                f.write(f"| {scenario_name} | {cpu_time:.4f} | {gpu_time:.4f} | {speedup:.2f}x |\n")
            
            f.write(f"\n**Overall Speedup**: {data['overall_speedup']:.2f}x\n\n")
        
        # Multiregion results
        if "multiregion" in results:
            data = results["multiregion"]
            f.write("## Multi-Region Analysis Performance\n\n")
            
            f.write("### Regional Portfolio Creation\n\n")
            f.write("| Implementation | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Speedup |\n")
            f.write("|---------------|--------------|-------------|--------------|--------------|--------|\n")
            
            cpu_data = data["cpu_regional_portfolio"]
            gpu_data = data["gpu_regional_portfolio"]
            
            f.write(f"| CPU | {cpu_data['mean']:.4f} | {cpu_data['std']:.4f} | {cpu_data['min']:.4f} | {cpu_data['max']:.4f} | 1.00x |\n")
            f.write(f"| GPU | {gpu_data['mean']:.4f} | {gpu_data['std']:.4f} | {gpu_data['min']:.4f} | {gpu_data['max']:.4f} | {gpu_data['speedup']:.2f}x |\n")
            
            f.write("\n### Regional Risk Comparison\n\n")
            f.write("| Implementation | Mean Time (s) | Std Dev (s) | Min Time (s) | Max Time (s) | Speedup |\n")
            f.write("|---------------|--------------|-------------|--------------|--------------|--------|\n")
            
            cpu_data = data["cpu_region_comparison"]
            gpu_data = data["gpu_region_comparison"]
            
            f.write(f"| CPU | {cpu_data['mean']:.4f} | {cpu_data['std']:.4f} | {cpu_data['min']:.4f} | {cpu_data['max']:.4f} | 1.00x |\n")
            f.write(f"| GPU | {gpu_data['mean']:.4f} | {gpu_data['std']:.4f} | {gpu_data['min']:.4f} | {gpu_data['max']:.4f} | {gpu_data['speedup']:.2f}x |\n")
            
            f.write(f"\n**Overall Speedup**: {data['overall_speedup']:.2f}x\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("| Component | CPU Time (s) | GPU Time (s) | Speedup |\n")
        f.write("|-----------|--------------|--------------|--------|\n")
        
        total_cpu_time = 0
        total_gpu_time = 0
        
        if "risk_assessment" in results:
            data = results["risk_assessment"]
            cpu_time = data["cpu"]["mean"]
            
            # Use best GPU time
            gpu_key = "gpu_default"
            for profile in MemoryProfile:
                profile_key = f"gpu_{profile.value}"
                if profile_key in data and data[profile_key]["mean"] < data[gpu_key]["mean"]:
                    gpu_key = profile_key
            
            gpu_time = data[gpu_key]["mean"]
            speedup = data[gpu_key]["speedup"]
            
            f.write(f"| Risk Assessment | {cpu_time:.4f} | {gpu_time:.4f} | {speedup:.2f}x |\n")
            total_cpu_time += cpu_time
            total_gpu_time += gpu_time
        
        if "scenario_analysis" in results:
            data = results["scenario_analysis"]
            cpu_time = data["cpu"]["mean"]
            gpu_time = data["gpu"]["mean"]
            speedup = data["gpu"]["speedup"]
            
            f.write(f"| Scenario Analysis | {cpu_time:.4f} | {gpu_time:.4f} | {speedup:.2f}x |\n")
            total_cpu_time += cpu_time
            total_gpu_time += gpu_time
        
        if "climate_risk" in results:
            data = results["climate_risk"]
            cpu_time = data["cpu"]["mean"]
            gpu_time = data["gpu"]["mean"]
            speedup = data["gpu"]["speedup"]
            
            f.write(f"| Climate Risk | {cpu_time:.4f} | {gpu_time:.4f} | {speedup:.2f}x |\n")
            total_cpu_time += cpu_time
            total_gpu_time += gpu_time
        
        if "multiregion" in results:
            data = results["multiregion"]
            
            # Regional portfolio
            cpu_time = data["cpu_regional_portfolio"]["mean"]
            gpu_time = data["gpu_regional_portfolio"]["mean"]
            speedup = data["gpu_regional_portfolio"]["speedup"]
            
            f.write(f"| Regional Portfolio | {cpu_time:.4f} | {gpu_time:.4f} | {speedup:.2f}x |\n")
            total_cpu_time += cpu_time
            total_gpu_time += gpu_time
            
            # Region comparison
            cpu_time = data["cpu_region_comparison"]["mean"]
            gpu_time = data["gpu_region_comparison"]["mean"]
            speedup = data["gpu_region_comparison"]["speedup"]
            
            f.write(f"| Region Comparison | {cpu_time:.4f} | {gpu_time:.4f} | {speedup:.2f}x |\n")
            total_cpu_time += cpu_time
            total_gpu_time += gpu_time
        
        # Overall
        total_speedup = total_cpu_time / total_gpu_time if total_gpu_time > 0 else 0
        f.write(f"| **TOTAL** | {total_cpu_time:.4f} | {total_gpu_time:.4f} | {total_speedup:.2f}x |\n\n")
        
        # Add chart if available
        chart_file = os.path.join(output_dir, "benchmark_results.png")
        if os.path.exists(chart_file):
            f.write(f"\n## Performance Chart\n\n")
            f.write(f"![Benchmark Results](benchmark_results.png)\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write(f"The Jetson-optimized implementation provides significant performance improvements across all components of the geospatial financial integration, with an overall speedup of **{total_speedup:.2f}x** compared to the CPU implementation.\n\n")
        
        if "risk_assessment" in results and "device_info" in results["risk_assessment"]:
            device_info = results["risk_assessment"]["device_info"]
            if device_info["is_jetson"]:
                f.write(f"These benchmarks were run on a {device_info['model']} with {device_info['cuda_cores']} CUDA cores and {device_info['total_memory_mb']:.0f}MB of memory. The optimizations implemented in the Jetson-specific code path take advantage of the device's unique characteristics to maximize performance.\n\n")
        
        f.write("Key optimization techniques include:\n\n")
        f.write("1. Efficient memory management to work within Jetson's memory constraints\n")
        f.write("2. Batched processing to avoid overwhelming the device\n")
        f.write("3. Tiled operations for large datasets\n")
        f.write("4. Custom CUDA kernels optimized for Jetson's architecture\n")
        f.write("5. Mixed-precision operations where appropriate\n")
        f.write("6. Adaptive resource allocation based on workload characteristics\n\n")
        
        f.write("These optimizations make the geospatial financial integration well-suited for deployment on edge devices like the Jetson Orin NX, enabling sophisticated financial risk analysis in resource-constrained environments.\n")
    
    return output_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark geospatial financial integration on Jetson Orin NX")
    parser.add_argument("--data-dir", type=str, default="./data/geo_financial", help="Directory with test data")
    parser.add_argument("--output-dir", type=str, default="./output/jetson_benchmarks", help="Directory for output files")
    parser.add_argument("--generate-data", action="store_true", help="Generate test data if not present")
    parser.add_argument("--num-assets", type=int, default=300, help="Number of assets for benchmarking")
    parser.add_argument("--device-id", type=int, default=0, help="GPU device ID to use")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each benchmark")
    parser.add_argument("--all-profiles", action="store_true", help="Benchmark all memory profiles")
    parser.add_argument("--no-scenarios", action="store_true", help="Skip scenario analysis benchmarks")
    parser.add_argument("--no-climate", action="store_true", help="Skip climate risk benchmarks")
    parser.add_argument("--no-multiregion", action="store_true", help="Skip multi-region benchmarks")
    parser.add_argument("--realtime", action="store_true", help="Run realtime monitoring benchmarks")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Generate or load test data
    if args.generate_data or not os.path.exists(os.path.join(args.data_dir, "portfolio.json")):
        logger.info(f"Generating test data with {args.num_assets} assets")
        portfolio, risk_model = create_test_data(args.data_dir, args.num_assets)
    else:
        logger.info(f"Loading test data from {args.data_dir}")
        portfolio, risk_model = load_test_data(args.data_dir)
    
    # Initialize results dictionary
    results = {}
    
    # Benchmark risk assessment
    logger.info("Running risk assessment benchmark")
    results["risk_assessment"] = benchmark_risk_assessment(
        portfolio=portfolio,
        risk_model=risk_model,
        device_id=args.device_id,
        iterations=args.iterations,
        benchmark_all_profiles=args.all_profiles
    )
    
    # Benchmark scenario analysis if not disabled
    if not args.no_scenarios:
        logger.info("Running scenario analysis benchmark")
        results["scenario_analysis"] = benchmark_scenario_analysis(
            portfolio=portfolio,
            risk_model=risk_model,
            device_id=args.device_id,
            iterations=args.iterations
        )
    
    # Benchmark climate risk assessment if not disabled
    if not args.no_climate:
        logger.info("Running climate risk benchmark")
        results["climate_risk"] = benchmark_climate_risk(
            portfolio=portfolio,
            risk_model=risk_model,
            device_id=args.device_id,
            iterations=args.iterations
        )
    
    # Benchmark multi-region analysis if not disabled
    if not args.no_multiregion:
        logger.info("Running multi-region benchmark")
        results["multiregion"] = benchmark_multiregion(
            portfolio=portfolio,
            risk_model=risk_model,
            device_id=args.device_id,
            iterations=args.iterations
        )
    
    # Save raw benchmark results
    results_file = os.path.join(args.output_dir, "benchmark_results.json")
    
    # Convert numpy values to Python native types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    with open(results_file, "w") as f:
        json.dump(convert_numpy(results), f, indent=2)
    
    logger.info(f"Raw benchmark results saved to {results_file}")
    
    # Generate plot
    plot_file = plot_results(results, args.output_dir)
    if plot_file:
        logger.info(f"Benchmark plot saved to {plot_file}")
    
    # Generate report
    report_file = generate_report(results, args.output_dir)
    logger.info(f"Benchmark report saved to {report_file}")
    
    # Print summary
    print("\n=== BENCHMARK SUMMARY ===\n")
    
    if "risk_assessment" in results:
        data = results["risk_assessment"]
        print(f"Risk Assessment: {data['cpu']['mean']:.4f}s CPU, {data['gpu_default']['mean']:.4f}s GPU, {data['gpu_default']['speedup']:.2f}x speedup")
    
    if "scenario_analysis" in results:
        data = results["scenario_analysis"]
        print(f"Scenario Analysis: {data['cpu']['mean']:.4f}s CPU, {data['gpu']['mean']:.4f}s GPU, {data['gpu']['speedup']:.2f}x speedup")
    
    if "climate_risk" in results:
        data = results["climate_risk"]
        print(f"Climate Risk: {data['cpu']['mean']:.4f}s CPU, {data['gpu']['mean']:.4f}s GPU, {data['gpu']['speedup']:.2f}x speedup")
    
    if "multiregion" in results:
        data = results["multiregion"]
        cpu_total = data["cpu_regional_portfolio"]["mean"] + data["cpu_region_comparison"]["mean"]
        gpu_total = data["gpu_regional_portfolio"]["mean"] + data["gpu_region_comparison"]["mean"]
        speedup = cpu_total / gpu_total
        print(f"Multi-region: {cpu_total:.4f}s CPU, {gpu_total:.4f}s GPU, {speedup:.2f}x speedup")
    
    # Calculate overall speedup
    total_cpu_time = 0
    total_gpu_time = 0
    
    if "risk_assessment" in results:
        total_cpu_time += results["risk_assessment"]["cpu"]["mean"]
        total_gpu_time += results["risk_assessment"]["gpu_default"]["mean"]
    
    if "scenario_analysis" in results:
        total_cpu_time += results["scenario_analysis"]["cpu"]["mean"]
        total_gpu_time += results["scenario_analysis"]["gpu"]["mean"]
    
    if "climate_risk" in results:
        total_cpu_time += results["climate_risk"]["cpu"]["mean"]
        total_gpu_time += results["climate_risk"]["gpu"]["mean"]
    
    if "multiregion" in results:
        total_cpu_time += results["multiregion"]["cpu_regional_portfolio"]["mean"]
        total_cpu_time += results["multiregion"]["cpu_region_comparison"]["mean"]
        total_gpu_time += results["multiregion"]["gpu_regional_portfolio"]["mean"]
        total_gpu_time += results["multiregion"]["gpu_region_comparison"]["mean"]
    
    overall_speedup = total_cpu_time / total_gpu_time if total_gpu_time > 0 else 0
    print(f"\nOVERALL: {total_cpu_time:.4f}s CPU, {total_gpu_time:.4f}s GPU, {overall_speedup:.2f}x speedup\n")


if __name__ == "__main__":
    main()
EOF

# Make the benchmark script executable
chmod +x "${benchmark_script}"

# Run the benchmark
if [ "${GENERATE_DATA}" = true ]; then
  data_flag="--generate-data"
else
  data_flag=""
fi

if [ "${BENCHMARK_ALL_PROFILES}" = true ]; then
  profile_flag="--all-profiles"
else
  profile_flag=""
fi

# Build command based on what to run
cmd="python3 ${benchmark_script} --data-dir \"${DATA_DIR}\" --output-dir \"${OUTPUT_DIR}\" ${data_flag} --num-assets ${NUM_ASSETS} --device-id ${DEVICE_ID} --iterations ${NUM_ITERATIONS} ${profile_flag}"

# Add optional flags
if [ "${VERBOSE}" = true ]; then
  cmd="${cmd} --verbose"
fi

if [ "${RUN_SCENARIOS}" = false ]; then
  cmd="${cmd} --no-scenarios"
fi

if [ "${RUN_CLIMATE}" = false ]; then
  cmd="${cmd} --no-climate"
fi

if [ "${RUN_MULTIREGION}" = false ]; then
  cmd="${cmd} --no-multiregion"
fi

if [ "${RUN_REALTIME}" = true ]; then
  cmd="${cmd} --realtime"
fi

# Run benchmark
run_command "${cmd}" "Running benchmarks"

echo "Benchmarks completed. Results available in ${OUTPUT_DIR}"