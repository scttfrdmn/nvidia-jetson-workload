# Geospatial Financial Risk Analysis Integration Guide

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This guide provides detailed information on how to use the Geospatial Financial Risk Analysis integration in the NVIDIA Jetson Workload project.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Key Concepts](#key-concepts)
4. [Tutorials](#tutorials)
5. [Advanced Usage](#advanced-usage)
6. [API Reference](#api-reference)
7. [Performance Considerations](#performance-considerations)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)

## Introduction

The Geospatial Financial Risk Analysis integration combines the Geospatial Analysis and Financial Modeling workloads to provide a comprehensive solution for assessing financial risks with geospatial context. This integration enables:

- Analyzing geospatial risk factors in financial portfolios
- Assessing risk exposure to natural hazards for investments
- Optimizing portfolios to minimize exposure to geospatial risks
- Evaluating climate-related financial risks across different scenarios
- Visualizing financial data with geospatial context

This integration is particularly useful for:

- Insurance companies assessing risk exposure to natural disasters
- Real estate investment trusts (REITs) optimizing property portfolios
- Infrastructure investors evaluating project locations considering environmental hazards
- Asset managers incorporating climate change risks into investment decisions
- Financial institutions performing climate-related stress testing and scenario analysis

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended, but CPU-only mode is supported)
- Base NVIDIA Jetson Workload installation

### Installation

The Geospatial Financial Risk Analysis integration is included in the base NVIDIA Jetson Workload installation. No additional installation steps are required.

For the interactive dashboard, you'll need to install additional dependencies:

```bash
pip install dash dash-bootstrap-components plotly
```

### Basic Usage

Here's a simple example of how to use the integration:

```python
from src.integrations.geo_financial.geo_risk import GeospatialRiskModel, GeospatialPortfolio
from src.integrations.geo_financial.data_connectors import AssetLocationDataLoader
from src.integrations.geo_financial.risk_aggregation import create_combined_risk_factor

# Load asset data
loader = AssetLocationDataLoader()
assets_df = loader.load_asset_csv(
    file_path='path/to/assets.csv',
    id_col='id',
    name_col='name',
    value_col='value',
    x_col='x',
    y_col='y'
)

# Create portfolio
portfolio = GeospatialPortfolio()
portfolio.add_assets_from_dataframe(assets_df, 'id', 'name', 'value', 'x', 'y')

# Create risk factors
flood_risk = create_elevation_risk_factor('path/to/dem.tif', risk_weight=0.6)
landslide_risk = create_slope_risk_factor('path/to/dem.tif', risk_weight=0.4)

# Create combined risk factor
combined_risk = create_combined_risk_factor(
    risk_factors=[flood_risk, landslide_risk],
    name="Combined Natural Hazard Risk",
    description="Combined risk from multiple natural hazards"
)

# Create risk model and assess portfolio risk
risk_model = GeospatialRiskModel()
risk_model.add_risk_factor(combined_risk)
risk_scores = portfolio.assess_risk(risk_model)

# Print risk scores
for asset_id, score in risk_scores.items():
    print(f"Asset {asset_id}: Risk Score = {score:.4f}")
```

## Key Concepts

### Spatial Risk Factors

A spatial risk factor represents a specific type of risk that varies across geographic space. Examples include:

- **Flood Risk**: Higher in low-elevation areas near water bodies
- **Landslide Risk**: Higher on steep slopes with certain soil types
- **Earthquake Risk**: Varies based on proximity to fault lines
- **Wildfire Risk**: Depends on vegetation, climate, and topography

The integration provides utilities for creating risk factors from various data sources:

- Digital Elevation Models (DEMs)
- Slope and aspect data
- Distance to features (coastlines, rivers, etc.)
- Climate and weather data
- Real-time event data streams

### Risk Aggregation

Multiple risk factors can be combined into a comprehensive risk assessment using various aggregation methods:

- **Weighted Average**: Simple weighted average of risk factors
- **Weighted Maximum**: Maximum risk value, weighted by factor importance
- **Weighted Product**: Product of risk factors, raised to power of weight
- **Copula-based**: Statistical methods that preserve dependencies between risks

### Climate Scenario Analysis

The integration supports climate risk assessment under different scenarios:

- **IPCC Scenarios**: SSP1-RCP2.6 (optimistic) to SSP5-RCP8.5 (pessimistic)
- **NGFS Scenarios**: Orderly, Disorderly, and Hothouse World
- **Time Horizons**: Short (1-5 years), Medium (5-15 years), Long (15-30 years), Very Long (30+ years)

### Portfolio Optimization

The integration can optimize investment portfolios to minimize exposure to geospatial risks while maintaining target financial returns:

- **Risk-Adjusted Weights**: Reduce exposure to high-risk assets
- **Constrained Optimization**: Apply constraints on sector allocation, maximum weights, etc.
- **Multi-Objective Optimization**: Balance risk reduction with return maximization

## Tutorials

### 1. Basic Geospatial Risk Analysis

Follow this tutorial to perform a basic geospatial risk analysis on a portfolio of assets:

1. **Load Data**: Load asset locations and DEM data
2. **Create Risk Factors**: Define relevant risk factors for your analysis
3. **Assess Portfolio Risk**: Calculate risk scores for each asset
4. **Visualize Results**: Create maps and charts to visualize risk exposure

See the [Basic Geospatial Risk Analysis Notebook](../notebooks/basic_geo_risk_analysis.ipynb) for a step-by-step guide.

### 2. Climate Risk Assessment

This tutorial demonstrates how to assess climate-related financial risks:

1. **Define Climate Scenarios**: Choose relevant climate scenarios and time horizons
2. **Create Climate Risk Factors**: Define physical risk factors (flooding, heat stress, etc.)
3. **Assess Transition Risks**: Evaluate sector-based transition risks
4. **Compare Scenarios**: Analyze portfolio performance across scenarios

See the [Climate Risk Assessment Notebook](../notebooks/climate_risk_assessment.ipynb) for details.

### 3. Interactive Dashboard

Learn how to use the interactive dashboard for geospatial financial risk analysis:

1. **Launch Dashboard**: Start the dashboard application
2. **Load Your Data**: Import your own asset and geospatial data
3. **Configure Analysis**: Set up risk factors and parameters
4. **Explore Results**: Interact with visualizations and optimization tools

See the [Dashboard Tutorial](../notebooks/dashboard_tutorial.ipynb) for instructions.

### 4. Real-Time Risk Monitoring

This tutorial shows how to use real-time data streaming for continuous risk monitoring:

1. **Set Up Data Streams**: Configure market data and geospatial event streams
2. **Connect to Risk Model**: Link streaming data to your risk assessment model
3. **Monitor Risk Changes**: Track how risk evolves in response to events
4. **Create Alerts**: Set up notifications for significant risk changes

Run the realtime monitoring example:

```bash
python -m src.integrations.geo_financial.examples.realtime_monitoring \
  --data-dir data/geo_financial \
  --output-dir results/geo_financial/realtime_monitor \
  --generate-data \
  --duration 900 \
  --visualization-interval 180
```

For a fully interactive experience, launch the real-time dashboard:

```bash
./scripts/run_realtime_dashboard.sh --generate-data --port 8050
```

### 5. Multi-Region Risk Analysis

This tutorial demonstrates how to perform comparative analysis across different geographic regions:

1. **Define Regions**: Create multiple geographic regions for analysis
2. **Allocate Assets**: Assign assets to appropriate regions
3. **Perform Region-Specific Analysis**: Calculate risk metrics for each region
4. **Compare Across Regions**: Analyze regional differences and correlations
5. **Optimize Allocation**: Determine optimal allocation across regions

Run the multi-region analysis example:

```bash
python -m src.integrations.geo_financial.examples.multiregion_analysis \
  --data-dir data/geo_financial/multiregion \
  --output-dir results/geo_financial/multiregion \
  --generate-data \
  --num-regions 4 \
  --assets-per-region 25
```

This example demonstrates region-based:
- Risk distribution analysis
- Cross-region correlations
- Diversification benefits
- Optimal portfolio allocation

## Advanced Usage

### Custom Risk Factors

You can create custom risk factors by subclassing the `SpatialRiskFactor` class:

```python
from src.integrations.geo_financial.geo_risk import SpatialRiskFactor

class WildfireRiskFactor(SpatialRiskFactor):
    """Custom risk factor for wildfire risk."""
    
    def __init__(self, vegetation_data, climate_data, risk_weight=1.0):
        """Initialize wildfire risk factor."""
        # Process input data to create risk surface
        risk_data = self._calculate_wildfire_risk(vegetation_data, climate_data)
        
        super().__init__(
            name="Wildfire Risk",
            description="Risk based on vegetation and climate data",
            risk_weight=risk_weight,
            spatial_data=risk_data,
            geo_transform=vegetation_data.geo_transform
        )
    
    def _calculate_wildfire_risk(self, vegetation_data, climate_data):
        """Custom method to calculate wildfire risk."""
        # Implementation details here
        return combined_risk_data
```

### Advanced Risk Aggregation

For sophisticated risk modeling, you can use copula-based methods:

```python
from src.integrations.geo_financial.risk_aggregation import RiskAggregator, AggregationMethod

# Create risk aggregator
aggregator = RiskAggregator()

# Compute correlation matrix
correlation_matrix = np.array([
    [1.0, 0.3, 0.5],
    [0.3, 1.0, 0.2],
    [0.5, 0.2, 1.0]
])

# Aggregate using Gaussian copula
aggregated_risk, stats = aggregator.aggregate_risk_factors(
    risk_factors=[factor1, factor2, factor3],
    method=AggregationMethod.COPULA_GAUSSIAN,
    correlation_matrix=correlation_matrix
)
```

### Custom Climate Scenarios

You can define custom climate scenarios for specialized analyses:

```python
from src.integrations.geo_financial.climate_risk_assessment import ClimateRiskAssessor

# Create custom scenario parameters
custom_params = {
    "temperature_increase": 2.0,
    "physical_risk_factor": 1.2,
    "transition_risk_factor": 0.8,
    "time_factors": {
        "short": 0.4,
        "medium": 0.8,
        "long": 1.2,
        "very_long": 1.6
    }
}

# Initialize assessor with custom data path
assessor = ClimateRiskAssessor(
    device_id=0,
    data_path='path/to/custom/scenarios'
)

# Use custom parameter directly in risk adjustment
base_risk = 0.5
adjusted_risk = assessor.adjust_risk_by_scenario(
    base_risk, 
    risk_type="physical",
    custom_params=custom_params
)
```

## API Reference

### Core Modules

#### `geo_risk` Module

- `SpatialRiskFactor`: Base class for spatial risk factors
- `GeospatialRiskModel`: Model for combining risk factors
- `GeospatialPortfolio`: Financial portfolio with geospatial properties
- Helper functions: `create_elevation_risk_factor`, `create_slope_risk_factor`, etc.

#### `risk_aggregation` Module

- `RiskAggregator`: Class for aggregating multiple risk factors
- `RiskSurfaceGenerator`: Creates continuous risk surfaces from point data
- `AggregationMethod`: Enumeration of aggregation methods
- Helper functions: `create_combined_risk_factor`, `interpolate_asset_risk_surface`, etc.

#### `climate_risk_assessment` Module

- `ClimateRiskAssessor`: Class for climate risk assessment
- `ClimateScenario`: Enumeration of climate scenarios
- `TimeHorizon`: Enumeration of time horizons
- `ClimateHazardType`: Enumeration of physical climate hazards
- `TransitionRiskType`: Enumeration of transition risk types
- Helper functions for climate risk factor creation

#### `data_connectors` Module

- `AssetLocationDataLoader`: Loads asset location data
- `FinancialDataLoader`: Loads financial time series data
- `GeoRiskDataLoader`: Loads geospatial risk data

#### `visualization` Module

- `GeoFinancialVisualizer`: Creates visualizations for geospatial financial analysis

#### `dashboard_app` Module

- `DashboardApp`: Interactive dashboard application

### Key Classes and Methods

#### Real-time Module

- `RealTimeStreamManager`: Manages multiple real-time data streams
- `RealTimeRiskAnalyzer`: Updates risk assessments based on streaming data
- `MarketDataStream`: Provides real-time market data
- `GeospatialEventStream`: Streams real-time geospatial events
- Helper functions for creating and configuring data streams

#### Multi-region Module

- `RegionDefinition`: Defines a geographic region for analysis
- `RegionalPortfolio`: Extends portfolio to separate assets by region
- `MultiRegionRiskModel`: Applies risk models across multiple regions
- `RegionalRiskComparator`: Compares risk characteristics across regions
- `MultiRegionVisualizer`: Creates region-based visualizations
- Helper functions for creating region grids and regional portfolios

See the [API Reference](./api_reference.md) for detailed documentation of all classes and methods.

## Performance Considerations

### GPU Acceleration

The integration leverages GPU acceleration for computationally intensive operations:

- Risk factor computation and transformation
- Risk aggregation, especially for large geospatial datasets
- Portfolio optimization
- Climate scenario analysis

To enable GPU acceleration, make sure you have a CUDA-capable GPU and specify the device ID when creating objects:

```python
# Use GPU device 0
risk_model = GeospatialRiskModel(device_id=0)
portfolio = GeospatialPortfolio(device_id=0)
```

For CPU-only execution, use a negative device ID:

```python
# Use CPU only
risk_model = GeospatialRiskModel(device_id=-1)
```

### Memory Management

When working with large geospatial datasets:

- Use appropriate resolution for your analysis to avoid excessive memory usage
- Process data in tiles or chunks when possible
- Release large data arrays when no longer needed
- Monitor memory usage, especially when using multiple GPUs

### Parallel Processing

For analyzing multiple scenarios or portfolios:

- Use the `concurrent.futures` module to parallelize independent tasks
- Consider using multi-GPU setups for large-scale analyses
- Use the dashboard's headless mode for batch processing

Example of parallel scenario analysis:

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def analyze_scenario(scenario, portfolio, risk_factors):
    # Create assessor for this scenario
    assessor = ClimateRiskAssessor(scenario=scenario)
    
    # Assess risk and return results
    risk_scores = assessor.assess_physical_climate_risk(
        portfolio=portfolio,
        risk_factors=risk_factors
    )
    return scenario, risk_scores

# Analyze multiple scenarios in parallel
scenarios = list(ClimateScenario)[:4]
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(analyze_scenario, scenario, portfolio, risk_factors)
        for scenario in scenarios
    ]
    
    results = {}
    for future in concurrent.futures.as_completed(futures):
        scenario, risk_scores = future.result()
        results[scenario.value] = risk_scores
```

## Examples

### Insurance Risk Assessment

```python
# Load insurance portfolio
portfolio = GeospatialPortfolio(device_id=0)
portfolio.add_assets_from_dataframe(insurance_df, 'id', 'property', 'value', 'lon', 'lat')

# Create flood risk factor
flood_risk = create_flood_risk_factor(
    dem_path='path/to/dem.tif',
    risk_weight=0.7,
    low_elevation_threshold=10.0
)

# Create landslide risk factor
landslide_risk = create_slope_risk_factor(
    dem_path='path/to/dem.tif',
    risk_weight=0.3
)

# Create combined risk factor
combined_risk = create_combined_risk_factor(
    risk_factors=[flood_risk, landslide_risk],
    name="Natural Hazard Risk",
    method=AggregationMethod.WEIGHTED_AVERAGE
)

# Create risk model and assess portfolio
risk_model = GeospatialRiskModel(device_id=0)
risk_model.add_risk_factor(combined_risk)
risk_scores = portfolio.assess_risk(risk_model)

# Calculate expected annual loss
expected_loss = {
    asset_id: score * portfolio.assets[i]['value'] * 0.01  # 1% annual probability
    for i, (asset_id, score) in enumerate(risk_scores.items())
}

# Total expected annual loss
total_expected_loss = sum(expected_loss.values())
print(f"Total Expected Annual Loss: ${total_expected_loss:,.2f}")
```

### Real Estate Portfolio Analysis

```python
# Load real estate portfolio
portfolio = GeospatialPortfolio(device_id=0)
portfolio.add_assets_from_dataframe(properties_df, 'id', 'address', 'value', 'lon', 'lat')

# Create climate risk assessor
assessor = ClimateRiskAssessor(
    scenario=ClimateScenario.SSP2_RCP45,
    time_horizon=TimeHorizon.MEDIUM
)

# Create climate risk factors
flood_risk = create_flood_risk_factor(
    dem_path='path/to/dem.tif',
    risk_weight=0.5,
    assessor=assessor
)

heatwave_risk = create_heatwave_risk_factor(
    temperature_data=temperature_data,
    geo_transform=geo_transform,
    risk_weight=0.3,
    assessor=assessor
)

sea_level_risk = create_sea_level_rise_risk_factor(
    dem_path='path/to/dem.tif',
    coastline_distance=coastline_distance,
    geo_transform=geo_transform,
    risk_weight=0.2,
    assessor=assessor
)

# Assess climate risk
risk_factors = [flood_risk, heatwave_risk, sea_level_risk]
physical_risk_scores = assessor.assess_physical_climate_risk(
    portfolio=portfolio,
    risk_factors=risk_factors
)

# Optimize portfolio to minimize climate risk
optimized_weights = portfolio.optimize_for_geo_risk(
    risk_model=risk_model,
    target_return=0.05  # 5% annual return
)

# Print optimization results
print("Recommended Portfolio Adjustments:")
for asset_id, opt_weight in optimized_weights.items():
    asset = next((a for a in portfolio.assets if a['id'] == asset_id), None)
    orig_weight = asset['value'] / sum(a['value'] for a in portfolio.assets)
    change = (opt_weight - orig_weight) * 100
    
    print(f"{asset['name']}: {change:+.1f}% allocation change")
```

### Climate Scenario Dashboard

```python
from src.integrations.geo_financial.dashboard_app import DashboardApp

# Create dashboard application
app = DashboardApp(
    data_dir='data/geo_financial',
    output_dir='results/geo_financial/dashboard',
    device_id=0
)

# Load data
app.load_data()

# Create portfolio and risk factors
app.create_portfolio()
app.create_risk_factors()
app.create_climate_risk_factors()

# Launch interactive dashboard
app.create_dashboard()
```

## Troubleshooting

### Common Issues

#### Missing CUDA or GPU

If you see errors related to CUDA or GPU availability:

```
ImportError: No module named 'cupy'
```

or

```
CUDA error: no CUDA-capable device is detected
```

Solutions:
- Ensure you have a CUDA-capable GPU
- Install the CUDA toolkit and drivers
- Install CuPy: `pip install cupy-cuda12x` (replace with your CUDA version)
- Use CPU mode by setting `device_id=-1` in all objects

#### Memory Errors

If you encounter out-of-memory errors:

Solutions:
- Reduce the resolution of your geospatial data
- Process data in smaller chunks
- Use CPU mode for very large datasets
- Close other GPU-intensive applications
- Use a GPU with more memory

#### Data Import Issues

If you have problems loading data:

Solutions:
- Check file paths and formats
- Ensure GeoTIFF files have the correct projection information
- Verify CSV files have the required columns
- Use the data connector utilities for proper import

### Getting Help

If you encounter issues not covered here:

- Check the [FAQ](./faq.md) for additional guidance
- Review the [GitHub Issues](https://github.com/scttfrdmn/nvidia-jetson-workload/issues) for similar problems
- File a new issue with detailed information about the problem
- Join the [Community Forum](https://community.example.com/nvidia-jetson-workload) for discussion

## References

- [Financial Modeling Workload Documentation](../financial_modeling/README.md)
- [Geospatial Analysis Workload Documentation](../geospatial/README.md)
- [IPCC Climate Scenarios](https://www.ipcc.ch/report/ar6/wg1/)
- [NGFS Climate Scenarios](https://www.ngfs.net/en/ngfs-climate-scenarios)
- [Task Force on Climate-related Financial Disclosures (TCFD)](https://www.fsb-tcfd.org/)