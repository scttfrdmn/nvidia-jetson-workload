# Geospatial Financial Risk Analysis Integration

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This module integrates the Financial Modeling and Geospatial Analysis workloads to create a comprehensive geospatial financial risk analysis solution.

## Overview

The Geospatial Financial Risk Analysis integration enables:

- Analyzing geospatial risk factors in financial portfolios
- Assessing risk exposure to natural hazards for investments
- Optimizing portfolios to minimize exposure to geospatial risks
- Visualizing financial data with geospatial context
- Leveraging GPU acceleration for both financial and geospatial computations
- Advanced risk aggregation using copulas and other statistical methods
- Sensitivity analysis for risk factors
- Risk surface interpolation from point data
- Climate risk assessment with standardized scenarios (IPCC, NGFS)
- Physical and transition climate risk evaluation
- Interactive dashboards for portfolio risk analysis and optimization

## Key Components

The integration includes the following components:

### `geo_risk` Module

- `SpatialRiskFactor`: Represents a geospatial risk factor (e.g., flood risk, landslide risk)
- `GeospatialRiskModel`: Combines multiple risk factors into a comprehensive model
- `GeospatialPortfolio`: Financial portfolio with geospatial asset locations
- Utility functions for creating risk factors from elevation, slope, and other data

### `risk_aggregation` Module

- `RiskAggregator`: Advanced methods for aggregating multiple risk factors
- `RiskSurfaceGenerator`: Generates continuous risk surfaces from point data
- `AggregationMethod`: Supported aggregation methods (weighted average, max, product, copula)
- Utility functions for creating combined risk factors and interpolated surfaces
- Statistical methods for modeling risk dependencies using copulas

### `climate_risk_assessment` Module

- `ClimateRiskAssessor`: Specialized class for assessing climate-related financial risks
- `ClimateScenario`: Standardized climate scenarios (IPCC SSPs, NGFS scenarios)
- `TimeHorizon`: Different time horizons for climate projections (short, medium, long-term)
- `ClimateHazardType`: Types of physical climate hazards (flooding, heat stress, etc.)
- `TransitionRiskType`: Types of climate transition risks (policy, legal, market, etc.)
- Utility functions for creating specialized climate risk factors
- Methods for scenario-based analysis and climate-adjusted Value-at-Risk (VaR)

### `dashboard_app` Module

- Interactive web-based dashboard for geospatial financial risk analysis
- Configurable climate scenarios and time horizons for risk assessment
- Portfolio optimization with risk-adjusted weights
- Climate scenario comparison and stress testing
- Integration of physical and transition climate risks
- Export capabilities for results and visualizations
- Headless operation mode for automated reporting

### `data_connectors` Module

- `AssetLocationDataLoader`: Loads financial asset data with geospatial locations
- `FinancialDataLoader`: Loads financial time series data (returns, prices)
- `GeoRiskDataLoader`: Loads and processes geospatial data for risk analysis
- Data conversion and transformation utilities

### `visualization` Module

- `GeoFinancialVisualizer`: Creates visualizations for geospatial financial analysis
- Risk map visualization with asset overlays
- Portfolio optimization visualizations
- Comprehensive dashboards combining financial and geospatial data

### `generate_test_data` Module

- Utilities for generating synthetic test data for development and testing
- Creates digital elevation models (DEMs) with realistic terrain features
- Generates asset location data with sector-based distributions
- Produces correlated financial returns data by sector

## Usage

The integration can be used through the provided Jupyter notebook, example scripts, or by importing the modules directly in Python.

### Jupyter Notebook

The `geospatial_financial_analysis.ipynb` notebook provides a comprehensive demonstration of the integration, including:

1. Loading and preparing geospatial and financial data
2. Creating spatial risk factors based on elevation and slope
3. Building a geospatial risk model
4. Analyzing portfolio risk using both financial and geospatial factors
5. Optimizing portfolios to minimize exposure to geospatial risks
6. Visualizing the results through comprehensive dashboards
7. Comparing performance between CPU and GPU executions

### Example Scripts

#### Generate Test Data

The `generate_test_data.py` script creates synthetic data for testing and development:

```bash
python -m src.integrations.geo_financial.generate_test_data \
  --output-dir data/geo_financial \
  --dem-size 500 \
  --num-assets 20 \
  --days 252 \
  --seed 42
```

#### Advanced Risk Analysis

The `examples/advanced_risk_analysis.py` script demonstrates advanced risk aggregation and analysis techniques:

```bash
python -m src.integrations.geo_financial.examples.advanced_risk_analysis \
  --data-dir data/geo_financial \
  --output-dir results/geo_financial \
  --device-id 0 \
  --num-assets 30 \
  --seed 42
```

This script provides examples of:
- Comparing different risk aggregation methods (weighted average, maximum, product, copula)
- Performing sensitivity analysis on risk factors
- Generating risk surfaces through interpolation
- Creating advanced visualizations of risk analysis results

#### Climate Risk Analysis

The `examples/climate_risk_analysis.py` script demonstrates how to assess climate-related financial risks:

```bash
python -m src.integrations.geo_financial.examples.climate_risk_analysis \
  --data-dir data/geo_financial \
  --output-dir results/geo_financial/climate \
  --device-id 0 \
  --num-assets 30 \
  --seed 42
```

This script provides examples of:
- Creating climate hazard risk factors (flooding, heat stress, sea level rise) 
- Comparing risks across different climate scenarios (SSPs, NGFS)
- Assessing risks over multiple time horizons (short, medium, long-term)
- Evaluating both physical and transition climate risks
- Calculating climate-adjusted Value-at-Risk (VaR)
- Visualizing climate risk impacts on sector-based portfolios

#### Interactive Dashboard

The `dashboard_app.py` script provides an interactive dashboard for geospatial financial risk analysis:

```bash
python -m src.integrations.geo_financial.dashboard_app \
  --data-dir data/geo_financial \
  --output-dir results/geo_financial/dashboard \
  --device-id 0 \
  --generate-data \
  --num-assets 30
```

This dashboard application includes:
- Interactive visualization of geospatial risk factors and asset exposures
- Portfolio optimization with risk-adjusted weights
- Climate scenario analysis with configurable parameters
- Comparison of physical and transition climate risks by sector
- Climate-adjusted financial metrics and Value-at-Risk (VaR)
- Ability to export results and visualizations

The dashboard requires additional Python packages:
```bash
pip install dash dash-bootstrap-components plotly
```

For headless operation (generating data and visualizations without launching the dashboard):
```bash
python -m src.integrations.geo_financial.dashboard_app \
  --headless \
  --output-dir results/geo_financial/dashboard
```

### Python API

#### Basic Risk Analysis

```python
from src.integrations.geo_financial.geo_risk import GeospatialRiskModel, GeospatialPortfolio
from src.integrations.geo_financial.data_connectors import AssetLocationDataLoader
from src.integrations.geo_financial.visualization import GeoFinancialVisualizer

# Create geospatial risk model
risk_model = GeospatialRiskModel(device_id=0)  # Use GPU if available

# Add risk factors
risk_model.add_risk_factor(create_elevation_risk_factor('path/to/dem.tif'))
risk_model.add_risk_factor(create_slope_risk_factor('path/to/dem.tif'))

# Create geospatial portfolio
portfolio = GeospatialPortfolio(device_id=0)

# Load asset data
asset_loader = AssetLocationDataLoader()
assets_df = asset_loader.load_asset_csv(
    file_path='path/to/assets.csv',
    id_col='id',
    name_col='name',
    value_col='value',
    x_col='x',
    y_col='y'
)

# Add assets to portfolio
portfolio.add_assets_from_dataframe(assets_df, 'id', 'name', 'value', 'x', 'y')

# Assess geospatial risk
risk_scores = portfolio.assess_risk(risk_model)

# Optimize portfolio to minimize geospatial risk
optimized_weights = portfolio.optimize_for_geo_risk(
    risk_model=risk_model,
    target_return=0.0005  # 0.05% daily return
)

# Visualize results
visualizer = GeoFinancialVisualizer()
fig = visualizer.plot_portfolio_optimization_results(
    portfolio=portfolio,
    original_weights={a['id']: a['value']/total_value for a in portfolio.assets},
    optimized_weights=optimized_weights,
    risk_scores=risk_scores
)
```

#### Advanced Risk Aggregation

```python
from src.integrations.geo_financial.geo_risk import SpatialRiskFactor
from src.integrations.geo_financial.risk_aggregation import (
    RiskAggregator,
    AggregationMethod,
    create_combined_risk_factor
)

# Create risk factors
flood_risk = create_elevation_risk_factor('path/to/dem.tif', risk_weight=0.5)
landslide_risk = create_slope_risk_factor('path/to/dem.tif', risk_weight=0.3)

# Create risk aggregator
aggregator = RiskAggregator(device_id=0)  # Use GPU if available

# Compare different aggregation methods
risk_factors = [flood_risk, landslide_risk]
weighted_avg, stats = aggregator.aggregate_risk_factors(
    risk_factors=risk_factors,
    method=AggregationMethod.WEIGHTED_AVERAGE
)

# Create combined risk factor using Gaussian copula
combined_risk = create_combined_risk_factor(
    risk_factors=risk_factors,
    name="Combined Natural Hazard Risk",
    description="Comprehensive natural hazard risk",
    method=AggregationMethod.COPULA_GAUSSIAN,
    device_id=0
)

# Create geospatial risk model with combined risk
risk_model = GeospatialRiskModel(device_id=0)
risk_model.add_risk_factor(combined_risk)
```

## Applications

The integration is useful for various applications:

- **Insurance Companies**: Assess risk exposure to natural disasters and climate hazards
- **Real Estate Investment Trusts (REITs)**: Optimize property portfolios based on geospatial risks
- **Infrastructure Investors**: Evaluate project locations considering environmental hazards
- **Asset Managers**: Incorporate climate change risks into investment decisions
- **Government Agencies**: Allocate resources for disaster mitigation and resilience
- **Banks and Financial Institutions**: Perform climate stress testing and scenario analysis
- **Pension Funds**: Evaluate long-term climate risks in investment portfolios
- **ESG Investors**: Assess physical and transition climate risks in sustainability-focused portfolios
- **Regulatory Compliance**: Meet emerging climate risk disclosure requirements (TCFD, etc.)
- **Credit Rating Agencies**: Incorporate climate factors into credit risk assessments

## GPU Acceleration

This integration leverages the GPU adaptability pattern from both workloads to provide efficient performance on:

- NVIDIA Jetson Orin NX (SM 8.7)
- AWS Graviton g5g with NVIDIA T4 GPU (SM 7.5)
- Other CUDA-capable GPUs

The integration automatically selects the optimal code paths based on the available hardware, with fallback to CPU execution when no GPU is available.

## Requirements

- Geospatial Analysis workload
- Financial Modeling workload
- NumPy, Pandas, Matplotlib
- Jupyter (for the interactive notebook)

## License

SPDX-License-Identifier: Apache-2.0  
SPDX-FileCopyrightText: 2025 Scott Friedman and Project Contributors