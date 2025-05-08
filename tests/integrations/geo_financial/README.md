# Geospatial Financial Integration Tests

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This directory contains integration tests for the Geospatial Financial integration, which combines the Geospatial Analysis and Financial Modeling workloads.

## Test Categories

### Risk Aggregation Tests (`test_risk_aggregation.py`)

Tests for the risk aggregation module, which provides advanced methods for combining multiple risk factors:

- Weighted average aggregation
- Weighted maximum aggregation
- Weighted product aggregation
- Copula-based aggregation (Gaussian and Student's t)
- Risk surface generation and interpolation
- Integration with the geospatial risk model

### Climate Risk Tests (`test_climate_risk.py`)

Tests for the climate risk assessment module, which focuses on climate-related financial risks:

- Climate scenario handling (IPCC and NGFS scenarios)
- Time horizon effects on risk assessment
- Creation of climate hazard risk factors (flood, heat, sea level rise)
- Physical and transition climate risk evaluation
- Combined climate risk assessment
- Climate-adjusted Value-at-Risk (VaR) calculation

### Dashboard Tests (`test_dashboard.py`)

Tests for the interactive dashboard application:

- Dashboard initialization and data loading
- Portfolio creation and risk factor generation
- Risk assessment and portfolio optimization
- Climate scenario comparison
- Dashboard data generation and JSON serialization
- Plot generation and file saving
- Dashboard UI creation (when Dash is available)

## Running Tests

### Run All Geo-Financial Tests

```bash
./run_tests.py
```

### Run a Specific Test Module

```bash
./run_tests.py --test test_risk_aggregation
```

### Run with Verbose Output

```bash
./run_tests.py --verbose
```

## Test Dependencies

These tests require the following dependencies:

- NumPy: For numerical calculations
- Matplotlib: For visualization tests
- SciPy: For some advanced statistical operations
- CuPy (optional): For GPU acceleration tests

Optional dependencies for full testing:

- Dash and Plotly: For dashboard UI tests
- GDAL: For GeoTIFF file handling

## GPU vs. CPU Testing

The tests automatically adapt to the available hardware:

- If CUDA/CuPy is available, they will test GPU-accelerated functionality
- If not, they will test CPU-only functionality
- Some tests may be skipped if specific dependencies are not available