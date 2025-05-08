# Integration Tests

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This directory contains integration tests for cross-workload integrations in the NVIDIA Jetson Workload project.

## Overview

Integration tests verify that different workloads and their components work together correctly. These tests focus on the interactions between different workloads and ensure that the cross-workload functionality works as expected.

## Integrations

### Geospatial Financial Integration

Tests for the integration between the Geospatial Analysis and Financial Modeling workloads, focused on geospatial financial risk analysis.

- **Risk Aggregation Tests**: Verify the functionality of various risk aggregation methods, including weighted averages, weighted maximums, weighted products, and copula-based methods.
- **Climate Risk Tests**: Test the climate risk assessment functionality, including physical and transition risk evaluation.
- **Dashboard Tests**: Verify the interactive dashboard application works correctly and integrates with other components.

### Running Tests

#### Run All Integration Tests

```bash
./run_all_tests.py
```

#### Run Tests for a Specific Integration

```bash
./run_all_tests.py --integration geo_financial
```

#### Run a Specific Test Module

```bash
cd geo_financial
./run_tests.py --test test_risk_aggregation
```

## Adding New Integration Tests

To add tests for a new integration:

1. Create a directory for the integration: `mkdir <integration_name>`
2. Add an `__init__.py` file to make it a package
3. Create test files with the `test_*.py` naming convention
4. Optionally add a `run_tests.py` script for the integration

## Best Practices

- Focus on testing the interactions between different workloads
- Test with realistic data that represents typical usage
- Verify that GPU acceleration works correctly (when available)
- Include fallback mechanisms for CPU-only environments
- Test with different configurations and scenarios