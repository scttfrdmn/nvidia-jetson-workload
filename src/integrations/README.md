# Cross-Workload Integrations

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This directory contains cross-workload integrations that demonstrate how different workloads in the NVIDIA Jetson Workload project can be combined to create powerful, integrated solutions.

## Available Integrations

### [Geospatial Financial Risk Analysis](geo_financial/)

Integration between Financial Modeling and Geospatial Analysis workloads for comprehensive geospatial financial risk assessment.

**Key Features:**
- Spatial risk factors (flood risk, landslide risk) based on terrain data
- Asset location mapping with geospatial risk exposure
- Portfolio optimization considering geospatial risks
- Data connectors for joining geospatial and financial data
- GPU-accelerated analysis with adaptability pattern
- Comprehensive visualization dashboards

**Applications:**
- Insurance companies assessing risk exposure to natural disasters
- Real estate investment trusts (REITs) optimizing property portfolios
- Infrastructure investors evaluating project locations
- Asset managers incorporating climate change risks into investment decisions

## Running Integrations

Each integration includes detailed documentation on how to run it. Most integrations provide convenient scripts in the `/scripts` directory for easy execution.

For example, to run the Geospatial Financial Risk Analysis:

```bash
./scripts/run_geo_financial_analysis.sh
```

## Creating New Integrations

To create a new integration:

1. Create a subdirectory in `/src/integrations/` with a descriptive name
2. Implement integration modules that import from existing workloads
3. Create a README.md file documenting the integration
4. Add a convenient run script to `/scripts/`
5. Update documentation to reference the new integration

## Benefits of Cross-Workload Integration

The integrations demonstrate several benefits of the NVIDIA Jetson Workload architecture:

1. **Code Reusability**: Leveraging existing, well-tested components from individual workloads
2. **GPU Adaptability**: Seamlessly using the GPU adaptability pattern across workloads
3. **Performance Synergy**: Combining optimized components for even better overall performance
4. **Domain Crossing**: Creating solutions that span multiple application domains
5. **Demonstration of Extensibility**: Showing how the architecture can grow beyond its initial scope

## License

SPDX-License-Identifier: Apache-2.0  
SPDX-FileCopyrightText: 2025 Scott Friedman and Project Contributors