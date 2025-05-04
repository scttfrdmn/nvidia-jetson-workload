# Visualization Dashboard Guide

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This guide provides detailed instructions for using the visualization dashboard to monitor workload performance, view benchmark results, and manage deployments.

## Table of Contents

1. [Overview](#overview)
2. [Dashboard Architecture](#dashboard-architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Dashboard Interface](#dashboard-interface)
5. [Workload Visualization](#workload-visualization)
6. [Benchmark Results](#benchmark-results)
7. [Cluster Monitoring](#cluster-monitoring)
8. [Custom Visualizations](#custom-visualizations)
9. [Integration with Other Tools](#integration-with-other-tools)
10. [Troubleshooting](#troubleshooting)

## Overview

The visualization dashboard is a web-based interface that provides real-time monitoring, visualization, and management capabilities for the GPU-accelerated scientific workloads. It integrates with the benchmarking suite to display performance metrics and comparison charts.

## Dashboard Architecture

The dashboard is built with modern web technologies:

- **Frontend**: React with TypeScript
- **UI Components**: Material-UI
- **Charting**: Chart.js for visualization
- **State Management**: React Context API
- **API Integration**: Fetch API for data retrieval
- **Real-time Updates**: Socket.IO for live data

The dashboard follows a modular design with separate components for:
- Main dashboard overview
- Workload-specific visualizations
- Benchmark result analysis
- Cluster monitoring and management
- Job submission and control

## Installation and Setup

### Prerequisites

- Node.js 16+
- npm 8+
- Modern web browser

### Installation

```bash
# Navigate to visualization directory
cd src/visualization

# Install dependencies
npm install

# Start development server
npm start
```

### Production Deployment

```bash
# Build production version
npm run build

# Serve using a web server
npx serve -s build
```

### Configuration

The dashboard can be configured through environment variables:

- `REACT_APP_API_URL`: URL for the backend API
- `REACT_APP_SOCKET_URL`: URL for WebSocket connection
- `REACT_APP_REFRESH_INTERVAL`: Data refresh interval in milliseconds

Create a `.env` file in the `src/visualization` directory:

```
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_SOCKET_URL=http://localhost:5000
REACT_APP_REFRESH_INTERVAL=5000
```

## Dashboard Interface

### Main Dashboard

The main dashboard provides an overview of:
- Running workloads
- Completed workloads
- Performance metrics
- Cluster status
- Quick launch options

![Main Dashboard](images/main-dashboard.png)

### Navigation

The top navigation bar provides access to:
- Dashboard home
- Workload-specific pages
- Benchmark results
- Cluster status
- Slurm job management

### Responsive Design

The dashboard is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile devices

## Workload Visualization

Each workload has a dedicated visualization page with specific charts and controls.

### N-body Simulation Visualization

The N-body simulation visualization shows:
- 3D particle positions and velocities
- Energy conservation over time
- Particle distribution statistics
- System center of mass movement

![N-body Visualization](images/nbody-visualization.png)

### Weather Simulation Visualization

The weather simulation visualization shows:
- Fluid flow velocity fields
- Pressure contours
- Temperature maps
- Precipitation patterns

![Weather Visualization](images/weather-visualization.png)

### Medical Imaging Visualization

The medical imaging visualization shows:
- 2D/3D image slices
- Segmentation results
- Registration overlays
- Filtering effects

![Medical Imaging Visualization](images/medical-visualization.png)

### Interactive Controls

Each visualization includes interactive controls:
- Zoom and pan
- Rotation (for 3D visualizations)
- Time slider for simulations
- Parameter adjustments
- Color mapping options

## Benchmark Results

The benchmark results page provides comprehensive performance analysis with multiple visualization options.

### Performance Charts

- **Execution Time Comparison**: Bar chart comparing execution time across workloads and devices
- **GPU Utilization**: Bar chart showing GPU utilization percentages
- **Memory Usage**: Bar chart showing host and device memory usage
- **Energy Consumption**: Bar chart showing energy consumption in joules
- **Throughput**: Bar chart showing workload-specific throughput metrics

### Performance Radar

The performance radar chart provides a multi-dimensional view of performance across metrics, allowing quick identification of strengths and weaknesses.

![Performance Radar](images/performance-radar.png)

### Hardware Comparison

The hardware comparison table shows relative performance metrics between different hardware configurations:
- Speedup ratios
- Memory efficiency
- Energy efficiency
- Overall performance score

### Filtering and Customization

The benchmark results visualization allows:
- Filtering by workload
- Filtering by device
- Selecting specific metrics
- Adjusting chart settings
- Exporting data in various formats

## Cluster Monitoring

The cluster status page provides real-time monitoring of the Jetson cluster.

### Node Status

For each node in the cluster:
- Online/offline status
- CPU utilization
- GPU utilization
- Memory usage
- Temperature
- Power consumption

![Cluster Status](images/cluster-status.png)

### Resource Allocation

Visual representation of resource allocation:
- CPU cores in use
- GPU memory allocation
- Network utilization
- Storage usage

### Job Queue

Overview of running and pending jobs:
- Job ID and name
- Assigned node
- Start time and duration
- Resource usage
- Status and progress

## Custom Visualizations

You can create custom visualizations for specific analysis needs.

### Creating a Custom Chart

1. Navigate to the "Custom Analysis" page
2. Select data sources (workloads, devices)
3. Choose chart type (bar, line, radar, etc.)
4. Select metrics to display
5. Configure chart options (colors, labels, etc.)
6. Generate and save the chart

### Data Export

Export data for external analysis:
- CSV format for spreadsheets
- JSON format for custom tools
- PNG/SVG for high-quality chart images

## Integration with Other Tools

The dashboard integrates with other tools in the ecosystem.

### Slurm Integration

Submit and manage Slurm jobs directly from the dashboard:
- Create new jobs with custom parameters
- Monitor job status
- View job output
- Cancel or resubmit jobs

### AWS Integration

If deployed on AWS, the dashboard can show AWS-specific metrics:
- EC2 instance status
- Billing information
- Scaling recommendations
- Resource optimization suggestions

### Alerts and Notifications

Configure alerts for:
- Node failures
- Job completions
- Performance thresholds
- System errors

## Troubleshooting

### Common Dashboard Issues

#### Dashboard Not Loading

```
Failed to load dashboard
```

Solutions:
- Check that the development server is running
- Verify network connectivity
- Clear browser cache and reload
- Check for JavaScript console errors

#### No Data Displayed

```
No data available or charts empty
```

Solutions:
- Ensure workloads have been run at least once
- Check API connectivity in network tab
- Verify that benchmark results exist
- Check permissions on result files

#### Visualization Errors

```
Error rendering visualization
```

Solutions:
- Update to the latest browser version
- Enable WebGL for 3D visualizations
- Reduce dataset size for complex visualizations
- Check for dataset format issues

### Browser Compatibility

The dashboard is tested with:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

For best performance, use Chrome or Firefox with WebGL enabled.