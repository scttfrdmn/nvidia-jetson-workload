# Cost Comparison Guide

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This guide explains how to use the cost comparison features in the NVIDIA Jetson Workload suite to make informed decisions about local versus cloud-based computing for scientific workloads.

## Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Understanding Cost Metrics](#understanding-cost-metrics)
- [Break-Even Analysis](#break-even-analysis)
- [Cloud Instance Reference](#cloud-instance-reference)
- [Example Workflows](#example-workflows)
- [Advanced Usage](#advanced-usage)
  - [Customizing Cloud Instances](#customizing-cloud-instances)
  - [Adjusting Jetson Parameters](#adjusting-jetson-parameters)
  - [DGX Spark System Configuration](#dgx-spark-system-configuration)
  - [Slurm Cluster Configuration](#slurm-cluster-configuration)
  - [Using Configuration Files](#using-configuration-files)
- [Sample Reports](#sample-reports)
- [Methodology](#methodology)

## Overview

The cost comparison framework allows you to:

1. **Compare execution costs** between NVIDIA Jetson devices, cloud providers (AWS, Azure, GCP), DGX systems, and Slurm clusters
2. **Analyze break-even points** to determine when local processing becomes more cost-effective
3. **Measure cost efficiency** in terms of operations per dollar
4. **Project long-term costs** based on your expected workload volume and frequency
5. **Evaluate enterprise options** including DGX systems and custom Slurm clusters

This analysis helps you make data-driven decisions about hardware investments versus cloud service utilization.

## Getting Started

### Prerequisites

- NVIDIA Jetson Workload suite installed (see main [README.md](../../README.md))
- Python 3.8+ with required dependencies

### Basic Usage

Enable cost comparison by adding the `--cost-analysis` flag to your benchmark command:

```bash
# Run all benchmarks with cost analysis
python benchmark/benchmark_suite.py --all --cost-analysis

# Run a specific workload with cost analysis
python benchmark/benchmark_suite.py --nbody --cost-analysis

# Run geospatial benchmarks with cost analysis
python src/geospatial/benchmark/geospatial_benchmark.py --cost-analysis
```

Alternatively, use the convenience script:

```bash
# Run all benchmarks with cost analysis
./benchmark/scripts/run_benchmarks.sh --all --cost-analysis

# Run with DGX and Slurm comparisons
./benchmark/scripts/run_benchmarks.sh --nbody --cost-analysis \
  --dgx-system-type dgx_a100 --slurm-node-type highend_gpu
```

### Output

The cost comparison generates:

1. **Detailed JSON data** with cost metrics for each benchmark
2. **Visual reports** with cost comparison charts
3. **Break-even analysis** showing when local computing becomes more economical

## Understanding Cost Metrics

The framework calculates several key metrics:

### Total Cost

The total cost of running a workload, including:

- **Local (Jetson)**: Hardware amortization + power consumption + maintenance
- **Cloud**: Instance costs + data transfer + storage costs

### Cost Efficiency (Operations per Dollar)

A measure of value showing how many operations you get per dollar spent:

```
Cost Efficiency = Throughput / Total Cost
```

Higher values indicate better cost efficiency.

### Cost Ratio

The ratio of cloud costs to local costs:

```
Cost Ratio = Cloud Cost / Local Cost
```

Values greater than 1.0 indicate that local computing is more cost-effective.

### Per-Hour Costs

The ongoing costs for continuous operation:

- **Local**: Mostly power and periodic maintenance
- **Cloud**: Primarily instance costs (billed per hour/second)

## Break-Even Analysis

Break-even analysis shows how many hours of operation are required before the upfront cost of Jetson hardware is offset by savings compared to cloud computing.

### Interpretation

- **Short break-even time** (days/weeks): Local computing is highly favorable for regular usage
- **Medium break-even time** (months): Local computing makes sense for regular workloads
- **Long break-even time** (years): Cloud computing may be more economical unless you have continuous workloads

### Example

If a Jetson device costs $599 upfront but saves $0.50 per hour compared to cloud computing:

```
Break-Even Hours = $599 / $0.50 = 1,198 hours (about 50 days of 24/7 operation)
```

After 1,198 hours of cumulative usage, the Jetson becomes the more economical option.

## Cloud Instance Reference

### AWS Instances

| Instance Type | vCPUs | Memory | GPU | Use Case | Cost Range |
|---------------|-------|--------|-----|----------|------------|
| g4dn.xlarge   | 4     | 16 GB  | 1 NVIDIA T4 | ML inference, small batch training, video encoding | $0.526/hr |
| g4dn.2xlarge  | 8     | 32 GB  | 1 NVIDIA T4 | ML inference with larger models | $0.752/hr |
| g5.xlarge     | 4     | 16 GB  | 1 NVIDIA A10G | ML training, ML inference | $1.006/hr |
| p3.2xlarge    | 8     | 61 GB  | 1 NVIDIA V100 | ML training, HPC | $3.06/hr |

### Azure Instances

| Instance Type | vCPUs | Memory | GPU | Use Case | Cost Range |
|---------------|-------|--------|-----|----------|------------|
| Standard_NC4as_T4_v3 | 4 | 28 GB | 1 NVIDIA T4 | ML inference, video encoding | $0.526/hr |
| Standard_NC6s_v3 | 6 | 112 GB | 1 NVIDIA V100 | ML training, HPC | $3.06/hr |
| Standard_ND96asr_A100_v4 | 96 | 900 GB | 8 NVIDIA A100 80GB | Large-scale ML training | $32.77/hr |

### GCP Instances

| Instance Type | vCPUs | Memory | GPU | Use Case | Cost Range |
|---------------|-------|--------|-----|----------|------------|
| n1-standard-4-t4 | 4 | 15 GB | 1 NVIDIA T4 | ML inference, video encoding | $0.571/hr |
| n1-standard-8-v100 | 8 | 30 GB | 1 NVIDIA V100 | ML training, HPC | $2.98/hr |
| a2-highgpu-1g | 12 | 85 GB | 1 NVIDIA A100 | ML training, HPC | $4.10/hr |

### Recommendations

- **T4-based instances** (AWS g4dn.xlarge, Azure NC4as_T4_v3) are most comparable to Jetson Orin NX in terms of capabilities
- For workloads requiring more memory, consider instances with higher memory-to-GPU ratios
- For large batch processing, A100-based instances may provide better cost efficiency despite higher hourly costs

## Example Workflows

### Scenario 1: Occasional Processing

For users who run workloads occasionally (e.g., a few hours per week):

```bash
# Run benchmarks most relevant to your workload
python benchmark/benchmark_suite.py --nbody --weather --cost-analysis

# Examine break-even analysis in the report
# If break-even time is > 1 year, cloud computing may be more economical
```

### Scenario 2: Regular Processing

For users who run workloads regularly (e.g., several hours per day):

```bash
# Run full benchmark suite with your expected workload parameters
python benchmark/benchmark_suite.py --all --cost-analysis \
  --nbody-particles 100000 --weather-grid 1024

# Focus on cost efficiency metrics in the report
# If break-even time is < 6 months, local computing is likely more economical
```

### Scenario 3: Continuous Operation

For users who run workloads continuously:

```bash
# Run benchmarks with parameters matching your production workload
python benchmark/benchmark_suite.py --medical --cost-analysis \
  --medical-size 1024 --medical-task segmentation

# Compare per-hour costs in the report
# Local computing is almost always more economical for 24/7 operation
```

### Scenario 4: Enterprise Deployment

For organizations considering enterprise-grade hardware:

```bash
# Compare Jetson cluster with DGX systems and cloud options
./benchmark/scripts/run_benchmarks.sh --all --cost-analysis \
  --dgx-system-type dgx_a100 --dgx-quantity 1 \
  --slurm-node-type jetson_cluster --slurm-nodes 32

# For large-scale deployments, use configuration files
./benchmark/scripts/run_benchmarks.sh --all --cost-analysis \
  --dgx-config benchmark/configs/dgx_superpod.yaml \
  --slurm-config benchmark/configs/slurm_cluster_highend.yaml
```

## Advanced Usage

### Customizing Cloud Instances

You can specify which cloud instances to use for comparison:

```bash
python benchmark/benchmark_suite.py --nbody --cost-analysis \
  --aws-instance g5.xlarge \
  --azure-instance Standard_NC6s_v3 \
  --gcp-instance a2-highgpu-1g
```

### Adjusting Jetson Parameters

The default cost model assumes:
- Hardware cost: $599 (Jetson Orin NX Developer Kit)
- Power cost: $0.12 per kWh
- Amortization period: 3 years
- Maintenance factor: 10% of hardware cost per year

To modify these parameters, you need to edit the `JetsonCostModel` initialization in the `cost_modeling.py` file.

### DGX Spark System Configuration

The benchmark suite includes support for comparing with NVIDIA DGX systems:

```bash
# Basic configuration for DGX A100
python benchmark/benchmark_suite.py --nbody --cost-analysis \
  --dgx-system-type dgx_a100

# Configure DGX quantity
python benchmark/benchmark_suite.py --nbody --cost-analysis \
  --dgx-system-type dgx_h100 --dgx-quantity 2

# Exclude DGX Spark from comparison
python benchmark/benchmark_suite.py --nbody --cost-analysis --no-dgx-spark
```

You can also use a YAML configuration file:

```bash
python benchmark/benchmark_suite.py --nbody --cost-analysis \
  --dgx-config benchmark/configs/dgx_superpod.yaml
```

Available DGX system types:
- `dgx_a100`: Standard DGX A100 with 8x A100 80GB GPUs
- `dgx_h100`: DGX H100 with 8x H100 80GB GPUs
- `dgx_station_a100`: DGX Station A100 with 4x A100 80GB GPUs
- `dgx_station_h100`: DGX Station H100 with 4x H100 80GB GPUs
- `dgx_superpod`: DGX SuperPOD configuration with multiple nodes

### Slurm Cluster Configuration

For comparing with Slurm clusters, use these options:

```bash
# Basic configuration with specific node type
python benchmark/benchmark_suite.py --nbody --cost-analysis \
  --slurm-node-type highend_gpu --slurm-nodes 16

# Exclude Slurm cluster from comparison
python benchmark/benchmark_suite.py --nbody --cost-analysis --no-slurm-cluster
```

Available Slurm node types:
- `basic_cpu`: Standard CPU-only compute nodes
- `basic_gpu`: Standard nodes with 1x NVIDIA T4 GPU
- `highend_gpu`: High-performance nodes with 4x NVIDIA A100 GPUs
- `jetson_cluster`: Cluster built from NVIDIA Jetson Orin modules
- `custom`: Custom node configuration (requires custom config file)

You can also use a YAML configuration file for detailed cluster specs:

```bash
python benchmark/benchmark_suite.py --nbody --cost-analysis \
  --slurm-config benchmark/configs/slurm_cluster_custom.yaml
```

### Using Configuration Files

The benchmark suite provides example configuration files in the `benchmark/configs/` directory:

```
benchmark/configs/
├── README.md                   # Configuration guide
├── dgx_a100.yaml               # DGX A100 configuration
├── dgx_h100.yaml               # DGX H100 configuration
├── dgx_station_a100.yaml       # DGX Station A100 configuration
├── dgx_superpod.yaml           # DGX SuperPOD configuration
├── slurm_cluster_basic.yaml    # Basic Slurm cluster
├── slurm_cluster_custom.yaml   # Custom Slurm cluster
├── slurm_cluster_highend.yaml  # High-end Slurm cluster
└── slurm_cluster_jetson.yaml   # Jetson-based Slurm cluster
```

You can create your own configuration files by copying and modifying these examples.

## Methodology

### Local Computing Cost Models

#### Jetson Cost Model

The Jetson cost model includes:

1. **Hardware Amortization**:
   ```
   Daily Hardware Cost = Hardware Cost / Amortization Period (days)
   ```

2. **Power Consumption**:
   ```
   Power Cost = Energy Consumption (kWh) * Cost per kWh
   ```
   
   If energy consumption data is not available, it's estimated based on GPU utilization:
   ```
   Power (watts) = Max Power * (0.3 + 0.7 * Utilization Factor)
   Energy (kWh) = Power * Time / 3600000
   ```

3. **Maintenance**:
   ```
   Daily Maintenance Cost = (Hardware Cost * Maintenance Factor) / 365
   ```

4. **Total Cost**:
   ```
   Total Cost = Hardware Cost (amortized for execution time) + 
               Maintenance Cost + 
               Power Cost
   ```

#### Slurm Cluster Cost Model

The Slurm cluster cost model includes:

1. **Hardware Amortization**:
   ```
   Hourly Hardware Cost = (Nodes * Cost Per Node + Network Cost) / Amortization Period / 24
   ```

2. **Power Consumption**:
   ```
   Power Cost = Energy Consumption (kWh) * Cost per kWh
   ```
   
   Estimated based on node power and utilization:
   ```
   Power (watts) = Power Per Node * Nodes Used * (0.4 + 0.6 * Utilization Factor)
   ```

3. **Maintenance & Administration**:
   ```
   Hourly Maintenance Cost = (Hardware Cost * Maintenance Factor) / (365 * 24)
   Hourly Admin Cost = Admin Cost Per Year / (365 * 24)
   ```

4. **Total Cost**:
   ```
   Total Cost = Hardware Cost + Maintenance Cost + Admin Cost + Power Cost
   ```

5. **Node Allocation**:
   The model estimates how many nodes are needed based on memory requirements or user-specified values.

#### DGX System Cost Model

The DGX system cost model includes:

1. **Hardware Amortization**:
   ```
   Hourly Hardware Cost = (System Base Cost * Quantity + Network Cost) / Amortization Period / 24
   ```
   
   For DGX SuperPOD:
   ```
   System Cost = Per Node Cost * Number of Nodes + Base Infrastructure Cost
   ```

2. **Power Consumption with Datacenter Overhead**:
   ```
   Power Cost = Energy Consumption (kWh) * Cost per kWh * (1 + Datacenter Overhead Factor)
   ```

3. **Maintenance & Administration**:
   ```
   Hourly Maintenance Cost = (Hardware Cost * Maintenance Factor) / (365 * 24)
   Hourly Admin Cost = Admin Cost Per Year / (365 * 24)
   ```

4. **Total Cost**:
   ```
   Total Cost = Hardware Cost + Maintenance Cost + Admin Cost + Power Cost
   ```

### Cloud Computing Cost Model

The cloud computing cost model includes:

1. **Instance Cost**:
   ```
   Instance Cost = Instance Hourly Rate * (Execution Time / 3600)
   ```
   With a minimum billing time of 60 seconds (varies by provider).

2. **Storage Cost**:
   ```
   Storage GB = (Host Memory + Device Memory) / 1024
   Storage Cost = Storage GB * Storage Rate ($/GB-month) * (1/720) hours
   ```

3. **Data Transfer Cost**:
   ```
   Data Transfer Cost = Storage GB * Data Transfer Rate ($/GB)
   ```

4. **Total Cost**:
   ```
   Total Cost = Instance Cost + Storage Cost + Data Transfer Cost
   ```

### Break-Even Calculation

Break-even time is calculated as:

```
Break-Even Hours = Jetson Hardware Cost / (Cloud Hourly Cost - Jetson Hourly Cost)
```

Where:
- Jetson Hourly Cost = Power Cost + Maintenance Cost (per hour)
- Cloud Hourly Cost = Instance Cost + Storage Cost + Data Transfer Cost (per hour)

If Cloud Hourly Cost < Jetson Hourly Cost, there is no break-even point (cloud is always cheaper).

## Sample Reports

The following sample reports demonstrate how to interpret and use the cost comparison data:

- [Sample Cost Comparison Report](sample-reports/cost-comparison-sample.md) - Example reports with interpretation
- [Detailed Cost Analysis Example](sample-reports/detailed-cost-analysis-example.md) - Step-by-step walkthrough of cost calculations
- [Interpreting Cost Results](sample-reports/interpreting-cost-results.md) - Guide to understanding metrics and visualizations

For sample configuration files, refer to:

- [DGX Custom Configuration](sample-configs/dgx_custom.yaml) - Example DGX system configuration
- [Slurm Cluster Configuration](sample-configs/slurm_cluster_custom.yaml) - Example Slurm cluster configuration

## Further Reading

- [Benchmarking Guide](benchmarking.md) - More details on running benchmarks
- [Deployment Guide](deployment.md) - Information on deploying workloads
- [AWS Graviton Cost Optimization](https://aws.amazon.com/blogs/compute/optimizing-costs-for-arm-based-applications-with-aws-graviton-ec2-instances/) - AWS guide on Graviton cost optimization

---

For questions or feedback on the cost comparison feature, please [create an issue](https://github.com/scttfrdmn/nvidia-jetson-workload/issues) in the project repository.