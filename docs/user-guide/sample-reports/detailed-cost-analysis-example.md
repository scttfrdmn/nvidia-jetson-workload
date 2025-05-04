# Detailed Cost Analysis Example

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This document provides a detailed walk-through of the cost comparison process using a concrete example. We'll analyze the N-body simulation workload with 100,000 particles across different computing environments.

## Workload Specification

The benchmark parameters for this example:

- **Workload**: N-body gravitational simulation
- **Particle Count**: 100,000
- **Simulation Steps**: 1,000
- **Visualization**: Disabled
- **Execution Time**: 53.4 seconds on Jetson Orin NX
- **GPU Utilization**: 87% on Jetson Orin NX
- **Memory Usage**: Host: 245 MB, Device: 1,228 MB
- **Throughput**: 18.7 iterations/second

## Step 1: Jetson Cost Calculation

The Jetson cost model includes hardware amortization, power consumption, and maintenance costs:

```python
# Jetson Model Parameters
hardware_cost = 599.0  # Jetson Orin NX Developer Kit cost in USD
power_cost = 0.12  # Cost per kWh in USD
amortization_period = 1095  # 3 years in days
maintenance_factor = 0.1  # 10% of hardware cost per year for maintenance
max_power_watts = 25.0  # Maximum power consumption in watts

# Calculate hardware cost (amortized for execution time)
execution_days = 53.4 / (24 * 3600)  # Convert 53.4 seconds to days
hardware_cost_amortized = (hardware_cost / amortization_period) * execution_days
# = (599.0 / 1095) * (53.4 / 86400) = $0.00029

# Calculate maintenance cost for execution period
maintenance_cost = (hardware_cost * maintenance_factor / 365) * execution_days
# = (599.0 * 0.1 / 365) * (53.4 / 86400) = $0.00002

# Calculate power cost with 87% GPU utilization
power_watts = max_power_watts * (0.3 + 0.7 * 0.87)  # Base load + utilization-dependent load
# = 25.0 * (0.3 + 0.7 * 0.87) = 22.7 watts
energy_kwh = (power_watts * 53.4) / 3600000  # Convert watt-seconds to kWh
# = (22.7 * 53.4) / 3600000 = 0.000336 kWh
power_cost_execution = energy_kwh * power_cost
# = 0.000336 * 0.12 = $0.00004

# Total Jetson cost
total_jetson_cost = hardware_cost_amortized + maintenance_cost + power_cost_execution
# = $0.00029 + $0.00002 + $0.00004 = $0.00035
```

Hourly cost for continuous operation:
```python
# Daily hardware amortization
daily_hardware_cost = hardware_cost / amortization_period
# = 599.0 / 1095 = $0.547 per day

# Daily maintenance
daily_maintenance = hardware_cost * maintenance_factor / 365
# = 599.0 * 0.1 / 365 = $0.164 per day

# Hourly costs
hourly_hardware_cost = daily_hardware_cost / 24
# = $0.547 / 24 = $0.0228 per hour
hourly_maintenance_cost = daily_maintenance / 24
# = $0.164 / 24 = $0.0068 per hour

# Hourly power consumption at average utilization (50%)
avg_power_watts = max_power_watts * (0.3 + 0.7 * 0.5)
# = 25.0 * (0.3 + 0.7 * 0.5) = 16.25 watts
hourly_energy_kwh = avg_power_watts / 1000  # Convert watts to kW for 1 hour
# = 16.25 / 1000 = 0.01625 kWh
hourly_power_cost = hourly_energy_kwh * power_cost
# = 0.01625 * 0.12 = $0.00195 per hour

# Total hourly cost for Jetson
hourly_jetson_cost = hourly_power_cost + hourly_maintenance_cost
# = $0.00195 + $0.0068 = $0.00875 per hour
```

## Step 2: Cloud Cost Calculation

For AWS g4dn.xlarge instance:

```python
# AWS Model Parameters
instance_cost = 0.526  # Cost per hour for g4dn.xlarge
storage_cost = 0.08  # Cost per GB-month for storage
data_transfer_cost = 0.09  # Cost per GB for data transfer
minimum_billing_time = 60.0  # Minimum billing time in seconds

# Calculate billable time
billable_time = max(53.4, minimum_billing_time)
# = max(53.4, 60.0) = 60.0 seconds

# Calculate instance cost
instance_cost_execution = instance_cost * (billable_time / 3600)
# = 0.526 * (60.0 / 3600) = $0.00877

# Calculate storage cost
storage_gb = (245 + 1228) / 1024  # Convert MB to GB
# = 1473 / 1024 = 1.438 GB
hourly_storage_rate = storage_cost / (30 * 24)  # Convert GB-month to GB-hour
# = 0.08 / 720 = $0.000111 per GB-hour
storage_cost_execution = storage_gb * hourly_storage_rate * (billable_time / 3600)
# = 1.438 * 0.000111 * (60.0 / 3600) = $0.000003

# Calculate data transfer cost
data_transfer_gb = storage_gb  # Assume all data is transferred out
# = 1.438 GB
data_transfer_cost_execution = data_transfer_gb * data_transfer_cost
# = 1.438 * 0.09 = $0.12942

# Total AWS cost
total_aws_cost = instance_cost_execution + storage_cost_execution + data_transfer_cost_execution
# = $0.00877 + $0.000003 + $0.12942 = $0.13819
```

Hourly AWS cost:
```python
hourly_aws_cost = instance_cost + (storage_cost / 720) * storage_gb + data_transfer_cost * storage_gb
# = $0.526 + (0.08 / 720) * 1.438 + 0.09 * 1.438
# = $0.526 + $0.00016 + $0.12942 = $0.65558 per hour
```

## Step 3: DGX Spark System Cost Calculation

For a single DGX A100 system:

```python
# DGX A100 Parameters
system_base_cost = 199000.0  # Base cost for DGX A100
quantity = 1
power_cost = 0.12  # Cost per kWh in USD
amortization_period = 1095  # 3 years in days
maintenance_factor = 0.15  # 15% of hardware cost per year for maintenance
utilization_factor = 0.5  # Average system utilization
networking_infrastructure_cost = 20000.0  # Additional networking infrastructure
datacenter_overhead_factor = 0.2  # Datacenter overhead (cooling, space)
admin_cost_per_year = 120000.0  # System administrator cost per year
max_power_watts = 6500.0  # Maximum power consumption in watts

# Calculate total hardware cost
hardware_cost = (system_base_cost * quantity) + networking_infrastructure_cost
# = (199000.0 * 1) + 20000.0 = $219000.0

# Hourly amortized hardware cost
hourly_hardware_cost = (hardware_cost / amortization_period) / 24
# = (219000.0 / 1095) / 24 = $8.334 per hour

# Hourly maintenance cost
hourly_maintenance_cost = (hardware_cost * maintenance_factor) / (365 * 24)
# = (219000.0 * 0.15) / (365 * 24) = $3.75 per hour

# Hourly admin cost
hourly_admin_cost = admin_cost_per_year / (365 * 24)
# = 120000.0 / (365 * 24) = $13.70 per hour

# For execution: amortized costs considering utilization
execution_hours = 53.4 / 3600  # Convert seconds to hours
# = 53.4 / 3600 = 0.01483 hours
hardware_cost_execution = hourly_hardware_cost * execution_hours / utilization_factor
# = $8.334 * 0.01483 / 0.5 = $0.2471
maintenance_cost_execution = hourly_maintenance_cost * execution_hours / utilization_factor
# = $3.75 * 0.01483 / 0.5 = $0.1112
admin_cost_execution = hourly_admin_cost * execution_hours / utilization_factor
# = $13.70 * 0.01483 / 0.5 = $0.4064

# Power cost with 90% GPU utilization
power_watts = max_power_watts * (0.3 + 0.7 * 0.9)
# = 6500.0 * (0.3 + 0.7 * 0.9) = 6045.0 watts
energy_kwh = (power_watts * 53.4) / 3600000  # Convert watt-seconds to kWh
# = (6045.0 * 53.4) / 3600000 = 0.08977 kWh
power_cost_with_overhead = (energy_kwh * power_cost) * (1 + datacenter_overhead_factor)
# = (0.08977 * 0.12) * 1.2 = $0.01293

# Total DGX cost
total_dgx_cost = hardware_cost_execution + maintenance_cost_execution + admin_cost_execution + power_cost_with_overhead
# = $0.2471 + $0.1112 + $0.4064 + $0.01293 = $0.77763
```

## Step 4: Slurm Cluster Cost Calculation

For a 16-node Jetson-based Slurm cluster:

```python
# Slurm Cluster Parameters (Jetson-based)
nodes = 16
cost_per_node = 800.0  # Cost per Jetson node
power_per_node = 30.0  # Power per node in watts
power_cost = 0.12  # Cost per kWh in USD
amortization_period = 1095  # 3 years in days
maintenance_factor = 0.1  # 10% of hardware cost per year for maintenance
utilization_factor = 0.7  # Average cluster utilization
network_cost = 5000.0  # Network infrastructure cost
admin_cost_per_year = 10000.0  # System administrator cost per year

# Total hardware cost
hardware_cost = (nodes * cost_per_node) + network_cost
# = (16 * 800.0) + 5000.0 = $17800.0

# Hourly amortized hardware cost
hourly_hardware_cost = (hardware_cost / amortization_period) / 24
# = (17800.0 / 1095) / 24 = $0.678 per hour

# Hourly maintenance cost
hourly_maintenance_cost = (hardware_cost * maintenance_factor) / (365 * 24)
# = (17800.0 * 0.1) / (365 * 24) = $0.203 per hour

# Hourly admin cost
hourly_admin_cost = admin_cost_per_year / (365 * 24)
# = 10000.0 / (365 * 24) = $1.142 per hour

# Determine number of nodes used for this workload
# Based on memory requirements: 1.4 GB needed, 16 GB per node
nodes_used = ceil(1.4 / 16) = 1

# Allocation factor
allocation_factor = nodes_used / nodes
# = 1 / 16 = 0.0625

# For execution: amortized costs with allocation and utilization
execution_hours = 53.4 / 3600
# = 53.4 / 3600 = 0.01483 hours
hardware_cost_execution = hourly_hardware_cost * execution_hours * allocation_factor / utilization_factor
# = $0.678 * 0.01483 * 0.0625 / 0.7 = $0.000898
maintenance_cost_execution = hourly_maintenance_cost * execution_hours * allocation_factor / utilization_factor
# = $0.203 * 0.01483 * 0.0625 / 0.7 = $0.000269
admin_cost_execution = hourly_admin_cost * execution_hours * allocation_factor / utilization_factor
# = $1.142 * 0.01483 * 0.0625 / 0.7 = $0.001512

# Power cost for nodes used (87% GPU utilization)
power_watts = power_per_node * nodes_used * (0.4 + 0.6 * 0.87)
# = 30.0 * 1 * (0.4 + 0.6 * 0.87) = 27.66 watts
energy_kwh = (power_watts * 53.4) / 3600000
# = (27.66 * 53.4) / 3600000 = 0.000411 kWh
power_cost_execution = energy_kwh * power_cost
# = 0.000411 * 0.12 = $0.0000493

# Total Slurm cluster cost
total_slurm_cost = hardware_cost_execution + maintenance_cost_execution + admin_cost_execution + power_cost_execution
# = $0.000898 + $0.000269 + $0.001512 + $0.0000493 = $0.00273
```

## Step 5: Cost Comparison Metrics

Let's calculate key comparison metrics:

### Cost Ratios (Cloud vs. Jetson)

```python
aws_ratio = total_aws_cost / total_jetson_cost
# = $0.13819 / $0.00035 = 394.8x

azure_ratio = total_azure_cost / total_jetson_cost
# = $0.13926 / $0.00035 = 397.9x (Azure slightly more expensive)

gcp_ratio = total_gcp_cost / total_jetson_cost
# = $0.13671 / $0.00035 = 390.6x
```

### Cost per Operation

With throughput of 18.7 iterations/second:

```python
total_operations = 18.7 * 53.4
# = 998.58 operations

jetson_cost_per_op = total_jetson_cost / total_operations
# = $0.00035 / 998.58 = $0.00000035 per operation

aws_cost_per_op = total_aws_cost / total_operations
# = $0.13819 / 998.58 = $0.000138 per operation

dgx_cost_per_op = total_dgx_cost / total_operations
# = $0.77763 / 998.58 = $0.000779 per operation

slurm_cost_per_op = total_slurm_cost / total_operations
# = $0.00273 / 998.58 = $0.0000027 per operation
```

### Break-Even Analysis

```python
# Jetson hardware cost
jetson_hardware_cost = 599.0

# Hourly cost difference (cloud - Jetson)
aws_hourly_diff = hourly_aws_cost - hourly_jetson_cost
# = $0.65558 - $0.00875 = $0.64683 per hour

# Break-even time (hours)
aws_break_even_hours = jetson_hardware_cost / aws_hourly_diff
# = $599.0 / $0.64683 = 926 hours ≈ 38.6 days

# For DGX (if assuming 50% utilization on both)
dgx_hourly_cost = hourly_hardware_cost + hourly_maintenance_cost + hourly_admin_cost + hourly_power_dgx
# = $8.334 + $3.75 + $13.70 + $0.936 = $26.72 per hour

dgx_hourly_diff = dgx_hourly_cost - hourly_jetson_cost
# = $26.72 - $0.00875 = $26.71 per hour

dgx_break_even_hours = (system_base_cost - jetson_hardware_cost) / dgx_hourly_diff
# = ($199000.0 - $599.0) / $26.71 = 7,426 hours ≈ 309 days
```

## Step 6: Visualization

The cost comparison generates several visualizations:

1. **Cost Comparison Bar Chart**:
   - Shows the absolute cost for each computing environment
   - Jetson: $0.00035
   - AWS: $0.13819
   - Azure: $0.13926
   - GCP: $0.13671
   - DGX A100: $0.77763
   - Slurm Cluster: $0.00273

2. **Cost Ratio Chart**:
   - Shows relative cost compared to Jetson
   - AWS: 394.8x
   - Azure: 397.9x
   - GCP: 390.6x
   - DGX A100: 2,221.8x
   - Slurm Cluster: 7.8x

3. **Break-Even Analysis Chart**:
   - AWS: 926 hours (38.6 days)
   - Azure: 942 hours (39.3 days)
   - GCP: 913 hours (38.0 days)
   - DGX A100: 7,426 hours (309 days)
   - Slurm Cluster: 2,165 hours (90.2 days)

## Step 7: Interpretation and Decision Making

Based on this analysis, we can make the following observations:

1. **For small workloads on a single device**:
   - Jetson is by far the most cost-effective option at $0.00035 per execution
   - Cloud solutions cost approximately 400x more for this specific workload
   - Break-even with cloud occurs after approximately 40 days of regular use

2. **For scaled enterprise workloads**:
   - A Jetson-based Slurm cluster offers excellent cost efficiency at 7.8x the cost of a single Jetson
   - DGX systems are significantly more expensive but offer massively higher computational density
   - For this specific workload, Jetson-based solutions are more cost-effective

3. **Decision factors beyond cost**:
   - **Time sensitivity**: If results are needed very quickly, DGX systems can complete workloads faster
   - **Scale of data**: For very large datasets, DGX systems offer more memory and I/O bandwidth
   - **Operational constraints**: Cloud solutions require no hardware management
   - **Deployment timeline**: Cloud solutions offer immediate access versus hardware procurement time

## Conclusion

The cost comparison analysis provides a comprehensive view of the financial implications of different computing environments. For the N-body simulation workload with 100,000 particles:

- **Jetson Orin NX** offers the best cost efficiency for single-device deployments
- **Jetson-based Slurm clusters** provide an excellent balance of cost and scalability
- **Cloud solutions** may be preferable for irregular usage below the break-even threshold
- **DGX systems** are justified when computational density and performance are critical priorities

This detailed analysis demonstrates how to apply the cost modeling framework to make informed infrastructure decisions based on specific workload characteristics and operational requirements.