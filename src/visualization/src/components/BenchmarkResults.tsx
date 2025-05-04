// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

import React, { useState, useEffect } from 'react';
import { 
  Container, Grid, Typography, Paper, Box, 
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
  Tabs, Tab, CircularProgress, Divider, Chip, Button,
  Card, CardContent, CardHeader
} from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';

// Import context
import { useWorkload } from '../context/WorkloadContext';
import {
  Science as ScienceIcon,
  Biotech as BiotechIcon,
  Public as PublicIcon,
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  BatteryChargingFull as PowerIcon,
  Schedule as TimeIcon
} from '@mui/icons-material';

// Import charts
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend,
  LineElement,
  PointElement,
  RadialLinearScale,
  Filler,
  ArcElement
} from 'chart.js';
import { Bar, Line, Radar, Pie } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  RadialLinearScale,
  ArcElement,
  Filler,
  Title,
  Tooltip,
  Legend
);

// Type definitions for benchmark results
interface PerformanceMetrics {
  totalTimeMs: number;
  gpuMemoryMb: number;
  cpuMemoryMb: number;
  gpuUtilization: number;
  cpuUtilization: number;
  powerConsumptionWatts: number;
  temperatureCelsius: number;
  operationsPerSecond: number;
  timeBreakdownMs: Record<string, number>;
}

interface DeviceCapabilities {
  name: string;
  compute_capability: string;
  total_memory: number;
  clock_rate: number;
  num_multiprocessors: number;
}

interface BenchmarkResult {
  workload_name: string;
  device_name: string;
  device_capabilities: DeviceCapabilities;
  execution_time: number;
  memory_usage: {
    host: number;
    device: number;
  };
  gpu_utilization: number | null;
  energy_consumption: number | null;
  throughput: number | null;
  additional_metrics: Record<string, any>;
  timestamp: string;
}

// Benchmark types defined in WorkloadContext

// Function to get workload icon
const WorkloadIcon = ({ type }: { type: string }) => {
  if (type.includes('nbody')) {
    return <ScienceIcon fontSize="large" color="info" />;
  } else if (type.includes('molecular')) {
    return <ScienceIcon fontSize="large" color="warning" />;
  } else if (type.includes('weather')) {
    return <PublicIcon fontSize="large" color="primary" />;
  } else if (type.includes('medical')) {
    return <BiotechIcon fontSize="large" color="secondary" />;
  } else {
    return <MemoryIcon fontSize="large" />;
  }
};

// Function to get human-readable workload name
const getWorkloadDisplayName = (name: string): string => {
  if (name === 'nbody_sim') return 'N-Body Simulation';
  if (name === 'molecular_dynamics') return 'Molecular Dynamics';
  if (name === 'weather_sim') return 'Weather Simulation';
  if (name.includes('medical_imaging')) {
    const task = name.replace('medical_imaging_', '').replace('_', ' ');
    return `Medical Imaging: ${task.charAt(0).toUpperCase() + task.slice(1)}`;
  }
  return name;
};

const BenchmarkResults: React.FC = () => {
  // Get data from context
  const { benchmarkResults, isLoading, refreshBenchmarkData } = useWorkload();
  
  // State
  const [activeTab, setActiveTab] = useState<number>(0);
  const [selectedWorkload, setSelectedWorkload] = useState<string | null>(null);

  // Tabs
  const tabs = [
    'Execution Time',
    'GPU Utilization',
    'Memory Usage',
    'Energy Consumption',
    'Throughput'
  ];

  // Set default selected workload
  useEffect(() => {
    if (benchmarkResults.length > 0 && !selectedWorkload) {
      setSelectedWorkload(benchmarkResults[0].workload_name);
    }
  }, [benchmarkResults, selectedWorkload]);

  // Get unique workload names
  const workloadNames = [...new Set(benchmarkResults.map(r => r.workload_name))];
  
  // Get unique device names
  const deviceNames = [...new Set(benchmarkResults.map(r => r.device_name))];
  
  // Handle tab change
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };
  
  // Handle workload selection
  const handleWorkloadSelect = (workload: string) => {
    setSelectedWorkload(workload);
  };
  
  // Filter results for selected workload
  const filteredResults = selectedWorkload
    ? benchmarkResults.filter(r => r.workload_name === selectedWorkload)
    : benchmarkResults;

  // Prepare data for execution time chart
  const executionTimeData = {
    labels: deviceNames,
    datasets: workloadNames.map((workload, index) => {
      const workloadResults = benchmarkResults.filter(r => r.workload_name === workload);
      const data = deviceNames.map(device => {
        const result = workloadResults.find(r => r.device_name === device);
        return result ? result.execution_time : 0;
      });
      
      return {
        label: getWorkloadDisplayName(workload),
        data,
        backgroundColor: [
          'rgba(53, 162, 235, 0.7)',
          'rgba(255, 99, 132, 0.7)',
          'rgba(75, 192, 192, 0.7)',
          'rgba(153, 102, 255, 0.7)',
        ][index % 4],
        borderColor: [
          'rgb(53, 162, 235)',
          'rgb(255, 99, 132)',
          'rgb(75, 192, 192)',
          'rgb(153, 102, 255)',
        ][index % 4],
        borderWidth: 1,
      };
    }),
  };

  // Prepare data for GPU utilization chart
  const gpuUtilizationData = {
    labels: deviceNames,
    datasets: workloadNames.map((workload, index) => {
      const workloadResults = benchmarkResults.filter(r => r.workload_name === workload);
      const data = deviceNames.map(device => {
        const result = workloadResults.find(r => r.device_name === device);
        return result && result.gpu_utilization ? result.gpu_utilization : 0;
      });
      
      return {
        label: getWorkloadDisplayName(workload),
        data,
        backgroundColor: [
          'rgba(53, 162, 235, 0.7)',
          'rgba(255, 99, 132, 0.7)',
          'rgba(75, 192, 192, 0.7)',
          'rgba(153, 102, 255, 0.7)',
        ][index % 4],
        borderColor: [
          'rgb(53, 162, 235)',
          'rgb(255, 99, 132)',
          'rgb(75, 192, 192)',
          'rgb(153, 102, 255)',
        ][index % 4],
        borderWidth: 1,
      };
    }),
  };

  // Prepare data for memory usage chart
  const memoryUsageData = {
    labels: deviceNames,
    datasets: workloadNames.flatMap((workload, index) => {
      const workloadResults = benchmarkResults.filter(r => r.workload_name === workload);
      
      // Host memory dataset
      const hostData = deviceNames.map(device => {
        const result = workloadResults.find(r => r.device_name === device);
        return result ? result.memory_usage.host : 0;
      });
      
      // Device memory dataset
      const deviceData = deviceNames.map(device => {
        const result = workloadResults.find(r => r.device_name === device);
        return result ? result.memory_usage.device : 0;
      });
      
      return [
        {
          label: `${getWorkloadDisplayName(workload)} - Host`,
          data: hostData,
          backgroundColor: [
            'rgba(53, 162, 235, 0.7)',
            'rgba(255, 99, 132, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(153, 102, 255, 0.7)',
          ][index % 4],
          borderColor: [
            'rgb(53, 162, 235)',
            'rgb(255, 99, 132)',
            'rgb(75, 192, 192)',
            'rgb(153, 102, 255)',
          ][index % 4],
          borderWidth: 1,
        },
        {
          label: `${getWorkloadDisplayName(workload)} - Device`,
          data: deviceData,
          backgroundColor: [
            'rgba(53, 162, 235, 0.4)',
            'rgba(255, 99, 132, 0.4)',
            'rgba(75, 192, 192, 0.4)',
            'rgba(153, 102, 255, 0.4)',
          ][index % 4],
          borderColor: [
            'rgb(53, 162, 235)',
            'rgb(255, 99, 132)',
            'rgb(75, 192, 192)',
            'rgb(153, 102, 255)',
          ][index % 4],
          borderWidth: 1,
          borderDash: [5, 5],
        }
      ];
    }),
  };

  // Prepare data for energy consumption chart
  const energyConsumptionData = {
    labels: deviceNames,
    datasets: workloadNames.map((workload, index) => {
      const workloadResults = benchmarkResults.filter(r => r.workload_name === workload);
      const data = deviceNames.map(device => {
        const result = workloadResults.find(r => r.device_name === device);
        return result && result.energy_consumption ? result.energy_consumption : 0;
      });
      
      return {
        label: getWorkloadDisplayName(workload),
        data,
        backgroundColor: [
          'rgba(53, 162, 235, 0.7)',
          'rgba(255, 99, 132, 0.7)',
          'rgba(75, 192, 192, 0.7)',
          'rgba(153, 102, 255, 0.7)',
        ][index % 4],
        borderColor: [
          'rgb(53, 162, 235)',
          'rgb(255, 99, 132)',
          'rgb(75, 192, 192)',
          'rgb(153, 102, 255)',
        ][index % 4],
        borderWidth: 1,
      };
    }),
  };

  // Prepare data for throughput chart
  const throughputData = {
    labels: deviceNames,
    datasets: workloadNames.map((workload, index) => {
      const workloadResults = benchmarkResults.filter(r => r.workload_name === workload);
      const data = deviceNames.map(device => {
        const result = workloadResults.find(r => r.device_name === device);
        return result && result.throughput ? result.throughput : 0;
      });
      
      return {
        label: getWorkloadDisplayName(workload),
        data,
        backgroundColor: [
          'rgba(53, 162, 235, 0.7)',
          'rgba(255, 99, 132, 0.7)',
          'rgba(75, 192, 192, 0.7)',
          'rgba(153, 102, 255, 0.7)',
        ][index % 4],
        borderColor: [
          'rgb(53, 162, 235)',
          'rgb(255, 99, 132)',
          'rgb(75, 192, 192)',
          'rgb(153, 102, 255)',
        ][index % 4],
        borderWidth: 1,
      };
    }),
  };

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += context.parsed.y.toFixed(2);
            }
            return label;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
      }
    }
  };

  // Get radar chart data for selected workload
  const getRadarChartData = (workloadName: string) => {
    const workloadResults = benchmarkResults.filter(r => r.workload_name === workloadName);
    
    return {
      labels: ['Execution Time', 'GPU Utilization', 'Memory Usage', 'Energy Efficiency', 'Throughput'],
      datasets: deviceNames.map((device, index) => {
        const result = workloadResults.find(r => r.device_name === device);
        
        // Normalize values (lower is better for execution time, higher is better for others)
        const maxExecutionTime = Math.max(...workloadResults.map(r => r.execution_time));
        const normalizedExecutionTime = result ? 100 - ((result.execution_time / maxExecutionTime) * 100) : 0;
        
        const gpuUtilization = result && result.gpu_utilization ? result.gpu_utilization : 0;
        
        const maxMemoryUsage = Math.max(...workloadResults.map(r => r.memory_usage.device));
        const normalizedMemoryUsage = result ? 100 - ((result.memory_usage.device / maxMemoryUsage) * 100) : 0;
        
        const maxEnergyConsumption = Math.max(...workloadResults.filter(r => r.energy_consumption !== null).map(r => r.energy_consumption!));
        const normalizedEnergyEfficiency = result && result.energy_consumption 
          ? 100 - ((result.energy_consumption / maxEnergyConsumption) * 100) 
          : 0;
        
        const maxThroughput = Math.max(...workloadResults.filter(r => r.throughput !== null).map(r => r.throughput!));
        const normalizedThroughput = result && result.throughput 
          ? (result.throughput / maxThroughput) * 100 
          : 0;
        
        return {
          label: device,
          data: [
            normalizedExecutionTime,
            gpuUtilization,
            normalizedMemoryUsage,
            normalizedEnergyEfficiency,
            normalizedThroughput
          ],
          backgroundColor: [
            'rgba(53, 162, 235, 0.2)',
            'rgba(255, 99, 132, 0.2)',
          ][index % 2],
          borderColor: [
            'rgb(53, 162, 235)',
            'rgb(255, 99, 132)',
          ][index % 2],
          borderWidth: 2,
        };
      }),
    };
  };

  // Get hardware comparison for selected workload
  const getHardwareComparison = (workloadName: string) => {
    const workloadResults = benchmarkResults.filter(r => r.workload_name === workloadName);
    
    // Calculate speedup relative to baseline device (Jetson Orin NX)
    const baselineResult = workloadResults.find(r => r.device_name === 'NVIDIA Jetson Orin NX');
    if (!baselineResult) return [];
    
    return workloadResults.map(result => {
      const speedup = baselineResult.execution_time / result.execution_time;
      const throughputRatio = result.throughput && baselineResult.throughput 
        ? result.throughput / baselineResult.throughput 
        : 1;
      const energyEfficiencyRatio = result.energy_consumption && baselineResult.energy_consumption
        ? baselineResult.energy_consumption / result.energy_consumption
        : 1;
      
      return {
        device: result.device_name,
        speedup,
        throughputRatio,
        energyEfficiencyRatio,
        gpuUtilization: result.gpu_utilization,
        memoryUsage: result.memory_usage.device
      };
    });
  };

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="xl">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          Benchmark Results
        </Typography>
        <Button 
          variant="outlined" 
          startIcon={<RefreshIcon />}
          onClick={refreshBenchmarkData}
          disabled={isLoading}
        >
          Refresh Data
        </Button>
      </Box>
      
      <Grid container spacing={3}>
        {/* Workload Selection */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Select Workload
            </Typography>
            <Divider sx={{ mb: 2 }} />
            
            <Grid container spacing={2}>
              {workloadNames.map((workload) => (
                <Grid item key={workload} xs={6} sm={3}>
                  <Card 
                    variant={selectedWorkload === workload ? "elevation" : "outlined"}
                    elevation={4}
                    sx={{ 
                      cursor: 'pointer',
                      transition: 'all 0.2s',
                      transform: selectedWorkload === workload ? 'scale(1.02)' : 'scale(1)',
                      bgcolor: selectedWorkload === workload ? 'action.selected' : 'background.paper',
                    }}
                    onClick={() => handleWorkloadSelect(workload)}
                  >
                    <CardContent sx={{ textAlign: 'center', p: 2 }}>
                      <WorkloadIcon type={workload} />
                      <Typography variant="subtitle1" sx={{ mt: 1 }}>
                        {getWorkloadDisplayName(workload)}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
        
        {/* Summary Cards */}
        {selectedWorkload && (
          <Grid item xs={12}>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography color="text.secondary" gutterBottom>
                        Best Execution Time
                      </Typography>
                      <TimeIcon color="primary" />
                    </Box>
                    <Typography variant="h4" component="div">
                      {Math.min(...filteredResults.map(r => r.execution_time)).toFixed(2)}s
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {filteredResults.reduce((best, current) => 
                        current.execution_time < best.execution_time ? current : best
                      ).device_name}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography color="text.secondary" gutterBottom>
                        Max GPU Utilization
                      </Typography>
                      <SpeedIcon color="secondary" />
                    </Box>
                    <Typography variant="h4" component="div">
                      {Math.max(...filteredResults.filter(r => r.gpu_utilization !== null).map(r => r.gpu_utilization!)).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {filteredResults.reduce((best, current) => 
                        (current.gpu_utilization || 0) > (best.gpu_utilization || 0) ? current : best
                      ).device_name}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography color="text.secondary" gutterBottom>
                        Max Throughput
                      </Typography>
                      <MemoryIcon color="info" />
                    </Box>
                    <Typography variant="h4" component="div">
                      {Math.max(...filteredResults.filter(r => r.throughput !== null).map(r => r.throughput!)).toFixed(1)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      ops/s or iterations/s
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              
              <Grid item xs={12} sm={6} md={3}>
                <Card>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography color="text.secondary" gutterBottom>
                        Energy Efficiency
                      </Typography>
                      <PowerIcon color="warning" />
                    </Box>
                    <Typography variant="h4" component="div">
                      {filteredResults.length > 0 
                        ? (Math.min(...filteredResults.filter(r => r.energy_consumption !== null).map(r => r.energy_consumption!)) / 
                           Math.min(...filteredResults.filter(r => r.execution_time !== null).map(r => r.execution_time))).toFixed(1)
                        : 0} W
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Average power consumption
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Grid>
        )}
        
        {/* Charts */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Tabs 
              value={activeTab} 
              onChange={handleTabChange}
              variant="scrollable"
              scrollButtons="auto"
              sx={{ mb: 2 }}
            >
              {tabs.map((tab, index) => (
                <Tab key={index} label={tab} />
              ))}
            </Tabs>
            
            <Box sx={{ height: 400, pt: 2 }}>
              {activeTab === 0 && (
                <Bar data={executionTimeData} options={{
                  ...chartOptions,
                  scales: {
                    ...chartOptions.scales,
                    y: {
                      ...chartOptions.scales.y,
                      title: {
                        display: true,
                        text: 'Execution Time (seconds)'
                      }
                    }
                  }
                }} />
              )}
              
              {activeTab === 1 && (
                <Bar data={gpuUtilizationData} options={{
                  ...chartOptions,
                  scales: {
                    ...chartOptions.scales,
                    y: {
                      ...chartOptions.scales.y,
                      title: {
                        display: true,
                        text: 'GPU Utilization (%)'
                      }
                    }
                  }
                }} />
              )}
              
              {activeTab === 2 && (
                <Bar data={memoryUsageData} options={{
                  ...chartOptions,
                  scales: {
                    ...chartOptions.scales,
                    y: {
                      ...chartOptions.scales.y,
                      title: {
                        display: true,
                        text: 'Memory Usage (MB)'
                      }
                    }
                  }
                }} />
              )}
              
              {activeTab === 3 && (
                <Bar data={energyConsumptionData} options={{
                  ...chartOptions,
                  scales: {
                    ...chartOptions.scales,
                    y: {
                      ...chartOptions.scales.y,
                      title: {
                        display: true,
                        text: 'Energy Consumption (Joules)'
                      }
                    }
                  }
                }} />
              )}
              
              {activeTab === 4 && (
                <Bar data={throughputData} options={{
                  ...chartOptions,
                  scales: {
                    ...chartOptions.scales,
                    y: {
                      ...chartOptions.scales.y,
                      title: {
                        display: true,
                        text: 'Throughput (iterations/s)'
                      }
                    }
                  }
                }} />
              )}
            </Box>
          </Paper>
        </Grid>
        
        {/* Workload Details */}
        {selectedWorkload && (
          <>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Hardware Comparison
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Device</TableCell>
                        <TableCell align="right">Speedup</TableCell>
                        <TableCell align="right">Throughput Ratio</TableCell>
                        <TableCell align="right">GPU Utilization</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {getHardwareComparison(selectedWorkload).map((row) => (
                        <TableRow key={row.device}>
                          <TableCell component="th" scope="row">
                            {row.device}
                          </TableCell>
                          <TableCell align="right">
                            {row.speedup.toFixed(2)}x
                          </TableCell>
                          <TableCell align="right">
                            {row.throughputRatio.toFixed(2)}x
                          </TableCell>
                          <TableCell align="right">
                            {row.gpuUtilization ? `${row.gpuUtilization.toFixed(1)}%` : 'N/A'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 2, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Performance Radar
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <Box sx={{ height: 350 }}>
                  <Radar 
                    data={getRadarChartData(selectedWorkload)} 
                    options={{
                      scales: {
                        r: {
                          min: 0,
                          max: 100,
                          beginAtZero: true,
                          ticks: {
                            display: false
                          }
                        }
                      }
                    }}
                  />
                </Box>
              </Paper>
            </Grid>
            
            <Grid item xs={12}>
              <Paper sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Detailed Metrics
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Device</TableCell>
                        <TableCell align="right">Execution Time (s)</TableCell>
                        <TableCell align="right">Host Memory (MB)</TableCell>
                        <TableCell align="right">Device Memory (MB)</TableCell>
                        <TableCell align="right">GPU Utilization (%)</TableCell>
                        <TableCell align="right">Energy (J)</TableCell>
                        <TableCell align="right">Throughput</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {filteredResults.map((result) => (
                        <TableRow key={result.device_name}>
                          <TableCell component="th" scope="row">
                            {result.device_name}
                          </TableCell>
                          <TableCell align="right">{result.execution_time.toFixed(2)}</TableCell>
                          <TableCell align="right">{result.memory_usage.host.toFixed(0)}</TableCell>
                          <TableCell align="right">{result.memory_usage.device.toFixed(0)}</TableCell>
                          <TableCell align="right">{result.gpu_utilization ? result.gpu_utilization.toFixed(1) : 'N/A'}</TableCell>
                          <TableCell align="right">{result.energy_consumption ? result.energy_consumption.toFixed(1) : 'N/A'}</TableCell>
                          <TableCell align="right">{result.throughput ? result.throughput.toFixed(2) : 'N/A'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
                
                {filteredResults.length > 0 && filteredResults[0].additional_metrics && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Workload Configuration
                    </Typography>
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {Object.entries(filteredResults[0].additional_metrics).map(([key, value]) => (
                        <Chip 
                          key={key} 
                          label={`${key}: ${typeof value === 'number' && value > 1e6 
                            ? `${(value / 1e9).toFixed(2)}B` 
                            : value}`} 
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  </Box>
                )}
              </Paper>
            </Grid>
          </>
        )}
      </Grid>
    </Container>
  );
};

export default BenchmarkResults;