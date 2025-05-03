// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 nvidia-jetson-workload contributors

import React from 'react';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend 
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import { Box, MenuItem, FormControl, InputLabel, Select, Typography } from '@mui/material';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// Component props
interface PerformanceChartProps {
  workloads: any[];
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ workloads }) => {
  // State for selected metric
  const [metric, setMetric] = React.useState('totalTimeMs');

  // Available metrics to display
  const metrics = [
    { value: 'totalTimeMs', label: 'Execution Time (ms)' },
    { value: 'gpuUtilization', label: 'GPU Utilization (%)' },
    { value: 'cpuUtilization', label: 'CPU Utilization (%)' },
    { value: 'gpuMemoryMb', label: 'GPU Memory (MB)' },
    { value: 'powerConsumptionWatts', label: 'Power Consumption (W)' },
    { value: 'operationsPerSecond', label: 'Operations Per Second' },
  ];

  // Filter workloads that have metrics
  const workloadsWithMetrics = workloads.filter(w => w.metrics);

  // Prepare data for the chart
  const chartData = {
    labels: workloadsWithMetrics.map(w => w.name),
    datasets: [
      {
        label: metrics.find(m => m.value === metric)?.label || metric,
        data: workloadsWithMetrics.map(w => {
          // For operations per second, use scientific notation for readability
          if (metric === 'operationsPerSecond') {
            return w.metrics[metric] / 1e9; // Convert to billions
          }
          return w.metrics[metric];
        }),
        backgroundColor: workloadsWithMetrics.map(w => {
          switch(w.type) {
            case 'weather': return 'rgba(53, 162, 235, 0.7)';
            case 'medical': return 'rgba(153, 102, 255, 0.7)';
            case 'nbody': return 'rgba(75, 192, 192, 0.7)';
            default: return 'rgba(201, 203, 207, 0.7)';
          }
        }),
        borderColor: workloadsWithMetrics.map(w => {
          switch(w.type) {
            case 'weather': return 'rgb(53, 162, 235)';
            case 'medical': return 'rgb(153, 102, 255)';
            case 'nbody': return 'rgb(75, 192, 192)';
            default: return 'rgb(201, 203, 207)';
          }
        }),
        borderWidth: 1,
      },
    ],
  };

  // Chart options
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: false,
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              if (metric === 'operationsPerSecond') {
                label += context.parsed.y.toFixed(2) + ' billion ops/sec';
              } else if (metric === 'totalTimeMs') {
                label += (context.parsed.y / 1000).toFixed(2) + ' seconds';
              } else {
                label += context.parsed.y.toFixed(2);
              }
            }
            return label;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: metric === 'operationsPerSecond' ? 'Billions of Operations per Second' : 
                metrics.find(m => m.value === metric)?.label || metric,
        }
      }
    }
  };

  // Handle metric change
  const handleMetricChange = (event: any) => {
    setMetric(event.target.value);
  };

  return (
    <Box>
      {workloadsWithMetrics.length === 0 ? (
        <Typography variant="body1" align="center" sx={{ mt: 10 }}>
          No performance data available
        </Typography>
      ) : (
        <>
          <Box sx={{ minWidth: 150, mb: 2 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Metric</InputLabel>
              <Select
                value={metric}
                label="Metric"
                onChange={handleMetricChange}
              >
                {metrics.map(option => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
          <Box sx={{ height: '100%' }}>
            <Bar data={chartData} options={options} />
          </Box>
        </>
      )}
    </Box>
  );
};

export default PerformanceChart;