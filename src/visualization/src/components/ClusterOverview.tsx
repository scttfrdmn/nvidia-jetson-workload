// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

import React from 'react';
import { 
  Paper, Typography, Grid, Box, 
  Card, CardContent, Chip, 
  LinearProgress, Divider 
} from '@mui/material';
import {
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Thermostat as ThermostatIcon,
  BatteryChargingFull as PowerIcon
} from '@mui/icons-material';

interface Node {
  id: string;
  name: string;
  status: 'online' | 'offline' | 'busy';
  ipAddress: string;
  gpuModel: string;
  cpuModel: string;
  memoryTotal: number;
  memoryAvailable: number;
  gpuMemoryTotal: number;
  gpuMemoryAvailable: number;
  temperature: number;
  powerUsage: number;
}

interface ClusterOverviewProps {
  nodes: Node[];
}

const ClusterOverview: React.FC<ClusterOverviewProps> = ({ nodes }) => {
  // Format memory values to GB with 1 decimal place
  const formatMemory = (memoryMB: number) => {
    return (memoryMB / 1024).toFixed(1) + ' GB';
  };

  // Calculate status count
  const statusCount = {
    online: nodes.filter(n => n.status === 'online').length,
    busy: nodes.filter(n => n.status === 'busy').length,
    offline: nodes.filter(n => n.status === 'offline').length
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Cluster Status
      </Typography>
      <Divider sx={{ mb: 2 }} />

      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="body1">
          Total Nodes: {nodes.length}
        </Typography>
        <Box>
          <Chip 
            label={`${statusCount.online} Online`} 
            color="success" 
            size="small" 
            sx={{ mr: 1 }} 
          />
          <Chip 
            label={`${statusCount.busy} Busy`} 
            color="primary" 
            size="small" 
            sx={{ mr: 1 }} 
          />
          <Chip 
            label={`${statusCount.offline} Offline`} 
            color="error" 
            size="small" 
          />
        </Box>
      </Box>

      <Grid container spacing={2}>
        {nodes.map(node => (
          <Grid item xs={12} sm={6} md={3} key={node.id}>
            <Card 
              variant="outlined"
              sx={{ 
                borderColor: 
                  node.status === 'online' 
                    ? 'success.main' 
                    : node.status === 'busy' 
                      ? 'primary.main' 
                      : 'error.main',
                opacity: node.status === 'offline' ? 0.7 : 1
              }}
            >
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                  <Typography variant="h6">
                    {node.name}
                  </Typography>
                  <Chip 
                    label={node.status.charAt(0).toUpperCase() + node.status.slice(1)} 
                    color={
                      node.status === 'online' 
                        ? 'success' 
                        : node.status === 'busy' 
                          ? 'primary' 
                          : 'error'
                    } 
                    size="small" 
                  />
                </Box>
                
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {node.ipAddress}
                </Typography>
                
                {node.status !== 'offline' && (
                  <>
                    <Box sx={{ mt: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                        <MemoryIcon fontSize="small" sx={{ mr: 1, color: 'primary.main' }} />
                        <Typography variant="body2">
                          Memory: {formatMemory(node.memoryAvailable)} / {formatMemory(node.memoryTotal)}
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={(1 - node.memoryAvailable / node.memoryTotal) * 100} 
                        sx={{ height: 6, borderRadius: 1, mb: 1 }}
                      />
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                        <SpeedIcon fontSize="small" sx={{ mr: 1, color: 'secondary.main' }} />
                        <Typography variant="body2">
                          GPU Memory: {formatMemory(node.gpuMemoryAvailable)} / {formatMemory(node.gpuMemoryTotal)}
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={(1 - node.gpuMemoryAvailable / node.gpuMemoryTotal) * 100} 
                        color="secondary"
                        sx={{ height: 6, borderRadius: 1, mb: 1 }}
                      />
                      
                      <Grid container spacing={1} sx={{ mt: 0.5 }}>
                        <Grid item xs={6}>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <ThermostatIcon fontSize="small" sx={{ mr: 0.5, color: 'error.main' }} />
                            <Typography variant="body2">
                              {node.temperature}°C
                            </Typography>
                          </Box>
                        </Grid>
                        <Grid item xs={6}>
                          <Box sx={{ display: 'flex', alignItems: 'center' }}>
                            <PowerIcon fontSize="small" sx={{ mr: 0.5, color: 'warning.main' }} />
                            <Typography variant="body2">
                              {node.powerUsage} W
                            </Typography>
                          </Box>
                        </Grid>
                      </Grid>
                    </Box>
                  </>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

export default ClusterOverview;