// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Container, Grid, Typography, Paper, Box, 
  Button, Card, CardContent, CardActions,
  Chip, Divider, LinearProgress, CircularProgress,
  Stack, IconButton
} from '@mui/material';
import { 
  Refresh as RefreshIcon,
  Science as ScienceIcon,
  Biotech as BiotechIcon,
  Public as PublicIcon,
  PlayArrow as PlayArrowIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  HourglassFull as HourglassIcon,
  Speed as SpeedIcon
} from '@mui/icons-material';

import { useWorkload } from '../context/WorkloadContext';
import ClusterOverview from './ClusterOverview';
import PerformanceChart from './charts/PerformanceChart';

// Status chip component
const StatusChip = ({ status }: { status: string }) => {
  switch (status) {
    case 'COMPLETED':
      return <Chip icon={<CheckCircleIcon />} label="Completed" color="success" size="small" />;
    case 'RUNNING':
      return <Chip icon={<HourglassIcon />} label="Running" color="primary" size="small" />;
    case 'PENDING':
      return <Chip icon={<HourglassIcon />} label="Pending" color="secondary" size="small" />;
    case 'FAILED':
      return <Chip icon={<ErrorIcon />} label="Failed" color="error" size="small" />;
    default:
      return <Chip label={status} size="small" />;
  }
};

// Icon for workload type
const WorkloadIcon = ({ type }: { type: string }) => {
  switch (type) {
    case 'weather':
      return <PublicIcon color="primary" />;
    case 'medical':
      return <BiotechIcon color="secondary" />;
    case 'nbody':
      return <ScienceIcon color="info" />;
    default:
      return <ScienceIcon />;
  }
};

const Dashboard = () => {
  const { 
    workloads, runningWorkloads, completedWorkloads, 
    nodes, isLoading, refreshData 
  } = useWorkload();
  const navigate = useNavigate();

  // Function to get completion percentage of a running workload
  const getProgressPercent = (workload: any) => {
    if (workload.status === 'COMPLETED') return 100;
    if (workload.status === 'FAILED') return 0;
    
    // Mock progress calculation - in a real app, would come from the workload data
    if (workload.type === 'weather') {
      return Math.min(95, ((Date.now() - workload.startTime) / (300000)) * 100);
    } else if (workload.type === 'medical') {
      return Math.min(95, ((Date.now() - workload.startTime) / (240000)) * 100);
    } else {
      return Math.min(95, ((Date.now() - workload.startTime) / (180000)) * 100);
    }
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" component="h1">
          NVIDIA Jetson Workload Dashboard
        </Typography>
        <Button 
          variant="outlined" 
          startIcon={<RefreshIcon />}
          onClick={refreshData}
          disabled={isLoading}
        >
          Refresh
        </Button>
      </Box>

      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <CircularProgress />
        </Box>
      ) : (
        <Grid container spacing={3}>
          {/* Cluster Overview */}
          <Grid item xs={12}>
            <ClusterOverview nodes={nodes} />
          </Grid>
          
          {/* Running Workloads */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Running Workloads
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              {runningWorkloads.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography color="text.secondary">
                    No workloads currently running
                  </Typography>
                  <Button 
                    variant="contained" 
                    sx={{ mt: 2 }}
                    onClick={() => navigate('/slurm')}
                  >
                    Launch Workload
                  </Button>
                </Box>
              ) : (
                <Stack spacing={2}>
                  {runningWorkloads.map(workload => (
                    <Card key={workload.id} variant="outlined">
                      <CardContent sx={{ pb: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                          <WorkloadIcon type={workload.type} />
                          <Typography variant="h6" sx={{ ml: 1 }}>
                            {workload.name}
                          </Typography>
                          <Box sx={{ ml: 'auto' }}>
                            <StatusChip status={workload.status} />
                          </Box>
                        </Box>
                        
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Node: {nodes.find(n => n.id === workload.nodeId)?.name || workload.nodeId}
                        </Typography>
                        
                        <Box sx={{ mt: 2 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                            <Typography variant="body2" sx={{ mr: 1 }}>
                              Progress: {Math.round(getProgressPercent(workload))}%
                            </Typography>
                          </Box>
                          <LinearProgress 
                            variant="determinate" 
                            value={getProgressPercent(workload)} 
                            sx={{ height: 8, borderRadius: 1 }}
                          />
                        </Box>
                      </CardContent>
                      <CardActions>
                        <Button 
                          size="small" 
                          startIcon={<InfoIcon />}
                          onClick={() => navigate(`/workload/${workload.id}`)}
                        >
                          Details
                        </Button>
                      </CardActions>
                    </Card>
                  ))}
                </Stack>
              )}
            </Paper>
          </Grid>
          
          {/* Performance Overview */}
          <Grid item xs={12} md={6}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Performance Overview
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              {completedWorkloads.length === 0 ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography color="text.secondary">
                    No completed workloads to analyze
                  </Typography>
                </Box>
              ) : (
                <Box sx={{ height: 300 }}>
                  <PerformanceChart workloads={completedWorkloads} />
                </Box>
              )}
            </Paper>
          </Grid>
          
          {/* Quick Launch */}
          <Grid item xs={12}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Quick Launch
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <PublicIcon color="primary" fontSize="large" />
                        <Typography variant="h6" sx={{ ml: 1 }}>
                          Weather Simulation
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        Fluid dynamics simulation showcasing atmospheric modeling parallelized across GPU cores.
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button 
                        variant="contained" 
                        color="primary"
                        startIcon={<PlayArrowIcon />}
                        onClick={() => navigate('/weather')}
                        fullWidth
                      >
                        Launch
                      </Button>
                    </CardActions>
                  </Card>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <BiotechIcon color="secondary" fontSize="large" />
                        <Typography variant="h6" sx={{ ml: 1 }}>
                          Medical Imaging
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        AI-driven segmentation and analysis of medical imaging data with GPU-accelerated neural networks.
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button 
                        variant="contained" 
                        color="secondary"
                        startIcon={<PlayArrowIcon />}
                        onClick={() => navigate('/medical')}
                        fullWidth
                      >
                        Launch
                      </Button>
                    </CardActions>
                  </Card>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <ScienceIcon color="info" fontSize="large" />
                        <Typography variant="h6" sx={{ ml: 1 }}>
                          N-Body Simulation
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        Physics-based simulation of gravitational interactions optimized for Jetson's CUDA capabilities.
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button 
                        variant="contained" 
                        color="info"
                        startIcon={<PlayArrowIcon />}
                        onClick={() => navigate('/nbody')}
                        fullWidth
                      >
                        Launch
                      </Button>
                    </CardActions>
                  </Card>
                </Grid>

                <Grid item xs={12} sm={6} md={3}>
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <SpeedIcon color="warning" fontSize="large" />
                        <Typography variant="h6" sx={{ ml: 1 }}>
                          Benchmarks
                        </Typography>
                      </Box>
                      <Typography variant="body2" color="text.secondary">
                        Performance analysis and comparison of all workloads across different hardware configurations.
                      </Typography>
                    </CardContent>
                    <CardActions>
                      <Button 
                        variant="contained" 
                        color="warning"
                        startIcon={<PlayArrowIcon />}
                        onClick={() => navigate('/benchmarks')}
                        fullWidth
                      >
                        View
                      </Button>
                    </CardActions>
                  </Card>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Container>
  );
};

export default Dashboard;