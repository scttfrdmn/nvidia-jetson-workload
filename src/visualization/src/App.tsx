// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 nvidia-jetson-workload contributors

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';

// Components
import Dashboard from './components/Dashboard';
import Navbar from './components/Navbar';
import WeatherVisualization from './components/visualizations/WeatherVisualization';
import MedicalVisualization from './components/visualizations/MedicalVisualization';
import NBodyVisualization from './components/visualizations/NBodyVisualization';
import WorkloadDetail from './components/WorkloadDetail';
import ClusterStatus from './components/ClusterStatus';
import SlurmPanel from './components/SlurmPanel';

// Context
import { WorkloadProvider } from './context/WorkloadContext';

// Create a theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#76B900', // NVIDIA green
    },
    secondary: {
      main: '#00A3E0', // NVIDIA blue
    },
    background: {
      default: '#121212',
      paper: '#1E1E1E',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          scrollbarColor: "#6b6b6b #2b2b2b",
          "&::-webkit-scrollbar, & *::-webkit-scrollbar": {
            backgroundColor: "#2b2b2b",
            width: '8px',
            height: '8px',
          },
          "&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb": {
            borderRadius: 8,
            backgroundColor: "#6b6b6b",
            border: "2px solid #2b2b2b",
          },
          "&::-webkit-scrollbar-thumb:focus, & *::-webkit-scrollbar-thumb:focus": {
            backgroundColor: "#959595",
          },
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <WorkloadProvider>
        <Router>
          <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
            <Navbar />
            <Box sx={{ flexGrow: 1, p: 3 }}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/weather" element={<WeatherVisualization />} />
                <Route path="/medical" element={<MedicalVisualization />} />
                <Route path="/nbody" element={<NBodyVisualization />} />
                <Route path="/workload/:id" element={<WorkloadDetail />} />
                <Route path="/cluster" element={<ClusterStatus />} />
                <Route path="/slurm" element={<SlurmPanel />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </WorkloadProvider>
    </ThemeProvider>
  );
}

export default App;