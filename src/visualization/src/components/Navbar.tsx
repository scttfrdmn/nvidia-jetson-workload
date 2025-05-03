// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 nvidia-jetson-workload contributors

import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  AppBar, Toolbar, Typography, Button, Box, 
  Drawer, List, ListItem, ListItemButton,
  ListItemIcon, ListItemText, Divider,
  IconButton, useMediaQuery, useTheme
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Public as WeatherIcon,
  Biotech as MedicalIcon,
  Science as NBodyIcon,
  Computer as ClusterIcon,
  Code as SlurmIcon,
  Info as InfoIcon
} from '@mui/icons-material';

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [drawerOpen, setDrawerOpen] = React.useState(false);

  const toggleDrawer = (open: boolean) => () => {
    setDrawerOpen(open);
  };

  const navItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'Weather Simulation', icon: <WeatherIcon />, path: '/weather' },
    { text: 'Medical Imaging', icon: <MedicalIcon />, path: '/medical' },
    { text: 'N-Body Simulation', icon: <NBodyIcon />, path: '/nbody' },
    { text: 'Cluster Status', icon: <ClusterIcon />, path: '/cluster' },
    { text: 'Slurm Jobs', icon: <SlurmIcon />, path: '/slurm' },
  ];

  const handleNavigate = (path: string) => {
    navigate(path);
    if (isMobile) {
      setDrawerOpen(false);
    }
  };

  const drawer = (
    <Box sx={{ width: 250 }} onClick={toggleDrawer(false)}>
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" component="div">
          NVIDIA Jetson Workload
        </Typography>
      </Box>
      <Divider />
      <List>
        {navItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton 
              selected={location.pathname === item.path}
              onClick={() => handleNavigate(item.path)}
            >
              <ListItemIcon>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Box>
  );

  return (
    <>
      <AppBar position="static">
        <Toolbar>
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={toggleDrawer(true)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          
          <Typography 
            variant="h6" 
            component="div" 
            sx={{ cursor: 'pointer' }}
            onClick={() => navigate('/')}
          >
            NVIDIA Jetson Workload
          </Typography>
          
          {!isMobile && (
            <Box sx={{ display: 'flex', ml: 4 }}>
              {navItems.map((item) => (
                <Button 
                  key={item.text}
                  color="inherit"
                  startIcon={item.icon}
                  onClick={() => handleNavigate(item.path)}
                  sx={{ 
                    ml: 1, 
                    opacity: location.pathname === item.path ? 1 : 0.8,
                    borderBottom: location.pathname === item.path ? '2px solid white' : 'none',
                    borderRadius: 0,
                    paddingBottom: '4px'
                  }}
                >
                  {item.text}
                </Button>
              ))}
            </Box>
          )}
          
          <Box sx={{ flexGrow: 1 }} />
          
          <Button
            color="inherit"
            startIcon={<InfoIcon />}
            onClick={() => window.open('https://github.com/scttfrdmn/nvidia-jetson-workload', '_blank')}
          >
            About
          </Button>
        </Toolbar>
      </AppBar>
      
      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={toggleDrawer(false)}
      >
        {drawer}
      </Drawer>
    </>
  );
};

export default Navbar;