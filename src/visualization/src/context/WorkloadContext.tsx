// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import io from 'socket.io-client';

// Type definitions for our workloads
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

enum WorkloadStatus {
  PENDING = 'PENDING',
  RUNNING = 'RUNNING',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
}

interface WorkloadBase {
  id: string;
  name: string;
  type: 'weather' | 'medical' | 'nbody';
  status: WorkloadStatus;
  nodeId: string;
  startTime: number;
  endTime?: number;
  metrics?: PerformanceMetrics;
  error?: string;
}

// Workload with type-specific properties
interface WeatherWorkload extends WorkloadBase {
  type: 'weather';
  config: {
    gridSizeX: number;
    gridSizeY: number;
    gridSizeZ: number;
    timeStep: number;
  };
  results?: {
    maxTemperature: number;
    minTemperature: number;
    maxWindSpeed: number;
    totalPrecipitation: number;
  };
}

interface MedicalWorkload extends WorkloadBase {
  type: 'medical';
  config: {
    imagingType: 'MRI' | 'CT' | 'ULTRASOUND' | 'XRAY' | 'PET';
    width: number;
    height: number;
    depth: number;
    datasetSize: number;
  };
  results?: {
    segmentations: number;
    classification?: string;
    accuracy: number;
  };
}

interface NBodyWorkload extends WorkloadBase {
  type: 'nbody';
  config: {
    numParticles: number;
    timeStep: number;
    totalSimulationTime: number;
    integrationAlgorithm: string;
  };
  results?: {
    energyConservationError: number;
    maximumVelocity: number;
    systemRadius: number;
  };
}

type Workload = WeatherWorkload | MedicalWorkload | NBodyWorkload;

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

// Benchmark result type definitions
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

interface WorkloadContextType {
  workloads: Workload[];
  runningWorkloads: Workload[];
  completedWorkloads: Workload[];
  nodes: Node[];
  benchmarkResults: BenchmarkResult[];
  isLoading: boolean;
  error: string | null;
  refreshData: () => void;
  launchWorkload: (type: 'weather' | 'medical' | 'nbody', config: any) => Promise<string>;
  cancelWorkload: (id: string) => Promise<boolean>;
  getWorkloadById: (id: string) => Workload | undefined;
  refreshBenchmarkData: () => void;
}

const WorkloadContext = createContext<WorkloadContextType | undefined>(undefined);

interface WorkloadProviderProps {
  children: ReactNode;
}

// Mock server URL - would be replaced with actual API endpoint
const API_URL = 'http://localhost:5000/api';
const SOCKET_URL = 'http://localhost:5000';

export const WorkloadProvider = ({ children }: WorkloadProviderProps) => {
  const [workloads, setWorkloads] = useState<Workload[]>([]);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [socket, setSocket] = useState<any>(null);

  // Filter workloads by status
  const runningWorkloads = workloads.filter(w => 
    w.status === WorkloadStatus.PENDING || w.status === WorkloadStatus.RUNNING
  );
  const completedWorkloads = workloads.filter(w => 
    w.status === WorkloadStatus.COMPLETED || w.status === WorkloadStatus.FAILED
  );

  // Initialize socket connection
  useEffect(() => {
    // In a real implementation, this would connect to your backend
    // For now, we'll just simulate it
    console.log('Setting up socket connection (simulated)');
    
    // Placeholder for actual socket connection
    // const newSocket = io(SOCKET_URL);
    // setSocket(newSocket);
    
    // Mock socket to simulate connection
    const mockSocket = {
      on: (event: string, callback: Function) => {
        console.log(`Registered handler for ${event} event`);
        return mockSocket;
      },
      off: (event: string) => {
        console.log(`Removed handler for ${event} event`);
        return mockSocket;
      },
      disconnect: () => {
        console.log('Socket disconnected');
      }
    };
    setSocket(mockSocket);
    
    // Return cleanup function
    return () => {
      console.log('Cleaning up socket connection');
      if (socket) {
        socket.disconnect();
      }
    };
  }, []);

  // Fetch initial data
  useEffect(() => {
    refreshData();
    refreshBenchmarkData();
  }, []);

  // Set up socket listeners for real-time updates
  useEffect(() => {
    if (socket) {
      // Listen for workload updates
      socket.on('workload:update', (updatedWorkload: Workload) => {
        setWorkloads(prevWorkloads => 
          prevWorkloads.map(w => w.id === updatedWorkload.id ? updatedWorkload : w)
        );
      });

      // Listen for new workloads
      socket.on('workload:new', (newWorkload: Workload) => {
        setWorkloads(prevWorkloads => [...prevWorkloads, newWorkload]);
      });

      // Listen for node updates
      socket.on('node:update', (updatedNode: Node) => {
        setNodes(prevNodes => 
          prevNodes.map(n => n.id === updatedNode.id ? updatedNode : n)
        );
      });

      // Clean up listeners when component unmounts
      return () => {
        socket.off('workload:update');
        socket.off('workload:new');
        socket.off('node:update');
      };
    }
  }, [socket]);

  // Function to refresh data from server
  const refreshData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, these would be actual API calls
      // For now we'll use mock data
      setTimeout(() => {
        // Mock workload data
        const mockWorkloads: Workload[] = [
          {
            id: 'w1',
            name: 'Weather Simulation 1',
            type: 'weather',
            status: WorkloadStatus.COMPLETED,
            nodeId: 'node1',
            startTime: Date.now() - 300000,
            endTime: Date.now() - 60000,
            config: {
              gridSizeX: 128,
              gridSizeY: 128,
              gridSizeZ: 64,
              timeStep: 0.1
            },
            results: {
              maxTemperature: 305.2,
              minTemperature: 273.1,
              maxWindSpeed: 15.7,
              totalPrecipitation: 12.3
            },
            metrics: {
              totalTimeMs: 240000,
              gpuMemoryMb: 4096,
              cpuMemoryMb: 2048,
              gpuUtilization: 95.2,
              cpuUtilization: 65.3,
              powerConsumptionWatts: 25.7,
              temperatureCelsius: 72.5,
              operationsPerSecond: 2.5e9,
              timeBreakdownMs: {
                'initialization': 500,
                'computation': 238000,
                'finalization': 1500
              }
            }
          },
          {
            id: 'w2',
            name: 'Medical Imaging 1',
            type: 'medical',
            status: WorkloadStatus.RUNNING,
            nodeId: 'node2',
            startTime: Date.now() - 120000,
            config: {
              imagingType: 'MRI',
              width: 512,
              height: 512,
              depth: 128,
              datasetSize: 10
            }
          },
          {
            id: 'w3',
            name: 'N-Body Simulation 1',
            type: 'nbody',
            status: WorkloadStatus.PENDING,
            nodeId: 'node3',
            startTime: Date.now(),
            config: {
              numParticles: 100000,
              timeStep: 0.01,
              totalSimulationTime: 100,
              integrationAlgorithm: 'LEAPFROG'
            }
          }
        ];
        
        // Mock node data
        const mockNodes: Node[] = [
          {
            id: 'node1',
            name: 'jetson-1',
            status: 'online',
            ipAddress: '192.168.1.101',
            gpuModel: 'NVIDIA Jetson Orin NX',
            cpuModel: 'ARMv8 Cortex-A78AE',
            memoryTotal: 16384,
            memoryAvailable: 12000,
            gpuMemoryTotal: 8192,
            gpuMemoryAvailable: 6000,
            temperature: 65.2,
            powerUsage: 15.3
          },
          {
            id: 'node2',
            name: 'jetson-2',
            status: 'busy',
            ipAddress: '192.168.1.102',
            gpuModel: 'NVIDIA Jetson Orin NX',
            cpuModel: 'ARMv8 Cortex-A78AE',
            memoryTotal: 16384,
            memoryAvailable: 8000,
            gpuMemoryTotal: 8192,
            gpuMemoryAvailable: 2000,
            temperature: 75.8,
            powerUsage: 25.7
          },
          {
            id: 'node3',
            name: 'jetson-3',
            status: 'online',
            ipAddress: '192.168.1.103',
            gpuModel: 'NVIDIA Jetson Orin NX',
            cpuModel: 'ARMv8 Cortex-A78AE',
            memoryTotal: 16384,
            memoryAvailable: 14000,
            gpuMemoryTotal: 8192,
            gpuMemoryAvailable: 7500,
            temperature: 62.1,
            powerUsage: 10.2
          },
          {
            id: 'node4',
            name: 'jetson-4',
            status: 'offline',
            ipAddress: '192.168.1.104',
            gpuModel: 'NVIDIA Jetson Orin NX',
            cpuModel: 'ARMv8 Cortex-A78AE',
            memoryTotal: 16384,
            memoryAvailable: 16384,
            gpuMemoryTotal: 8192,
            gpuMemoryAvailable: 8192,
            temperature: 0,
            powerUsage: 0
          }
        ];
        
        setWorkloads(mockWorkloads);
        setNodes(mockNodes);
        setIsLoading(false);
      }, 500);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Failed to fetch data from server');
      setIsLoading(false);
    }
  };

  // Launch a new workload
  const launchWorkload = async (
    type: 'weather' | 'medical' | 'nbody', 
    config: any
  ): Promise<string> => {
    // In a real app, this would make an API call to launch the workload
    // For now, we'll simulate it
    console.log(`Launching ${type} workload with config:`, config);
    
    // Generate a unique ID for the new workload
    const id = `w${workloads.length + 1}`;
    
    // Create a new workload object
    const newWorkload: Workload = {
      id,
      name: `${type.charAt(0).toUpperCase() + type.slice(1)} Simulation ${workloads.length + 1}`,
      type,
      status: WorkloadStatus.PENDING,
      nodeId: nodes.find(n => n.status === 'online')?.id || 'node1',
      startTime: Date.now(),
      config
    };
    
    // Add the new workload to our state
    setWorkloads(prevWorkloads => [...prevWorkloads, newWorkload as any]);
    
    // In a real app, we'd return the ID from the API response
    return id;
  };

  // Cancel a running workload
  const cancelWorkload = async (id: string): Promise<boolean> => {
    // In a real app, this would make an API call to cancel the workload
    // For now, we'll simulate it
    console.log(`Cancelling workload ${id}`);
    
    // Update the workload status in our state
    setWorkloads(prevWorkloads => 
      prevWorkloads.map(w => {
        if (w.id === id && (w.status === WorkloadStatus.PENDING || w.status === WorkloadStatus.RUNNING)) {
          return {
            ...w,
            status: WorkloadStatus.FAILED,
            endTime: Date.now(),
            error: 'Cancelled by user'
          };
        }
        return w;
      })
    );
    
    // In a real app, we'd return the success status from the API response
    return true;
  };

  // Get a workload by ID
  const getWorkloadById = (id: string): Workload | undefined => {
    return workloads.find(w => w.id === id);
  };

  // Function to refresh benchmark data
  const refreshBenchmarkData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      // In a real app, this would make an API call to fetch benchmark data
      // For now, we'll use mock data
      setTimeout(() => {
        // Mock benchmark results
        const mockBenchmarkResults: BenchmarkResult[] = [
          {
            workload_name: 'nbody_sim',
            device_name: 'NVIDIA Jetson Orin NX',
            device_capabilities: {
              name: 'NVIDIA Jetson Orin NX',
              compute_capability: '8.7',
              total_memory: 16384,
              clock_rate: 1000,
              num_multiprocessors: 16
            },
            execution_time: 12.5,
            memory_usage: {
              host: 256,
              device: 2048
            },
            gpu_utilization: 95.2,
            energy_consumption: 350.75,
            throughput: 80.0,
            additional_metrics: {
              num_particles: 10000,
              num_steps: 1000,
              interactions_per_second: 8.0e9
            },
            timestamp: '2025-01-15T12:30:45'
          },
          {
            workload_name: 'nbody_sim',
            device_name: 'NVIDIA T4',
            device_capabilities: {
              name: 'NVIDIA T4',
              compute_capability: '7.5',
              total_memory: 16384,
              clock_rate: 1590,
              num_multiprocessors: 40
            },
            execution_time: 4.8,
            memory_usage: {
              host: 312,
              device: 2560
            },
            gpu_utilization: 97.5,
            energy_consumption: 520.45,
            throughput: 208.3,
            additional_metrics: {
              num_particles: 10000,
              num_steps: 1000,
              interactions_per_second: 2.1e10
            },
            timestamp: '2025-01-15T12:45:30'
          },
          {
            workload_name: 'molecular_dynamics',
            device_name: 'NVIDIA Jetson Orin NX',
            device_capabilities: {
              name: 'NVIDIA Jetson Orin NX',
              compute_capability: '8.7',
              total_memory: 16384,
              clock_rate: 1000,
              num_multiprocessors: 16
            },
            execution_time: 18.2,
            memory_usage: {
              host: 420,
              device: 1800
            },
            gpu_utilization: 92.4,
            energy_consumption: 410.35,
            throughput: 54.9,
            additional_metrics: {
              num_atoms: 5000,
              num_steps: 1000,
              interactions_per_second: 1.37e9
            },
            timestamp: '2025-01-15T13:15:10'
          },
          {
            workload_name: 'molecular_dynamics',
            device_name: 'NVIDIA T4',
            device_capabilities: {
              name: 'NVIDIA T4',
              compute_capability: '7.5',
              total_memory: 16384,
              clock_rate: 1590,
              num_multiprocessors: 40
            },
            execution_time: 7.1,
            memory_usage: {
              host: 480,
              device: 2240
            },
            gpu_utilization: 96.8,
            energy_consumption: 610.25,
            throughput: 140.8,
            additional_metrics: {
              num_atoms: 5000,
              num_steps: 1000,
              interactions_per_second: 3.52e9
            },
            timestamp: '2025-01-15T13:30:22'
          },
          {
            workload_name: 'weather_sim',
            device_name: 'NVIDIA Jetson Orin NX',
            device_capabilities: {
              name: 'NVIDIA Jetson Orin NX',
              compute_capability: '8.7',
              total_memory: 16384,
              clock_rate: 1000,
              num_multiprocessors: 16
            },
            execution_time: 21.5,
            memory_usage: {
              host: 850,
              device: 3200
            },
            gpu_utilization: 94.7,
            energy_consumption: 490.85,
            throughput: 46.5,
            additional_metrics: {
              grid_size: 512,
              num_steps: 1000,
              grid_points_per_second: 1.22e7
            },
            timestamp: '2025-01-15T14:10:05'
          },
          {
            workload_name: 'weather_sim',
            device_name: 'NVIDIA T4',
            device_capabilities: {
              name: 'NVIDIA T4',
              compute_capability: '7.5',
              total_memory: 16384,
              clock_rate: 1590,
              num_multiprocessors: 40
            },
            execution_time: 8.6,
            memory_usage: {
              host: 920,
              device: 4096
            },
            gpu_utilization: 98.1,
            energy_consumption: 720.55,
            throughput: 116.3,
            additional_metrics: {
              grid_size: 512,
              num_steps: 1000,
              grid_points_per_second: 3.05e7
            },
            timestamp: '2025-01-15T14:25:35'
          },
          {
            workload_name: 'medical_imaging_ct_reconstruction',
            device_name: 'NVIDIA Jetson Orin NX',
            device_capabilities: {
              name: 'NVIDIA Jetson Orin NX',
              compute_capability: '8.7',
              total_memory: 16384,
              clock_rate: 1000,
              num_multiprocessors: 16
            },
            execution_time: 15.3,
            memory_usage: {
              host: 760,
              device: 2800
            },
            gpu_utilization: 93.8,
            energy_consumption: 380.45,
            throughput: 0.65,
            additional_metrics: {
              image_size: 512,
              num_iterations: 10,
              num_angles: 180,
              method: 'FilteredBackProjection'
            },
            timestamp: '2025-01-15T15:05:15'
          },
          {
            workload_name: 'medical_imaging_ct_reconstruction',
            device_name: 'NVIDIA T4',
            device_capabilities: {
              name: 'NVIDIA T4',
              compute_capability: '7.5',
              total_memory: 16384,
              clock_rate: 1590,
              num_multiprocessors: 40
            },
            execution_time: 5.9,
            memory_usage: {
              host: 840,
              device: 3500
            },
            gpu_utilization: 97.2,
            energy_consumption: 560.30,
            throughput: 1.69,
            additional_metrics: {
              image_size: 512,
              num_iterations: 10,
              num_angles: 180,
              method: 'FilteredBackProjection'
            },
            timestamp: '2025-01-15T15:20:40'
          }
        ];
        
        setBenchmarkResults(mockBenchmarkResults);
        setIsLoading(false);
      }, 500);
    } catch (err) {
      console.error('Error fetching benchmark data:', err);
      setError('Failed to fetch benchmark data from server');
      setIsLoading(false);
    }
  };

  // Context value
  const value = {
    workloads,
    runningWorkloads,
    completedWorkloads,
    nodes,
    benchmarkResults,
    isLoading,
    error,
    refreshData,
    launchWorkload,
    cancelWorkload,
    getWorkloadById,
    refreshBenchmarkData
  };

  return (
    <WorkloadContext.Provider value={value}>
      {children}
    </WorkloadContext.Provider>
  );
};

// Custom hook to use the workload context
export const useWorkload = () => {
  const context = useContext(WorkloadContext);
  if (context === undefined) {
    throw new Error('useWorkload must be used within a WorkloadProvider');
  }
  return context;
};