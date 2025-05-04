# GPU Adaptability Design Pattern

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors

## Overview

This document defines the common GPU adaptability pattern implemented across all workloads in the NVIDIA Jetson Workload project. This pattern ensures that all applications efficiently utilize hardware resources and scale appropriately across different GPU architectures.

## Design Pattern Structure

1. **Device Capability Detection**
   ```cpp
   struct DeviceCapabilities {
       DeviceType device_type;              // Enum: JetsonOrin, T4, HighEnd, etc.
       int compute_capability_major;        // CUDA compute capability major version
       int compute_capability_minor;        // CUDA compute capability minor version
       size_t global_memory_bytes;          // Total global memory size
       int multiprocessor_count;            // Number of streaming multiprocessors
       // Additional capability info...
   };
   
   // Function to query device and initialize capabilities
   DeviceCapabilities detect_device_capabilities();
   ```

2. **Kernel Selection and Parameter Tuning**
   ```cpp
   // Kernel launcher function
   void launch_optimized_kernel(
       /* kernel parameters */
       const DeviceCapabilities& capabilities,
       WorkloadStrategy strategy,
       cudaStream_t stream
   ) {
       // Select based on device capabilities
       if (capabilities.device_type == DeviceType::JetsonOrin) {
           // Parameters optimized for Jetson Orin
           launch_kernel_for_jetson(/* ... */);
       } 
       else if (capabilities.device_type == DeviceType::T4) {
           // Parameters optimized for T4
           launch_kernel_for_t4(/* ... */);
       }
       else if (capabilities.compute_capability_major >= 7) {
           // Generic parameters for SM 7.0+
           launch_kernel_for_sm7(/* ... */);
       }
       else {
           // Fallback implementation
           launch_kernel_basic(/* ... */);
       }
   }
   ```

3. **Workload Distribution Strategy**
   ```cpp
   enum class WorkloadStrategy {
       GPUOnly,          // Use only GPU for computation
       CPUOnly,          // Use only CPU for computation
       Hybrid,           // Use both CPU and GPU with static division
       AdaptiveHybrid    // Use both CPU and GPU with dynamic balancing
   };
   ```

4. **CPU-GPU Hybrid Computing**
   ```cpp
   void compute_hybrid(
       /* data and parameters */
       const DeviceCapabilities& capabilities,
       WorkloadStrategy strategy
   ) {
       // Determine CPU/GPU split ratio based on capabilities
       float cpu_ratio = determine_optimal_ratio(capabilities, strategy);
       
       // Launch CPU threads for portion of work
       launch_cpu_computation(/* portion of data */);
       
       // Launch GPU kernels for remainder of work
       launch_gpu_computation(/* remainder of data */);
       
       // Synchronize results
       wait_for_completion();
   }
   ```

5. **Algorithm Specialization**
   - Template-based kernel specialization for different tile sizes
   - Memory access pattern optimization per architecture
   - Arithmetic throughput tuning based on available instructions

## Implementation Guidelines

### Device-Specific Optimizations

1. **Jetson Orin (SM 8.7)**
   - Tile size: 16
   - Block size: 128-256
   - Shared memory: Up to 48 KB
   - Special considerations: Power efficiency optimizations

2. **T4 (SM 7.5)**
   - Tile size: 8-16
   - Block size: 128
   - Shared memory: Up to 32 KB
   - Special considerations: Texture memory usage for bandwidth optimization

3. **High-End GPUs (SM 8.0+)**
   - Tile size: 32
   - Block size: 256-512
   - Shared memory: Up to 96 KB
   - Special considerations: Higher occupancy and concurrency

4. **CPU Fallback**
   - Vectorization where available (AVX2/NEON)
   - Multi-core parallelization
   - Cache optimization

### Memory Optimization

1. **Memory Transfer Patterns**
   ```cpp
   // Device memory management pattern
   class DeviceMemoryManager {
   public:
       // Asynchronous memory transfers with events
       void transfer_host_to_device_async(void* host_ptr, void* device_ptr, 
                                          size_t size, cudaStream_t stream);
       
       // Staged transfers for large data
       void transfer_staged(void* host_ptr, void* device_ptr, 
                           size_t total_size, size_t chunk_size);
       
       // Pinned memory management
       void* allocate_pinned(size_t size);
       void free_pinned(void* ptr);
   };
   ```

2. **Memory Access Patterns**
   - Coalesced memory access for global memory
   - Shared memory banking optimizations
   - Texture memory for read-only data with spatial locality
   - Constant memory for small, read-only parameters

### Dynamic Workload Balancing

1. **Performance Monitoring**
   ```cpp
   struct PerformanceMetrics {
       float gpu_compute_time;
       float cpu_compute_time;
       float transfer_time;
       float total_time;
       float energy_consumption;
   };
   
   // Collect metrics during execution
   PerformanceMetrics collect_metrics();
   
   // Adjust workload based on metrics
   void adjust_workload_distribution(const PerformanceMetrics& metrics);
   ```

2. **Feedback-driven Optimization**
   - Runtime performance monitoring
   - Dynamic adjustment of CPU/GPU workload ratio
   - Adaptive kernel parameter selection

## Example Implementations

All workloads in this project implement this pattern, with specific examples in:

1. **N-body Simulation**
   - `src/nbody_sim/cpp/src/adaptive_kernels.cu`
   - `src/nbody_sim/cpp/src/device_adaptor.cpp`

2. **Molecular Dynamics**
   - `src/molecular-dynamics/cpp/src/cuda_kernels.cu`
   - `src/molecular-dynamics/cpp/include/molecular_dynamics/common.hpp`

## Performance Testing

To validate the adaptability pattern:

1. Run each workload across target hardware:
   - Jetson Orin NX
   - AWS Graviton g5g with T4 GPU
   - Desktop/server with other NVIDIA GPUs

2. Collect metrics:
   - Execution time
   - Energy consumption
   - Resource utilization (CPU, GPU, memory)

3. Compare different strategies:
   - GPUOnly vs. CPUOnly vs. Hybrid vs. AdaptiveHybrid
   - Different tile sizes and kernel parameters

## Conclusion

This GPU adaptability pattern ensures that all workloads in the NVIDIA Jetson Workload project demonstrate excellent utilization of available hardware resources, maintain high arithmetic throughput regardless of underlying hardware, achieve optimal load balance across disparate system components, and scale efficiently across different hardware capabilities.