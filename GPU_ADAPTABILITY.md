# GPU Adaptability and System Utilization

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors

## Cross-Workload GPU Adaptability Strategy

All workloads in this repository are designed with the same core principles of GPU adaptability and system utilization in mind. Each application serves as an example of excellent multi-core and GPU programming that maintains high arithmetic throughput and coordination across heterogeneous hardware environments.

### Key Principles

1. **Automatic Hardware Detection and Optimization**
   - Dynamic detection of GPU compute capabilities
   - Runtime adaptation of kernel parameters based on available hardware
   - Graceful fallback to CPU when GPU is unavailable or insufficient

2. **Balanced Resource Utilization**
   - Concurrent CPU and GPU execution where appropriate
   - Asynchronous memory transfers overlapped with computation
   - Dynamic work distribution between host and device

3. **Compute Throughput Maximization**
   - Tiled algorithm implementations with tuned tile sizes per architecture
   - Arithmetic intensity optimization targeting different instruction sets
   - Memory access patterns optimized per device memory hierarchy

4. **Memory Hierarchy Utilization**
   - Register usage optimization per SM architecture
   - Shared memory allocation scaled to hardware capabilities
   - Automatic texture memory usage where beneficial
   - Coalesced memory access patterns

5. **Scaling Across Device Capabilities**
   - Jetson Orin NX (SM 8.7) specific optimizations
   - AWS Graviton g5g with T4 GPU (SM 7.5) specific optimizations
   - Auto-tuning for other NVIDIA architectures
   - CPU fallback with vectorization (AVX2/NEON)

## Implementation Details

### CUDA Kernel Adaptability

All workloads implement similar patterns for kernel adaptability:

```cpp
// Example from N-body simulation
void launch_compute_accelerations(
    const Vec3* positions,
    const scalar_t* masses,
    Vec3* accelerations,
    index_t num_particles,
    scalar_t G,
    scalar_t softening,
    const DeviceCapabilities& capabilities,
    cudaStream_t stream
) {
    // Choose implementation based on device capabilities
    if (capabilities.device_type == GPUDeviceType::JetsonOrin) {
        // Parameters optimized for Jetson Orin
        index_t block_size = 128;
        index_t tile_size = 16;
        launch_compute_accelerations_tiled(
            positions, masses, accelerations, num_particles, 
            G, softening, block_size, tile_size, stream
        );
    } 
    else if (capabilities.device_type == GPUDeviceType::T4) {
        // Parameters optimized for T4
        index_t block_size = 128;
        index_t tile_size = 16;
        bool use_texture_memory = true;
        launch_compute_accelerations_textured(
            positions, masses, accelerations, num_particles,
            G, softening, block_size, tile_size, use_texture_memory, stream
        );
    }
    else if (capabilities.compute_capability_major >= 7) {
        // Generic parameters for SM 7.0+
        index_t block_size = 256;
        launch_compute_accelerations_warp_optimized(
            positions, masses, accelerations, num_particles,
            G, softening, block_size, stream
        );
    }
    else if (capabilities.device_type == GPUDeviceType::CPU) {
        // CPU fallback with vectorization hints
        compute_accelerations_cpu_parallel(
            positions, masses, accelerations, num_particles,
            G, softening
        );
    }
    else {
        // Baseline implementation for any device
        index_t block_size = 256;
        launch_compute_accelerations_basic(
            positions, masses, accelerations, num_particles,
            G, softening, block_size, stream
        );
    }
}
```

### Advanced Kernel Specialization for Medical Imaging

The Medical Imaging workload extends this pattern with template-based kernel specialization:

```cpp
// Example from Medical Imaging workload's CT reconstruction
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void backprojectionKernel_SM87(
    const float* __restrict__ sinogram,
    float* __restrict__ image,
    const float* __restrict__ angles,
    int width, int height, int num_angles
) {
    // Specialized for Jetson Orin NX (SM 8.7)
    // Uses larger register count, cooperative groups, and shared memory optimizations
    
    // ... implementation details ...
}

template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void backprojectionKernel_SM75(
    const float* __restrict__ sinogram,
    float* __restrict__ image,
    const float* __restrict__ angles,
    int width, int height, int num_angles
) {
    // Specialized for T4 GPU (SM 7.5)
    // Uses texture memory for sinogram, optimized memory access pattern
    
    // ... implementation details ...
}

void launchBackprojection(
    const float* sinogram,
    float* image,
    const float* angles,
    int width, int height, int num_angles,
    const DeviceCapabilities& capabilities,
    cudaStream_t stream
) {
    // Choose block size based on device type
    dim3 blockSize;
    
    if (capabilities.device_type == GPUDeviceType::JetsonOrin) {
        // Optimized for Jetson Orin NX
        blockSize = dim3(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
                     
        backprojectionKernel_SM87<16, 16><<<gridSize, blockSize, 0, stream>>>(
            sinogram, image, angles, width, height, num_angles);
    }
    else if (capabilities.device_type == GPUDeviceType::T4) {
        // Optimized for T4
        blockSize = dim3(32, 8);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
                     
        backprojectionKernel_SM75<32, 8><<<gridSize, blockSize, 0, stream>>>(
            sinogram, image, angles, width, height, num_angles);
    }
    else if (capabilities.compute_capability_major >= 7) {
        // Generic SM 7.0+
        blockSize = dim3(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
                     
        backprojectionKernel_SM70<16, 16><<<gridSize, blockSize, 0, stream>>>(
            sinogram, image, angles, width, height, num_angles);
    }
    else {
        // Fallback for any GPU
        blockSize = dim3(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                     (height + blockSize.y - 1) / blockSize.y);
                     
        backprojectionKernel_Generic<16, 16><<<gridSize, blockSize, 0, stream>>>(
            sinogram, image, angles, width, height, num_angles);
    }
}
```

### CPU-GPU Work Distribution

Each workload implements balanced work distribution between CPU and GPU:

1. **Asynchronous Processing Pipeline:**
   - Input data preprocessing on CPU while GPU works on previous batch
   - Multiple CUDA streams for concurrent kernel execution
   - Pinned memory for efficient data transfer

2. **Hybrid Computation Strategy:**
   - Complex control flow handled on CPU
   - Compute-intensive operations offloaded to GPU
   - Dynamic work partitioning based on current load and capabilities

3. **Automatic Load Balancing:**
   - Runtime performance monitoring and feedback
   - Work stealing queue for idle processors
   - Adaptive batch sizing based on throughput measurements

## Performance Metrics and Monitoring

All workloads include comprehensive performance metrics:

1. **Hardware Utilization:**
   - SM occupancy monitoring
   - Memory bandwidth utilization
   - Compute vs. memory bound analysis

2. **Scaling Efficiency:**
   - Strong scaling (fixed problem size, varying resources)
   - Weak scaling (fixed work per resource, varying problem size)
   - Resource efficiency analysis

3. **Energy Efficiency:**
   - Performance per watt measurements
   - Idle vs. active power draw
   - Energy-optimal configuration detection

## Conclusion

The GPU adaptability strategy implemented across all workloads ensures that each application:
- Demonstrates excellent utilization of available hardware resources
- Maintains high arithmetic throughput regardless of underlying hardware
- Achieves optimal load balance across disparate system components
- Scales efficiently across different hardware capabilities
- Provides a learning resource for best practices in heterogeneous computing

This approach ensures that the workloads not only perform optimally on the target Jetson and AWS Graviton platforms but also serve as reference implementations of high-performance heterogeneous computing.