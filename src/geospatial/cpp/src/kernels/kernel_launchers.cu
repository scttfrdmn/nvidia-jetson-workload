/**
 * @file kernel_launchers.cu
 * @brief Launcher functions for CUDA kernels for DEM processing
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace geospatial {
namespace kernels {

// Forward declarations of the actual kernel functions from dem_kernels.cu
__global__ void viewshedKernel(
    const float* __restrict__ dem,
    float* __restrict__ viewshed,
    int width,
    int height,
    float observer_x,
    float observer_y,
    float observer_height,
    float max_radius,
    float cell_size,
    float curvature_coeff);

__global__ void terrainDerivativesKernel(
    const float* __restrict__ dem,
    float* __restrict__ slope,
    float* __restrict__ aspect,
    float* __restrict__ curvature,
    int width,
    int height,
    float cell_size,
    float z_factor);

__global__ void initFillSinksKernel(
    const float* __restrict__ dem,
    float* __restrict__ filled,
    int width,
    int height);

__global__ void fillSinksIterationKernel(
    const float* __restrict__ dem,
    float* __restrict__ filled,
    int width,
    int height,
    float z_limit,
    int* changed);

// Launcher functions that can be called from C++ code

extern "C" {

void launchViewshedKernel(
    const float* d_dem,
    float* d_viewshed,
    int width,
    int height,
    float observer_x,
    float observer_y,
    float observer_height,
    float max_radius,
    float cell_size,
    float curvature_coeff,
    cudaStream_t stream)
{
    // Calculate grid dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Launch kernel
    viewshedKernel<<<grid, block, 0, stream>>>(
        d_dem,
        d_viewshed,
        width,
        height,
        observer_x,
        observer_y,
        observer_height,
        max_radius,
        cell_size,
        curvature_coeff
    );
    
    // Wait for kernel to finish
    cudaStreamSynchronize(stream);
}

void launchTerrainDerivativesKernel(
    const float* d_dem,
    float* d_slope,
    float* d_aspect,
    float* d_curvature,
    int width,
    int height,
    float cell_size,
    float z_factor,
    cudaStream_t stream)
{
    // Calculate grid dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Launch kernel
    terrainDerivativesKernel<<<grid, block, 0, stream>>>(
        d_dem,
        d_slope,
        d_aspect,
        d_curvature,
        width,
        height,
        cell_size,
        z_factor
    );
    
    // Wait for kernel to finish
    cudaStreamSynchronize(stream);
}

void launchFillSinksKernel(
    const float* d_dem,
    float* d_filled,
    int width,
    int height,
    float z_limit,
    cudaStream_t stream)
{
    // Calculate grid dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    // Initialize filled DEM
    initFillSinksKernel<<<grid, block, 0, stream>>>(
        d_dem,
        d_filled,
        width,
        height
    );
    
    // Allocate device memory for changed flag
    int* d_changed;
    cudaMalloc(&d_changed, sizeof(int));
    
    // Iteratively fill sinks
    const int MAX_ITERATIONS = 100;
    for (int i = 0; i < MAX_ITERATIONS; i++) {
        // Reset changed flag
        cudaMemset(d_changed, 0, sizeof(int));
        
        // Launch kernel
        fillSinksIterationKernel<<<grid, block, 0, stream>>>(
            d_dem,
            d_filled,
            width,
            height,
            z_limit,
            d_changed
        );
        
        // Check if any cells were changed
        int h_changed;
        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        
        // If no cells were changed, we're done
        if (h_changed == 0) {
            break;
        }
    }
    
    // Clean up
    cudaFree(d_changed);
    
    // Wait for kernel to finish
    cudaStreamSynchronize(stream);
}

} // extern "C"

} // namespace kernels
} // namespace geospatial