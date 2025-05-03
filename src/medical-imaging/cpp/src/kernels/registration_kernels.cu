/**
 * @file registration_kernels.cu
 * @brief CUDA kernels for medical image registration.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <vector_types.h>
#include <cmath>
#include <stdio.h>

#include "../../include/medical_imaging/medical_imaging.hpp"
#include "../../include/medical_imaging/gpu_adaptability.hpp"

namespace cg = cooperative_groups;

namespace medical_imaging {

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaError_t::cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

// Device constants for registration parameters
__constant__ int d_width;                     // Image width
__constant__ int d_height;                    // Image height
__constant__ int d_depth;                     // Image depth (for 3D)
__constant__ int d_channels;                  // Number of channels
__constant__ float d_transform_matrix[16];    // Transformation matrix (4x4)
__constant__ int d_interpolation_mode;        // Interpolation mode (0: nearest, 1: linear, 2: cubic)

/**
 * @brief Image warping kernel for 2D images (optimized for high-end GPUs, SM >= 8.0).
 * 
 * This kernel applies a geometric transformation to an image.
 * 
 * @param input Input image
 * @param output Output transformed image
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void imageWarpingKernel_SM80(
    const float* __restrict__ input,
    float* __restrict__ output
) {
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    const int c = blockIdx.z;  // Channel index
    
    // Check boundaries
    if (x >= d_width || y >= d_height || c >= d_channels) {
        return;
    }
    
    // Output index
    const int out_idx = (y * d_width + x) * d_channels + c;
    
    // Apply transform (inverse mapping)
    // Calculate source coordinates by applying inverse transformation
    
    // Homogeneous coordinates
    float src_x, src_y, src_z, w;
    
    // Apply transformation matrix (inverse mapping)
    // We are using a 3x3 or 4x4 transformation matrix in row-major order
    if (d_depth > 1) {
        // 3D transformation - not implemented in this example
        output[out_idx] = 0.0f;
        return;
    } else {
        // 2D transformation using 3x3 matrix (stored in the first 3x3 part of 4x4 matrix)
        src_x = d_transform_matrix[0] * x + d_transform_matrix[1] * y + d_transform_matrix[2];
        src_y = d_transform_matrix[3] * x + d_transform_matrix[4] * y + d_transform_matrix[5];
        w = d_transform_matrix[6] * x + d_transform_matrix[7] * y + d_transform_matrix[8];
        
        // Perspective division
        if (fabs(w) > 1e-10f) {
            src_x /= w;
            src_y /= w;
        }
    }
    
    // Check if source coordinates are within bounds
    if (src_x < 0 || src_x >= d_width - 1 || src_y < 0 || src_y >= d_height - 1) {
        output[out_idx] = 0.0f;  // Outside image bounds
        return;
    }
    
    // Interpolate based on mode
    float result = 0.0f;
    
    switch (d_interpolation_mode) {
        case 0:  // Nearest neighbor
        {
            const int nx = __float2int_rd(src_x + 0.5f);
            const int ny = __float2int_rd(src_y + 0.5f);
            const int src_idx = (ny * d_width + nx) * d_channels + c;
            result = input[src_idx];
            break;
        }
        
        case 1:  // Bilinear interpolation
        {
            const int x0 = __float2int_rd(src_x);
            const int y0 = __float2int_rd(src_y);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            
            const float wx = src_x - x0;
            const float wy = src_y - y0;
            
            const float val00 = input[(y0 * d_width + x0) * d_channels + c];
            const float val01 = input[(y0 * d_width + x1) * d_channels + c];
            const float val10 = input[(y1 * d_width + x0) * d_channels + c];
            const float val11 = input[(y1 * d_width + x1) * d_channels + c];
            
            const float val0 = (1.0f - wx) * val00 + wx * val01;
            const float val1 = (1.0f - wx) * val10 + wx * val11;
            
            result = (1.0f - wy) * val0 + wy * val1;
            break;
        }
        
        case 2:  // Bicubic interpolation (simplified)
        {
            // This is a simplified bicubic implementation
            // A full implementation would use 16 neighbors and cubic weighting
            
            const int x0 = __float2int_rd(src_x);
            const int y0 = __float2int_rd(src_y);
            
            // Calculate fractional parts
            const float dx = src_x - x0;
            const float dy = src_y - y0;
            
            // Cubic weights
            const float wx0 = -0.5f * dx * dx * dx + dx * dx - 0.5f * dx;
            const float wx1 = 1.5f * dx * dx * dx - 2.5f * dx * dx + 1.0f;
            const float wx2 = -1.5f * dx * dx * dx + 2.0f * dx * dx + 0.5f * dx;
            const float wx3 = 0.5f * dx * dx * dx - 0.5f * dx * dx;
            
            const float wy0 = -0.5f * dy * dy * dy + dy * dy - 0.5f * dy;
            const float wy1 = 1.5f * dy * dy * dy - 2.5f * dy * dy + 1.0f;
            const float wy2 = -1.5f * dy * dy * dy + 2.0f * dy * dy + 0.5f * dy;
            const float wy3 = 0.5f * dy * dy * dy - 0.5f * dy * dy;
            
            // Initialize accumulator
            result = 0.0f;
            
            // Accumulate contributions from 16 neighbors
            for (int j = -1; j <= 2; j++) {
                const int y_idx = y0 + j;
                
                // Skip out-of-bounds pixels
                if (y_idx < 0 || y_idx >= d_height) {
                    continue;
                }
                
                // Select y weight
                float wy;
                switch (j) {
                    case -1: wy = wy0; break;
                    case 0:  wy = wy1; break;
                    case 1:  wy = wy2; break;
                    case 2:  wy = wy3; break;
                }
                
                for (int i = -1; i <= 2; i++) {
                    const int x_idx = x0 + i;
                    
                    // Skip out-of-bounds pixels
                    if (x_idx < 0 || x_idx >= d_width) {
                        continue;
                    }
                    
                    // Select x weight
                    float wx;
                    switch (i) {
                        case -1: wx = wx0; break;
                        case 0:  wx = wx1; break;
                        case 1:  wx = wx2; break;
                        case 2:  wx = wx3; break;
                    }
                    
                    // Get pixel value
                    const float val = input[(y_idx * d_width + x_idx) * d_channels + c];
                    
                    // Accumulate weighted value
                    result += wx * wy * val;
                }
            }
            break;
        }
        
        default:  // Default to nearest neighbor
        {
            const int nx = __float2int_rd(src_x + 0.5f);
            const int ny = __float2int_rd(src_y + 0.5f);
            const int src_idx = (ny * d_width + nx) * d_channels + c;
            result = input[src_idx];
            break;
        }
    }
    
    // Store result
    output[out_idx] = result;
}

/**
 * @brief Image warping kernel for 2D images (optimized for T4 GPU, SM 7.5).
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void imageWarpingKernel_SM75(
    const float* __restrict__ input,
    float* __restrict__ output
) {
    // Use cooperative groups for better warp-level control
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int c = blockIdx.z;  // Channel index
    
    // Check boundaries
    if (x >= d_width || y >= d_height || c >= d_channels) {
        return;
    }
    
    // Output index
    const int out_idx = (y * d_width + x) * d_channels + c;
    
    // Apply transform (inverse mapping)
    float src_x, src_y, src_z, w;
    
    // Apply transformation matrix (inverse mapping)
    if (d_depth > 1) {
        // 3D transformation - not implemented in this example
        output[out_idx] = 0.0f;
        return;
    } else {
        // 2D transformation using 3x3 matrix
        src_x = d_transform_matrix[0] * x + d_transform_matrix[1] * y + d_transform_matrix[2];
        src_y = d_transform_matrix[3] * x + d_transform_matrix[4] * y + d_transform_matrix[5];
        w = d_transform_matrix[6] * x + d_transform_matrix[7] * y + d_transform_matrix[8];
        
        // Perspective division
        if (fabs(w) > 1e-10f) {
            src_x /= w;
            src_y /= w;
        }
    }
    
    // Check if source coordinates are within bounds
    if (src_x < 0 || src_x >= d_width - 1 || src_y < 0 || src_y >= d_height - 1) {
        output[out_idx] = 0.0f;  // Outside image bounds
        return;
    }
    
    // Interpolate based on mode
    float result = 0.0f;
    
    // Use warp-level optimization for T4
    // Each warp handles interpolation for 32 consecutive pixels
    
    // Use the same interpolation logic as SM80 kernel but with warp-optimized access
    switch (d_interpolation_mode) {
        case 0:  // Nearest neighbor
        {
            const int nx = __float2int_rd(src_x + 0.5f);
            const int ny = __float2int_rd(src_y + 0.5f);
            const int src_idx = (ny * d_width + nx) * d_channels + c;
            result = input[src_idx];
            break;
        }
        
        case 1:  // Bilinear interpolation
        {
            const int x0 = __float2int_rd(src_x);
            const int y0 = __float2int_rd(src_y);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            
            const float wx = src_x - x0;
            const float wy = src_y - y0;
            
            const float val00 = input[(y0 * d_width + x0) * d_channels + c];
            const float val01 = input[(y0 * d_width + x1) * d_channels + c];
            const float val10 = input[(y1 * d_width + x0) * d_channels + c];
            const float val11 = input[(y1 * d_width + x1) * d_channels + c];
            
            const float val0 = (1.0f - wx) * val00 + wx * val01;
            const float val1 = (1.0f - wx) * val10 + wx * val11;
            
            result = (1.0f - wy) * val0 + wy * val1;
            break;
        }
        
        case 2:  // Bicubic interpolation (simplified)
        {
            // Simplified bicubic implementation as in SM80 kernel
            const int x0 = __float2int_rd(src_x);
            const int y0 = __float2int_rd(src_y);
            
            // Calculate fractional parts
            const float dx = src_x - x0;
            const float dy = src_y - y0;
            
            // Cubic weights
            const float wx0 = -0.5f * dx * dx * dx + dx * dx - 0.5f * dx;
            const float wx1 = 1.5f * dx * dx * dx - 2.5f * dx * dx + 1.0f;
            const float wx2 = -1.5f * dx * dx * dx + 2.0f * dx * dx + 0.5f * dx;
            const float wx3 = 0.5f * dx * dx * dx - 0.5f * dx * dx;
            
            const float wy0 = -0.5f * dy * dy * dy + dy * dy - 0.5f * dy;
            const float wy1 = 1.5f * dy * dy * dy - 2.5f * dy * dy + 1.0f;
            const float wy2 = -1.5f * dy * dy * dy + 2.0f * dy * dy + 0.5f * dy;
            const float wy3 = 0.5f * dy * dy * dy - 0.5f * dy * dy;
            
            // Initialize accumulator
            result = 0.0f;
            
            // Loop unrolling for T4 performance
            #pragma unroll 4
            for (int j = -1; j <= 2; j++) {
                const int y_idx = y0 + j;
                
                // Skip out-of-bounds pixels
                if (y_idx < 0 || y_idx >= d_height) {
                    continue;
                }
                
                // Select y weight
                float wy;
                switch (j) {
                    case -1: wy = wy0; break;
                    case 0:  wy = wy1; break;
                    case 1:  wy = wy2; break;
                    case 2:  wy = wy3; break;
                }
                
                #pragma unroll 4
                for (int i = -1; i <= 2; i++) {
                    const int x_idx = x0 + i;
                    
                    // Skip out-of-bounds pixels
                    if (x_idx < 0 || x_idx >= d_width) {
                        continue;
                    }
                    
                    // Select x weight
                    float wx;
                    switch (i) {
                        case -1: wx = wx0; break;
                        case 0:  wx = wx1; break;
                        case 1:  wx = wx2; break;
                        case 2:  wx = wx3; break;
                    }
                    
                    // Get pixel value
                    const float val = input[(y_idx * d_width + x_idx) * d_channels + c];
                    
                    // Accumulate weighted value
                    result += wx * wy * val;
                }
            }
            break;
        }
        
        default:  // Default to nearest neighbor
        {
            const int nx = __float2int_rd(src_x + 0.5f);
            const int ny = __float2int_rd(src_y + 0.5f);
            const int src_idx = (ny * d_width + nx) * d_channels + c;
            result = input[src_idx];
            break;
        }
    }
    
    // Store result
    output[out_idx] = result;
}

/**
 * @brief Image warping kernel for 2D images (optimized for Jetson Orin NX, SM 8.7).
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void imageWarpingKernel_SM87(
    const float* __restrict__ input,
    float* __restrict__ output
) {
    // Use cooperative groups
    cg::thread_block block = cg::this_thread_block();
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    const int c = blockIdx.z;  // Channel index
    
    // Check boundaries
    if (x >= d_width || y >= d_height || c >= d_channels) {
        return;
    }
    
    // Output index
    const int out_idx = (y * d_width + x) * d_channels + c;
    
    // Apply transform (inverse mapping)
    float src_x, src_y, src_z, w;
    
    // Apply transformation matrix (inverse mapping)
    if (d_depth > 1) {
        // 3D transformation - not implemented in this example
        output[out_idx] = 0.0f;
        return;
    } else {
        // 2D transformation using 3x3 matrix with fused multiply-add for Ampere
        src_x = fmaf(d_transform_matrix[0], x, fmaf(d_transform_matrix[1], y, d_transform_matrix[2]));
        src_y = fmaf(d_transform_matrix[3], x, fmaf(d_transform_matrix[4], y, d_transform_matrix[5]));
        w = fmaf(d_transform_matrix[6], x, fmaf(d_transform_matrix[7], y, d_transform_matrix[8]));
        
        // Perspective division
        if (fabs(w) > 1e-10f) {
            const float inv_w = 1.0f / w;
            src_x *= inv_w;
            src_y *= inv_w;
        }
    }
    
    // Check if source coordinates are within bounds
    if (src_x < 0 || src_x >= d_width - 1 || src_y < 0 || src_y >= d_height - 1) {
        output[out_idx] = 0.0f;  // Outside image bounds
        return;
    }
    
    // Interpolate based on mode
    float result = 0.0f;
    
    // Use the same interpolation logic as SM80 kernel but with Ampere optimizations
    switch (d_interpolation_mode) {
        case 0:  // Nearest neighbor
        {
            const int nx = __float2int_rd(src_x + 0.5f);
            const int ny = __float2int_rd(src_y + 0.5f);
            const int src_idx = (ny * d_width + nx) * d_channels + c;
            result = input[src_idx];
            break;
        }
        
        case 1:  // Bilinear interpolation
        {
            const int x0 = __float2int_rd(src_x);
            const int y0 = __float2int_rd(src_y);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            
            const float wx = src_x - x0;
            const float wy = src_y - y0;
            const float wx1 = 1.0f - wx;
            const float wy1 = 1.0f - wy;
            
            const float val00 = input[(y0 * d_width + x0) * d_channels + c];
            const float val01 = input[(y0 * d_width + x1) * d_channels + c];
            const float val10 = input[(y1 * d_width + x0) * d_channels + c];
            const float val11 = input[(y1 * d_width + x1) * d_channels + c];
            
            // Use fused multiply-add for better performance on Ampere
            const float val0 = fmaf(wx1, val00, wx * val01);
            const float val1 = fmaf(wx1, val10, wx * val11);
            
            result = fmaf(wy1, val0, wy * val1);
            break;
        }
        
        case 2:  // Bicubic interpolation (simplified)
        {
            // Simplified bicubic implementation optimized for Ampere
            const int x0 = __float2int_rd(src_x);
            const int y0 = __float2int_rd(src_y);
            
            // Calculate fractional parts
            const float dx = src_x - x0;
            const float dy = src_y - y0;
            
            // Cubic weights with fused multiply-add
            const float dx2 = dx * dx;
            const float dx3 = dx2 * dx;
            const float dy2 = dy * dy;
            const float dy3 = dy2 * dy;
            
            const float wx0 = fmaf(-0.5f, dx3, fmaf(dx2, 1.0f, -0.5f * dx));
            const float wx1 = fmaf(1.5f, dx3, fmaf(-2.5f, dx2, 1.0f));
            const float wx2 = fmaf(-1.5f, dx3, fmaf(2.0f, dx2, 0.5f * dx));
            const float wx3 = fmaf(0.5f, dx3, -0.5f * dx2);
            
            const float wy0 = fmaf(-0.5f, dy3, fmaf(dy2, 1.0f, -0.5f * dy));
            const float wy1 = fmaf(1.5f, dy3, fmaf(-2.5f, dy2, 1.0f));
            const float wy2 = fmaf(-1.5f, dy3, fmaf(2.0f, dy2, 0.5f * dy));
            const float wy3 = fmaf(0.5f, dy3, -0.5f * dy2);
            
            // Initialize accumulator
            result = 0.0f;
            
            // Manual loop unrolling for Ampere
            for (int j = -1; j <= 2; j++) {
                const int y_idx = y0 + j;
                
                // Skip out-of-bounds pixels
                if (y_idx < 0 || y_idx >= d_height) {
                    continue;
                }
                
                // Select y weight
                float wy;
                switch (j) {
                    case -1: wy = wy0; break;
                    case 0:  wy = wy1; break;
                    case 1:  wy = wy2; break;
                    case 2:  wy = wy3; break;
                }
                
                for (int i = -1; i <= 2; i++) {
                    const int x_idx = x0 + i;
                    
                    // Skip out-of-bounds pixels
                    if (x_idx < 0 || x_idx >= d_width) {
                        continue;
                    }
                    
                    // Select x weight
                    float wx;
                    switch (i) {
                        case -1: wx = wx0; break;
                        case 0:  wx = wx1; break;
                        case 1:  wx = wx2; break;
                        case 2:  wx = wx3; break;
                    }
                    
                    // Get pixel value
                    const float val = input[(y_idx * d_width + x_idx) * d_channels + c];
                    
                    // Accumulate weighted value with fused multiply-add
                    result = fmaf(wx * wy, val, result);
                }
            }
            break;
        }
        
        default:  // Default to nearest neighbor
        {
            const int nx = __float2int_rd(src_x + 0.5f);
            const int ny = __float2int_rd(src_y + 0.5f);
            const int src_idx = (ny * d_width + nx) * d_channels + c;
            result = input[src_idx];
            break;
        }
    }
    
    // Store result
    output[out_idx] = result;
}

/**
 * @brief Generic image warping CUDA kernel (works on all CUDA-capable GPUs).
 */
__global__ void imageWarpingKernel_Generic(
    const float* __restrict__ input,
    float* __restrict__ output
) {
    // Global indices
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z;  // Channel index
    
    // Check boundaries
    if (x >= d_width || y >= d_height || c >= d_channels) {
        return;
    }
    
    // Output index
    const int out_idx = (y * d_width + x) * d_channels + c;
    
    // Apply transform (inverse mapping)
    float src_x, src_y, w;
    
    // Apply transformation matrix (inverse mapping)
    if (d_depth > 1) {
        // 3D transformation - not implemented in this example
        output[out_idx] = 0.0f;
        return;
    } else {
        // 2D transformation using 3x3 matrix
        src_x = d_transform_matrix[0] * x + d_transform_matrix[1] * y + d_transform_matrix[2];
        src_y = d_transform_matrix[3] * x + d_transform_matrix[4] * y + d_transform_matrix[5];
        w = d_transform_matrix[6] * x + d_transform_matrix[7] * y + d_transform_matrix[8];
        
        // Perspective division
        if (fabs(w) > 1e-10f) {
            src_x /= w;
            src_y /= w;
        }
    }
    
    // Check if source coordinates are within bounds
    if (src_x < 0 || src_x >= d_width - 1 || src_y < 0 || src_y >= d_height - 1) {
        output[out_idx] = 0.0f;  // Outside image bounds
        return;
    }
    
    // Interpolate based on mode
    float result = 0.0f;
    
    switch (d_interpolation_mode) {
        case 0:  // Nearest neighbor
        {
            const int nx = __float2int_rd(src_x + 0.5f);
            const int ny = __float2int_rd(src_y + 0.5f);
            const int src_idx = (ny * d_width + nx) * d_channels + c;
            result = input[src_idx];
            break;
        }
        
        case 1:  // Bilinear interpolation
        {
            const int x0 = __float2int_rd(src_x);
            const int y0 = __float2int_rd(src_y);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            
            const float wx = src_x - x0;
            const float wy = src_y - y0;
            
            const float val00 = input[(y0 * d_width + x0) * d_channels + c];
            const float val01 = input[(y0 * d_width + x1) * d_channels + c];
            const float val10 = input[(y1 * d_width + x0) * d_channels + c];
            const float val11 = input[(y1 * d_width + x1) * d_channels + c];
            
            const float val0 = (1.0f - wx) * val00 + wx * val01;
            const float val1 = (1.0f - wx) * val10 + wx * val11;
            
            result = (1.0f - wy) * val0 + wy * val1;
            break;
        }
        
        default:  // Default to nearest neighbor
        {
            const int nx = __float2int_rd(src_x + 0.5f);
            const int ny = __float2int_rd(src_y + 0.5f);
            const int src_idx = (ny * d_width + nx) * d_channels + c;
            result = input[src_idx];
            break;
        }
    }
    
    // Store result
    output[out_idx] = result;
}

/**
 * @brief Mutual information kernel (optimized for high-end GPUs, SM >= 8.0).
 * 
 * This kernel calculates joint histogram for mutual information computation.
 * 
 * @param image1 First image
 * @param image2 Second image
 * @param histogram Joint histogram (output)
 * @param num_bins Number of histogram bins
 * @param max_val Maximum image value
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void mutualInformationKernel_SM80(
    const float* __restrict__ image1,
    const float* __restrict__ image2,
    int* __restrict__ histogram,
    int num_bins,
    float max_val
) {
    // Shared memory for block-level histogram
    extern __shared__ int shared_histogram[];
    
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Linear thread index within block
    const int thread_idx = ty * BLOCK_SIZE_X + tx;
    
    // Initialize shared histogram to zero
    for (int i = thread_idx; i < num_bins * num_bins; i += BLOCK_SIZE_X * BLOCK_SIZE_Y) {
        shared_histogram[i] = 0;
    }
    
    // Ensure all threads have initialized shared memory
    __syncthreads();
    
    // Calculate histogram
    if (x < d_width && y < d_height) {
        // Linear index
        const int idx = y * d_width + x;
        
        // Get pixel values
        const float val1 = image1[idx];
        const float val2 = image2[idx];
        
        // Normalize to bin indices
        const int bin1 = min(static_cast<int>(val1 * num_bins / max_val), num_bins - 1);
        const int bin2 = min(static_cast<int>(val2 * num_bins / max_val), num_bins - 1);
        
        // Calculate joint histogram index
        const int hist_idx = bin1 * num_bins + bin2;
        
        // Atomically increment histogram
        atomicAdd(&shared_histogram[hist_idx], 1);
    }
    
    // Ensure all threads have updated shared histogram
    __syncthreads();
    
    // Merge block-level histograms into global histogram
    for (int i = thread_idx; i < num_bins * num_bins; i += BLOCK_SIZE_X * BLOCK_SIZE_Y) {
        if (shared_histogram[i] > 0) {
            atomicAdd(&histogram[i], shared_histogram[i]);
        }
    }
}

// Wrapper functions to launch the appropriate kernel based on device capabilities

/**
 * @brief Launch image warping kernel for the appropriate device.
 * @param d_input Device pointer to input image
 * @param d_output Device pointer to output image
 * @param width Image width
 * @param height Image height
 * @param depth Image depth (for 3D)
 * @param channels Number of channels
 * @param transformation_matrix Transformation matrix (4x4 or 3x3)
 * @param interpolation_mode Interpolation mode (0: nearest, 1: linear, 2: cubic)
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchImageWarpingKernel(
    float* d_input,
    float* d_output,
    int width,
    int height,
    int depth,
    int channels,
    const std::vector<float>& transformation_matrix,
    int interpolation_mode,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Validate input
    if (transformation_matrix.size() < 9) {
        fprintf(stderr, "Error: Transformation matrix must have at least 9 elements for 2D or 16 for 3D\n");
        return false;
    }
    
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_depth, &depth, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_channels, &channels, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_interpolation_mode, &interpolation_mode, sizeof(int)));
    
    // Copy transformation matrix (using at most 16 elements)
    float transform_matrix[16] = {0.0f};
    for (size_t i = 0; i < std::min(transformation_matrix.size(), size_t(16)); ++i) {
        transform_matrix[i] = transformation_matrix[i];
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_transform_matrix, transform_matrix, 16 * sizeof(float)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Determine block and grid dimensions
    dim3 blockDim(params.block_size_x, params.block_size_y, 1);
    dim3 gridDim(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        channels
    );
    
    // Launch appropriate kernel based on device capabilities
    if (device_caps.compute_capability_major > 8 || 
        (device_caps.compute_capability_major == 8 && device_caps.compute_capability_minor >= 7)) {
        // Jetson Orin NX (SM 8.7)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            imageWarpingKernel_SM87<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            imageWarpingKernel_SM87<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output
            );
        } else {
            imageWarpingKernel_SM87<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output
            );
        }
    } else if (device_caps.compute_capability_major == 8) {
        // High-end GPUs (SM 8.0+)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            imageWarpingKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            imageWarpingKernel_SM80<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output
            );
        } else {
            imageWarpingKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output
            );
        }
    } else if (device_caps.compute_capability_major == 7 && device_caps.compute_capability_minor >= 5) {
        // T4 GPU (SM 7.5)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            imageWarpingKernel_SM75<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            imageWarpingKernel_SM75<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output
            );
        } else {
            imageWarpingKernel_SM75<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output
            );
        }
    } else {
        // Generic version for older GPUs
        dim3 genericBlockDim(16, 16, 1);  // Default block size for older GPUs
        dim3 genericGridDim(
            (width + genericBlockDim.x - 1) / genericBlockDim.x,
            (height + genericBlockDim.y - 1) / genericBlockDim.y,
            channels
        );
        
        imageWarpingKernel_Generic<<<genericGridDim, genericBlockDim, 0, stream>>>(
            d_input, d_output
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

/**
 * @brief Launch mutual information kernel for the appropriate device.
 * @param d_image1 Device pointer to first image
 * @param d_image2 Device pointer to second image
 * @param width Image width
 * @param height Image height
 * @param num_bins Number of histogram bins
 * @param max_val Maximum image value
 * @param mi_value Output mutual information value
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchMutualInformationKernel(
    float* d_image1,
    float* d_image2,
    int width,
    int height,
    int num_bins,
    float max_val,
    float& mi_value,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Allocate device memory for histogram
    int* d_histogram = nullptr;
    CUDA_CHECK(cudaMalloc(&d_histogram, num_bins * num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_histogram, 0, num_bins * num_bins * sizeof(int)));
    
    // Determine block and grid dimensions
    dim3 blockDim(params.block_size_x, params.block_size_y, 1);
    dim3 gridDim(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        1
    );
    
    // Shared memory size
    size_t sharedMemSize = num_bins * num_bins * sizeof(int);
    
    // Launch kernel for mutual information calculation
    if (device_caps.compute_capability_major >= 8) {
        // High-end GPUs (SM >= 8.0)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            mutualInformationKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_image1, d_image2, d_histogram, num_bins, max_val
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            mutualInformationKernel_SM80<32, 8><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_image1, d_image2, d_histogram, num_bins, max_val
            );
        } else {
            mutualInformationKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_image1, d_image2, d_histogram, num_bins, max_val
            );
        }
    } else {
        // Generic version for older GPUs
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            mutualInformationKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_image1, d_image2, d_histogram, num_bins, max_val
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            mutualInformationKernel_SM80<32, 8><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_image1, d_image2, d_histogram, num_bins, max_val
            );
        } else {
            mutualInformationKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_image1, d_image2, d_histogram, num_bins, max_val
            );
        }
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Copy the histogram back to host
    std::vector<int> h_histogram(num_bins * num_bins);
    CUDA_CHECK(cudaMemcpy(h_histogram.data(), d_histogram, num_bins * num_bins * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Compute marginal histograms and mutual information on host
    std::vector<int> h_hist1(num_bins, 0);
    std::vector<int> h_hist2(num_bins, 0);
    
    int total_count = 0;
    
    // Compute marginal histograms
    for (int i = 0; i < num_bins; ++i) {
        for (int j = 0; j < num_bins; ++j) {
            int count = h_histogram[i * num_bins + j];
            h_hist1[i] += count;
            h_hist2[j] += count;
            total_count += count;
        }
    }
    
    // Compute mutual information
    mi_value = 0.0f;
    
    if (total_count > 0) {
        const float total_count_inv = 1.0f / total_count;
        
        for (int i = 0; i < num_bins; ++i) {
            for (int j = 0; j < num_bins; ++j) {
                const int count = h_histogram[i * num_bins + j];
                
                if (count > 0) {
                    const float p_xy = count * total_count_inv;
                    const float p_x = h_hist1[i] * total_count_inv;
                    const float p_y = h_hist2[j] * total_count_inv;
                    
                    mi_value += p_xy * log2f(p_xy / (p_x * p_y));
                }
            }
        }
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_histogram));
    
    return true;
}

} // namespace medical_imaging