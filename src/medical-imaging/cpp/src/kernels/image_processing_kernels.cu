/**
 * @file image_processing_kernels.cu
 * @brief CUDA kernels for medical image processing.
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

// Device constants for image processing parameters
__constant__ int d_width;               // Image width
__constant__ int d_height;              // Image height
__constant__ int d_depth;               // Image depth
__constant__ int d_channels;            // Number of channels
__constant__ int d_kernel_size;         // Convolution kernel size
__constant__ float d_kernel[121];       // Convolution kernel (up to 11x11)

/**
 * @brief 2D convolution kernel (optimized for high-end GPUs, SM >= 8.0).
 * 
 * This kernel applies a 2D convolution filter to an image.
 * 
 * @param input Input image
 * @param output Output image
 * @param kernel_radius Radius of the convolution kernel
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void convolution2DKernel_SM80(
    const float* __restrict__ input,
    float* __restrict__ output,
    int kernel_radius
) {
    // Shared memory for input image tile
    extern __shared__ float shared_input[];
    
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
    
    // Kernel dimensions
    const int kernel_diameter = 2 * kernel_radius + 1;
    
    // Calculate shared memory tile size
    const int tile_width = BLOCK_SIZE_X + 2 * kernel_radius;
    const int tile_height = BLOCK_SIZE_Y + 2 * kernel_radius;
    
    // Local indices for shared memory
    const int lx = tx + kernel_radius;
    const int ly = ty + kernel_radius;
    
    // Load input image tile to shared memory
    // Each thread loads its own pixel and potentially some halo pixels
    for (int dy = 0; dy < tile_height; dy += BLOCK_SIZE_Y) {
        int sy = ly + dy - kernel_radius;
        int gy = y + dy - kernel_radius;
        
        if (sy >= 0 && sy < tile_height) {
            for (int dx = 0; dx < tile_width; dx += BLOCK_SIZE_X) {
                int sx = lx + dx - kernel_radius;
                int gx = x + dx - kernel_radius;
                
                if (sx >= 0 && sx < tile_width) {
                    // Handle boundary conditions with clamping
                    int src_x = max(0, min(d_width - 1, gx));
                    int src_y = max(0, min(d_height - 1, gy));
                    
                    // Load pixel to shared memory
                    shared_input[sy * tile_width + sx] = input[(src_y * d_width + src_x) * d_channels + c];
                }
            }
        }
    }
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Apply convolution
    float sum = 0.0f;
    
    for (int ky = 0; ky < kernel_diameter; ++ky) {
        for (int kx = 0; kx < kernel_diameter; ++kx) {
            const int kernel_idx = ky * kernel_diameter + kx;
            const int image_x = lx + kx - kernel_radius;
            const int image_y = ly + ky - kernel_radius;
            
            sum += shared_input[image_y * tile_width + image_x] * d_kernel[kernel_idx];
        }
    }
    
    // Store result
    output[(y * d_width + x) * d_channels + c] = sum;
}

/**
 * @brief 2D convolution kernel (optimized for T4 GPU, SM 7.5).
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void convolution2DKernel_SM75(
    const float* __restrict__ input,
    float* __restrict__ output,
    int kernel_radius
) {
    // Use cooperative groups for better warp-level control
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Shared memory for input image tile
    extern __shared__ float shared_input[];
    
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
    
    // Kernel dimensions
    const int kernel_diameter = 2 * kernel_radius + 1;
    
    // Calculate shared memory tile size
    const int tile_width = BLOCK_SIZE_X + 2 * kernel_radius;
    const int tile_height = BLOCK_SIZE_Y + 2 * kernel_radius;
    
    // Local indices for shared memory
    const int lx = tx + kernel_radius;
    const int ly = ty + kernel_radius;
    
    // Load input image tile to shared memory with warp-optimized loading
    // Each warp loads a contiguous region for better memory coalescing
    for (int i = warp.thread_rank(); i < tile_width * tile_height; i += warp.size()) {
        int sx = i % tile_width;
        int sy = i / tile_width;
        
        int gx = blockIdx.x * BLOCK_SIZE_X + sx - kernel_radius;
        int gy = blockIdx.y * BLOCK_SIZE_Y + sy - kernel_radius;
        
        // Handle boundary conditions with clamping
        int src_x = max(0, min(d_width - 1, gx));
        int src_y = max(0, min(d_height - 1, gy));
        
        // Load pixel to shared memory
        shared_input[sy * tile_width + sx] = input[(src_y * d_width + src_x) * d_channels + c];
    }
    
    // Ensure all threads have loaded their data
    block.sync();
    
    // Apply convolution with loop unrolling and optimization for T4
    float sum = 0.0f;
    
    // Unroll small kernel sizes for better performance on T4
    if (kernel_radius <= 2) {
        // 5x5 kernel or smaller
        #pragma unroll
        for (int ky = 0; ky < kernel_diameter; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < kernel_diameter; ++kx) {
                const int kernel_idx = ky * kernel_diameter + kx;
                const int image_x = lx + kx - kernel_radius;
                const int image_y = ly + ky - kernel_radius;
                
                sum += shared_input[image_y * tile_width + image_x] * d_kernel[kernel_idx];
            }
        }
    } else {
        // Larger kernels
        for (int ky = 0; ky < kernel_diameter; ++ky) {
            for (int kx = 0; kx < kernel_diameter; ++kx) {
                const int kernel_idx = ky * kernel_diameter + kx;
                const int image_x = lx + kx - kernel_radius;
                const int image_y = ly + ky - kernel_radius;
                
                sum += shared_input[image_y * tile_width + image_x] * d_kernel[kernel_idx];
            }
        }
    }
    
    // Store result
    output[(y * d_width + x) * d_channels + c] = sum;
}

/**
 * @brief 2D convolution kernel (optimized for Jetson Orin NX, SM 8.7).
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void convolution2DKernel_SM87(
    const float* __restrict__ input,
    float* __restrict__ output,
    int kernel_radius
) {
    // Use cooperative groups
    cg::thread_block block = cg::this_thread_block();
    
    // Shared memory for input image tile
    extern __shared__ float shared_input[];
    
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
    
    // Kernel dimensions
    const int kernel_diameter = 2 * kernel_radius + 1;
    
    // Calculate shared memory tile size
    const int tile_width = BLOCK_SIZE_X + 2 * kernel_radius;
    const int tile_height = BLOCK_SIZE_Y + 2 * kernel_radius;
    
    // Local indices for shared memory
    const int lx = tx + kernel_radius;
    const int ly = ty + kernel_radius;
    
    // Calculate linear thread index for efficient loading
    const int thread_idx = ty * BLOCK_SIZE_X + tx;
    const int total_threads = BLOCK_SIZE_X * BLOCK_SIZE_Y;
    
    // Load input image tile to shared memory using all threads
    // Each thread loads multiple pixels for better utilization
    for (int i = thread_idx; i < tile_width * tile_height; i += total_threads) {
        int sx = i % tile_width;
        int sy = i / tile_width;
        
        int gx = blockIdx.x * BLOCK_SIZE_X + sx - kernel_radius;
        int gy = blockIdx.y * BLOCK_SIZE_Y + sy - kernel_radius;
        
        // Handle boundary conditions with clamping
        int src_x = max(0, min(d_width - 1, gx));
        int src_y = max(0, min(d_height - 1, gy));
        
        // Load pixel to shared memory
        shared_input[sy * tile_width + sx] = input[(src_y * d_width + src_x) * d_channels + c];
    }
    
    // Ensure all threads have loaded their data
    block.sync();
    
    // Apply convolution with optimized memory access for Ampere
    float sum = 0.0f;
    
    // Use a more efficient memory access pattern for Ampere architecture
    for (int k = 0; k < kernel_diameter * kernel_diameter; ++k) {
        const int ky = k / kernel_diameter;
        const int kx = k % kernel_diameter;
        
        const int image_x = lx + kx - kernel_radius;
        const int image_y = ly + ky - kernel_radius;
        
        // Use fused multiply-add for better performance on Ampere
        sum = fmaf(shared_input[image_y * tile_width + image_x], d_kernel[k], sum);
    }
    
    // Store result
    output[(y * d_width + x) * d_channels + c] = sum;
}

/**
 * @brief Generic CUDA kernel for 2D convolution (works on all CUDA-capable GPUs).
 */
__global__ void convolution2DKernel_Generic(
    const float* __restrict__ input,
    float* __restrict__ output,
    int kernel_radius
) {
    // Global indices
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z;  // Channel index
    
    // Check boundaries
    if (x >= d_width || y >= d_height || c >= d_channels) {
        return;
    }
    
    // Kernel dimensions
    const int kernel_diameter = 2 * kernel_radius + 1;
    
    // Apply convolution
    float sum = 0.0f;
    
    for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
        for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
            // Calculate kernel index
            const int kernel_idx = (ky + kernel_radius) * kernel_diameter + (kx + kernel_radius);
            
            // Calculate image coordinates with boundary clamping
            const int image_x = max(0, min(d_width - 1, x + kx));
            const int image_y = max(0, min(d_height - 1, y + ky));
            
            // Get pixel value
            const float pixel = input[(image_y * d_width + image_x) * d_channels + c];
            
            // Add weighted contribution
            sum += pixel * d_kernel[kernel_idx];
        }
    }
    
    // Store result
    output[(y * d_width + x) * d_channels + c] = sum;
}

/**
 * @brief Median filter kernel (optimized for high-end GPUs, SM >= 8.0).
 * 
 * This kernel applies a median filter to an image.
 * 
 * @param input Input image
 * @param output Output image
 * @param radius Filter radius
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void medianFilterKernel_SM80(
    const float* __restrict__ input,
    float* __restrict__ output,
    int radius
) {
    // Shared memory for input image tile
    extern __shared__ float shared_input[];
    
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
    
    // Filter size
    const int filter_size = 2 * radius + 1;
    const int window_area = filter_size * filter_size;
    
    // Calculate shared memory tile size
    const int tile_width = BLOCK_SIZE_X + 2 * radius;
    const int tile_height = BLOCK_SIZE_Y + 2 * radius;
    
    // Local indices for shared memory
    const int lx = tx + radius;
    const int ly = ty + radius;
    
    // Load input image tile to shared memory
    // Each thread loads its own pixel and potentially some halo pixels
    for (int dy = 0; dy < tile_height; dy += BLOCK_SIZE_Y) {
        int sy = ly + dy - radius;
        int gy = y + dy - radius;
        
        if (sy >= 0 && sy < tile_height) {
            for (int dx = 0; dx < tile_width; dx += BLOCK_SIZE_X) {
                int sx = lx + dx - radius;
                int gx = x + dx - radius;
                
                if (sx >= 0 && sx < tile_width) {
                    // Handle boundary conditions with clamping
                    int src_x = max(0, min(d_width - 1, gx));
                    int src_y = max(0, min(d_height - 1, gy));
                    
                    // Load pixel to shared memory
                    shared_input[sy * tile_width + sx] = input[(src_y * d_width + src_x) * d_channels + c];
                }
            }
        }
    }
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Extract window values
    float window[121];  // Max window size (11x11)
    int count = 0;
    
    for (int wy = -radius; wy <= radius; ++wy) {
        for (int wx = -radius; wx <= radius; ++wx) {
            const int image_x = lx + wx;
            const int image_y = ly + wy;
            
            window[count++] = shared_input[image_y * tile_width + image_x];
        }
    }
    
    // Sort window values using odd-even sort (more predictable for GPU)
    // This is efficient for small window sizes
    for (int i = 0; i < window_area; ++i) {
        for (int j = i % 2; j < window_area - 1; j += 2) {
            if (window[j] > window[j + 1]) {
                float temp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = temp;
            }
        }
    }
    
    // Get median value
    float median = window[window_area / 2];
    
    // Store result
    output[(y * d_width + x) * d_channels + c] = median;
}

/**
 * @brief Bilateral filter kernel (optimized for high-end GPUs, SM >= 8.0).
 * 
 * This kernel applies a bilateral filter to an image.
 * 
 * @param input Input image
 * @param output Output image
 * @param spatial_sigma Spatial sigma parameter
 * @param range_sigma Range sigma parameter
 * @param radius Filter radius
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void bilateralFilterKernel_SM80(
    const float* __restrict__ input,
    float* __restrict__ output,
    float spatial_sigma,
    float range_sigma,
    int radius
) {
    // Shared memory for input image tile
    extern __shared__ float shared_input[];
    
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
    
    // Calculate shared memory tile size
    const int tile_width = BLOCK_SIZE_X + 2 * radius;
    const int tile_height = BLOCK_SIZE_Y + 2 * radius;
    
    // Local indices for shared memory
    const int lx = tx + radius;
    const int ly = ty + radius;
    
    // Load input image tile to shared memory
    // Each thread loads its own pixel and potentially some halo pixels
    for (int dy = 0; dy < tile_height; dy += BLOCK_SIZE_Y) {
        int sy = ly + dy - radius;
        int gy = y + dy - radius;
        
        if (sy >= 0 && sy < tile_height) {
            for (int dx = 0; dx < tile_width; dx += BLOCK_SIZE_X) {
                int sx = lx + dx - radius;
                int gx = x + dx - radius;
                
                if (sx >= 0 && sx < tile_width) {
                    // Handle boundary conditions with clamping
                    int src_x = max(0, min(d_width - 1, gx));
                    int src_y = max(0, min(d_height - 1, gy));
                    
                    // Load pixel to shared memory
                    shared_input[sy * tile_width + sx] = input[(src_y * d_width + src_x) * d_channels + c];
                }
            }
        }
    }
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Get center pixel value
    const float center_val = shared_input[ly * tile_width + lx];
    
    // Precompute constants
    const float spatial_sigma_sq_2 = 2.0f * spatial_sigma * spatial_sigma;
    const float range_sigma_sq_2 = 2.0f * range_sigma * range_sigma;
    
    // Apply bilateral filter
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    for (int wy = -radius; wy <= radius; ++wy) {
        for (int wx = -radius; wx <= radius; ++wx) {
            const int image_x = lx + wx;
            const int image_y = ly + wy;
            
            // Get pixel value
            const float val = shared_input[image_y * tile_width + image_x];
            
            // Calculate spatial weight
            const float spatial_dist_sq = wx * wx + wy * wy;
            const float spatial_weight = expf(-spatial_dist_sq / spatial_sigma_sq_2);
            
            // Calculate range weight
            const float range_dist_sq = (val - center_val) * (val - center_val);
            const float range_weight = expf(-range_dist_sq / range_sigma_sq_2);
            
            // Combined weight
            const float weight = spatial_weight * range_weight;
            
            // Accumulate
            sum += val * weight;
            weight_sum += weight;
        }
    }
    
    // Normalize and store result
    output[(y * d_width + x) * d_channels + c] = sum / weight_sum;
}

/**
 * @brief Non-local means filter kernel (high-end GPUs, SM >= 8.0).
 * 
 * This kernel applies a non-local means filter to an image.
 * 
 * @param input Input image
 * @param output Output image
 * @param search_radius Search window radius
 * @param patch_radius Patch radius
 * @param h Filter parameter (controls filtering strength)
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void nlmFilterKernel_SM80(
    const float* __restrict__ input,
    float* __restrict__ output,
    int search_radius,
    int patch_radius,
    float h
) {
    // Shared memory for input image tile
    extern __shared__ float shared_input[];
    
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
    
    // Total radius for shared memory
    const int total_radius = search_radius + patch_radius;
    
    // Calculate shared memory tile size
    const int tile_width = BLOCK_SIZE_X + 2 * total_radius;
    const int tile_height = BLOCK_SIZE_Y + 2 * total_radius;
    
    // Local indices for shared memory
    const int lx = tx + total_radius;
    const int ly = ty + total_radius;
    
    // Load input image tile to shared memory
    // Each thread loads its own pixel and potentially some halo pixels
    for (int dy = 0; dy < tile_height; dy += BLOCK_SIZE_Y) {
        int sy = ly + dy - total_radius;
        int gy = y + dy - total_radius;
        
        if (sy >= 0 && sy < tile_height) {
            for (int dx = 0; dx < tile_width; dx += BLOCK_SIZE_X) {
                int sx = lx + dx - total_radius;
                int gx = x + dx - total_radius;
                
                if (sx >= 0 && sx < tile_width) {
                    // Handle boundary conditions with clamping
                    int src_x = max(0, min(d_width - 1, gx));
                    int src_y = max(0, min(d_height - 1, gy));
                    
                    // Load pixel to shared memory
                    shared_input[sy * tile_width + sx] = input[(src_y * d_width + src_x) * d_channels + c];
                }
            }
        }
    }
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Apply non-local means filter
    float sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Precompute constants
    const float h_squared = h * h;
    const int patch_size = 2 * patch_radius + 1;
    const int patch_area = patch_size * patch_size;
    
    // Loop over search window
    for (int sy = -search_radius; sy <= search_radius; ++sy) {
        for (int sx = -search_radius; sx <= search_radius; ++sx) {
            // Calculate patch center
            const int nx = lx + sx;
            const int ny = ly + sy;
            
            // Calculate patch distance (SSD)
            float patch_dist = 0.0f;
            
            for (int py = -patch_radius; py <= patch_radius; ++py) {
                for (int px = -patch_radius; px <= patch_radius; ++px) {
                    const float center_val = shared_input[(ly + py) * tile_width + (lx + px)];
                    const float neighbor_val = shared_input[(ny + py) * tile_width + (nx + px)];
                    
                    const float diff = center_val - neighbor_val;
                    patch_dist += diff * diff;
                }
            }
            
            // Normalize patch distance
            patch_dist /= patch_area;
            
            // Calculate weight
            const float weight = expf(-patch_dist / h_squared);
            
            // Get pixel value
            const float val = shared_input[ny * tile_width + nx];
            
            // Accumulate
            sum += val * weight;
            weight_sum += weight;
        }
    }
    
    // Normalize and store result
    output[(y * d_width + x) * d_channels + c] = sum / weight_sum;
}

// Wrapper functions to launch the appropriate kernel based on device capabilities

/**
 * @brief Launch the 2D convolution kernel for the appropriate device.
 * @param d_input Device pointer to input image
 * @param d_output Device pointer to output image
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param d_kernel_ptr Device pointer to convolution kernel
 * @param kernel_size Kernel size
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launch2DConvolutionKernel(
    float* d_input,
    float* d_output,
    int width,
    int height,
    int channels,
    float* d_kernel_ptr,
    int kernel_size,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Calculate kernel radius
    int kernel_radius = kernel_size / 2;
    
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_channels, &channels, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel_size, &kernel_size, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_kernel, d_kernel_ptr, kernel_size * kernel_size * sizeof(float)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Determine block and grid dimensions
    dim3 blockDim(params.block_size_x, params.block_size_y, 1);
    dim3 gridDim(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        channels
    );
    
    // Calculate shared memory size
    size_t sharedMemSize = (params.block_size_x + 2 * kernel_radius) * 
                         (params.block_size_y + 2 * kernel_radius) * 
                         sizeof(float);
    
    // Launch appropriate kernel based on device capabilities
    if (device_caps.compute_capability_major > 8 || 
        (device_caps.compute_capability_major == 8 && device_caps.compute_capability_minor >= 7)) {
        // Jetson Orin NX (SM 8.7)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            convolution2DKernel_SM87<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_input, d_output, kernel_radius
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            convolution2DKernel_SM87<32, 8><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_input, d_output, kernel_radius
            );
        } else {
            convolution2DKernel_SM87<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_input, d_output, kernel_radius
            );
        }
    } else if (device_caps.compute_capability_major == 8) {
        // High-end GPUs (SM 8.0+)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            convolution2DKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_input, d_output, kernel_radius
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            convolution2DKernel_SM80<32, 8><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_input, d_output, kernel_radius
            );
        } else {
            convolution2DKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_input, d_output, kernel_radius
            );
        }
    } else if (device_caps.compute_capability_major == 7 && device_caps.compute_capability_minor >= 5) {
        // T4 GPU (SM 7.5)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            convolution2DKernel_SM75<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_input, d_output, kernel_radius
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            convolution2DKernel_SM75<32, 8><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_input, d_output, kernel_radius
            );
        } else {
            convolution2DKernel_SM75<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_input, d_output, kernel_radius
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
        
        convolution2DKernel_Generic<<<genericGridDim, genericBlockDim, 0, stream>>>(
            d_input, d_output, kernel_radius
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

/**
 * @brief Launch the median filter kernel for the appropriate device.
 * @param d_input Device pointer to input image
 * @param d_output Device pointer to output image
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param radius Filter radius
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchMedianFilterKernel(
    float* d_input,
    float* d_output,
    int width,
    int height,
    int channels,
    int radius,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_channels, &channels, sizeof(int)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Determine block and grid dimensions
    dim3 blockDim(params.block_size_x, params.block_size_y, 1);
    dim3 gridDim(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        channels
    );
    
    // Calculate shared memory size
    size_t sharedMemSize = (params.block_size_x + 2 * radius) * 
                         (params.block_size_y + 2 * radius) * 
                         sizeof(float);
    
    // For now, we only have the SM80 version implemented
    // This can be extended similarly to the convolution kernel
    if (params.block_size_x == 16 && params.block_size_y == 16) {
        medianFilterKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
            d_input, d_output, radius
        );
    } else if (params.block_size_x == 32 && params.block_size_y == 8) {
        medianFilterKernel_SM80<32, 8><<<gridDim, blockDim, sharedMemSize, stream>>>(
            d_input, d_output, radius
        );
    } else {
        medianFilterKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
            d_input, d_output, radius
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

/**
 * @brief Launch the bilateral filter kernel for the appropriate device.
 * @param d_input Device pointer to input image
 * @param d_output Device pointer to output image
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param spatial_sigma Spatial sigma parameter
 * @param range_sigma Range sigma parameter
 * @param radius Filter radius
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchBilateralFilterKernel(
    float* d_input,
    float* d_output,
    int width,
    int height,
    int channels,
    float spatial_sigma,
    float range_sigma,
    int radius,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_channels, &channels, sizeof(int)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Determine block and grid dimensions
    dim3 blockDim(params.block_size_x, params.block_size_y, 1);
    dim3 gridDim(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        channels
    );
    
    // Calculate shared memory size
    size_t sharedMemSize = (params.block_size_x + 2 * radius) * 
                         (params.block_size_y + 2 * radius) * 
                         sizeof(float);
    
    // For now, we only have the SM80 version implemented
    // This can be extended similarly to the convolution kernel
    if (params.block_size_x == 16 && params.block_size_y == 16) {
        bilateralFilterKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
            d_input, d_output, spatial_sigma, range_sigma, radius
        );
    } else if (params.block_size_x == 32 && params.block_size_y == 8) {
        bilateralFilterKernel_SM80<32, 8><<<gridDim, blockDim, sharedMemSize, stream>>>(
            d_input, d_output, spatial_sigma, range_sigma, radius
        );
    } else {
        bilateralFilterKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
            d_input, d_output, spatial_sigma, range_sigma, radius
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

/**
 * @brief Launch the non-local means filter kernel for the appropriate device.
 * @param d_input Device pointer to input image
 * @param d_output Device pointer to output image
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param search_radius Search window radius
 * @param patch_radius Patch radius
 * @param h Filter parameter (controls filtering strength)
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchNLMFilterKernel(
    float* d_input,
    float* d_output,
    int width,
    int height,
    int channels,
    int search_radius,
    int patch_radius,
    float h,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_channels, &channels, sizeof(int)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Determine block and grid dimensions
    dim3 blockDim(params.block_size_x, params.block_size_y, 1);
    dim3 gridDim(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        channels
    );
    
    // Calculate total radius
    int total_radius = search_radius + patch_radius;
    
    // Calculate shared memory size
    size_t sharedMemSize = (params.block_size_x + 2 * total_radius) * 
                         (params.block_size_y + 2 * total_radius) * 
                         sizeof(float);
    
    // For now, we only have the SM80 version implemented
    // This can be extended similarly to the convolution kernel
    if (params.block_size_x == 16 && params.block_size_y == 16) {
        nlmFilterKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
            d_input, d_output, search_radius, patch_radius, h
        );
    } else if (params.block_size_x == 32 && params.block_size_y == 8) {
        nlmFilterKernel_SM80<32, 8><<<gridDim, blockDim, sharedMemSize, stream>>>(
            d_input, d_output, search_radius, patch_radius, h
        );
    } else {
        nlmFilterKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
            d_input, d_output, search_radius, patch_radius, h
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

} // namespace medical_imaging