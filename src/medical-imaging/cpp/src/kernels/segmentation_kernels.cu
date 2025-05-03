/**
 * @file segmentation_kernels.cu
 * @brief CUDA kernels for medical image segmentation.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
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

// Device constants for segmentation parameters
__constant__ int d_width;               // Image width
__constant__ int d_height;              // Image height
__constant__ int d_depth;               // Image depth (for 3D images)
__constant__ int d_channels;            // Number of channels
__constant__ int d_num_segments;        // Number of segments
__constant__ float d_threshold;         // Threshold value

// Threshold segmentation kernels

/**
 * @brief Simple thresholding kernel (optimized for high-end GPUs, SM >= 8.0).
 * 
 * @param input Input image
 * @param output Output segmentation
 * @param threshold Threshold value
 * @param max_value Maximum value for segmentation
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void thresholdingKernel_SM80(
    const float* __restrict__ input,
    float* __restrict__ output,
    float threshold,
    float max_value
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
    
    // Compute input and output indices
    const int idx = (y * d_width + x) * d_channels + c;
    
    // Apply threshold
    output[idx] = (input[idx] > threshold) ? max_value : 0.0f;
}

/**
 * @brief Optimized thresholding kernel for T4 GPUs (SM 7.5).
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void thresholdingKernel_SM75(
    const float* __restrict__ input,
    float* __restrict__ output,
    float threshold,
    float max_value
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
    
    // Compute input and output indices
    const int idx = (y * d_width + x) * d_channels + c;
    
    // Apply threshold with warp-level optimization
    // Process 32 elements per warp
    const int warp_idx = warp.thread_rank();
    const int warp_offset = warp_idx * 4; // Each thread processes 4 consecutive elements
    
    if (warp_offset + 3 < d_width * d_height * d_channels) {
        float4 input_data;
        float4 output_data;
        
        // Load 4 consecutive elements
        reinterpret_cast<float4*>(&input_data)[0] = reinterpret_cast<const float4*>(&input[idx & ~3])[0];
        
        // Apply threshold
        output_data.x = (input_data.x > threshold) ? max_value : 0.0f;
        output_data.y = (input_data.y > threshold) ? max_value : 0.0f;
        output_data.z = (input_data.z > threshold) ? max_value : 0.0f;
        output_data.w = (input_data.w > threshold) ? max_value : 0.0f;
        
        // Store results
        reinterpret_cast<float4*>(&output[idx & ~3])[0] = output_data;
    }
    else {
        // Handle boundary case
        output[idx] = (input[idx] > threshold) ? max_value : 0.0f;
    }
}

/**
 * @brief Optimized thresholding kernel for Jetson Orin NX (SM 8.7).
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void thresholdingKernel_SM87(
    const float* __restrict__ input,
    float* __restrict__ output,
    float threshold,
    float max_value
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
    
    // Compute input and output indices
    const int idx = (y * d_width + x) * d_channels + c;
    
    // Apply threshold with Ampere architecture optimization
    const float val = input[idx];
    
    // Use predication instead of branching for better performance on Ampere
    const float result = __float_as_int(val > threshold) & __float_as_int(max_value);
    
    // Store result
    output[idx] = result;
}

/**
 * @brief Generic thresholding CUDA kernel (works on all CUDA-capable GPUs).
 */
__global__ void thresholdingKernel_Generic(
    const float* __restrict__ input,
    float* __restrict__ output,
    float threshold,
    float max_value
) {
    // Global indices
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int c = blockIdx.z;  // Channel index
    
    // Check boundaries
    if (x >= d_width || y >= d_height || c >= d_channels) {
        return;
    }
    
    // Compute input and output indices
    const int idx = (y * d_width + x) * d_channels + c;
    
    // Apply threshold
    output[idx] = (input[idx] > threshold) ? max_value : 0.0f;
}

// Watershed segmentation kernels

/**
 * @brief Structure for pixel information in watershed algorithm.
 */
struct PixelInfo {
    float height;    // Pixel intensity
    int label;       // Segment label
    int x, y;        // Pixel coordinates
    
    // Comparison operator for sorting
    __device__ bool operator<(const PixelInfo& other) const {
        return height < other.height;
    }
};

/**
 * @brief Initialize watershed algorithm (optimized for high-end GPUs, SM >= 8.0).
 * 
 * @param input Input image
 * @param markers Marker image with initial seeds
 * @param pixel_info Array of pixel information (sorted by height)
 * @param num_pixels Number of pixels
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void watershedInitKernel_SM80(
    const float* __restrict__ input,
    const float* __restrict__ markers,
    PixelInfo* __restrict__ pixel_info,
    int* __restrict__ num_pixels
) {
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Check boundaries
    if (x >= d_width || y >= d_height) {
        return;
    }
    
    // Calculate linear index
    const int idx = y * d_width + x;
    
    // Initialize pixel info
    pixel_info[idx].height = input[idx];
    pixel_info[idx].x = x;
    pixel_info[idx].y = y;
    
    // Check if pixel is a marker
    const float marker_val = markers[idx];
    
    if (marker_val > 0.0f) {
        // Marker pixel, assign label
        pixel_info[idx].label = static_cast<int>(marker_val);
    } else {
        // Non-marker pixel, assign label -1 (undefined)
        pixel_info[idx].label = -1;
    }
    
    // Update total number of pixels (for debug)
    if (x == 0 && y == 0) {
        *num_pixels = d_width * d_height;
    }
}

/**
 * @brief Watershed flooding step (optimized for high-end GPUs, SM >= 8.0).
 * 
 * @param pixel_info Sorted array of pixel information
 * @param output Output segmentation
 * @param width Image width
 * @param height Image height
 */
template <int BLOCK_SIZE>
__global__ void watershedFloodKernel_SM80(
    PixelInfo* __restrict__ pixel_info,
    float* __restrict__ output,
    int width,
    int height
) {
    // Thread index
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // Number of pixels
    const int num_pixels = width * height;
    
    // Check boundaries
    if (tid >= num_pixels) {
        return;
    }
    
    // Get current pixel info
    const PixelInfo pixel = pixel_info[tid];
    
    // Skip if already labeled
    if (pixel.label > 0) {
        // Copy label to output
        output[pixel.y * width + pixel.x] = static_cast<float>(pixel.label);
        return;
    }
    
    // Neighbor offsets (4-connectivity)
    const int dx[4] = {-1, 0, 1, 0};
    const int dy[4] = {0, -1, 0, 1};
    
    // Check neighbors
    int label = -1;
    
    for (int i = 0; i < 4; ++i) {
        const int nx = pixel.x + dx[i];
        const int ny = pixel.y + dy[i];
        
        // Check boundaries
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            // Get neighbor index
            const int nidx = ny * width + nx;
            
            // Get neighbor's label
            const int neighbor_label = pixel_info[nidx].label;
            
            if (neighbor_label > 0) {
                if (label == -1) {
                    // First labeled neighbor, adopt its label
                    label = neighbor_label;
                } else if (label != neighbor_label) {
                    // Boundary pixel (neighbors with different labels)
                    label = 0;  // Watershed boundary
                    break;
                }
            }
        }
    }
    
    // Update pixel label
    pixel_info[tid].label = label;
    
    // Copy label to output
    output[pixel.y * width + pixel.x] = static_cast<float>(label);
}

/**
 * @brief Level set update kernel (optimized for high-end GPUs, SM >= 8.0).
 * 
 * This kernel implements one iteration of the level set evolution for image segmentation.
 * 
 * @param phi Level set function (signed distance function)
 * @param input Input image
 * @param new_phi Updated level set function
 * @param alpha Balloon force coefficient
 * @param beta Curvature coefficient
 * @param gamma Edge coefficient
 * @param dt Time step
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void levelSetUpdateKernel_SM80(
    const float* __restrict__ phi,
    const float* __restrict__ input,
    float* __restrict__ new_phi,
    float alpha,
    float beta,
    float gamma,
    float dt
) {
    // Shared memory for local neighborhood
    extern __shared__ float shared_data[];
    
    // Split shared memory
    float* s_phi = shared_data;
    float* s_input = &s_phi[BLOCK_SIZE_X * BLOCK_SIZE_Y];
    
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Check boundaries
    if (x >= d_width || y >= d_height) {
        return;
    }
    
    // Linear index
    const int idx = y * d_width + x;
    
    // Local linear index in shared memory
    const int s_idx = ty * BLOCK_SIZE_X + tx;
    
    // Load level set and input image to shared memory
    s_phi[s_idx] = phi[idx];
    s_input[s_idx] = input[idx];
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Skip boundary pixels
    if (x == 0 || x == d_width - 1 || y == 0 || y == d_height - 1) {
        new_phi[idx] = phi[idx];
        return;
    }
    
    // Calculate derivatives for curvature
    const float phi_x = (phi[(y) * d_width + (x + 1)] - phi[(y) * d_width + (x - 1)]) / 2.0f;
    const float phi_y = (phi[(y + 1) * d_width + (x)] - phi[(y - 1) * d_width + (x)]) / 2.0f;
    
    const float phi_xx = phi[(y) * d_width + (x + 1)] - 2.0f * phi[idx] + phi[(y) * d_width + (x - 1)];
    const float phi_yy = phi[(y + 1) * d_width + (x)] - 2.0f * phi[idx] + phi[(y - 1) * d_width + (x)];
    
    const float phi_xy = (phi[(y + 1) * d_width + (x + 1)] - phi[(y + 1) * d_width + (x - 1)] - 
                          phi[(y - 1) * d_width + (x + 1)] + phi[(y - 1) * d_width + (x - 1)]) / 4.0f;
    
    // Calculate curvature (mean curvature)
    const float grad_sq = phi_x * phi_x + phi_y * phi_y + 1e-6f;  // Add small constant to avoid division by zero
    const float curvature = (phi_xx * phi_y * phi_y - 2 * phi_xy * phi_x * phi_y + phi_yy * phi_x * phi_x) / 
                          (grad_sq * sqrtf(grad_sq));
    
    // Calculate gradient magnitude of input image (for edge stopping)
    const float img_x = (input[(y) * d_width + (x + 1)] - input[(y) * d_width + (x - 1)]) / 2.0f;
    const float img_y = (input[(y + 1) * d_width + (x)] - input[(y - 1) * d_width + (x)]) / 2.0f;
    
    const float edge_indicator = 1.0f / (1.0f + (img_x * img_x + img_y * img_y));
    
    // Calculate speed function
    const float speed = alpha + beta * curvature + gamma * edge_indicator;
    
    // Update level set function
    new_phi[idx] = phi[idx] + dt * speed * sqrtf(grad_sq);
}

/**
 * @brief Graph cut segmentation kernel (optimized for high-end GPUs, SM >= 8.0).
 * 
 * This kernel implements one iteration of the max-flow/min-cut algorithm
 * for graph-based image segmentation.
 * 
 * @param input Input image
 * @param seeds Seed image (positive for foreground, negative for background)
 * @param flows Flow capacities (4 directions per pixel)
 * @param residual_flows Residual flow capacities
 * @param labels Current pixel labels
 * @param active_pixel_mask Mask of active pixels for the push-relabel algorithm
 * @param heights Height function for push-relabel
 * @param excess Excess flow at each pixel
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void graphCutKernel_SM80(
    const float* __restrict__ input,
    const float* __restrict__ seeds,
    float* __restrict__ flows,
    float* __restrict__ residual_flows,
    int* __restrict__ labels,
    int* __restrict__ active_pixel_mask,
    int* __restrict__ heights,
    float* __restrict__ excess
) {
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Check boundaries
    if (x >= d_width || y >= d_height) {
        return;
    }
    
    // Linear index
    const int p = y * d_width + x;
    
    // Skip inactive pixels
    if (active_pixel_mask[p] == 0) {
        return;
    }
    
    // Check if pixel has excess flow
    if (excess[p] <= 0.0f) {
        active_pixel_mask[p] = 0;
        return;
    }
    
    // Direction offsets (right, down, left, up)
    const int dx[4] = {1, 0, -1, 0};
    const int dy[4] = {0, 1, 0, -1};
    
    // Indices for flow array (4 flow capacities per pixel)
    const int flow_idx[4] = {
        p * 4 + 0,  // Right flow
        p * 4 + 1,  // Down flow
        p * 4 + 2,  // Left flow
        p * 4 + 3   // Up flow
    };
    
    // Push excess flow to neighboring pixels with lower height
    bool pushed = false;
    
    for (int dir = 0; dir < 4; ++dir) {
        const int nx = x + dx[dir];
        const int ny = y + dy[dir];
        
        // Check boundaries
        if (nx >= 0 && nx < d_width && ny >= 0 && ny < d_height) {
            const int q = ny * d_width + nx;
            
            // Can push flow if: 
            // 1. Height of p is greater than height of q
            // 2. Residual capacity from p to q is positive
            if (heights[p] > heights[q] && residual_flows[flow_idx[dir]] > 0.0f) {
                // Calculate flow to push (minimum of excess at p and residual capacity)
                const float push_flow = min(excess[p], residual_flows[flow_idx[dir]]);
                
                if (push_flow > 0.0f) {
                    // Update excess flow
                    excess[p] -= push_flow;
                    excess[q] += push_flow;
                    
                    // Update residual capacities
                    residual_flows[flow_idx[dir]] -= push_flow;
                    
                    // Opposite direction index
                    const int rev_dir = (dir + 2) % 4;
                    const int rev_flow_idx = q * 4 + rev_dir;
                    residual_flows[rev_flow_idx] += push_flow;
                    
                    // Mark q as active
                    active_pixel_mask[q] = 1;
                    
                    pushed = true;
                    
                    // If all excess has been pushed, we're done
                    if (excess[p] <= 0.0f) {
                        active_pixel_mask[p] = 0;
                        break;
                    }
                }
            }
        }
    }
    
    // If couldn't push, relabel (increase height)
    if (!pushed && excess[p] > 0.0f) {
        int min_height = INT_MAX;
        
        for (int dir = 0; dir < 4; ++dir) {
            const int nx = x + dx[dir];
            const int ny = y + dy[dir];
            
            // Check boundaries
            if (nx >= 0 && nx < d_width && ny >= 0 && ny < d_height) {
                const int q = ny * d_width + nx;
                
                // Look for minimum height among neighbors with positive residual capacity
                if (residual_flows[flow_idx[dir]] > 0.0f) {
                    min_height = min(min_height, heights[q]);
                }
            }
        }
        
        // Relabel (set height to one more than minimum neighbor height)
        if (min_height != INT_MAX) {
            heights[p] = min_height + 1;
        }
    }
}

/**
 * @brief Min-cut classification kernel (optimized for high-end GPUs, SM >= 8.0).
 * 
 * This kernel determines the final segmentation based on the min-cut solution.
 * 
 * @param excess Excess flow at each pixel
 * @param output Output segmentation mask
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void graphCutFinalizeKernel_SM80(
    const float* __restrict__ excess,
    float* __restrict__ output
) {
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Check boundaries
    if (x >= d_width || y >= d_height) {
        return;
    }
    
    // Linear index
    const int p = y * d_width + x;
    
    // Pixels with positive excess are on the source side (foreground)
    output[p] = (excess[p] > 0.0f) ? 1.0f : 0.0f;
}

// Wrapper functions to launch the appropriate kernel based on device capabilities

/**
 * @brief Launch thresholding kernel for the appropriate device.
 * @param d_input Device pointer to input image
 * @param d_output Device pointer to output segmentation
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels
 * @param threshold Threshold value
 * @param max_value Maximum value for segmentation
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchThresholdingKernel(
    float* d_input,
    float* d_output,
    int width,
    int height,
    int channels,
    float threshold,
    float max_value,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_channels, &channels, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float)));
    
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
            thresholdingKernel_SM87<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output, threshold, max_value
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            thresholdingKernel_SM87<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output, threshold, max_value
            );
        } else {
            thresholdingKernel_SM87<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output, threshold, max_value
            );
        }
    } else if (device_caps.compute_capability_major == 8) {
        // High-end GPUs (SM 8.0+)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            thresholdingKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output, threshold, max_value
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            thresholdingKernel_SM80<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output, threshold, max_value
            );
        } else {
            thresholdingKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output, threshold, max_value
            );
        }
    } else if (device_caps.compute_capability_major == 7 && device_caps.compute_capability_minor >= 5) {
        // T4 GPU (SM 7.5)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            thresholdingKernel_SM75<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output, threshold, max_value
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            thresholdingKernel_SM75<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output, threshold, max_value
            );
        } else {
            thresholdingKernel_SM75<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_input, d_output, threshold, max_value
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
        
        thresholdingKernel_Generic<<<genericGridDim, genericBlockDim, 0, stream>>>(
            d_input, d_output, threshold, max_value
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

/**
 * @brief Launch watershed segmentation kernels for the appropriate device.
 * @param d_input Device pointer to input image
 * @param d_markers Device pointer to marker image
 * @param d_output Device pointer to output segmentation
 * @param width Image width
 * @param height Image height
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchWatershedKernel(
    float* d_input,
    float* d_markers,
    float* d_output,
    int width,
    int height,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Allocate device memory for pixel info and helper arrays
    PixelInfo* d_pixel_info = nullptr;
    int* d_num_pixels = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_pixel_info, width * height * sizeof(PixelInfo)));
    CUDA_CHECK(cudaMalloc(&d_num_pixels, sizeof(int)));
    
    // Initialize with zero
    CUDA_CHECK(cudaMemset(d_num_pixels, 0, sizeof(int)));
    
    // Determine block and grid dimensions for init kernel
    dim3 init_blockDim(params.block_size_x, params.block_size_y, 1);
    dim3 init_gridDim(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        1
    );
    
    // Initialize watershed
    if (device_caps.compute_capability_major >= 8) {
        // High-end GPUs (SM >= 8.0)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            watershedInitKernel_SM80<16, 16><<<init_gridDim, init_blockDim, 0, stream>>>(
                d_input, d_markers, d_pixel_info, d_num_pixels
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            watershedInitKernel_SM80<32, 8><<<init_gridDim, init_blockDim, 0, stream>>>(
                d_input, d_markers, d_pixel_info, d_num_pixels
            );
        } else {
            watershedInitKernel_SM80<16, 16><<<init_gridDim, init_blockDim, 0, stream>>>(
                d_input, d_markers, d_pixel_info, d_num_pixels
            );
        }
    } else {
        // Generic initialization for older GPUs
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            watershedInitKernel_SM80<16, 16><<<init_gridDim, init_blockDim, 0, stream>>>(
                d_input, d_markers, d_pixel_info, d_num_pixels
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            watershedInitKernel_SM80<32, 8><<<init_gridDim, init_blockDim, 0, stream>>>(
                d_input, d_markers, d_pixel_info, d_num_pixels
            );
        } else {
            watershedInitKernel_SM80<16, 16><<<init_gridDim, init_blockDim, 0, stream>>>(
                d_input, d_markers, d_pixel_info, d_num_pixels
            );
        }
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Sort pixels by height using Thrust
    thrust::device_ptr<PixelInfo> thrust_pixel_info(d_pixel_info);
    thrust::sort(thrust::device, thrust_pixel_info, thrust_pixel_info + (width * height));
    
    // Determine block and grid dimensions for flood kernel
    const int FLOOD_BLOCK_SIZE = 256;
    
    dim3 flood_blockDim(FLOOD_BLOCK_SIZE, 1, 1);
    dim3 flood_gridDim(
        (width * height + FLOOD_BLOCK_SIZE - 1) / FLOOD_BLOCK_SIZE,
        1,
        1
    );
    
    // Flood watershed
    if (device_caps.compute_capability_major >= 8) {
        // High-end GPUs (SM >= 8.0)
        watershedFloodKernel_SM80<FLOOD_BLOCK_SIZE><<<flood_gridDim, flood_blockDim, 0, stream>>>(
            d_pixel_info, d_output, width, height
        );
    } else {
        // Generic flood for older GPUs
        watershedFloodKernel_SM80<FLOOD_BLOCK_SIZE><<<flood_gridDim, flood_blockDim, 0, stream>>>(
            d_pixel_info, d_output, width, height
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Free temporary device memory
    CUDA_CHECK(cudaFree(d_pixel_info));
    CUDA_CHECK(cudaFree(d_num_pixels));
    
    return true;
}

/**
 * @brief Launch level set segmentation kernels for the appropriate device.
 * @param d_input Device pointer to input image
 * @param d_init_phi Device pointer to initial level set function
 * @param d_output Device pointer to output segmentation
 * @param width Image width
 * @param height Image height
 * @param iterations Number of level set iterations
 * @param alpha Balloon force coefficient
 * @param beta Curvature coefficient
 * @param gamma Edge coefficient
 * @param dt Time step
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchLevelSetKernel(
    float* d_input,
    float* d_init_phi,
    float* d_output,
    int width,
    int height,
    int iterations,
    float alpha,
    float beta,
    float gamma,
    float dt,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Allocate device memory for intermediate results
    float* d_phi = nullptr;
    float* d_new_phi = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_phi, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_new_phi, width * height * sizeof(float)));
    
    // Copy initial level set function
    CUDA_CHECK(cudaMemcpy(d_phi, d_init_phi, width * height * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Determine block and grid dimensions
    dim3 blockDim(params.block_size_x, params.block_size_y, 1);
    dim3 gridDim(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        1
    );
    
    // Calculate shared memory size
    size_t sharedMemSize = 2 * params.block_size_x * params.block_size_y * sizeof(float);
    
    // Iterate level set evolution
    for (int i = 0; i < iterations; i++) {
        // Update level set
        if (device_caps.compute_capability_major >= 8) {
            // High-end GPUs (SM >= 8.0)
            if (params.block_size_x == 16 && params.block_size_y == 16) {
                levelSetUpdateKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                    d_phi, d_input, d_new_phi, alpha, beta, gamma, dt
                );
            } else if (params.block_size_x == 32 && params.block_size_y == 8) {
                levelSetUpdateKernel_SM80<32, 8><<<gridDim, blockDim, sharedMemSize, stream>>>(
                    d_phi, d_input, d_new_phi, alpha, beta, gamma, dt
                );
            } else {
                levelSetUpdateKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                    d_phi, d_input, d_new_phi, alpha, beta, gamma, dt
                );
            }
        } else {
            // Generic update for older GPUs
            if (params.block_size_x == 16 && params.block_size_y == 16) {
                levelSetUpdateKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                    d_phi, d_input, d_new_phi, alpha, beta, gamma, dt
                );
            } else if (params.block_size_x == 32 && params.block_size_y == 8) {
                levelSetUpdateKernel_SM80<32, 8><<<gridDim, blockDim, sharedMemSize, stream>>>(
                    d_phi, d_input, d_new_phi, alpha, beta, gamma, dt
                );
            } else {
                levelSetUpdateKernel_SM80<16, 16><<<gridDim, blockDim, sharedMemSize, stream>>>(
                    d_phi, d_input, d_new_phi, alpha, beta, gamma, dt
                );
            }
        }
        
        // Swap phi and new_phi
        float* temp = d_phi;
        d_phi = d_new_phi;
        d_new_phi = temp;
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Threshold final level set function to get segmentation
    launchThresholdingKernel(d_phi, d_output, width, height, 1, 0.0f, 1.0f, device_caps, params);
    
    // Free temporary device memory
    CUDA_CHECK(cudaFree(d_phi));
    CUDA_CHECK(cudaFree(d_new_phi));
    
    return true;
}

/**
 * @brief Launch graph cut segmentation kernels for the appropriate device.
 * @param d_input Device pointer to input image
 * @param d_seeds Device pointer to seed image
 * @param d_output Device pointer to output segmentation
 * @param width Image width
 * @param height Image height
 * @param max_iterations Maximum number of iterations
 * @param lambda Regularization parameter
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchGraphCutKernel(
    float* d_input,
    float* d_seeds,
    float* d_output,
    int width,
    int height,
    int max_iterations,
    float lambda,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_width, &width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_height, &height, sizeof(int)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Allocate device memory for graph cut algorithm
    float* d_flows = nullptr;          // Edge capacities (4 directions per pixel)
    float* d_residual_flows = nullptr; // Residual flow capacities
    int* d_labels = nullptr;           // Pixel labels
    int* d_active_pixel_mask = nullptr; // Mask of active pixels
    int* d_heights = nullptr;          // Height function for push-relabel
    float* d_excess = nullptr;         // Excess flow at each pixel
    
    const int num_pixels = width * height;
    const int num_edges = num_pixels * 4; // 4 edges per pixel (right, down, left, up)
    
    CUDA_CHECK(cudaMalloc(&d_flows, num_edges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_residual_flows, num_edges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, num_pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active_pixel_mask, num_pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_heights, num_pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_excess, num_pixels * sizeof(float)));
    
    // Initialize arrays with zeros
    CUDA_CHECK(cudaMemset(d_flows, 0, num_edges * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_residual_flows, 0, num_edges * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_labels, 0, num_pixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_active_pixel_mask, 0, num_pixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_heights, 0, num_pixels * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_excess, 0, num_pixels * sizeof(float)));
    
    // TODO: Setup edge capacities based on input image and lambda
    // This would involve a separate initialization kernel
    
    // Determine block and grid dimensions
    dim3 blockDim(params.block_size_x, params.block_size_y, 1);
    dim3 gridDim(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        1
    );
    
    // Iterate graph cut algorithm
    for (int i = 0; i < max_iterations; i++) {
        // Check if there are any active pixels
        int h_any_active = 0;
        CUDA_CHECK(cudaMemcpy(&h_any_active, d_active_pixel_mask, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (h_any_active == 0) {
            break;  // No active pixels, algorithm has converged
        }
        
        // Run graph cut iteration
        if (device_caps.compute_capability_major >= 8) {
            // High-end GPUs (SM >= 8.0)
            if (params.block_size_x == 16 && params.block_size_y == 16) {
                graphCutKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                    d_input, d_seeds, d_flows, d_residual_flows, d_labels,
                    d_active_pixel_mask, d_heights, d_excess
                );
            } else if (params.block_size_x == 32 && params.block_size_y == 8) {
                graphCutKernel_SM80<32, 8><<<gridDim, blockDim, 0, stream>>>(
                    d_input, d_seeds, d_flows, d_residual_flows, d_labels,
                    d_active_pixel_mask, d_heights, d_excess
                );
            } else {
                graphCutKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                    d_input, d_seeds, d_flows, d_residual_flows, d_labels,
                    d_active_pixel_mask, d_heights, d_excess
                );
            }
        } else {
            // Generic version for older GPUs
            if (params.block_size_x == 16 && params.block_size_y == 16) {
                graphCutKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                    d_input, d_seeds, d_flows, d_residual_flows, d_labels,
                    d_active_pixel_mask, d_heights, d_excess
                );
            } else if (params.block_size_x == 32 && params.block_size_y == 8) {
                graphCutKernel_SM80<32, 8><<<gridDim, blockDim, 0, stream>>>(
                    d_input, d_seeds, d_flows, d_residual_flows, d_labels,
                    d_active_pixel_mask, d_heights, d_excess
                );
            } else {
                graphCutKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                    d_input, d_seeds, d_flows, d_residual_flows, d_labels,
                    d_active_pixel_mask, d_heights, d_excess
                );
            }
        }
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Finalize segmentation
    if (device_caps.compute_capability_major >= 8) {
        // High-end GPUs (SM >= 8.0)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            graphCutFinalizeKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_excess, d_output
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            graphCutFinalizeKernel_SM80<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_excess, d_output
            );
        } else {
            graphCutFinalizeKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_excess, d_output
            );
        }
    } else {
        // Generic version for older GPUs
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            graphCutFinalizeKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_excess, d_output
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            graphCutFinalizeKernel_SM80<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_excess, d_output
            );
        } else {
            graphCutFinalizeKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_excess, d_output
            );
        }
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Free temporary device memory
    CUDA_CHECK(cudaFree(d_flows));
    CUDA_CHECK(cudaFree(d_residual_flows));
    CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_active_pixel_mask));
    CUDA_CHECK(cudaFree(d_heights));
    CUDA_CHECK(cudaFree(d_excess));
    
    return true;
}

} // namespace medical_imaging