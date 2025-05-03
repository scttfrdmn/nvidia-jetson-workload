/**
 * @file ct_reconstruction_kernels.cu
 * @brief CUDA kernels for CT image reconstruction.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cufft.h>
#include <cmath>
#include <stdio.h>

#include "../../include/medical_imaging/medical_imaging.hpp"
#include "../../include/medical_imaging/gpu_adaptability.hpp"

namespace cg = cooperative_groups;

namespace medical_imaging {

// Constants
#define PI 3.14159265358979323846f

// Device constants for reconstruction parameters
__constant__ float d_angles[2048];      // Projection angles (radians)
__constant__ float d_filter_coeffs[4096]; // Filter coefficients
__constant__ int d_num_angles;          // Number of angles
__constant__ int d_filter_size;         // Filter size
__constant__ int d_proj_width;          // Projection width
__constant__ int d_proj_height;         // Projection height (number of projections)
__constant__ int d_img_width;           // Output image width
__constant__ int d_img_height;          // Output image height

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

/**
 * @brief Applies a ramp filter to projection data (optimized for high-end GPUs, SM >= 8.0).
 * 
 * This kernel applies a 1D ramp filter to each row of the projection data.
 * 
 * @param projections Input projection data
 * @param filtered_projections Output filtered projection data
 * @param filter_type Filter type (0: Ram-Lak, 1: Shepp-Logan, 2: Cosine, 3: Hamming)
 */
template <int BLOCK_SIZE>
__global__ void applyRampFilterKernel_SM80(
    const float* __restrict__ projections,
    float* __restrict__ filtered_projections,
    int filter_type
) {
    // Shared memory for input projections and filter coefficients
    extern __shared__ float shared_data[];
    
    // Local thread index
    const int tx = threadIdx.x;
    
    // Global indices
    const int proj_idx = blockIdx.y;    // Projection index
    const int row_idx = blockIdx.x;     // Row index within the projection
    
    // Check boundaries
    if (proj_idx >= d_num_angles || row_idx >= d_proj_height) {
        return;
    }
    
    // Compute start index for this projection row
    const int row_start = (proj_idx * d_proj_height + row_idx) * d_proj_width;
    
    // Load projection data to shared memory with cooperative loading
    float* s_proj = shared_data;
    
    for (int i = tx; i < d_proj_width; i += BLOCK_SIZE) {
        s_proj[i] = projections[row_start + i];
    }
    
    // Ensure all threads have loaded their data
    __syncthreads();
    
    // Perform 1D convolution with filter
    for (int i = tx; i < d_proj_width; i += BLOCK_SIZE) {
        float sum = 0.0f;
        
        // Apply convolution with filter coefficients
        for (int j = 0; j < d_filter_size; ++j) {
            int idx = i - d_filter_size / 2 + j;
            
            // Handle boundary conditions
            if (idx >= 0 && idx < d_proj_width) {
                sum += s_proj[idx] * d_filter_coeffs[j];
            }
        }
        
        // Store result
        filtered_projections[row_start + i] = sum;
    }
}

/**
 * @brief Applies a ramp filter to projection data (optimized for T4 GPU, SM 7.5).
 */
template <int BLOCK_SIZE>
__global__ void applyRampFilterKernel_SM75(
    const float* __restrict__ projections,
    float* __restrict__ filtered_projections,
    int filter_type
) {
    // Use cooperative groups for better warp-level control
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Shared memory for input projections
    extern __shared__ float shared_data[];
    
    // Local thread index
    const int tx = threadIdx.x;
    
    // Global indices
    const int proj_idx = blockIdx.y;    // Projection index
    const int row_idx = blockIdx.x;     // Row index within the projection
    
    // Check boundaries
    if (proj_idx >= d_num_angles || row_idx >= d_proj_height) {
        return;
    }
    
    // Compute start index for this projection row
    const int row_start = (proj_idx * d_proj_height + row_idx) * d_proj_width;
    
    // Load projection data to shared memory
    float* s_proj = shared_data;
    
    for (int i = tx; i < d_proj_width; i += BLOCK_SIZE) {
        s_proj[i] = projections[row_start + i];
    }
    
    // Ensure all threads have loaded their data
    block.sync();
    
    // Perform 1D convolution with filter, optimized for warp execution
    for (int i = tx; i < d_proj_width; i += BLOCK_SIZE) {
        float sum = 0.0f;
        
        // Apply convolution with filter coefficients
        for (int j = 0; j < d_filter_size; ++j) {
            int idx = i - d_filter_size / 2 + j;
            
            // Handle boundary conditions
            if (idx >= 0 && idx < d_proj_width) {
                sum += s_proj[idx] * d_filter_coeffs[j];
            }
        }
        
        // Store result
        filtered_projections[row_start + i] = sum;
    }
}

/**
 * @brief Applies a ramp filter to projection data (optimized for Jetson Orin NX, SM 8.7).
 */
template <int BLOCK_SIZE>
__global__ void applyRampFilterKernel_SM87(
    const float* __restrict__ projections,
    float* __restrict__ filtered_projections,
    int filter_type
) {
    // Use cooperative groups
    cg::thread_block block = cg::this_thread_block();
    
    // Shared memory for input projections
    extern __shared__ float shared_data[];
    
    // Local thread index
    const int tx = threadIdx.x;
    
    // Global indices
    const int proj_idx = blockIdx.y;    // Projection index
    const int row_idx = blockIdx.x;     // Row index within the projection
    
    // Check boundaries
    if (proj_idx >= d_num_angles || row_idx >= d_proj_height) {
        return;
    }
    
    // Compute start index for this projection row
    const int row_start = (proj_idx * d_proj_height + row_idx) * d_proj_width;
    
    // Load projection data to shared memory with coalesced access
    float* s_proj = shared_data;
    
    for (int i = tx; i < d_proj_width; i += BLOCK_SIZE) {
        s_proj[i] = projections[row_start + i];
    }
    
    // Ensure all threads have loaded their data
    block.sync();
    
    // Perform 1D convolution with filter using fused multiply-add operations for better performance on Ampere architecture
    for (int i = tx; i < d_proj_width; i += BLOCK_SIZE) {
        float sum = 0.0f;
        
        // Apply convolution with filter coefficients
        for (int j = 0; j < d_filter_size; ++j) {
            int idx = i - d_filter_size / 2 + j;
            
            // Handle boundary conditions
            if (idx >= 0 && idx < d_proj_width) {
                sum = fmaf(s_proj[idx], d_filter_coeffs[j], sum);
            }
        }
        
        // Store result
        filtered_projections[row_start + i] = sum;
    }
}

/**
 * @brief Generic CUDA kernel for ramp filtering (works on all CUDA-capable GPUs).
 */
__global__ void applyRampFilterKernel_Generic(
    const float* __restrict__ projections,
    float* __restrict__ filtered_projections,
    int filter_type
) {
    // Global indices
    const int proj_idx = blockIdx.y;    // Projection index
    const int row_idx = blockIdx.x;     // Row index within the projection
    const int col_idx = threadIdx.x + blockIdx.z * blockDim.x; // Column index
    
    // Check boundaries
    if (proj_idx >= d_num_angles || row_idx >= d_proj_height || col_idx >= d_proj_width) {
        return;
    }
    
    // Compute index for this point
    const int idx = (proj_idx * d_proj_height + row_idx) * d_proj_width + col_idx;
    
    // Perform 1D convolution with filter
    float sum = 0.0f;
    
    // Apply convolution with filter coefficients
    for (int j = 0; j < d_filter_size; ++j) {
        int offset = col_idx - d_filter_size / 2 + j;
        
        // Handle boundary conditions
        if (offset >= 0 && offset < d_proj_width) {
            sum += projections[(proj_idx * d_proj_height + row_idx) * d_proj_width + offset] 
                  * d_filter_coeffs[j];
        }
    }
    
    // Store result
    filtered_projections[idx] = sum;
}

/**
 * @brief Backprojects filtered projections to reconstruct the image (optimized for high-end GPUs, SM >= 8.0).
 * 
 * This kernel implements the backprojection step of the filtered backprojection algorithm.
 * 
 * @param filtered_projections Input filtered projection data
 * @param output Output reconstructed image
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void backprojectKernel_SM80(
    const float* __restrict__ filtered_projections,
    float* __restrict__ output
) {
    // Shared memory for filtered projections
    extern __shared__ float shared_filtered[];
    
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Check boundaries
    if (x >= d_img_width || y >= d_img_height) {
        return;
    }
    
    // Calculate global index
    const int img_idx = y * d_img_width + x;
    
    // Image coordinates relative to center
    const float cx = x - d_img_width / 2.0f + 0.5f;
    const float cy = y - d_img_height / 2.0f + 0.5f;
    
    // Initialize output value
    float sum = 0.0f;
    
    // Loop over all projection angles
    for (int i = 0; i < d_num_angles; ++i) {
        // Calculate angle in radians
        const float angle = d_angles[i];
        
        // Calculate rotated coordinates
        const float cos_angle = cosf(angle);
        const float sin_angle = sinf(angle);
        
        // Calculate projection position (perpendicular distance from origin)
        const float t = cx * cos_angle + cy * sin_angle;
        
        // Convert to projection index
        const float proj_idx = t + d_proj_width / 2.0f;
        
        // Check if projection is within bounds
        if (proj_idx >= 0 && proj_idx < d_proj_width - 1) {
            // Compute interpolation weights
            const int idx_low = static_cast<int>(proj_idx);
            const int idx_high = idx_low + 1;
            const float weight = proj_idx - idx_low;
            
            // Get projection values
            const float val_low = filtered_projections[i * d_proj_width + idx_low];
            const float val_high = filtered_projections[i * d_proj_width + idx_high];
            
            // Linear interpolation
            const float val = (1.0f - weight) * val_low + weight * val_high;
            
            // Add to sum
            sum += val;
        }
    }
    
    // Scale and store result
    output[img_idx] = sum * (PI / (2.0f * d_num_angles));
}

/**
 * @brief Backprojects filtered projections to reconstruct the image (optimized for T4 GPU, SM 7.5).
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void backprojectKernel_SM75(
    const float* __restrict__ filtered_projections,
    float* __restrict__ output
) {
    // Use cooperative groups
    cg::thread_block block = cg::this_thread_block();
    
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Check boundaries
    if (x >= d_img_width || y >= d_img_height) {
        return;
    }
    
    // Calculate global index
    const int img_idx = y * d_img_width + x;
    
    // Image coordinates relative to center
    const float cx = x - d_img_width / 2.0f + 0.5f;
    const float cy = y - d_img_height / 2.0f + 0.5f;
    
    // Initialize output value
    float sum = 0.0f;
    
    // Loop over all projection angles with loop unrolling (helps T4)
    #pragma unroll 4
    for (int i = 0; i < d_num_angles; ++i) {
        // Calculate angle in radians
        const float angle = d_angles[i];
        
        // Calculate rotated coordinates
        const float cos_angle = cosf(angle);
        const float sin_angle = sinf(angle);
        
        // Calculate projection position
        const float t = cx * cos_angle + cy * sin_angle;
        
        // Convert to projection index
        const float proj_idx = t + d_proj_width / 2.0f;
        
        // Check if projection is within bounds
        if (proj_idx >= 0 && proj_idx < d_proj_width - 1) {
            // Compute interpolation weights
            const int idx_low = static_cast<int>(proj_idx);
            const int idx_high = idx_low + 1;
            const float weight = proj_idx - idx_low;
            
            // Get projection values with coalesced memory access
            const float val_low = filtered_projections[i * d_proj_width + idx_low];
            const float val_high = filtered_projections[i * d_proj_width + idx_high];
            
            // Linear interpolation
            const float val = (1.0f - weight) * val_low + weight * val_high;
            
            // Add to sum
            sum += val;
        }
    }
    
    // Scale and store result
    output[img_idx] = sum * (PI / (2.0f * d_num_angles));
}

/**
 * @brief Backprojects filtered projections to reconstruct the image (optimized for Jetson Orin NX, SM 8.7).
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void backprojectKernel_SM87(
    const float* __restrict__ filtered_projections,
    float* __restrict__ output
) {
    // Use cooperative groups
    cg::thread_block block = cg::this_thread_block();
    
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Check boundaries
    if (x >= d_img_width || y >= d_img_height) {
        return;
    }
    
    // Calculate global index
    const int img_idx = y * d_img_width + x;
    
    // Image coordinates relative to center
    const float cx = x - d_img_width / 2.0f + 0.5f;
    const float cy = y - d_img_height / 2.0f + 0.5f;
    
    // Initialize output value
    float sum = 0.0f;
    
    // Loop over all projection angles (optimized for Ampere architecture)
    for (int i = 0; i < d_num_angles; ++i) {
        // Calculate angle in radians
        const float angle = d_angles[i];
        
        // Calculate rotated coordinates
        const float cos_angle = cosf(angle);
        const float sin_angle = sinf(angle);
        
        // Calculate projection position with fused multiply-add for better performance on Ampere
        const float t = fmaf(cx, cos_angle, cy * sin_angle);
        
        // Convert to projection index
        const float proj_idx = t + d_proj_width / 2.0f;
        
        // Check if projection is within bounds
        if (proj_idx >= 0 && proj_idx < d_proj_width - 1) {
            // Compute interpolation weights
            const int idx_low = static_cast<int>(proj_idx);
            const int idx_high = idx_low + 1;
            const float weight = proj_idx - idx_low;
            
            // Get projection values
            const float val_low = filtered_projections[i * d_proj_width + idx_low];
            const float val_high = filtered_projections[i * d_proj_width + idx_high];
            
            // Linear interpolation with fused multiply-add
            const float val = fmaf(weight, val_high, (1.0f - weight) * val_low);
            
            // Add to sum
            sum += val;
        }
    }
    
    // Scale and store result
    output[img_idx] = sum * (PI / (2.0f * d_num_angles));
}

/**
 * @brief Generic CUDA kernel for backprojection (works on all CUDA-capable GPUs).
 */
__global__ void backprojectKernel_Generic(
    const float* __restrict__ filtered_projections,
    float* __restrict__ output
) {
    // Global indices
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check boundaries
    if (x >= d_img_width || y >= d_img_height) {
        return;
    }
    
    // Calculate global index
    const int img_idx = y * d_img_width + x;
    
    // Image coordinates relative to center
    const float cx = x - d_img_width / 2.0f + 0.5f;
    const float cy = y - d_img_height / 2.0f + 0.5f;
    
    // Initialize output value
    float sum = 0.0f;
    
    // Loop over all projection angles
    for (int i = 0; i < d_num_angles; ++i) {
        // Calculate angle in radians
        const float angle = d_angles[i];
        
        // Calculate rotated coordinates
        const float cos_angle = cosf(angle);
        const float sin_angle = sinf(angle);
        
        // Calculate projection position
        const float t = cx * cos_angle + cy * sin_angle;
        
        // Convert to projection index
        const float proj_idx = t + d_proj_width / 2.0f;
        
        // Check if projection is within bounds
        if (proj_idx >= 0 && proj_idx < d_proj_width - 1) {
            // Compute interpolation weights
            const int idx_low = static_cast<int>(proj_idx);
            const int idx_high = idx_low + 1;
            const float weight = proj_idx - idx_low;
            
            // Get projection values
            const float val_low = filtered_projections[i * d_proj_width + idx_low];
            const float val_high = filtered_projections[i * d_proj_width + idx_high];
            
            // Linear interpolation
            const float val = (1.0f - weight) * val_low + weight * val_high;
            
            // Add to sum
            sum += val;
        }
    }
    
    // Scale and store result
    output[img_idx] = sum * (PI / (2.0f * d_num_angles));
}

/**
 * @brief Compute forward projection (ray casting) from image to projections.
 * 
 * This is used in iterative reconstruction algorithms.
 * 
 * @param image Input image
 * @param projections Output projections
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void forwardProjectionKernel_SM80(
    const float* __restrict__ image,
    float* __restrict__ projections
) {
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int proj_idx = blockIdx.y;    // Projection index
    const int t_idx = blockIdx.x * BLOCK_SIZE_X + tx; // Position along detector
    
    // Check boundaries
    if (proj_idx >= d_num_angles || t_idx >= d_proj_width) {
        return;
    }
    
    // Calculate angle
    const float angle = d_angles[proj_idx];
    const float cos_angle = cosf(angle);
    const float sin_angle = sinf(angle);
    
    // Initialize sum
    float sum = 0.0f;
    
    // Ray position in detector coordinate system
    const float t = t_idx - d_proj_width / 2.0f + 0.5f;
    
    // Calculate line parameters
    const float rho = t;
    const float a = -sin_angle;
    const float b = cos_angle;
    
    // Ray integration step size (smaller for better quality)
    const float step_size = 0.5f;
    
    // Length of diagonal (max possible ray length)
    const float max_length = sqrtf(d_img_width * d_img_width + d_img_height * d_img_height);
    
    // Center of image
    const float cx = d_img_width / 2.0f - 0.5f;
    const float cy = d_img_height / 2.0f - 0.5f;
    
    // Ray integration
    for (float s = -max_length; s <= max_length; s += step_size) {
        // Calculate point along the ray
        const float x = t * cos_angle - s * sin_angle + cx;
        const float y = t * sin_angle + s * cos_angle + cy;
        
        // Check if point is within image bounds
        if (x >= 0 && x < d_img_width - 1 && y >= 0 && y < d_img_height - 1) {
            // Bilinear interpolation
            const int x0 = static_cast<int>(x);
            const int y0 = static_cast<int>(y);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;
            
            const float wx = x - x0;
            const float wy = y - y0;
            
            const float val00 = image[y0 * d_img_width + x0];
            const float val01 = image[y0 * d_img_width + x1];
            const float val10 = image[y1 * d_img_width + x0];
            const float val11 = image[y1 * d_img_width + x1];
            
            const float val0 = (1.0f - wx) * val00 + wx * val01;
            const float val1 = (1.0f - wx) * val10 + wx * val11;
            
            const float val = (1.0f - wy) * val0 + wy * val1;
            
            // Add to sum
            sum += val * step_size;
        }
    }
    
    // Store result
    projections[proj_idx * d_proj_width + t_idx] = sum;
}

/**
 * @brief SIRT (Simultaneous Iterative Reconstruction Technique) update kernel.
 * 
 * This kernel updates the image based on the difference between measured and computed projections.
 * 
 * @param image Current image estimate
 * @param measured_projections Measured projections
 * @param computed_projections Computed projections from forward projection
 * @param updated_image Updated image
 * @param relaxation_factor Relaxation factor (controls convergence speed)
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void sirtUpdateKernel_SM80(
    const float* __restrict__ image,
    const float* __restrict__ measured_projections,
    const float* __restrict__ computed_projections,
    float* __restrict__ updated_image,
    float relaxation_factor
) {
    // Local thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Check boundaries
    if (x >= d_img_width || y >= d_img_height) {
        return;
    }
    
    // Calculate global index
    const int img_idx = y * d_img_width + x;
    
    // Image coordinates relative to center
    const float cx = x - d_img_width / 2.0f + 0.5f;
    const float cy = y - d_img_height / 2.0f + 0.5f;
    
    // Initialize correction term
    float correction = 0.0f;
    
    // Loop over all projection angles
    for (int i = 0; i < d_num_angles; ++i) {
        // Calculate angle in radians
        const float angle = d_angles[i];
        
        // Calculate rotated coordinates
        const float cos_angle = cosf(angle);
        const float sin_angle = sinf(angle);
        
        // Calculate projection position
        const float t = cx * cos_angle + cy * sin_angle;
        
        // Convert to projection index
        const float proj_idx = t + d_proj_width / 2.0f;
        
        // Check if projection is within bounds
        if (proj_idx >= 0 && proj_idx < d_proj_width - 1) {
            // Compute interpolation weights
            const int idx_low = static_cast<int>(proj_idx);
            const int idx_high = idx_low + 1;
            const float weight = proj_idx - idx_low;
            
            // Get measured and computed projection values
            const float measured_low = measured_projections[i * d_proj_width + idx_low];
            const float measured_high = measured_projections[i * d_proj_width + idx_high];
            const float computed_low = computed_projections[i * d_proj_width + idx_low];
            const float computed_high = computed_projections[i * d_proj_width + idx_high];
            
            // Linear interpolation
            const float measured = (1.0f - weight) * measured_low + weight * measured_high;
            const float computed = (1.0f - weight) * computed_low + weight * computed_high;
            
            // Add to correction
            correction += (measured - computed);
        }
    }
    
    // Apply correction with relaxation factor
    const float current_value = image[img_idx];
    updated_image[img_idx] = current_value + relaxation_factor * correction / d_num_angles;
}

// Wrapper functions to launch the appropriate kernel based on device capabilities

/**
 * @brief Launch the ramp filter kernel for the appropriate device.
 * @param d_projections Device pointer to projection data
 * @param d_filtered_projections Device pointer to filtered projection data
 * @param num_angles Number of projection angles
 * @param proj_width Width of each projection
 * @param proj_height Height of each projection
 * @param d_angles_ptr Device pointer to projection angles (radians)
 * @param d_filter_coeffs_ptr Device pointer to filter coefficients
 * @param filter_size Size of the filter
 * @param filter_type Filter type (0: Ram-Lak, 1: Shepp-Logan, 2: Cosine, 3: Hamming)
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchRampFilterKernel(
    float* d_projections,
    float* d_filtered_projections,
    int num_angles,
    int proj_width,
    int proj_height,
    float* d_angles_ptr,
    float* d_filter_coeffs_ptr,
    int filter_size,
    int filter_type,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Copy constants to device
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_angles, &num_angles, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proj_width, &proj_width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proj_height, &proj_height, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter_size, &filter_size, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_angles, d_angles_ptr, num_angles * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter_coeffs, d_filter_coeffs_ptr, filter_size * sizeof(float)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Determine block and grid dimensions
    dim3 blockDim(params.block_size_x, 1, 1);
    dim3 gridDim(proj_height, num_angles, 1);
    
    // Calculate shared memory size
    size_t sharedMemSize = proj_width * sizeof(float);
    
    // Launch appropriate kernel based on device capabilities
    if (device_caps.compute_capability_major > 8 || 
        (device_caps.compute_capability_major == 8 && device_caps.compute_capability_minor >= 7)) {
        // Jetson Orin NX (SM 8.7)
        if (params.block_size_x == 256) {
            applyRampFilterKernel_SM87<256><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_projections, d_filtered_projections, filter_type
            );
        } else if (params.block_size_x == 512) {
            applyRampFilterKernel_SM87<512><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_projections, d_filtered_projections, filter_type
            );
        } else {
            applyRampFilterKernel_SM87<256><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_projections, d_filtered_projections, filter_type
            );
        }
    } else if (device_caps.compute_capability_major == 8) {
        // High-end GPUs (SM 8.0+)
        if (params.block_size_x == 256) {
            applyRampFilterKernel_SM80<256><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_projections, d_filtered_projections, filter_type
            );
        } else if (params.block_size_x == 512) {
            applyRampFilterKernel_SM80<512><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_projections, d_filtered_projections, filter_type
            );
        } else {
            applyRampFilterKernel_SM80<256><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_projections, d_filtered_projections, filter_type
            );
        }
    } else if (device_caps.compute_capability_major == 7 && device_caps.compute_capability_minor >= 5) {
        // T4 GPU (SM 7.5)
        if (params.block_size_x == 256) {
            applyRampFilterKernel_SM75<256><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_projections, d_filtered_projections, filter_type
            );
        } else if (params.block_size_x == 512) {
            applyRampFilterKernel_SM75<512><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_projections, d_filtered_projections, filter_type
            );
        } else {
            applyRampFilterKernel_SM75<256><<<gridDim, blockDim, sharedMemSize, stream>>>(
                d_projections, d_filtered_projections, filter_type
            );
        }
    } else {
        // Generic version for older GPUs
        // Use 3D grid to handle larger projections
        int max_threads = 512;  // Maximum threads per block for older GPUs
        int threads = std::min(max_threads, proj_width);
        int blocks = (proj_width + threads - 1) / threads;
        
        dim3 genericBlockDim(threads, 1, 1);
        dim3 genericGridDim(proj_height, num_angles, blocks);
        
        applyRampFilterKernel_Generic<<<genericGridDim, genericBlockDim, 0, stream>>>(
            d_projections, d_filtered_projections, filter_type
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

/**
 * @brief Launch the backprojection kernel for the appropriate device.
 * @param d_filtered_projections Device pointer to filtered projection data
 * @param d_output Device pointer to output image
 * @param num_angles Number of projection angles
 * @param proj_width Width of each projection
 * @param img_width Width of the output image
 * @param img_height Height of the output image
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchBackprojectionKernel(
    float* d_filtered_projections,
    float* d_output,
    int num_angles,
    int proj_width,
    int img_width,
    int img_height,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Copy constants to device (if not already copied)
    CUDA_CHECK(cudaMemcpyToSymbol(d_num_angles, &num_angles, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_proj_width, &proj_width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_img_width, &img_width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_img_height, &img_height, sizeof(int)));
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Determine block and grid dimensions
    dim3 blockDim(params.block_size_x, params.block_size_y, 1);
    dim3 gridDim(
        (img_width + params.block_size_x - 1) / params.block_size_x,
        (img_height + params.block_size_y - 1) / params.block_size_y,
        1
    );
    
    // Launch appropriate kernel based on device capabilities
    if (device_caps.compute_capability_major > 8 || 
        (device_caps.compute_capability_major == 8 && device_caps.compute_capability_minor >= 7)) {
        // Jetson Orin NX (SM 8.7)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            backprojectKernel_SM87<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_filtered_projections, d_output
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            backprojectKernel_SM87<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_filtered_projections, d_output
            );
        } else {
            backprojectKernel_SM87<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_filtered_projections, d_output
            );
        }
    } else if (device_caps.compute_capability_major == 8) {
        // High-end GPUs (SM 8.0+)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            backprojectKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_filtered_projections, d_output
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            backprojectKernel_SM80<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_filtered_projections, d_output
            );
        } else {
            backprojectKernel_SM80<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_filtered_projections, d_output
            );
        }
    } else if (device_caps.compute_capability_major == 7 && device_caps.compute_capability_minor >= 5) {
        // T4 GPU (SM 7.5)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            backprojectKernel_SM75<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_filtered_projections, d_output
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            backprojectKernel_SM75<32, 8><<<gridDim, blockDim, 0, stream>>>(
                d_filtered_projections, d_output
            );
        } else {
            backprojectKernel_SM75<16, 16><<<gridDim, blockDim, 0, stream>>>(
                d_filtered_projections, d_output
            );
        }
    } else {
        // Generic version for older GPUs
        dim3 genericBlockDim(16, 16, 1);  // Default block size for older GPUs
        dim3 genericGridDim(
            (img_width + genericBlockDim.x - 1) / genericBlockDim.x,
            (img_height + genericBlockDim.y - 1) / genericBlockDim.y,
            1
        );
        
        backprojectKernel_Generic<<<genericGridDim, genericBlockDim, 0, stream>>>(
            d_filtered_projections, d_output
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

/**
 * @brief Generate filter coefficients for different filter types.
 * @param filter_coeffs Output filter coefficients
 * @param filter_size Size of the filter
 * @param filter_type Filter type (0: Ram-Lak, 1: Shepp-Logan, 2: Cosine, 3: Hamming)
 */
void generateFilterCoefficients(std::vector<float>& filter_coeffs, int filter_size, int filter_type) {
    filter_coeffs.resize(filter_size);
    
    // Center index
    const int center = filter_size / 2;
    
    // Generate Ram-Lak (ramp) filter
    for (int i = 0; i < filter_size; ++i) {
        const int k = i - center;
        
        // Ram-Lak filter
        if (k == 0) {
            filter_coeffs[i] = 0.25f;  // DC component
        } else if (k % 2 == 0) {
            filter_coeffs[i] = 0.0f;   // Even harmonics are zero
        } else {
            filter_coeffs[i] = -1.0f / (PI * PI * k * k);  // Odd harmonics
        }
        
        // Apply window based on filter type
        switch (filter_type) {
            case 0:  // Ram-Lak (no additional window)
                break;
                
            case 1:  // Shepp-Logan
                if (k != 0) {
                    filter_coeffs[i] *= sinf(PI * k / (filter_size / 2)) / (PI * k / (filter_size / 2));
                }
                break;
                
            case 2:  // Cosine
                filter_coeffs[i] *= cosf(PI * k / filter_size);
                break;
                
            case 3:  // Hamming
                filter_coeffs[i] *= (0.54f + 0.46f * cosf(2.0f * PI * k / filter_size));
                break;
                
            default:  // Default to Ram-Lak
                break;
        }
    }
}

} // namespace medical_imaging