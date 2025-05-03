/**
 * @file shallow_water_kernels.cu
 * @brief CUDA kernels for shallow water equations simulation.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <stdio.h>

#include "../../include/weather_sim/weather_sim.hpp"
#include "../../include/weather_sim/gpu_adaptability.hpp"

namespace cg = cooperative_groups;

namespace weather_sim {

// Device constants for physical parameters
__constant__ float d_gravity;
__constant__ float d_dx;
__constant__ float d_dy;
__constant__ float d_dt;
__constant__ float d_coriolis_f;

// Error checking macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

/**
 * @brief CUDA kernel for shallow water model (optimized for high-end GPUs, SM >= 8.0).
 * @param u Input u velocity
 * @param v Input v velocity
 * @param h Input height field
 * @param u_out Output u velocity
 * @param v_out Output v velocity
 * @param h_out Output height field
 * @param width Grid width
 * @param height Grid height
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void shallowWaterStepKernel_SM80(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ h,
    float* __restrict__ u_out,
    float* __restrict__ v_out,
    float* __restrict__ h_out,
    int width,
    int height
) {
    // Shared memory for block data + halo
    extern __shared__ float shared_data[];
    
    // Organize shared memory: u, v, h for the block plus halo
    constexpr int SHARED_WIDTH = BLOCK_SIZE_X + 2;
    constexpr int SHARED_HEIGHT = BLOCK_SIZE_Y + 2;
    
    float* s_u = shared_data;
    float* s_v = &s_u[SHARED_WIDTH * SHARED_HEIGHT];
    float* s_h = &s_v[SHARED_WIDTH * SHARED_HEIGHT];
    
    // Local thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Shared memory index for current cell
    const int s_idx = (ty + 1) * SHARED_WIDTH + (tx + 1);
    
    // Load center data to shared memory
    if (x < width && y < height) {
        const int g_idx = y * width + x;
        s_u[s_idx] = u[g_idx];
        s_v[s_idx] = v[g_idx];
        s_h[s_idx] = h[g_idx];
    } else {
        s_u[s_idx] = 0.0f;
        s_v[s_idx] = 0.0f;
        s_h[s_idx] = 0.0f;
    }
    
    // Load halo cells
    // Top and bottom rows
    if (ty == 0) {
        // Top row
        const int y_top = max(0, y - 1);
        const int g_idx_top = y_top * width + x;
        const int s_idx_top = 0 * SHARED_WIDTH + (tx + 1);
        
        if (x < width) {
            s_u[s_idx_top] = u[g_idx_top];
            s_v[s_idx_top] = v[g_idx_top];
            s_h[s_idx_top] = h[g_idx_top];
        } else {
            s_u[s_idx_top] = 0.0f;
            s_v[s_idx_top] = 0.0f;
            s_h[s_idx_top] = 0.0f;
        }
        
        // Bottom row
        if (blockIdx.y * BLOCK_SIZE_Y + BLOCK_SIZE_Y < height) {
            // Not the last block, load next row's data
            const int y_bottom = min(height - 1, y + BLOCK_SIZE_Y);
            const int g_idx_bottom = y_bottom * width + x;
            const int s_idx_bottom = (BLOCK_SIZE_Y + 1) * SHARED_WIDTH + (tx + 1);
            
            if (x < width) {
                s_u[s_idx_bottom] = u[g_idx_bottom];
                s_v[s_idx_bottom] = v[g_idx_bottom];
                s_h[s_idx_bottom] = h[g_idx_bottom];
            } else {
                s_u[s_idx_bottom] = 0.0f;
                s_v[s_idx_bottom] = 0.0f;
                s_h[s_idx_bottom] = 0.0f;
            }
        } else {
            // Last block, duplicate boundary
            const int s_idx_bottom = (BLOCK_SIZE_Y + 1) * SHARED_WIDTH + (tx + 1);
            const int s_idx_last = BLOCK_SIZE_Y * SHARED_WIDTH + (tx + 1);
            
            if (tx < BLOCK_SIZE_X && blockIdx.x * BLOCK_SIZE_X + tx < width) {
                s_u[s_idx_bottom] = s_u[s_idx_last];
                s_v[s_idx_bottom] = s_v[s_idx_last];
                s_h[s_idx_bottom] = s_h[s_idx_last];
            }
        }
    }
    
    // Left and right columns
    if (tx == 0) {
        // Left column
        const int x_left = max(0, x - 1);
        const int g_idx_left = y * width + x_left;
        const int s_idx_left = (ty + 1) * SHARED_WIDTH + 0;
        
        if (y < height) {
            s_u[s_idx_left] = u[g_idx_left];
            s_v[s_idx_left] = v[g_idx_left];
            s_h[s_idx_left] = h[g_idx_left];
        } else {
            s_u[s_idx_left] = 0.0f;
            s_v[s_idx_left] = 0.0f;
            s_h[s_idx_left] = 0.0f;
        }
        
        // Right column
        if (blockIdx.x * BLOCK_SIZE_X + BLOCK_SIZE_X < width) {
            // Not the last block, load next column's data
            const int x_right = min(width - 1, x + BLOCK_SIZE_X);
            const int g_idx_right = y * width + x_right;
            const int s_idx_right = (ty + 1) * SHARED_WIDTH + (BLOCK_SIZE_X + 1);
            
            if (y < height) {
                s_u[s_idx_right] = u[g_idx_right];
                s_v[s_idx_right] = v[g_idx_right];
                s_h[s_idx_right] = h[g_idx_right];
            } else {
                s_u[s_idx_right] = 0.0f;
                s_v[s_idx_right] = 0.0f;
                s_h[s_idx_right] = 0.0f;
            }
        } else {
            // Last block, duplicate boundary
            const int s_idx_right = (ty + 1) * SHARED_WIDTH + (BLOCK_SIZE_X + 1);
            const int s_idx_last = (ty + 1) * SHARED_WIDTH + BLOCK_SIZE_X;
            
            if (ty < BLOCK_SIZE_Y && blockIdx.y * BLOCK_SIZE_Y + ty < height) {
                s_u[s_idx_right] = s_u[s_idx_last];
                s_v[s_idx_right] = s_v[s_idx_last];
                s_h[s_idx_right] = s_h[s_idx_last];
            }
        }
    }
    
    // Synchronize to ensure all data is loaded into shared memory
    __syncthreads();
    
    // Only compute for valid grid points
    if (x < width && y < height) {
        // Spatial indices for finite differences
        const int s_idx_l = s_idx - 1;               // left
        const int s_idx_r = s_idx + 1;               // right
        const int s_idx_t = s_idx - SHARED_WIDTH;    // top
        const int s_idx_b = s_idx + SHARED_WIDTH;    // bottom
        
        // Compute finite differences
        const float h_x = (s_h[s_idx_r] - s_h[s_idx_l]) / (2.0f * d_dx);
        const float h_y = (s_h[s_idx_b] - s_h[s_idx_t]) / (2.0f * d_dy);
        
        const float u_x = (s_u[s_idx_r] - s_u[s_idx_l]) / (2.0f * d_dx);
        const float u_y = (s_u[s_idx_b] - s_u[s_idx_t]) / (2.0f * d_dy);
        
        const float v_x = (s_v[s_idx_r] - s_v[s_idx_l]) / (2.0f * d_dx);
        const float v_y = (s_v[s_idx_b] - s_v[s_idx_t]) / (2.0f * d_dy);
        
        // Compute tendencies (time derivatives)
        const float du_dt = -s_u[s_idx] * u_x - s_v[s_idx] * u_y - d_gravity * h_x + d_coriolis_f * s_v[s_idx];
        const float dv_dt = -s_u[s_idx] * v_x - s_v[s_idx] * v_y - d_gravity * h_y - d_coriolis_f * s_u[s_idx];
        const float dh_dt = -s_h[s_idx] * (u_x + v_y) - s_u[s_idx] * h_x - s_v[s_idx] * h_y;
        
        // Update using forward Euler time stepping
        const int g_idx = y * width + x;
        u_out[g_idx] = s_u[s_idx] + d_dt * du_dt;
        v_out[g_idx] = s_v[s_idx] + d_dt * dv_dt;
        h_out[g_idx] = s_h[s_idx] + d_dt * dh_dt;
    }
}

/**
 * @brief CUDA kernel for shallow water model (optimized for T4 GPU, SM 7.5).
 * @param u Input u velocity
 * @param v Input v velocity
 * @param h Input height field
 * @param u_out Output u velocity
 * @param v_out Output v velocity
 * @param h_out Output height field
 * @param width Grid width
 * @param height Grid height
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void shallowWaterStepKernel_SM75(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ h,
    float* __restrict__ u_out,
    float* __restrict__ v_out,
    float* __restrict__ h_out,
    int width,
    int height
) {
    // T4-specific optimizations
    // Use cooperative groups and warp-level primitives for better performance
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    // Shared memory for block data with slightly different layout for SM 7.5
    extern __shared__ float shared_data[];
    
    // Organize shared memory more efficiently for T4
    constexpr int SHARED_WIDTH = BLOCK_SIZE_X + 2;
    constexpr int SHARED_HEIGHT = BLOCK_SIZE_Y + 2;
    
    float* s_u = shared_data;
    float* s_v = &s_u[SHARED_WIDTH * SHARED_HEIGHT];
    float* s_h = &s_v[SHARED_WIDTH * SHARED_HEIGHT];
    
    // Local thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Load data to shared memory (similar to SM80 version but with different memory access pattern)
    // This pattern is more suited to T4's memory system
    const int s_idx = (ty + 1) * SHARED_WIDTH + (tx + 1);
    
    if (x < width && y < height) {
        const int g_idx = y * width + x;
        s_u[s_idx] = u[g_idx];
        s_v[s_idx] = v[g_idx];
        s_h[s_idx] = h[g_idx];
    } else {
        s_u[s_idx] = 0.0f;
        s_v[s_idx] = 0.0f;
        s_h[s_idx] = 0.0f;
    }
    
    // Load halo (similar approach to SM80 version with minor adjustments)
    // Top and bottom rows
    if (ty < 1) {
        // Top row
        const int y_top = max(0, blockIdx.y * BLOCK_SIZE_Y - 1 + ty);
        const int s_row = ty;
        
        for (int i = tx; i < BLOCK_SIZE_X; i += BLOCK_SIZE_X / warp.size()) {
            const int x_i = blockIdx.x * BLOCK_SIZE_X + i;
            if (x_i < width && y_top < height) {
                const int g_idx = y_top * width + x_i;
                const int s_idx = s_row * SHARED_WIDTH + (i + 1);
                s_u[s_idx] = u[g_idx];
                s_v[s_idx] = v[g_idx];
                s_h[s_idx] = h[g_idx];
            }
        }
        
        // Bottom row
        const int y_bottom = min(height - 1, blockIdx.y * BLOCK_SIZE_Y + BLOCK_SIZE_Y + ty);
        const int s_row_bottom = BLOCK_SIZE_Y + 1 + ty;
        
        for (int i = tx; i < BLOCK_SIZE_X; i += BLOCK_SIZE_X / warp.size()) {
            const int x_i = blockIdx.x * BLOCK_SIZE_X + i;
            if (x_i < width && y_bottom < height) {
                const int g_idx = y_bottom * width + x_i;
                const int s_idx = s_row_bottom * SHARED_WIDTH + (i + 1);
                s_u[s_idx] = u[g_idx];
                s_v[s_idx] = v[g_idx];
                s_h[s_idx] = h[g_idx];
            }
        }
    }
    
    // Left and right columns
    if (tx < 1) {
        // Left column
        const int x_left = max(0, blockIdx.x * BLOCK_SIZE_X - 1 + tx);
        const int s_col = tx;
        
        for (int i = ty; i < BLOCK_SIZE_Y; i += BLOCK_SIZE_Y / warp.size()) {
            const int y_i = blockIdx.y * BLOCK_SIZE_Y + i;
            if (y_i < height && x_left < width) {
                const int g_idx = y_i * width + x_left;
                const int s_idx = (i + 1) * SHARED_WIDTH + s_col;
                s_u[s_idx] = u[g_idx];
                s_v[s_idx] = v[g_idx];
                s_h[s_idx] = h[g_idx];
            }
        }
        
        // Right column
        const int x_right = min(width - 1, blockIdx.x * BLOCK_SIZE_X + BLOCK_SIZE_X + tx);
        const int s_col_right = BLOCK_SIZE_X + 1 + tx;
        
        for (int i = ty; i < BLOCK_SIZE_Y; i += BLOCK_SIZE_Y / warp.size()) {
            const int y_i = blockIdx.y * BLOCK_SIZE_Y + i;
            if (y_i < height && x_right < width) {
                const int g_idx = y_i * width + x_right;
                const int s_idx = (i + 1) * SHARED_WIDTH + s_col_right;
                s_u[s_idx] = u[g_idx];
                s_v[s_idx] = v[g_idx];
                s_h[s_idx] = h[g_idx];
            }
        }
    }
    
    // Synchronize to ensure all data is loaded into shared memory
    block.sync();
    
    // Only compute for valid grid points
    if (x < width && y < height) {
        // Spatial indices for finite differences
        const int s_idx_l = s_idx - 1;               // left
        const int s_idx_r = s_idx + 1;               // right
        const int s_idx_t = s_idx - SHARED_WIDTH;    // top
        const int s_idx_b = s_idx + SHARED_WIDTH;    // bottom
        
        // Compute finite differences
        const float h_x = (s_h[s_idx_r] - s_h[s_idx_l]) / (2.0f * d_dx);
        const float h_y = (s_h[s_idx_b] - s_h[s_idx_t]) / (2.0f * d_dy);
        
        const float u_x = (s_u[s_idx_r] - s_u[s_idx_l]) / (2.0f * d_dx);
        const float u_y = (s_u[s_idx_b] - s_u[s_idx_t]) / (2.0f * d_dy);
        
        const float v_x = (s_v[s_idx_r] - s_v[s_idx_l]) / (2.0f * d_dx);
        const float v_y = (s_v[s_idx_b] - s_v[s_idx_t]) / (2.0f * d_dy);
        
        // Compute tendencies (time derivatives)
        const float du_dt = -s_u[s_idx] * u_x - s_v[s_idx] * u_y - d_gravity * h_x + d_coriolis_f * s_v[s_idx];
        const float dv_dt = -s_u[s_idx] * v_x - s_v[s_idx] * v_y - d_gravity * h_y - d_coriolis_f * s_u[s_idx];
        const float dh_dt = -s_h[s_idx] * (u_x + v_y) - s_u[s_idx] * h_x - s_v[s_idx] * h_y;
        
        // Update using forward Euler time stepping
        const int g_idx = y * width + x;
        u_out[g_idx] = s_u[s_idx] + d_dt * du_dt;
        v_out[g_idx] = s_v[s_idx] + d_dt * dv_dt;
        h_out[g_idx] = s_h[s_idx] + d_dt * dh_dt;
    }
}

/**
 * @brief CUDA kernel for shallow water model (optimized for Jetson Orin NX, SM 8.7).
 * @param u Input u velocity
 * @param v Input v velocity
 * @param h Input height field
 * @param u_out Output u velocity
 * @param v_out Output v velocity
 * @param h_out Output height field
 * @param width Grid width
 * @param height Grid height
 */
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void shallowWaterStepKernel_SM87(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ h,
    float* __restrict__ u_out,
    float* __restrict__ v_out,
    float* __restrict__ h_out,
    int width,
    int height
) {
    // Jetson Orin NX-specific optimizations
    // Use cooperative groups for better handling of thread synchronization
    cg::thread_block block = cg::this_thread_block();
    
    // Shared memory for block data optimized for Ampere architecture
    extern __shared__ float shared_data[];
    
    // Organize shared memory layout optimized for Ampere's memory architecture
    constexpr int SHARED_WIDTH = BLOCK_SIZE_X + 2;
    constexpr int SHARED_HEIGHT = BLOCK_SIZE_Y + 2;
    
    // Interleaved shared memory layout for better cache locality on Ampere
    // Each thread loads a 3-element vector (u, v, h) for better memory coalescence
    float* s_u = shared_data;
    float* s_v = &s_u[SHARED_WIDTH * SHARED_HEIGHT];
    float* s_h = &s_v[SHARED_WIDTH * SHARED_HEIGHT];
    
    // Local thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Global indices
    const int x = blockIdx.x * BLOCK_SIZE_X + tx;
    const int y = blockIdx.y * BLOCK_SIZE_Y + ty;
    
    // Shared memory index for current cell
    const int s_idx = (ty + 1) * SHARED_WIDTH + (tx + 1);
    
    // Load center data to shared memory
    if (x < width && y < height) {
        const int g_idx = y * width + x;
        s_u[s_idx] = u[g_idx];
        s_v[s_idx] = v[g_idx];
        s_h[s_idx] = h[g_idx];
    } else {
        s_u[s_idx] = 0.0f;
        s_v[s_idx] = 0.0f;
        s_h[s_idx] = 0.0f;
    }
    
    // Load halo using a more efficient approach for Ampere
    // Each thread loads multiple elements for better memory utilization
    
    // Handle top and bottom halos
    if (ty < 2) {
        // Top halo (ty == 0) or bottom halo (ty == 1)
        const int halo_y = ty == 0 ? 
            max(0, blockIdx.y * BLOCK_SIZE_Y - 1) : 
            min(height - 1, blockIdx.y * BLOCK_SIZE_Y + BLOCK_SIZE_Y);
            
        const int halo_s_row = ty == 0 ? 0 : BLOCK_SIZE_Y + 1;
        
        // Each thread loads multiple items horizontally
        for (int i = tx; i < BLOCK_SIZE_X; i += blockDim.x / 2) {
            const int halo_x = blockIdx.x * BLOCK_SIZE_X + i;
            if (halo_x < width) {
                const int g_idx = halo_y * width + halo_x;
                const int s_idx = halo_s_row * SHARED_WIDTH + (i + 1);
                s_u[s_idx] = u[g_idx];
                s_v[s_idx] = v[g_idx];
                s_h[s_idx] = h[g_idx];
            }
        }
    }
    
    // Handle left and right halos
    if (tx < 2) {
        // Left halo (tx == 0) or right halo (tx == 1)
        const int halo_x = tx == 0 ? 
            max(0, blockIdx.x * BLOCK_SIZE_X - 1) : 
            min(width - 1, blockIdx.x * BLOCK_SIZE_X + BLOCK_SIZE_X);
            
        const int halo_s_col = tx == 0 ? 0 : BLOCK_SIZE_X + 1;
        
        // Each thread loads multiple items vertically
        for (int i = ty; i < BLOCK_SIZE_Y; i += blockDim.y / 2) {
            const int halo_y = blockIdx.y * BLOCK_SIZE_Y + i;
            if (halo_y < height) {
                const int g_idx = halo_y * width + halo_x;
                const int s_idx = (i + 1) * SHARED_WIDTH + halo_s_col;
                s_u[s_idx] = u[g_idx];
                s_v[s_idx] = v[g_idx];
                s_h[s_idx] = h[g_idx];
            }
        }
    }
    
    // Synchronize to ensure all data is loaded into shared memory
    block.sync();
    
    // Only compute for valid grid points
    if (x < width && y < height) {
        // Spatial indices for finite differences
        const int s_idx_l = s_idx - 1;               // left
        const int s_idx_r = s_idx + 1;               // right
        const int s_idx_t = s_idx - SHARED_WIDTH;    // top
        const int s_idx_b = s_idx + SHARED_WIDTH;    // bottom
        
        // Compute finite differences with fused operations for better instruction utilization
        const float h_x = (s_h[s_idx_r] - s_h[s_idx_l]) * (0.5f / d_dx);
        const float h_y = (s_h[s_idx_b] - s_h[s_idx_t]) * (0.5f / d_dy);
        
        const float u_x = (s_u[s_idx_r] - s_u[s_idx_l]) * (0.5f / d_dx);
        const float u_y = (s_u[s_idx_b] - s_u[s_idx_t]) * (0.5f / d_dy);
        
        const float v_x = (s_v[s_idx_r] - s_v[s_idx_l]) * (0.5f / d_dx);
        const float v_y = (s_v[s_idx_b] - s_v[s_idx_t]) * (0.5f / d_dy);
        
        // Current cell values
        const float u_c = s_u[s_idx];
        const float v_c = s_v[s_idx];
        const float h_c = s_h[s_idx];
        
        // Compute tendencies (time derivatives)
        // Use fused multiply-add operations for better performance on Ampere
        float du_dt = -u_c * u_x;
        du_dt = fmaf(-v_c, u_y, du_dt);
        du_dt = fmaf(-d_gravity, h_x, du_dt);
        du_dt = fmaf(d_coriolis_f, v_c, du_dt);
        
        float dv_dt = -u_c * v_x;
        dv_dt = fmaf(-v_c, v_y, dv_dt);
        dv_dt = fmaf(-d_gravity, h_y, dv_dt);
        dv_dt = fmaf(-d_coriolis_f, u_c, dv_dt);
        
        float dh_dt = -h_c * (u_x + v_y);
        dh_dt = fmaf(-u_c, h_x, dh_dt);
        dh_dt = fmaf(-v_c, h_y, dh_dt);
        
        // Update using forward Euler time stepping
        const int g_idx = y * width + x;
        u_out[g_idx] = u_c + d_dt * du_dt;
        v_out[g_idx] = v_c + d_dt * dv_dt;
        h_out[g_idx] = h_c + d_dt * dh_dt;
    }
}

/**
 * @brief Generic CUDA kernel for shallow water model (works on all CUDA-capable GPUs).
 * @param u Input u velocity
 * @param v Input v velocity
 * @param h Input height field
 * @param u_out Output u velocity
 * @param v_out Output v velocity
 * @param h_out Output height field
 * @param width Grid width
 * @param height Grid height
 */
__global__ void shallowWaterStepKernel_Generic(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ h,
    float* __restrict__ u_out,
    float* __restrict__ v_out,
    float* __restrict__ h_out,
    int width,
    int height
) {
    // Global indices
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only compute for valid grid points
    if (x >= width || y >= height) return;
    
    // Global index
    const int idx = y * width + x;
    
    // Get current cell values
    const float u_c = u[idx];
    const float v_c = v[idx];
    const float h_c = h[idx];
    
    // Compute indices for neighboring cells with boundary handling
    const int left  = x > 0 ? idx - 1 : idx;
    const int right = x < width - 1 ? idx + 1 : idx;
    const int top   = y > 0 ? idx - width : idx;
    const int bottom = y < height - 1 ? idx + width : idx;
    
    // Compute finite differences
    const float h_x = (h[right] - h[left]) / (2.0f * d_dx);
    const float h_y = (h[bottom] - h[top]) / (2.0f * d_dy);
    
    const float u_x = (u[right] - u[left]) / (2.0f * d_dx);
    const float u_y = (u[bottom] - u[top]) / (2.0f * d_dy);
    
    const float v_x = (v[right] - v[left]) / (2.0f * d_dx);
    const float v_y = (v[bottom] - v[top]) / (2.0f * d_dy);
    
    // Compute tendencies (time derivatives)
    const float du_dt = -u_c * u_x - v_c * u_y - d_gravity * h_x + d_coriolis_f * v_c;
    const float dv_dt = -u_c * v_x - v_c * v_y - d_gravity * h_y - d_coriolis_f * u_c;
    const float dh_dt = -h_c * (u_x + v_y) - u_c * h_x - v_c * h_y;
    
    // Update using forward Euler time stepping
    u_out[idx] = u_c + d_dt * du_dt;
    v_out[idx] = v_c + d_dt * dv_dt;
    h_out[idx] = h_c + d_dt * dh_dt;
}

/**
 * @brief Calculate vorticity for the given velocity field.
 * @param u u-component of velocity
 * @param v v-component of velocity
 * @param vorticity Output vorticity field
 * @param width Grid width
 * @param height Grid height
 */
__global__ void calculateVorticityKernel(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ vorticity,
    int width,
    int height
) {
    // Global indices
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only compute for valid grid points
    if (x >= width || y >= height) return;
    
    // Global index
    const int idx = y * width + x;
    
    // Compute indices for neighboring cells with boundary handling
    const int left  = x > 0 ? idx - 1 : idx;
    const int right = x < width - 1 ? idx + 1 : idx;
    const int top   = y > 0 ? idx - width : idx;
    const int bottom = y < height - 1 ? idx + width : idx;
    
    // Compute finite differences
    const float v_x = (v[right] - v[left]) / (2.0f * d_dx);
    const float u_y = (u[bottom] - u[top]) / (2.0f * d_dy);
    
    // Vorticity = dv/dx - du/dy
    vorticity[idx] = v_x - u_y;
}

/**
 * @brief Calculate divergence for the given velocity field.
 * @param u u-component of velocity
 * @param v v-component of velocity
 * @param divergence Output divergence field
 * @param width Grid width
 * @param height Grid height
 */
__global__ void calculateDivergenceKernel(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ divergence,
    int width,
    int height
) {
    // Global indices
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Only compute for valid grid points
    if (x >= width || y >= height) return;
    
    // Global index
    const int idx = y * width + x;
    
    // Compute indices for neighboring cells with boundary handling
    const int left  = x > 0 ? idx - 1 : idx;
    const int right = x < width - 1 ? idx + 1 : idx;
    const int top   = y > 0 ? idx - width : idx;
    const int bottom = y < height - 1 ? idx + width : idx;
    
    // Compute finite differences
    const float u_x = (u[right] - u[left]) / (2.0f * d_dx);
    const float v_y = (v[bottom] - v[top]) / (2.0f * d_dy);
    
    // Divergence = du/dx + dv/dy
    divergence[idx] = u_x + v_y;
}

// Wrapper functions to launch the appropriate kernel based on device capabilities

/**
 * @brief Launch the shallow water step kernel for the appropriate device.
 * @param d_u Input u velocity
 * @param d_v Input v velocity
 * @param d_h Input height field
 * @param d_u_out Output u velocity
 * @param d_v_out Output v velocity
 * @param d_h_out Output height field
 * @param width Grid width
 * @param height Grid height
 * @param dt Time step
 * @param gravity Gravity constant
 * @param dx Grid spacing in x direction
 * @param dy Grid spacing in y direction
 * @param coriolis_f Coriolis parameter
 * @param device_caps Device capabilities
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchShallowWaterKernel(
    float* d_u,
    float* d_v,
    float* d_h,
    float* d_u_out,
    float* d_v_out,
    float* d_h_out,
    int width,
    int height,
    float dt,
    float gravity,
    float dx,
    float dy,
    float coriolis_f,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
) {
    // Set constant memory parameters
    float h_gravity = gravity;
    float h_dx = dx;
    float h_dy = dy;
    float h_dt = dt;
    float h_coriolis_f = coriolis_f;
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_gravity, &h_gravity, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dx, &h_dx, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dy, &h_dy, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dt, &h_dt, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_coriolis_f, &h_coriolis_f, sizeof(float)));
    
    // Create 3D grid and block dimensions
    dim3 block(params.block_size_x, params.block_size_y, params.block_size_z);
    dim3 grid(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        params.grid_size_z
    );
    
    // Calculate shared memory size
    int shared_width = params.block_size_x + 2;
    int shared_height = params.block_size_y + 2;
    size_t shared_mem_size = 3 * shared_width * shared_height * sizeof(float);
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Launch appropriate kernel based on device capabilities
    if (device_caps.compute_capability_major > 8 || 
        (device_caps.compute_capability_major == 8 && device_caps.compute_capability_minor >= 7)) {
        // Jetson Orin NX (SM 8.7)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            shallowWaterStepKernel_SM87<16, 16><<<grid, block, shared_mem_size, stream>>>(
                d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
            );
        } else if (params.block_size_x == 8 && params.block_size_y == 8) {
            shallowWaterStepKernel_SM87<8, 8><<<grid, block, shared_mem_size, stream>>>(
                d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            shallowWaterStepKernel_SM87<32, 8><<<grid, block, shared_mem_size, stream>>>(
                d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
            );
        } else {
            // Fall back to generic for unsupported block sizes
            shallowWaterStepKernel_Generic<<<grid, block, 0, stream>>>(
                d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
            );
        }
    } else if (device_caps.compute_capability_major == 8) {
        // High-end GPUs (SM 8.0+)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            shallowWaterStepKernel_SM80<16, 16><<<grid, block, shared_mem_size, stream>>>(
                d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            shallowWaterStepKernel_SM80<32, 8><<<grid, block, shared_mem_size, stream>>>(
                d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
            );
        } else {
            // Fall back to generic for unsupported block sizes
            shallowWaterStepKernel_Generic<<<grid, block, 0, stream>>>(
                d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
            );
        }
    } else if (device_caps.compute_capability_major == 7 && device_caps.compute_capability_minor >= 5) {
        // T4 GPU (SM 7.5)
        if (params.block_size_x == 16 && params.block_size_y == 16) {
            shallowWaterStepKernel_SM75<16, 16><<<grid, block, shared_mem_size, stream>>>(
                d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
            );
        } else if (params.block_size_x == 32 && params.block_size_y == 8) {
            shallowWaterStepKernel_SM75<32, 8><<<grid, block, shared_mem_size, stream>>>(
                d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
            );
        } else {
            // Fall back to generic for unsupported block sizes
            shallowWaterStepKernel_Generic<<<grid, block, 0, stream>>>(
                d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
            );
        }
    } else {
        // Generic version for older GPUs
        shallowWaterStepKernel_Generic<<<grid, block, 0, stream>>>(
            d_u, d_v, d_h, d_u_out, d_v_out, d_h_out, width, height
        );
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

/**
 * @brief Launch the diagnostics kernels.
 * @param d_u Input u velocity
 * @param d_v Input v velocity
 * @param d_vorticity Output vorticity field
 * @param d_divergence Output divergence field
 * @param width Grid width
 * @param height Grid height
 * @param dx Grid spacing in x direction
 * @param dy Grid spacing in y direction
 * @param params Kernel launch parameters
 * @return True if successful
 */
bool launchDiagnosticsKernels(
    float* d_u,
    float* d_v,
    float* d_vorticity,
    float* d_divergence,
    int width,
    int height,
    float dx,
    float dy,
    const KernelLaunchParams& params
) {
    // Set constant memory parameters
    float h_dx = dx;
    float h_dy = dy;
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_dx, &h_dx, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_dy, &h_dy, sizeof(float)));
    
    // Create 3D grid and block dimensions
    dim3 block(params.block_size_x, params.block_size_y, params.block_size_z);
    dim3 grid(
        (width + params.block_size_x - 1) / params.block_size_x,
        (height + params.block_size_y - 1) / params.block_size_y,
        params.grid_size_z
    );
    
    // Get CUDA stream
    cudaStream_t stream = 0;  // Default stream
    
    // Launch vorticity kernel
    calculateVorticityKernel<<<grid, block, 0, stream>>>(
        d_u, d_v, d_vorticity, width, height
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Launch divergence kernel
    calculateDivergenceKernel<<<grid, block, 0, stream>>>(
        d_u, d_v, d_divergence, width, height
    );
    CUDA_CHECK(cudaGetLastError());
    
    return true;
}

} // namespace weather_sim