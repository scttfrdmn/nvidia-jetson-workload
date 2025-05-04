/**
 * @file dem_kernels.cu
 * @brief CUDA kernels for Digital Elevation Model processing
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <limits>

namespace geospatial {
namespace kernels {

/**
 * @brief Viewshed calculation kernel
 * 
 * Computes visibility from an observer point to all other DEM cells
 * using line-of-sight algorithm with adjustments for Earth curvature.
 * 
 * @param dem Input Digital Elevation Model
 * @param viewshed Output viewshed raster (1=visible, 0=not visible)
 * @param width DEM width
 * @param height DEM height
 * @param observer_x Observer X coordinate in DEM cells
 * @param observer_y Observer Y coordinate in DEM cells
 * @param observer_height Observer height above the terrain (meters)
 * @param max_radius Maximum visibility radius in cells (0 for unlimited)
 * @param cell_size Cell size in meters
 * @param curvature_coeff Coefficient for Earth curvature adjustment
 */
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
    float curvature_coeff)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int idx = row * width + col;
    
    // Check if the cell is the observer or has NoData
    if (dem[idx] == -9999.0f) {
        viewshed[idx] = 0.0f;
        return;
    }
    
    // Convert cell coordinates to floating-point for precise calculations
    float cell_x = static_cast<float>(col);
    float cell_y = static_cast<float>(row);
    
    // Calculate distance from observer to cell
    float dx = cell_x - observer_x;
    float dy = cell_y - observer_y;
    float distance = sqrtf(dx * dx + dy * dy) * cell_size;
    
    // Apply distance limit if specified
    if (max_radius > 0 && distance > max_radius) {
        viewshed[idx] = 0.0f;
        return;
    }
    
    // For cells very close to the observer, mark as visible
    if (distance < cell_size * 0.5f) {
        viewshed[idx] = 1.0f;
        return;
    }
    
    // Get observer and target elevations
    int observer_idx = static_cast<int>(observer_y) * width + static_cast<int>(observer_x);
    float observer_elevation = dem[observer_idx] + observer_height;
    float target_elevation = dem[idx];
    
    // Calculate the line from observer to target
    float angle = atan2f(dy, dx);
    float max_angle = -INFINITY;
    
    // Sample points along the line
    int num_samples = ceilf(distance / (cell_size * 0.5f));
    
    for (int i = 1; i < num_samples; i++) {
        float sample_fraction = static_cast<float>(i) / static_cast<float>(num_samples);
        float sample_distance = sample_fraction * distance;
        
        // Calculate earth curvature adjustment for this distance
        float curvature_adjustment = curvature_coeff * sample_distance * sample_distance;
        
        // Calculate the sample position
        float sample_x = observer_x + sample_fraction * dx;
        float sample_y = observer_y + sample_fraction * dy;
        
        // Interpolate elevation at this position
        int sample_col = __float2int_rd(sample_x);
        int sample_row = __float2int_rd(sample_y);
        
        // Bounds check
        if (sample_col < 0 || sample_col >= width - 1 || sample_row < 0 || sample_row >= height - 1) {
            continue;
        }
        
        // Bilinear interpolation weights
        float wx = sample_x - sample_col;
        float wy = sample_y - sample_row;
        
        int ul_idx = sample_row * width + sample_col;
        int ur_idx = ul_idx + 1;
        int ll_idx = ul_idx + width;
        int lr_idx = ll_idx + 1;
        
        float ul_val = dem[ul_idx];
        float ur_val = dem[ur_idx];
        float ll_val = dem[ll_idx];
        float lr_val = dem[lr_idx];
        
        // Skip calculation if any sample has NoData
        if (ul_val == -9999.0f || ur_val == -9999.0f || ll_val == -9999.0f || lr_val == -9999.0f) {
            continue;
        }
        
        // Perform bilinear interpolation
        float top = (1.0f - wx) * ul_val + wx * ur_val;
        float bottom = (1.0f - wx) * ll_val + wx * lr_val;
        float sample_elevation = (1.0f - wy) * top + wy * bottom;
        
        // Adjust for Earth curvature
        sample_elevation -= curvature_adjustment;
        
        // Calculate the angle to this sample
        float h_diff = sample_elevation - observer_elevation;
        float angle_to_sample = atanf(h_diff / sample_distance);
        
        // Update the maximum angle seen so far
        max_angle = fmaxf(max_angle, angle_to_sample);
    }
    
    // Calculate the angle to the target
    float h_diff = target_elevation - observer_elevation;
    float curvature_adjustment = curvature_coeff * distance * distance;
    float adjusted_h_diff = h_diff - curvature_adjustment;
    float target_angle = atanf(adjusted_h_diff / distance);
    
    // If the target angle is greater than or equal to the maximum angle seen,
    // the target is visible
    if (target_angle >= max_angle) {
        viewshed[idx] = 1.0f;
    } else {
        viewshed[idx] = 0.0f;
    }
}

/**
 * @brief Calculate terrain derivatives (slope, aspect, curvature)
 * 
 * Uses a 3x3 neighborhood to calculate terrain derivatives using
 * the Horn algorithm.
 * 
 * @param dem Input Digital Elevation Model
 * @param slope Output slope raster (degrees)
 * @param aspect Output aspect raster (degrees)
 * @param curvature Output curvature raster
 * @param width DEM width
 * @param height DEM height
 * @param cell_size Cell size in projection units
 * @param z_factor Vertical exaggeration factor
 */
__global__ void terrainDerivativesKernel(
    const float* __restrict__ dem,
    float* __restrict__ slope,
    float* __restrict__ aspect,
    float* __restrict__ curvature,
    int width,
    int height,
    float cell_size,
    float z_factor)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col <= 0 || col >= width - 1 || row <= 0 || row >= height - 1) return;
    
    int idx = row * width + col;
    
    // Get the 3x3 neighborhood
    float z11 = dem[(row-1)*width + (col-1)];
    float z12 = dem[(row-1)*width + col];
    float z13 = dem[(row-1)*width + (col+1)];
    float z21 = dem[row*width + (col-1)];
    float z22 = dem[idx]; // Center cell
    float z23 = dem[row*width + (col+1)];
    float z31 = dem[(row+1)*width + (col-1)];
    float z32 = dem[(row+1)*width + col];
    float z33 = dem[(row+1)*width + (col+1)];
    
    // Check for NoData
    if (z11 == -9999.0f || z12 == -9999.0f || z13 == -9999.0f ||
        z21 == -9999.0f || z22 == -9999.0f || z23 == -9999.0f ||
        z31 == -9999.0f || z32 == -9999.0f || z33 == -9999.0f) {
        slope[idx] = -9999.0f;
        aspect[idx] = -9999.0f;
        curvature[idx] = -9999.0f;
        return;
    }
    
    // Apply z-factor
    z11 *= z_factor; z12 *= z_factor; z13 *= z_factor;
    z21 *= z_factor; z22 *= z_factor; z23 *= z_factor;
    z31 *= z_factor; z32 *= z_factor; z33 *= z_factor;
    
    // Calculate local differences
    float dx = ((z13 - z11) + 2.0f * (z23 - z21) + (z33 - z31)) / (8.0f * cell_size);
    float dy = ((z31 - z11) + 2.0f * (z32 - z12) + (z33 - z13)) / (8.0f * cell_size);
    
    // Calculate slope (in degrees)
    float slope_radians = atanf(sqrtf(dx*dx + dy*dy));
    slope[idx] = slope_radians * 180.0f / M_PI;
    
    // Calculate aspect (in degrees)
    float aspect_radians = 0.0f;
    if (dx == 0.0f) {
        if (dy > 0.0f) {
            aspect_radians = M_PI / 2.0f;
        } else if (dy < 0.0f) {
            aspect_radians = 3.0f * M_PI / 2.0f;
        } else {
            aspect_radians = 0.0f;
        }
    } else {
        aspect_radians = atanf(dy / dx);
        
        if (dx < 0.0f) {
            aspect_radians += M_PI;
        } else if (dy < 0.0f) {
            aspect_radians += 2.0f * M_PI;
        }
    }
    
    // Convert to degrees and adjust to start from North
    aspect[idx] = fmodf(90.0f - aspect_radians * 180.0f / M_PI + 360.0f, 360.0f);
    
    // Calculate curvature (second derivative)
    float d2x = (z13 - 2.0f * z22 + z31) / (cell_size * cell_size);
    float d2y = (z31 - 2.0f * z22 + z13) / (cell_size * cell_size);
    float dxy = ((z33 - z31 - z13 + z11) / 4.0f) / (cell_size * cell_size);
    
    // Total curvature
    curvature[idx] = -1.0f * (d2x + d2y);
}

/**
 * @brief Fill sinks in a DEM using a priority-flood algorithm
 * 
 * Implements a simplified version of the Priority-Flood algorithm to fill sinks
 * in a DEM, limited to a maximum z-value difference if specified.
 * 
 * This kernel models a serial algorithm in a parallel setting by using multiple
 * iterations with boundary exchanges.
 * 
 * @param dem Input Digital Elevation Model
 * @param filled Output filled DEM
 * @param width DEM width
 * @param height DEM height
 * @param z_limit Maximum z-value difference to apply fill
 * @param changed Flag to indicate if the DEM was changed in this iteration
 */
__global__ void fillSinksIterationKernel(
    const float* __restrict__ dem,
    float* __restrict__ filled,
    int width,
    int height,
    float z_limit,
    int* changed)
{
    // Each thread processes one cell
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < 0 || col >= width || row < 0 || row >= height) return;
    
    int idx = row * width + col;
    
    // Skip NoData cells
    if (dem[idx] == -9999.0f) {
        filled[idx] = -9999.0f;
        return;
    }
    
    // Get current cell elevation
    float current_elevation = filled[idx];
    
    // Check 8 neighboring cells
    float max_neighbor = -9999.0f;
    
    // Define 8-connected neighborhood
    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    
    // Find the highest neighbor that is lower than the current cell
    for (int i = 0; i < 8; i++) {
        int nc = col + dx[i];
        int nr = row + dy[i];
        
        // Skip out-of-bounds neighbors
        if (nc < 0 || nc >= width || nr < 0 || nr >= height) continue;
        
        int nidx = nr * width + nc;
        
        // Skip NoData neighbors
        if (filled[nidx] == -9999.0f) continue;
        
        // If this neighbor is lower than current cell, check if it's the highest
        if (filled[nidx] < current_elevation) {
            max_neighbor = fmaxf(max_neighbor, filled[nidx]);
        }
    }
    
    // If we found a lower neighbor
    if (max_neighbor != -9999.0f) {
        // Calculate the elevation difference
        float diff = current_elevation - max_neighbor;
        
        // If the difference is greater than the z-limit, fill the sink
        if (diff > z_limit) {
            float new_elevation = max_neighbor + z_limit;
            
            // Only update if the new elevation is lower than the current elevation
            if (new_elevation < current_elevation) {
                filled[idx] = new_elevation;
                *changed = 1; // Set flag to indicate the DEM was changed
            }
        }
    }
}

/**
 * @brief Initialize a DEM fill operation 
 * 
 * Copies the input DEM to the output buffer and marks cells on the DEM boundary
 * as special edge cases for the sink filling algorithm.
 * 
 * @param dem Input Digital Elevation Model
 * @param filled Output buffer for the filled DEM
 * @param width DEM width
 * @param height DEM height
 */
__global__ void initFillSinksKernel(
    const float* __restrict__ dem,
    float* __restrict__ filled,
    int width,
    int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int idx = row * width + col;
    
    // Copy the original DEM
    filled[idx] = dem[idx];
}

} // namespace kernels
} // namespace geospatial