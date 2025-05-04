/**
 * @file point_cloud_kernels.cu
 * @brief CUDA kernels for point cloud processing
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <limits>
#include <helper_math.h>

namespace geospatial {
namespace kernels {

// Point cloud structure for GPU processing
struct PointData {
    float x, y, z;        // Position
    uint8_t intensity;    // Intensity value
    uint8_t return_num;   // Return number
    uint8_t classification; // Point classification
};

/**
 * @brief Point cloud classification kernel
 * 
 * Classifies points based on geometric features:
 * - Ground: Lowest points in the neighborhood
 * - Vegetation: Points with multiple returns
 * - Buildings: Points above ground with planar characteristics
 * 
 * @param points Input point cloud
 * @param classified_points Output classified points (same as input with updated classification)
 * @param num_points Number of points in the cloud
 * @param min_x Minimum X coordinate
 * @param min_y Minimum Y coordinate
 * @param min_z Minimum Z coordinate
 * @param max_x Maximum X coordinate
 * @param max_y Maximum Y coordinate
 * @param max_z Maximum Z coordinate
 * @param grid_size Grid size for spatial partitioning
 */
__global__ void classifyPointsKernel(
    const PointData* __restrict__ points,
    PointData* __restrict__ classified_points,
    int num_points,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z,
    float grid_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // Copy point to output
    classified_points[idx] = points[idx];
    
    // Get point coordinates
    float x = points[idx].x;
    float y = points[idx].y;
    float z = points[idx].z;
    
    // Calculate grid cell indices
    int grid_width = __float2int_ru((max_x - min_x) / grid_size);
    int grid_height = __float2int_ru((max_y - min_y) / grid_size);
    
    int cell_x = __float2int_rd((x - min_x) / grid_size);
    int cell_y = __float2int_rd((y - min_y) / grid_size);
    
    // Initialize minimum heights for ground point detection
    float min_height = z;
    float max_height = z;
    float mean_height = z;
    int count = 1;
    
    // Check for multiple returns (vegetation indicator)
    bool is_multiple_return = points[idx].return_num > 1;
    
    // Scan neighborhood for min/max heights
    const int neighborhood_size = 1;
    for (int nx = max(0, cell_x - neighborhood_size); nx <= min(grid_width - 1, cell_x + neighborhood_size); nx++) {
        for (int ny = max(0, cell_y - neighborhood_size); ny <= min(grid_height - 1, cell_y + neighborhood_size); ny++) {
            
            // For each point, check if it's in this grid cell
            for (int i = 0; i < num_points; i++) {
                if (i == idx) continue; // Skip the current point
                
                int point_cell_x = __float2int_rd((points[i].x - min_x) / grid_size);
                int point_cell_y = __float2int_rd((points[i].y - min_y) / grid_size);
                
                if (point_cell_x == nx && point_cell_y == ny) {
                    min_height = min(min_height, points[i].z);
                    max_height = max(max_height, points[i].z);
                    mean_height += points[i].z;
                    count++;
                }
            }
        }
    }
    
    if (count > 1) {
        mean_height /= count;
    }
    
    // Height range in the neighborhood
    float height_range = max_height - min_height;
    
    // Distance from point to minimum height
    float height_from_ground = z - min_height;
    
    // Classify points
    uint8_t classification;
    
    // Ground points (lowest points)
    if (height_from_ground < 0.2f) {
        classification = 2; // Ground
    }
    // Vegetation (multiple returns or non-planar neighborhood)
    else if (is_multiple_return || height_range > 3.0f) {
        // Low, medium, or high vegetation based on height
        if (height_from_ground < 0.5f) {
            classification = 3; // Low vegetation
        } else if (height_from_ground < 2.0f) {
            classification = 4; // Medium vegetation
        } else {
            classification = 5; // High vegetation
        }
    }
    // Buildings (planar and above ground)
    else if (height_from_ground > 1.0f && height_range < 1.0f) {
        classification = 6; // Building
    }
    // Water (low intensity, near ground)
    else if (points[idx].intensity < 30 && height_from_ground < 0.3f) {
        classification = 9; // Water
    }
    // Default: unclassified
    else {
        classification = 1; // Unclassified
    }
    
    // Update classification
    classified_points[idx].classification = classification;
}

/**
 * @brief Create DEM from point cloud kernel
 * 
 * Creates a Digital Elevation Model from a point cloud using
 * the highest resolution (interpolation method can be specified)
 * 
 * @param points Input point cloud
 * @param num_points Number of points in the cloud
 * @param dem Output DEM
 * @param width DEM width
 * @param height DEM height
 * @param min_x Minimum X coordinate
 * @param min_y Minimum Y coordinate
 * @param cell_size Cell size in CRS units
 * @param algorithm Algorithm for DEM creation (0=TIN, 1=IDW, 2=natural neighbor)
 */
__global__ void createDEMKernel(
    const PointData* __restrict__ points,
    int num_points,
    float* __restrict__ dem,
    int width,
    int height,
    float min_x,
    float min_y,
    float cell_size,
    int algorithm)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= width || row >= height) return;
    
    int idx = row * width + col;
    
    // Calculate the geographic coordinates of this cell
    float geo_x = min_x + col * cell_size;
    float geo_y = min_y + row * cell_size;
    
    // Initialize elevation
    float elevation = -9999.0f; // NoData value
    
    // Interpolation parameters
    const float search_radius = cell_size * 2.0f;
    const float search_radius_sq = search_radius * search_radius;
    
    switch (algorithm) {
        case 0: // TIN-based interpolation (simplified)
        {
            // Find closest 3 points for triangulation
            float min_dist[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
            int closest_idx[3] = {-1, -1, -1};
            
            for (int i = 0; i < num_points; i++) {
                // Skip non-ground points
                if (points[i].classification != 2) continue;
                
                float dx = points[i].x - geo_x;
                float dy = points[i].y - geo_y;
                float dist_sq = dx * dx + dy * dy;
                
                // If this point is closer than any of the current closest points, update
                for (int j = 0; j < 3; j++) {
                    if (dist_sq < min_dist[j]) {
                        // Shift the other points
                        for (int k = 2; k > j; k--) {
                            min_dist[k] = min_dist[k-1];
                            closest_idx[k] = closest_idx[k-1];
                        }
                        
                        min_dist[j] = dist_sq;
                        closest_idx[j] = i;
                        break;
                    }
                }
            }
            
            // If we found at least 3 points, perform triangulation
            if (closest_idx[0] >= 0 && closest_idx[1] >= 0 && closest_idx[2] >= 0) {
                // Simplified barycentric interpolation
                float weight_sum = 0.0f;
                float weighted_elev = 0.0f;
                
                for (int j = 0; j < 3; j++) {
                    if (closest_idx[j] >= 0) {
                        float weight = 1.0f / sqrtf(min_dist[j] + 1e-6f);
                        weight_sum += weight;
                        weighted_elev += weight * points[closest_idx[j]].z;
                    }
                }
                
                if (weight_sum > 0.0f) {
                    elevation = weighted_elev / weight_sum;
                }
            }
            break;
        }
        
        case 1: // IDW interpolation
        {
            float weight_sum = 0.0f;
            float weighted_elev = 0.0f;
            
            for (int i = 0; i < num_points; i++) {
                // Skip non-ground points
                if (points[i].classification != 2) continue;
                
                float dx = points[i].x - geo_x;
                float dy = points[i].y - geo_y;
                float dist_sq = dx * dx + dy * dy;
                
                if (dist_sq < search_radius_sq) {
                    // Inverse distance weighting
                    float weight = 1.0f / (sqrtf(dist_sq) + 1e-6f);
                    weight_sum += weight;
                    weighted_elev += weight * points[i].z;
                }
            }
            
            if (weight_sum > 0.0f) {
                elevation = weighted_elev / weight_sum;
            }
            break;
        }
        
        case 2: // Natural neighbor (simplified)
        {
            // This is a simplified version - actual natural neighbor would use Voronoi diagrams
            // Here we implement a modified IDW that adjusts weights based on point density
            
            float weight_sum = 0.0f;
            float weighted_elev = 0.0f;
            int point_count = 0;
            
            for (int i = 0; i < num_points; i++) {
                // Skip non-ground points
                if (points[i].classification != 2) continue;
                
                float dx = points[i].x - geo_x;
                float dy = points[i].y - geo_y;
                float dist_sq = dx * dx + dy * dy;
                
                if (dist_sq < search_radius_sq) {
                    point_count++;
                }
            }
            
            // Adjust search radius based on point density
            float adjusted_radius_sq = search_radius_sq;
            if (point_count > 10) {
                adjusted_radius_sq *= 0.7f;
            } else if (point_count < 3) {
                adjusted_radius_sq *= 1.5f;
            }
            
            for (int i = 0; i < num_points; i++) {
                // Skip non-ground points
                if (points[i].classification != 2) continue;
                
                float dx = points[i].x - geo_x;
                float dy = points[i].y - geo_y;
                float dist_sq = dx * dx + dy * dy;
                
                if (dist_sq < adjusted_radius_sq) {
                    // Modified inverse distance weighting
                    float weight = 1.0f / powf(sqrtf(dist_sq) + 1e-6f, 2.0f);
                    weight_sum += weight;
                    weighted_elev += weight * points[i].z;
                }
            }
            
            if (weight_sum > 0.0f) {
                elevation = weighted_elev / weight_sum;
            }
            break;
        }
        
        default:
            elevation = -9999.0f; // NoData value
            break;
    }
    
    // Write result to DEM
    dem[idx] = elevation;
}

/**
 * @brief Kernel for computing the normal vectors for a point cloud
 * 
 * Computes normal vectors for each point in the cloud using
 * principal component analysis (PCA) on neighboring points.
 * 
 * @param points Input point cloud
 * @param normals Output normal vectors (x, y, z)
 * @param num_points Number of points in the cloud
 * @param radius Radius for normal estimation
 * @param min_x Minimum X coordinate
 * @param min_y Minimum Y coordinate
 * @param min_z Minimum Z coordinate
 * @param max_x Maximum X coordinate
 * @param max_y Maximum Y coordinate
 * @param max_z Maximum Z coordinate
 * @param grid_size Grid size for spatial partitioning
 */
__global__ void computeNormalsKernel(
    const PointData* __restrict__ points,
    float3* __restrict__ normals,
    int num_points,
    float radius,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z,
    float grid_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // Get point coordinates
    float x = points[idx].x;
    float y = points[idx].y;
    float z = points[idx].z;
    
    // Calculate grid cell indices
    int grid_width = __float2int_ru((max_x - min_x) / grid_size);
    int grid_height = __float2int_ru((max_y - min_y) / grid_size);
    
    int cell_x = __float2int_rd((x - min_x) / grid_size);
    int cell_y = __float2int_rd((y - min_y) / grid_size);
    
    // Collect neighboring points
    const int MAX_NEIGHBORS = 30;
    float3 neighbors[MAX_NEIGHBORS];
    int neighbor_count = 0;
    
    // Center point at origin for numerical stability
    float3 center = make_float3(x, y, z);
    
    // Add the current point as first neighbor
    neighbors[neighbor_count++] = make_float3(0, 0, 0);
    
    // Scan neighborhood for points within radius
    const float radius_sq = radius * radius;
    const int neighborhood_size = __float2int_ru(radius / grid_size);
    
    for (int nx = max(0, cell_x - neighborhood_size); nx <= min(grid_width - 1, cell_x + neighborhood_size); nx++) {
        for (int ny = max(0, cell_y - neighborhood_size); ny <= min(grid_height - 1, cell_y + neighborhood_size); ny++) {
            
            // For each point, check if it's in this grid cell and within radius
            for (int i = 0; i < num_points; i++) {
                if (i == idx) continue; // Skip the current point
                
                int point_cell_x = __float2int_rd((points[i].x - min_x) / grid_size);
                int point_cell_y = __float2int_rd((points[i].y - min_y) / grid_size);
                
                if (point_cell_x == nx && point_cell_y == ny) {
                    float dx = points[i].x - x;
                    float dy = points[i].y - y;
                    float dz = points[i].z - z;
                    float dist_sq = dx * dx + dy * dy + dz * dz;
                    
                    if (dist_sq < radius_sq && neighbor_count < MAX_NEIGHBORS) {
                        neighbors[neighbor_count++] = make_float3(dx, dy, dz);
                    }
                }
            }
        }
    }
    
    // Compute covariance matrix
    float cov[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // xx, xy, xz, yy, yz, zz
    
    for (int i = 0; i < neighbor_count; i++) {
        float3 p = neighbors[i];
        cov[0] += p.x * p.x;
        cov[1] += p.x * p.y;
        cov[2] += p.x * p.z;
        cov[3] += p.y * p.y;
        cov[4] += p.y * p.z;
        cov[5] += p.z * p.z;
    }
    
    for (int i = 0; i < 6; i++) {
        cov[i] /= neighbor_count;
    }
    
    // Find normal vector through simplified eigenvalue analysis
    // (This is a simplified approach - a full PCA would compute all eigenvectors)
    
    // Use power iteration to find the smallest eigenvector
    float3 normal = make_float3(1.0f, 0.0f, 0.0f); // Initial guess
    
    // Iteratively apply covariance matrix (simplified power iteration)
    for (int iter = 0; iter < 10; iter++) {
        float3 new_normal;
        new_normal.x = cov[0] * normal.x + cov[1] * normal.y + cov[2] * normal.z;
        new_normal.y = cov[1] * normal.x + cov[3] * normal.y + cov[4] * normal.z;
        new_normal.z = cov[2] * normal.x + cov[4] * normal.y + cov[5] * normal.z;
        
        // Normalize
        float len = sqrtf(new_normal.x * new_normal.x + 
                          new_normal.y * new_normal.y + 
                          new_normal.z * new_normal.z);
        
        if (len > 1e-6f) {
            normal.x = new_normal.x / len;
            normal.y = new_normal.y / len;
            normal.z = new_normal.z / len;
        }
    }
    
    // Ensure normal points up (assuming Z is up)
    if (normal.z < 0) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }
    
    // Write result
    normals[idx] = normal;
}

/**
 * @brief Extract buildings from point cloud
 * 
 * Detects building footprints by clustering building points
 * and extracting the boundary of each cluster.
 * 
 * @param points Input point cloud
 * @param num_points Number of points in the cloud
 * @param labels Output point labels (cluster IDs)
 * @param min_x Minimum X coordinate
 * @param min_y Minimum Y coordinate
 * @param min_z Minimum Z coordinate
 * @param max_x Maximum X coordinate
 * @param max_y Maximum Y coordinate
 * @param max_z Maximum Z coordinate
 * @param grid_size Grid size for spatial partitioning
 * @param min_height Minimum height difference for building detection
 */
__global__ void extractBuildingsKernel(
    const PointData* __restrict__ points,
    int num_points,
    int* __restrict__ labels,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z,
    float grid_size,
    float min_height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // Initialize with no cluster
    labels[idx] = -1;
    
    // Only process building points
    if (points[idx].classification != 6) {
        return;
    }
    
    // Get point coordinates
    float x = points[idx].x;
    float y = points[idx].y;
    float z = points[idx].z;
    
    // Calculate grid cell indices
    int grid_width = __float2int_ru((max_x - min_x) / grid_size);
    int grid_height = __float2int_ru((max_y - min_y) / grid_size);
    
    int cell_x = __float2int_rd((x - min_x) / grid_size);
    int cell_y = __float2int_rd((y - min_y) / grid_size);
    
    // Find ground height at this location
    float ground_height = min_z;
    
    for (int i = 0; i < num_points; i++) {
        if (points[i].classification != 2) continue; // Only ground points
        
        int point_cell_x = __float2int_rd((points[i].x - min_x) / grid_size);
        int point_cell_y = __float2int_rd((points[i].y - min_y) / grid_size);
        
        // If ground point is in same grid cell
        if (point_cell_x == cell_x && point_cell_y == cell_y) {
            ground_height = max(ground_height, points[i].z);
        }
    }
    
    // Check if point is high enough above ground
    if (z - ground_height < min_height) {
        return;
    }
    
    // Simplistic clustering by density-based proximity
    // This is a simplified version of DBSCAN
    int cluster_id = idx + 1; // Use point index + 1 as initial cluster ID
    
    // Look for nearby building points
    const float search_radius = grid_size * 2.0f;
    const float search_radius_sq = search_radius * search_radius;
    
    for (int i = 0; i < num_points; i++) {
        if (i == idx) continue;
        if (points[i].classification != 6) continue; // Only building points
        
        float dx = points[i].x - x;
        float dy = points[i].y - y;
        float dist_sq = dx*dx + dy*dy;
        
        if (dist_sq < search_radius_sq) {
            // If this is the first point we're processing, use its cluster ID
            // Otherwise, use our own ID
            atomicMin(&labels[i], cluster_id);
            cluster_id = min(cluster_id, labels[i]);
        }
    }
    
    // Update our own label
    labels[idx] = cluster_id;
}

// Kernel launcher functions for each CUDA kernel

extern "C" {

void launchClassifyPointsKernel(
    const PointData* d_points,
    PointData* d_classified_points,
    int num_points,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z,
    float grid_size,
    cudaStream_t stream = 0)
{
    // Calculate grid dimensions
    int blockSize = 256;
    int gridSize = (num_points + blockSize - 1) / blockSize;
    
    // Launch kernel
    classifyPointsKernel<<<gridSize, blockSize, 0, stream>>>(
        d_points,
        d_classified_points,
        num_points,
        min_x, min_y, min_z,
        max_x, max_y, max_z,
        grid_size
    );
    
    // Wait for kernel to finish
    cudaStreamSynchronize(stream);
}

void launchCreateDEMKernel(
    const PointData* d_points,
    int num_points,
    float* d_dem,
    int width,
    int height,
    float min_x,
    float min_y,
    float cell_size,
    int algorithm,
    cudaStream_t stream = 0)
{
    // Calculate grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                 (height + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    createDEMKernel<<<gridSize, blockSize, 0, stream>>>(
        d_points,
        num_points,
        d_dem,
        width,
        height,
        min_x,
        min_y,
        cell_size,
        algorithm
    );
    
    // Wait for kernel to finish
    cudaStreamSynchronize(stream);
}

void launchComputeNormalsKernel(
    const PointData* d_points,
    float3* d_normals,
    int num_points,
    float radius,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z,
    float grid_size,
    cudaStream_t stream = 0)
{
    // Calculate grid dimensions
    int blockSize = 256;
    int gridSize = (num_points + blockSize - 1) / blockSize;
    
    // Launch kernel
    computeNormalsKernel<<<gridSize, blockSize, 0, stream>>>(
        d_points,
        d_normals,
        num_points,
        radius,
        min_x, min_y, min_z,
        max_x, max_y, max_z,
        grid_size
    );
    
    // Wait for kernel to finish
    cudaStreamSynchronize(stream);
}

void launchExtractBuildingsKernel(
    const PointData* d_points,
    int num_points,
    int* d_labels,
    float min_x, float min_y, float min_z,
    float max_x, float max_y, float max_z,
    float grid_size,
    float min_height,
    cudaStream_t stream = 0)
{
    // Calculate grid dimensions
    int blockSize = 256;
    int gridSize = (num_points + blockSize - 1) / blockSize;
    
    // Launch kernel
    extractBuildingsKernel<<<gridSize, blockSize, 0, stream>>>(
        d_points,
        num_points,
        d_labels,
        min_x, min_y, min_z,
        max_x, max_y, max_z,
        grid_size,
        min_height
    );
    
    // Wait for kernel to finish
    cudaStreamSynchronize(stream);
}

} // extern "C"

} // namespace kernels
} // namespace geospatial