// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <array>
#include <unordered_map>

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace molecular_dynamics {

// Error checking macro for CUDA calls
#define CUDA_CHECK(call)                                                         \
    do {                                                                          \
        cudaError_t error = call;                                                 \
        if (error != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(error) << std::endl;                  \
            throw std::runtime_error("CUDA error");                               \
        }                                                                         \
    } while (0)

// Type definitions
using scalar_t = double;      // Use double for higher precision
using index_t = int32_t;      // Index type
using atom_id_t = uint32_t;   // Atom identifier type
using res_id_t = uint32_t;    // Residue identifier type

// Constants
constexpr scalar_t DEFAULT_TIMESTEP = 0.001;     // Default timestep in picoseconds (1 femtosecond)
constexpr scalar_t DEFAULT_TEMPERATURE = 300.0;  // Default temperature in Kelvin
constexpr scalar_t DEFAULT_CUTOFF = 10.0;        // Default cutoff distance in Angstroms
constexpr scalar_t BOLTZMANN = 8.31446261815324e-3; // Boltzmann constant in kJ/(mol·K)
constexpr index_t DEFAULT_BLOCK_SIZE = 256;      // Default CUDA block size

// 3D Vector class
class Vec3 {
public:
    scalar_t x, y, z;

    // Constructors
    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(scalar_t x_, scalar_t y_, scalar_t z_) : x(x_), y(y_), z(z_) {}

    // Vector operations
    __host__ __device__ Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vec3 operator*(scalar_t s) const {
        return Vec3(x * s, y * s, z * s);
    }

    __host__ __device__ Vec3 operator/(scalar_t s) const {
        scalar_t inv_s = 1.0 / s;
        return Vec3(x * inv_s, y * inv_s, z * inv_s);
    }

    __host__ __device__ Vec3& operator+=(const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__ Vec3& operator-=(const Vec3& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    __host__ __device__ Vec3& operator*=(scalar_t s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    __host__ __device__ Vec3& operator/=(scalar_t s) {
        scalar_t inv_s = 1.0 / s;
        x *= inv_s;
        y *= inv_s;
        z *= inv_s;
        return *this;
    }

    // Dot product
    __host__ __device__ scalar_t dot(const Vec3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    // Cross product
    __host__ __device__ Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    // Length squared
    __host__ __device__ scalar_t length_squared() const {
        return x * x + y * y + z * z;
    }

    // Length
    __host__ __device__ scalar_t length() const {
        return sqrt(length_squared());
    }

    // Normalize
    __host__ __device__ Vec3 normalized() const {
        scalar_t len = length();
        if (len > 0) {
            return *this / len;
        }
        return *this;
    }

    // Normalize in place
    __host__ __device__ void normalize() {
        scalar_t len = length();
        if (len > 0) {
            *this /= len;
        }
    }

    // Distance to another point
    __host__ __device__ scalar_t distance(const Vec3& v) const {
        return (*this - v).length();
    }

    // Distance squared to another point
    __host__ __device__ scalar_t distance_squared(const Vec3& v) const {
        return (*this - v).length_squared();
    }
};

// Define operator* for scalar * Vec3
__host__ __device__ inline Vec3 operator*(scalar_t s, const Vec3& v) {
    return v * s;
}

// Enumeration for atom types
enum class AtomType {
    Hydrogen,
    Carbon,
    Nitrogen,
    Oxygen,
    Sulfur,
    Phosphorus,
    Other
};

// Enumeration for force field types
enum class ForceFieldType {
    AMBER,
    CHARMM,
    OPLS,
    Custom
};

// Enumeration for integration methods
enum class IntegrationType {
    VelocityVerlet,
    Leapfrog,
    Beeman
};

// Enumeration for thermostat types
enum class ThermostatType {
    None,
    Berendsen,
    Andersen,
    NoseHoover
};

// Enumeration for GPU device types for scaling
enum class GPUDeviceType {
    Unknown,
    CPU,            // Fallback CPU implementation
    JetsonOrin,     // Nvidia Jetson Orin (SM 8.7)
    T4,             // Nvidia T4 (SM 7.5)
    HighEnd         // Other high-end GPUs (SM >= 8.0)
};

// Structure to hold device capabilities and scaling information
struct DeviceCapabilities {
    GPUDeviceType device_type;
    int compute_capability_major;
    int compute_capability_minor;
    size_t global_memory_bytes;
    int multiprocessor_count;
    int max_threads_per_multiprocessor;
    int max_threads_per_block;
    int max_shared_memory_per_block;
    
    // Optimal kernel launch parameters based on device capabilities
    int get_optimal_block_size() const {
        switch (device_type) {
            case GPUDeviceType::JetsonOrin:
                return 256;  // Optimal for SM 8.7
            case GPUDeviceType::T4:
                return 128;  // Optimal for SM 7.5
            case GPUDeviceType::HighEnd:
                return 256;  // General high-end GPU
            default:
                return DEFAULT_BLOCK_SIZE;
        }
    }
    
    // Get optimal tile size for tiled algorithms
    int get_optimal_tile_size() const {
        switch (device_type) {
            case GPUDeviceType::JetsonOrin:
                return 16;  // 16x16 tiles
            case GPUDeviceType::T4:
                return 8;   // 8x8 tiles
            case GPUDeviceType::HighEnd:
                return 32;  // 32x32 tiles
            default:
                return 8;
        }
    }
    
    // Get optimal shared memory usage based on device
    size_t get_optimal_shared_memory() const {
        switch (device_type) {
            case GPUDeviceType::JetsonOrin:
                return 48 * 1024;  // 48 KB
            case GPUDeviceType::T4:
                return 32 * 1024;  // 32 KB
            case GPUDeviceType::HighEnd:
                return 96 * 1024;  // 96 KB
            default:
                return 16 * 1024;  // 16 KB
        }
    }
    
    // Get optimal number of grid dimensions
    int get_optimal_grid_size(int total_elements) const {
        int block_size = get_optimal_block_size();
        return (total_elements + block_size - 1) / block_size;
    }
};

// Function to detect and initialize device capabilities
DeviceCapabilities detect_device_capabilities();

} // namespace molecular_dynamics