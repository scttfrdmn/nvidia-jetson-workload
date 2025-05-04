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

// CUDA includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace nbody_sim {

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
using scalar_t = double;  // Use double for higher precision
using index_t = int32_t;  // Index type

// Constants
constexpr scalar_t DEFAULT_G = 1.0;         // Default gravitational constant
constexpr scalar_t DEFAULT_SOFTENING = 1e-6; // Softening parameter to prevent singularities
constexpr scalar_t DEFAULT_TIMESTEP = 0.01;  // Default simulation time step
constexpr index_t DEFAULT_BLOCK_SIZE = 256;  // Default CUDA block size

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
};

// Define operator* for scalar * Vec3
__host__ __device__ inline Vec3 operator*(scalar_t s, const Vec3& v) {
    return v * s;
}

// Enumeration for integration methods
enum class IntegrationType {
    Euler,
    Leapfrog,
    Verlet,
    RungeKutta4
};

// Enumeration for system types
enum class SystemType {
    Random,
    Solar,
    Galaxy,
    Custom
};

// CPU-GPU Workload distribution strategy
enum class WorkloadStrategy {
    GPUOnly,          // Use only GPU for computation (default)
    CPUOnly,          // Use only CPU for computation
    Hybrid,           // Use both CPU and GPU with static workload division
    AdaptiveHybrid    // Use both CPU and GPU with dynamic workload balancing
};

} // namespace nbody_sim