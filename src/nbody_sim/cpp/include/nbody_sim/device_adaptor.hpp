// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#pragma once

#include "nbody_sim/common.hpp"

namespace nbody_sim {

// Enumeration for GPU device types for scaling
enum class DeviceType {
    Unknown,
    CPU,            // Fallback CPU implementation
    JetsonOrin,     // Nvidia Jetson Orin (SM 8.7)
    T4,             // Nvidia T4 (SM 7.5)
    HighEnd,        // Other high-end GPUs (SM >= 8.0)
    OtherGPU        // Other GPUs
};

// Structure to hold device capabilities and scaling information
struct DeviceCapabilities {
    DeviceType device_type;
    int compute_capability_major;
    int compute_capability_minor;
    size_t global_memory_bytes;
    int multiprocessor_count;
    int max_threads_per_multiprocessor;
    int max_threads_per_block;
    int max_shared_memory_per_block;
};

// Function to detect and initialize device capabilities
DeviceCapabilities detect_device_capabilities();

// Helper functions to get optimal parameters for different devices
int get_optimal_block_size(const DeviceCapabilities& caps);
int get_optimal_tile_size(const DeviceCapabilities& caps);
size_t get_optimal_shared_memory(const DeviceCapabilities& caps);
int get_optimal_grid_size(const DeviceCapabilities& caps, int total_elements);

} // namespace nbody_sim