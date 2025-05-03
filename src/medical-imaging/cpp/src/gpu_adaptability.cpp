/**
 * @file gpu_adaptability.cpp
 * @brief Implementation of GPU adaptability pattern for medical imaging.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>
#include <random>
#include <numeric>
#include <iomanip>
#include <sstream>

#include "../include/medical_imaging/gpu_adaptability.hpp"

// Function prototypes for external CUDA kernel launchers
namespace medical_imaging {

// CT reconstruction kernels
extern bool launchRampFilterKernel(
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
);

extern bool launchBackprojectionKernel(
    float* d_filtered_projections,
    float* d_output,
    int num_angles,
    int proj_width,
    int img_width,
    int img_height,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
);

// Image processing kernels
extern bool launch2DConvolutionKernel(
    float* d_input,
    float* d_output,
    int width,
    int height,
    int channels,
    float* d_kernel_ptr,
    int kernel_size,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
);

extern bool launchMedianFilterKernel(
    float* d_input,
    float* d_output,
    int width,
    int height,
    int channels,
    int radius,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
);

extern bool launchBilateralFilterKernel(
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
);

extern bool launchNLMFilterKernel(
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
);

// Segmentation kernels
extern bool launchThresholdingKernel(
    float* d_input,
    float* d_output,
    int width,
    int height,
    int channels,
    float threshold,
    float max_value,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
);

extern bool launchWatershedKernel(
    float* d_input,
    float* d_markers,
    float* d_output,
    int width,
    int height,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
);

extern bool launchLevelSetKernel(
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
);

extern bool launchGraphCutKernel(
    float* d_input,
    float* d_seeds,
    float* d_output,
    int width,
    int height,
    int max_iterations,
    float lambda,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
);

// Registration kernels
extern bool launchImageWarpingKernel(
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
);

extern bool launchMutualInformationKernel(
    float* d_image1,
    float* d_image2,
    int width,
    int height,
    int num_bins,
    float max_val,
    float& mi_value,
    const DeviceCapabilities& device_caps,
    const KernelLaunchParams& params
);

namespace {
    // Helper function to convert device type to string
    std::string deviceTypeToString(DeviceType type) {
        switch (type) {
            case DeviceType::CPU:
                return "CPU";
            case DeviceType::JetsonOrinNX:
                return "Jetson Orin NX";
            case DeviceType::T4:
                return "NVIDIA T4";
            case DeviceType::HighEndGPU:
                return "High-end NVIDIA GPU";
            case DeviceType::OtherGPU:
                return "Other NVIDIA GPU";
            default:
                return "Unknown";
        }
    }
}

/**
 * @brief Get a human-readable summary of device capabilities.
 * @return Summary string
 */
std::string DeviceCapabilities::getSummary() const {
    std::ostringstream oss;
    oss << "Device: " << device_name << " (" << deviceTypeToString(device_type) << ")" << std::endl;
    oss << "  Compute capability: " << compute_capability_major << "." << compute_capability_minor << std::endl;
    oss << "  CUDA cores: " << cuda_cores << " (" << multiprocessors << " SMs)" << std::endl;
    oss << "  Memory: " << (global_memory / (1024 * 1024)) << " MB" << std::endl;
    oss << "  Shared memory per block: " << (shared_memory_per_block / 1024) << " KB" << std::endl;
    oss << "  Max threads per block: " << max_threads_per_block << std::endl;
    oss << "  Max threads per SM: " << max_threads_per_multiprocessor << std::endl;
    oss << "  Clock rate: " << (clock_rate_khz / 1000.0f) << " MHz" << std::endl;
    oss << "  Memory clock rate: " << (memory_clock_rate_khz / 1000.0f) << " MHz" << std::endl;
    oss << "  Memory bus width: " << memory_bus_width << " bits" << std::endl;
    oss << "  Compute power ratio: " << std::fixed << std::setprecision(2) << compute_power_ratio << "x" << std::endl;
    return oss.str();
}

/**
 * @brief Compute grid dimensions based on problem size.
 * @param width Problem width
 * @param height Problem height
 * @param depth Problem depth
 */
void KernelLaunchParams::computeGridDimensions(int width, int height, int depth) {
    grid_size_x = (width + block_size_x - 1) / block_size_x;
    grid_size_y = (height + block_size_y - 1) / block_size_y;
    grid_size_z = (depth + block_size_z - 1) / block_size_z;
}

//--------------------------------------
// AdaptiveKernelManager implementation
//--------------------------------------

// Singleton instance
AdaptiveKernelManager& AdaptiveKernelManager::getInstance() {
    static AdaptiveKernelManager instance;
    return instance;
}

AdaptiveKernelManager::AdaptiveKernelManager() {
    // Initialize with default parameters
    cuda_available_ = false;
    
    // Set default CPU as fallback
    device_caps_ = DeviceCapabilities(
        DeviceType::CPU, 
        "CPU Fallback", 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f
    );
    
    // Initialize default workload distribution
    gpu_workload_ratios_["CT_Reconstruction"] = 0.9f;
    gpu_workload_ratios_["MRI_Reconstruction"] = 0.9f;
    gpu_workload_ratios_["Image_Filtering"] = 0.85f;
    gpu_workload_ratios_["Segmentation"] = 0.9f;
    gpu_workload_ratios_["Registration"] = 0.8f;
}

bool AdaptiveKernelManager::initialize(int device_id) {
    // Check if CUDA is available
    cuda_available_ = false;
    int device_count = 0;
    
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA not available or no CUDA devices found. Using CPU fallback." << std::endl;
        return false;
    }
    
    // Check if the requested device is valid
    if (device_id < 0 || device_id >= device_count) {
        std::cerr << "Invalid device ID: " << device_id << ". Using device 0." << std::endl;
        device_id = 0;
    }
    
    // Set the device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    cuda_available_ = true;
    
    // Detect device capabilities
    detectDeviceCapabilities(device_id);
    
    // Initialize default parameters
    initializeDefaultParams();
    
    // Tune parameters based on device type
    tuneParameters();
    
    return true;
}

bool AdaptiveKernelManager::isCudaAvailable() const {
    return cuda_available_;
}

const DeviceCapabilities& AdaptiveKernelManager::getDeviceCapabilities() const {
    return device_caps_;
}

void AdaptiveKernelManager::detectDeviceCapabilities(int device_id) {
    if (!cuda_available_) {
        return;
    }
    
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Determine device type based on compute capability and name
    DeviceType device_type = DeviceType::OtherGPU;
    
    if (props.major == 8 && props.minor >= 7) {
        // Jetson Orin NX has SM 8.7
        if (std::string(props.name).find("Orin") != std::string::npos) {
            device_type = DeviceType::JetsonOrinNX;
        } else {
            device_type = DeviceType::HighEndGPU;
        }
    } else if (props.major == 7 && props.minor >= 5) {
        // T4 has SM 7.5
        if (std::string(props.name).find("T4") != std::string::npos) {
            device_type = DeviceType::T4;
        } else {
            device_type = DeviceType::OtherGPU;
        }
    } else if (props.major >= 8) {
        device_type = DeviceType::HighEndGPU;
    }
    
    // Calculate CUDA cores per SM based on architecture
    int cuda_cores_per_sm = 0;
    
    if (props.major == 8) {
        // Ampere: SM 8.0, 8.6, 8.7
        cuda_cores_per_sm = 64;
    } else if (props.major == 7) {
        // Volta (SM 7.0) or Turing (SM 7.5)
        cuda_cores_per_sm = props.minor == 0 ? 64 : 64;
    } else if (props.major == 6) {
        // Pascal
        cuda_cores_per_sm = 64;
    } else if (props.major == 5) {
        // Maxwell
        cuda_cores_per_sm = 128;
    } else {
        // Default for unknown architecture
        cuda_cores_per_sm = 32;
    }
    
    int cuda_cores = props.multiProcessorCount * cuda_cores_per_sm;
    
    // Estimate compute power ratio relative to CPU
    // This is a rough estimate based on architectural differences
    float compute_power_ratio = 1.0f;
    
    switch (device_type) {
        case DeviceType::JetsonOrinNX:
            compute_power_ratio = 15.0f;  // Jetson Orin NX has decent compute power
            break;
        case DeviceType::T4:
            compute_power_ratio = 20.0f;  // T4 is optimized for inference
            break;
        case DeviceType::HighEndGPU:
            compute_power_ratio = 30.0f;  // High-end GPUs have substantial compute power
            break;
        case DeviceType::OtherGPU:
            compute_power_ratio = 10.0f;  // Conservative estimate for unknown GPUs
            break;
        default:
            compute_power_ratio = 1.0f;   // CPU baseline
            break;
    }
    
    // Store the capabilities
    device_caps_ = DeviceCapabilities(
        device_type,
        props.name,
        props.major,
        props.minor,
        cuda_cores,
        props.multiProcessorCount,
        props.totalGlobalMem,
        props.sharedMemPerBlock,
        props.maxThreadsPerBlock,
        props.maxThreadsPerMultiProcessor,
        props.clockRate,
        props.memoryClockRate,
        props.memoryBusWidth,
        compute_power_ratio
    );
    
    // Print device details
    std::cout << "CUDA Device: " << props.name << std::endl;
    std::cout << "  Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "  Type: " << deviceTypeToString(device_type) << std::endl;
    std::cout << "  CUDA Cores: " << cuda_cores << " (" << props.multiProcessorCount << " SMs)" << std::endl;
}

void AdaptiveKernelManager::initializeDefaultParams() {
    // Default parameters for different operations and problem sizes
    
    // Example key: "Convolution_512x512"
    // Default values will be overridden by device-specific tuning
    
    // Default block sizes for 2D operations
    KernelLaunchParams params_2d;
    params_2d.block_size_x = 16;
    params_2d.block_size_y = 16;
    params_2d.block_size_z = 1;
    
    // Default block sizes for 1D operations
    KernelLaunchParams params_1d;
    params_1d.block_size_x = 256;
    params_1d.block_size_y = 1;
    params_1d.block_size_z = 1;
    
    // Default block sizes for 3D operations
    KernelLaunchParams params_3d;
    params_3d.block_size_x = 8;
    params_3d.block_size_y = 8;
    params_3d.block_size_z = 4;
    
    // Set defaults for common problem sizes
    
    // 2D operations (various image sizes)
    for (const auto& op : {"Convolution", "MedianFilter", "BilateralFilter", "NLMFilter",
                          "Thresholding", "Watershed", "LevelSet", "GraphCut",
                          "ImageWarping", "MutualInformation"}) {
        for (const auto& size : {256, 512, 1024, 2048, 4096}) {
            ProblemSize problem_size(size, size, 1);
            optimal_params_[op][problem_size] = params_2d;
        }
    }
    
    // 1D operations (CT projection filtering)
    for (const auto& width : {512, 1024, 2048, 4096}) {
        for (const auto& height : {180, 360, 720, 1440}) {
            ProblemSize problem_size(width, height, 1);
            optimal_params_["RampFilter"][problem_size] = params_1d;
        }
    }
    
    // Specialized operations (CT backprojection)
    for (const auto& size : {256, 512, 1024, 2048}) {
        ProblemSize problem_size(size, size, 1);
        optimal_params_["Backprojection"][problem_size] = params_2d;
    }
    
    // 3D operations (volume processing)
    for (const auto& size : {64, 128, 256, 512}) {
        ProblemSize problem_size(size, size, size);
        for (const auto& op : {"Convolution3D", "VolumeFiltering", "VolumeSegmentation"}) {
            optimal_params_[op][problem_size] = params_3d;
        }
    }
}

void AdaptiveKernelManager::tuneParameters() {
    // Apply device-specific optimizations
    switch (device_caps_.device_type) {
        case DeviceType::JetsonOrinNX:
            optimizeForJetsonOrin();
            break;
        case DeviceType::T4:
            optimizeForT4();
            break;
        case DeviceType::HighEndGPU:
            optimizeForHighEndGPU();
            break;
        default:
            optimizeForGenericGPU();
            break;
    }
}

void AdaptiveKernelManager::optimizeForJetsonOrin() {
    // Jetson Orin NX-specific optimizations
    
    // Adjust workload distribution to favor CPU more on Jetson
    // as the CPU in Jetson Orin is quite powerful (8-core ARM CPU)
    gpu_workload_ratios_["CT_Reconstruction"] = 0.85f;
    gpu_workload_ratios_["MRI_Reconstruction"] = 0.85f;
    gpu_workload_ratios_["Image_Filtering"] = 0.8f;
    gpu_workload_ratios_["Segmentation"] = 0.85f;
    gpu_workload_ratios_["Registration"] = 0.75f;
    
    // Optimize block sizes for Jetson Orin NX
    // Ampere architecture (SM 8.7) in Jetson Orin performs well with larger block sizes
    
    // Optimize 2D operations
    KernelLaunchParams params_2d;
    params_2d.block_size_x = 16;
    params_2d.block_size_y = 16;
    params_2d.block_size_z = 1;
    
    // For larger operations, use more threads per block to utilize Ampere's capabilities
    KernelLaunchParams params_2d_large;
    params_2d_large.block_size_x = 32;
    params_2d_large.block_size_y = 8;
    params_2d_large.block_size_z = 1;
    
    // Optimize 1D operations
    KernelLaunchParams params_1d;
    params_1d.block_size_x = 128;
    params_1d.block_size_y = 1;
    params_1d.block_size_z = 1;
    
    // Apply optimized parameters
    for (const auto& op : {"Convolution", "MedianFilter", "BilateralFilter", "NLMFilter",
                          "Thresholding", "Watershed", "LevelSet", "GraphCut",
                          "ImageWarping", "MutualInformation"}) {
        // Use different block sizes based on problem size
        for (const auto& size : {256, 512}) {
            ProblemSize problem_size(size, size, 1);
            optimal_params_[op][problem_size] = params_2d;
        }
        
        for (const auto& size : {1024, 2048, 4096}) {
            ProblemSize problem_size(size, size, 1);
            optimal_params_[op][problem_size] = params_2d_large;
        }
    }
    
    // Optimize for CT operations
    KernelLaunchParams params_ct;
    params_ct.block_size_x = 16;
    params_ct.block_size_y = 16;
    params_ct.block_size_z = 1;
    
    for (const auto& size : {256, 512, 1024, 2048}) {
        ProblemSize problem_size(size, size, 1);
        optimal_params_["Backprojection"][problem_size] = params_ct;
    }
}

void AdaptiveKernelManager::optimizeForT4() {
    // T4-specific optimizations
    
    // Adjust workload distribution for T4 (which has less CPU resources in cloud environments)
    gpu_workload_ratios_["CT_Reconstruction"] = 0.95f;
    gpu_workload_ratios_["MRI_Reconstruction"] = 0.95f;
    gpu_workload_ratios_["Image_Filtering"] = 0.9f;
    gpu_workload_ratios_["Segmentation"] = 0.95f;
    gpu_workload_ratios_["Registration"] = 0.9f;
    
    // Optimize block sizes for T4
    // Turing architecture (SM 7.5) performs well with these block sizes
    
    // Optimize 2D operations
    KernelLaunchParams params_2d;
    params_2d.block_size_x = 16;
    params_2d.block_size_y = 16;
    params_2d.block_size_z = 1;
    
    // Optimize 1D operations
    KernelLaunchParams params_1d;
    params_1d.block_size_x = 256;
    params_1d.block_size_y = 1;
    params_1d.block_size_z = 1;
    
    // Apply optimized parameters
    for (const auto& op : {"Convolution", "MedianFilter", "BilateralFilter", "NLMFilter",
                          "Thresholding", "Watershed", "LevelSet", "GraphCut",
                          "ImageWarping", "MutualInformation"}) {
        for (const auto& size : {256, 512, 1024, 2048, 4096}) {
            ProblemSize problem_size(size, size, 1);
            optimal_params_[op][problem_size] = params_2d;
        }
    }
    
    // Optimize for CT operations
    KernelLaunchParams params_ct;
    params_ct.block_size_x = 16;
    params_ct.block_size_y = 16;
    params_ct.block_size_z = 1;
    
    for (const auto& size : {256, 512, 1024, 2048}) {
        ProblemSize problem_size(size, size, 1);
        optimal_params_["Backprojection"][problem_size] = params_ct;
    }
}

void AdaptiveKernelManager::optimizeForHighEndGPU() {
    // High-end GPU-specific optimizations
    
    // Adjust workload distribution for high-end GPUs (which usually have powerful CPUs too)
    gpu_workload_ratios_["CT_Reconstruction"] = 0.9f;
    gpu_workload_ratios_["MRI_Reconstruction"] = 0.9f;
    gpu_workload_ratios_["Image_Filtering"] = 0.85f;
    gpu_workload_ratios_["Segmentation"] = 0.9f;
    gpu_workload_ratios_["Registration"] = 0.85f;
    
    // Optimize block sizes for high-end GPUs
    // Ampere architecture (SM 8.0+) performs well with these block sizes
    
    // Optimize 2D operations
    KernelLaunchParams params_2d;
    params_2d.block_size_x = 16;
    params_2d.block_size_y = 16;
    params_2d.block_size_z = 1;
    
    // For larger operations, use more threads per block
    KernelLaunchParams params_2d_large;
    params_2d_large.block_size_x = 32;
    params_2d_large.block_size_y = 16;
    params_2d_large.block_size_z = 1;
    
    // Optimize 1D operations
    KernelLaunchParams params_1d;
    params_1d.block_size_x = 512;
    params_1d.block_size_y = 1;
    params_1d.block_size_z = 1;
    
    // Apply optimized parameters
    for (const auto& op : {"Convolution", "MedianFilter", "BilateralFilter", "NLMFilter",
                          "Thresholding", "Watershed", "LevelSet", "GraphCut",
                          "ImageWarping", "MutualInformation"}) {
        // Use different block sizes based on problem size
        for (const auto& size : {256, 512}) {
            ProblemSize problem_size(size, size, 1);
            optimal_params_[op][problem_size] = params_2d;
        }
        
        for (const auto& size : {1024, 2048, 4096}) {
            ProblemSize problem_size(size, size, 1);
            optimal_params_[op][problem_size] = params_2d_large;
        }
    }
    
    // Optimize for CT operations
    KernelLaunchParams params_ct;
    params_ct.block_size_x = 32;
    params_ct.block_size_y = 16;
    params_ct.block_size_z = 1;
    
    for (const auto& size : {256, 512, 1024, 2048}) {
        ProblemSize problem_size(size, size, 1);
        optimal_params_["Backprojection"][problem_size] = params_ct;
    }
}

void AdaptiveKernelManager::optimizeForGenericGPU() {
    // Generic optimizations for unknown GPUs
    
    // Conservative workload distribution
    gpu_workload_ratios_["CT_Reconstruction"] = 0.8f;
    gpu_workload_ratios_["MRI_Reconstruction"] = 0.8f;
    gpu_workload_ratios_["Image_Filtering"] = 0.75f;
    gpu_workload_ratios_["Segmentation"] = 0.8f;
    gpu_workload_ratios_["Registration"] = 0.7f;
    
    // Conservative block sizes for unknown GPUs
    
    // Optimize 2D operations
    KernelLaunchParams params_2d;
    params_2d.block_size_x = 16;
    params_2d.block_size_y = 16;
    params_2d.block_size_z = 1;
    
    // Optimize 1D operations
    KernelLaunchParams params_1d;
    params_1d.block_size_x = 256;
    params_1d.block_size_y = 1;
    params_1d.block_size_z = 1;
    
    // Apply conservative parameters
    for (const auto& op : {"Convolution", "MedianFilter", "BilateralFilter", "NLMFilter",
                          "Thresholding", "Watershed", "LevelSet", "GraphCut",
                          "ImageWarping", "MutualInformation"}) {
        for (const auto& size : {256, 512, 1024, 2048, 4096}) {
            ProblemSize problem_size(size, size, 1);
            optimal_params_[op][problem_size] = params_2d;
        }
    }
    
    // Optimize for CT operations
    KernelLaunchParams params_ct;
    params_ct.block_size_x = 16;
    params_ct.block_size_y = 16;
    params_ct.block_size_z = 1;
    
    for (const auto& size : {256, 512, 1024, 2048}) {
        ProblemSize problem_size(size, size, 1);
        optimal_params_["Backprojection"][problem_size] = params_ct;
    }
}

KernelLaunchParams AdaptiveKernelManager::getOptimalKernelParams(
    const std::string& operation_name,
    int width, int height, int depth
) {
    // Create problem size tuple
    ProblemSize problem_size(width, height, depth);
    
    // Check if we have optimized parameters for this operation and size
    auto op_it = optimal_params_.find(operation_name);
    if (op_it != optimal_params_.end()) {
        auto size_it = op_it->second.find(problem_size);
        if (size_it != op_it->second.end()) {
            // We have optimized parameters
            return size_it->second;
        }
        
        // Try to find the closest problem size
        ProblemSize closest_size = problem_size;
        float min_distance = std::numeric_limits<float>::max();
        
        for (const auto& [size, params] : op_it->second) {
            // Calculate Euclidean distance between problem sizes
            float distance = std::sqrt(
                std::pow(static_cast<float>(std::get<0>(size) - width), 2) +
                std::pow(static_cast<float>(std::get<1>(size) - height), 2) +
                std::pow(static_cast<float>(std::get<2>(size) - depth), 2)
            );
            
            if (distance < min_distance) {
                min_distance = distance;
                closest_size = size;
            }
        }
        
        if (min_distance < std::numeric_limits<float>::max()) {
            return op_it->second[closest_size];
        }
    }
    
    // If not found, use default parameters
    KernelLaunchParams default_params;
    
    // Set default block sizes based on dimension
    if (depth > 1) {
        // 3D problem
        default_params.block_size_x = 8;
        default_params.block_size_y = 8;
        default_params.block_size_z = 4;
    } else {
        // 2D problem
        default_params.block_size_x = 16;
        default_params.block_size_y = 16;
        default_params.block_size_z = 1;
    }
    
    // Compute grid dimensions
    default_params.computeGridDimensions(width, height, depth);
    
    return default_params;
}

float AdaptiveKernelManager::getGpuWorkloadRatio(const std::string& operation_name) {
    auto it = gpu_workload_ratios_.find(operation_name);
    if (it != gpu_workload_ratios_.end()) {
        return it->second;
    }
    
    // Default ratio
    return 0.9f;
}

void AdaptiveKernelManager::updatePerformanceMetrics(
    const std::string& operation_name,
    const KernelLaunchParams& params,
    double execution_time_ms
) {
    // This method can be used to collect performance metrics for auto-tuning
    // For now, we just track the metrics but don't use them for optimization
    
    // Convert params to problem size
    ProblemSize problem_size(params.grid_size_x * params.block_size_x,
                            params.grid_size_y * params.block_size_y,
                            params.grid_size_z * params.block_size_z);
    
    // Check if we already have records for this operation and problem size
    auto& records = performance_history_[operation_name][problem_size];
    
    // Check if we already have this exact parameter set
    bool found = false;
    for (auto& record : records) {
        if (record.params.block_size_x == params.block_size_x &&
            record.params.block_size_y == params.block_size_y &&
            record.params.block_size_z == params.block_size_z) {
            // Update existing record
            record.execution_time_ms = (record.execution_time_ms * record.sample_count + execution_time_ms) /
                                       (record.sample_count + 1);
            record.sample_count++;
            found = true;
            break;
        }
    }
    
    if (!found) {
        // Add new record
        PerformanceRecord record;
        record.params = params;
        record.execution_time_ms = execution_time_ms;
        record.sample_count = 1;
        records.push_back(record);
    }
    
    // TODO: Implement auto-tuning by analyzing performance metrics
    // and updating optimal_params_ map
}

ComputeBackend AdaptiveKernelManager::determineOptimalBackend(
    int width, int height, int depth,
    const std::string& operation_name
) {
    // Determine the optimal backend based on problem size and device capabilities
    
    if (!cuda_available_) {
        return ComputeBackend::CPU;
    }
    
    // For small problems, CPU may be more efficient
    const int small_size_threshold = 256;
    if (width <= small_size_threshold && height <= small_size_threshold && depth <= small_size_threshold) {
        return ComputeBackend::CPU;
    }
    
    // For very large problems, use adaptive hybrid approach
    const int large_size_threshold = 2048;
    if (width >= large_size_threshold || height >= large_size_threshold || depth >= large_size_threshold) {
        return ComputeBackend::AdaptiveHybrid;
    }
    
    // Get GPU workload ratio for this operation
    float gpu_ratio = getGpuWorkloadRatio(operation_name);
    
    // If GPU is significantly more powerful, use GPU-only
    if (gpu_ratio > 0.9f) {
        return ComputeBackend::CUDA;
    }
    
    // If GPU can handle most of the workload but CPU can help, use hybrid
    if (gpu_ratio > 0.7f) {
        return ComputeBackend::Hybrid;
    }
    
    // For operations where CPU is competitive, use adaptive approach
    return ComputeBackend::AdaptiveHybrid;
}

//-----------------------------------
// MemoryManager implementation
//-----------------------------------

MemoryManager& MemoryManager::getInstance() {
    static MemoryManager instance;
    return instance;
}

MemoryManager::MemoryManager() : initialized_(false), device_id_(0) {}

bool MemoryManager::initialize(int device_id) {
    // Check if CUDA is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA not available or no CUDA devices found." << std::endl;
        return false;
    }
    
    // Check if the requested device is valid
    if (device_id < 0 || device_id >= device_count) {
        std::cerr << "Invalid device ID: " << device_id << ". Using device 0." << std::endl;
        device_id = 0;
    }
    
    // Set the device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    device_id_ = device_id;
    initialized_ = true;
    
    return true;
}

void* MemoryManager::allocateDevice(size_t size) {
    if (!initialized_) {
        std::cerr << "Memory manager not initialized." << std::endl;
        return nullptr;
    }
    
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    
    return ptr;
}

void MemoryManager::freeDevice(void* ptr) {
    if (!initialized_ || ptr == nullptr) {
        return;
    }
    
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free device memory: " << cudaGetErrorString(err) << std::endl;
    }
}

bool MemoryManager::copyHostToDevice(void* dst, const void* src, size_t size) {
    if (!initialized_ || dst == nullptr || src == nullptr) {
        return false;
    }
    
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy host to device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool MemoryManager::copyDeviceToHost(void* dst, const void* src, size_t size) {
    if (!initialized_ || dst == nullptr || src == nullptr) {
        return false;
    }
    
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy device to host: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool MemoryManager::copyDeviceToDevice(void* dst, const void* src, size_t size) {
    if (!initialized_ || dst == nullptr || src == nullptr) {
        return false;
    }
    
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy device to device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

int MemoryManager::createStream() {
    if (!initialized_) {
        return -1;
    }
    
    // Check if there are any free IDs
    if (!free_stream_ids_.empty()) {
        int id = free_stream_ids_.back();
        free_stream_ids_.pop_back();
        return id;
    }
    
    // Create a new stream
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Add to streams vector
    streams_.push_back(stream);
    return static_cast<int>(streams_.size() - 1);
}

void MemoryManager::destroyStream(int stream_id) {
    if (!initialized_ || stream_id < 0 || stream_id >= static_cast<int>(streams_.size())) {
        return;
    }
    
    // Check if the stream is valid
    if (streams_[stream_id] != nullptr) {
        cudaStreamDestroy(static_cast<cudaStream_t>(streams_[stream_id]));
        streams_[stream_id] = nullptr;
        free_stream_ids_.push_back(stream_id);
    }
}

void MemoryManager::synchronizeStream(int stream_id) {
    if (!initialized_ || stream_id < 0 || stream_id >= static_cast<int>(streams_.size())) {
        return;
    }
    
    // Check if the stream is valid
    if (streams_[stream_id] != nullptr) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(streams_[stream_id]));
    }
}

void MemoryManager::synchronizeDevice() {
    if (!initialized_) {
        return;
    }
    
    cudaDeviceSynchronize();
}

//-----------------------------------
// DeviceImage implementation
//-----------------------------------

template <typename T>
DeviceImage<T>::DeviceImage(int width, int height, int depth, int channels)
    : width_(width), height_(height), depth_(depth), channels_(channels), data_(nullptr) {
    if (width <= 0 || height <= 0 || depth <= 0 || channels <= 0) {
        throw std::invalid_argument("Invalid dimensions for DeviceImage");
    }
    
    // Allocate device memory
    size_t size = width * height * depth * channels * sizeof(T);
    data_ = static_cast<T*>(MemoryManager::getInstance().allocateDevice(size));
    
    if (data_ == nullptr) {
        throw std::runtime_error("Failed to allocate device memory for DeviceImage");
    }
}

template <typename T>
DeviceImage<T>::DeviceImage(const MedicalImage& host_image)
    : width_(0), height_(0), depth_(1), channels_(1), data_(nullptr) {
    // Get dimensions from host image
    const auto& size = host_image.getSize();
    
    if (size.empty()) {
        throw std::invalid_argument("Empty host image");
    }
    
    width_ = size[0];
    height_ = size.size() > 1 ? size[1] : 1;
    depth_ = size.size() > 2 ? size[2] : 1;
    channels_ = host_image.getChannels();
    
    // Allocate device memory
    size_t mem_size = width_ * height_ * depth_ * channels_ * sizeof(T);
    data_ = static_cast<T*>(MemoryManager::getInstance().allocateDevice(mem_size));
    
    if (data_ == nullptr) {
        throw std::runtime_error("Failed to allocate device memory for DeviceImage");
    }
    
    // Copy data from host to device
    copyFromHostImage(host_image);
}

template <typename T>
DeviceImage<T>::~DeviceImage() {
    freeMemory();
}

template <typename T>
void DeviceImage<T>::freeMemory() {
    if (data_ != nullptr) {
        MemoryManager::getInstance().freeDevice(data_);
        data_ = nullptr;
    }
}

template <typename T>
bool DeviceImage<T>::copyFromHost(const T* host_data) {
    if (data_ == nullptr || host_data == nullptr) {
        return false;
    }
    
    size_t size = width_ * height_ * depth_ * channels_ * sizeof(T);
    return MemoryManager::getInstance().copyHostToDevice(data_, host_data, size);
}

template <typename T>
bool DeviceImage<T>::copyFromHostImage(const MedicalImage& host_image) {
    // Verify dimensions
    const auto& size = host_image.getSize();
    
    if (size.empty() || 
        static_cast<index_t>(width_) != size[0] || 
        (size.size() > 1 && static_cast<index_t>(height_) != size[1]) ||
        (size.size() > 2 && static_cast<index_t>(depth_) != size[2]) ||
        static_cast<index_t>(channels_) != host_image.getChannels()) {
        
        std::cerr << "Dimension mismatch between host and device image" << std::endl;
        return false;
    }
    
    // Copy data (assuming that T is compatible with scalar_t in MedicalImage)
    return copyFromHost(reinterpret_cast<const T*>(host_image.getData()));
}

template <typename T>
bool DeviceImage<T>::copyToHost(T* host_data) const {
    if (data_ == nullptr || host_data == nullptr) {
        return false;
    }
    
    size_t size = width_ * height_ * depth_ * channels_ * sizeof(T);
    return MemoryManager::getInstance().copyDeviceToHost(host_data, data_, size);
}

template <typename T>
bool DeviceImage<T>::copyToHostImage(MedicalImage& host_image) const {
    // Verify dimensions
    const auto& size = host_image.getSize();
    
    if (size.empty() || 
        static_cast<index_t>(width_) != size[0] || 
        (size.size() > 1 && static_cast<index_t>(height_) != size[1]) ||
        (size.size() > 2 && static_cast<index_t>(depth_) != size[2]) ||
        static_cast<index_t>(channels_) != host_image.getChannels()) {
        
        std::cerr << "Dimension mismatch between host and device image" << std::endl;
        return false;
    }
    
    // Copy data (assuming that T is compatible with scalar_t in MedicalImage)
    return copyToHost(reinterpret_cast<T*>(host_image.getData()));
}

template <typename T>
MedicalImage DeviceImage<T>::toHostImage() const {
    // Create dimensions vector
    std::vector<index_t> dims;
    dims.push_back(width_);
    dims.push_back(height_);
    if (depth_ > 1) {
        dims.push_back(depth_);
    }
    
    // Determine dimensionality
    ImageDimension dim;
    if (depth_ > 1) {
        dim = ImageDimension::D3;
    } else {
        dim = ImageDimension::D2;
    }
    
    // Create host image
    MedicalImage host_image(dims, dim, ImageType::Grayscale, channels_);
    
    // Copy data
    copyToHostImage(host_image);
    
    return host_image;
}

// Explicit template instantiations for common types
template class DeviceImage<float>;
template class DeviceImage<complex_t>;

//-----------------------------------
// KernelAdapterFactory implementation
//-----------------------------------

KernelAdapterFactory& KernelAdapterFactory::getInstance() {
    static KernelAdapterFactory instance;
    return instance;
}

void KernelAdapterFactory::registerAdapter(std::shared_ptr<KernelAdapter> adapter) {
    adapters_.push_back(adapter);
}

std::shared_ptr<KernelAdapter> KernelAdapterFactory::getBestAdapter(int device_id) {
    // Initialize adapters
    for (auto& adapter : adapters_) {
        adapter->initialize(device_id);
    }
    
    // Find the highest priority compatible adapter
    std::shared_ptr<KernelAdapter> best_adapter = nullptr;
    int highest_priority = -1;
    
    for (auto& adapter : adapters_) {
        if (adapter->isCompatible() && adapter->getPriority() > highest_priority) {
            best_adapter = adapter;
            highest_priority = adapter->getPriority();
        }
    }
    
    return best_adapter;
}

std::shared_ptr<KernelAdapter> KernelAdapterFactory::getAdapter(const std::string& name, int device_id) {
    // Find adapter by name
    for (auto& adapter : adapters_) {
        if (adapter->getName() == name) {
            adapter->initialize(device_id);
            return adapter;
        }
    }
    
    return nullptr;
}

std::vector<std::string> KernelAdapterFactory::getAvailableAdapters() const {
    std::vector<std::string> names;
    for (const auto& adapter : adapters_) {
        names.push_back(adapter->getName());
    }
    return names;
}

//-----------------------------------
// HighEndGPUAdapter implementation
//-----------------------------------

bool HighEndGPUAdapter::initialize(int device_id) {
    device_id_ = device_id;
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Get device properties
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Determine device type
    DeviceType device_type = DeviceType::OtherGPU;
    
    if (props.major == 8 && props.minor >= 0) {
        device_type = DeviceType::HighEndGPU;
    }
    
    // Store capabilities
    device_caps_.device_type = device_type;
    device_caps_.compute_capability_major = props.major;
    device_caps_.compute_capability_minor = props.minor;
    
    return true;
}

bool HighEndGPUAdapter::isCompatible() const {
    // Check if this adapter is compatible with the current device
    return device_caps_.device_type == DeviceType::HighEndGPU;
}

double HighEndGPUAdapter::executeFilteredBackProjection(
    const DeviceImage<scalar_t>& projections,
    const std::vector<scalar_t>& angles,
    DeviceImage<scalar_t>& output,
    int filter_type
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get dimensions
    int num_angles = static_cast<int>(angles.size());
    int proj_width = projections.getWidth();
    int proj_height = projections.getHeight();
    int img_width = output.getWidth();
    int img_height = output.getHeight();
    
    // Allocate device memory for angles and filter coefficients
    float* d_angles = nullptr;
    cudaMalloc(&d_angles, num_angles * sizeof(float));
    cudaMemcpy(d_angles, angles.data(), num_angles * sizeof(float), cudaMemcpyHostToDevice);
    
    // Generate filter coefficients
    int filter_size = proj_width;
    std::vector<float> filter_coeffs(filter_size, 0.0f);
    // TODO: Implement filter coefficient generation
    
    // Allocate device memory for filter coefficients
    float* d_filter_coeffs = nullptr;
    cudaMalloc(&d_filter_coeffs, filter_size * sizeof(float));
    cudaMemcpy(d_filter_coeffs, filter_coeffs.data(), filter_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Allocate device memory for filtered projections
    DeviceImage<scalar_t> filtered_projections(proj_width, proj_height, 1, 1);
    
    // Get optimal kernel parameters
    auto ramp_filter_params = AdaptiveKernelManager::getInstance().getOptimalKernelParams(
        "RampFilter", proj_width, proj_height, 1
    );
    
    auto backprojection_params = AdaptiveKernelManager::getInstance().getOptimalKernelParams(
        "Backprojection", img_width, img_height, 1
    );
    
    // Launch ramp filter kernel
    launchRampFilterKernel(
        projections.getData(),
        filtered_projections.getData(),
        num_angles,
        proj_width,
        proj_height,
        d_angles,
        d_filter_coeffs,
        filter_size,
        filter_type,
        device_caps_,
        ramp_filter_params
    );
    
    // Launch backprojection kernel
    launchBackprojectionKernel(
        filtered_projections.getData(),
        output.getData(),
        num_angles,
        proj_width,
        img_width,
        img_height,
        device_caps_,
        backprojection_params
    );
    
    // Free device memory
    cudaFree(d_angles);
    cudaFree(d_filter_coeffs);
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

double HighEndGPUAdapter::executeIterativeReconstruction(
    const DeviceImage<scalar_t>& projections,
    const std::vector<scalar_t>& angles,
    DeviceImage<scalar_t>& output,
    int num_iterations
) {
    // Not implemented yet
    return 0.0;
}

double HighEndGPUAdapter::executeFFT(
    const DeviceImage<complex_t>& input,
    DeviceImage<complex_t>& output,
    bool inverse
) {
    // Not implemented yet
    return 0.0;
}

double HighEndGPUAdapter::executeNonCartesianFFT(
    const DeviceImage<complex_t>& input,
    const std::vector<std::pair<scalar_t, scalar_t>>& trajectories,
    DeviceImage<complex_t>& output,
    bool inverse
) {
    // Not implemented yet
    return 0.0;
}

double HighEndGPUAdapter::executeConvolution(
    const DeviceImage<scalar_t>& input,
    const std::vector<scalar_t>& kernel,
    DeviceImage<scalar_t>& output
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get dimensions
    int width = input.getWidth();
    int height = input.getHeight();
    int channels = input.getChannels();
    
    // Calculate kernel size (assuming square kernel)
    int kernel_size = static_cast<int>(std::sqrt(kernel.size()));
    
    // Allocate device memory for kernel
    float* d_kernel = nullptr;
    cudaMalloc(&d_kernel, kernel.size() * sizeof(float));
    cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Get optimal kernel parameters
    auto conv_params = AdaptiveKernelManager::getInstance().getOptimalKernelParams(
        "Convolution", width, height, 1
    );
    
    // Launch convolution kernel
    launch2DConvolutionKernel(
        input.getData(),
        output.getData(),
        width,
        height,
        channels,
        d_kernel,
        kernel_size,
        device_caps_,
        conv_params
    );
    
    // Free device memory
    cudaFree(d_kernel);
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

double HighEndGPUAdapter::executeMedianFilter(
    const DeviceImage<scalar_t>& input,
    DeviceImage<scalar_t>& output,
    int radius
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get dimensions
    int width = input.getWidth();
    int height = input.getHeight();
    int channels = input.getChannels();
    
    // Get optimal kernel parameters
    auto filter_params = AdaptiveKernelManager::getInstance().getOptimalKernelParams(
        "MedianFilter", width, height, 1
    );
    
    // Launch median filter kernel
    launchMedianFilterKernel(
        input.getData(),
        output.getData(),
        width,
        height,
        channels,
        radius,
        device_caps_,
        filter_params
    );
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

double HighEndGPUAdapter::executeBilateralFilter(
    const DeviceImage<scalar_t>& input,
    DeviceImage<scalar_t>& output,
    scalar_t spatial_sigma,
    scalar_t range_sigma,
    int radius
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get dimensions
    int width = input.getWidth();
    int height = input.getHeight();
    int channels = input.getChannels();
    
    // Get optimal kernel parameters
    auto filter_params = AdaptiveKernelManager::getInstance().getOptimalKernelParams(
        "BilateralFilter", width, height, 1
    );
    
    // Launch bilateral filter kernel
    launchBilateralFilterKernel(
        input.getData(),
        output.getData(),
        width,
        height,
        channels,
        spatial_sigma,
        range_sigma,
        radius,
        device_caps_,
        filter_params
    );
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

double HighEndGPUAdapter::executeNLMFilter(
    const DeviceImage<scalar_t>& input,
    DeviceImage<scalar_t>& output,
    int search_radius,
    int patch_radius,
    scalar_t h
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get dimensions
    int width = input.getWidth();
    int height = input.getHeight();
    int channels = input.getChannels();
    
    // Get optimal kernel parameters
    auto filter_params = AdaptiveKernelManager::getInstance().getOptimalKernelParams(
        "NLMFilter", width, height, 1
    );
    
    // Launch NLM filter kernel
    launchNLMFilterKernel(
        input.getData(),
        output.getData(),
        width,
        height,
        channels,
        search_radius,
        patch_radius,
        h,
        device_caps_,
        filter_params
    );
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

double HighEndGPUAdapter::executeThresholding(
    const DeviceImage<scalar_t>& input,
    DeviceImage<scalar_t>& output,
    scalar_t threshold,
    scalar_t max_value
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get dimensions
    int width = input.getWidth();
    int height = input.getHeight();
    int channels = input.getChannels();
    
    // Get optimal kernel parameters
    auto thresh_params = AdaptiveKernelManager::getInstance().getOptimalKernelParams(
        "Thresholding", width, height, 1
    );
    
    // Launch thresholding kernel
    launchThresholdingKernel(
        input.getData(),
        output.getData(),
        width,
        height,
        channels,
        threshold,
        max_value,
        device_caps_,
        thresh_params
    );
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

double HighEndGPUAdapter::executeRegionGrowing(
    const DeviceImage<scalar_t>& input,
    DeviceImage<scalar_t>& output,
    const std::vector<std::vector<index_t>>& seeds,
    scalar_t threshold
) {
    // Not implemented yet
    return 0.0;
}

double HighEndGPUAdapter::executeImageWarping(
    const DeviceImage<scalar_t>& input,
    DeviceImage<scalar_t>& output,
    const std::vector<scalar_t>& transform_matrix,
    int interpolation_mode
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get dimensions
    int width = input.getWidth();
    int height = input.getHeight();
    int depth = input.getDepth();
    int channels = input.getChannels();
    
    // Get optimal kernel parameters
    auto warp_params = AdaptiveKernelManager::getInstance().getOptimalKernelParams(
        "ImageWarping", width, height, depth
    );
    
    // Launch image warping kernel
    launchImageWarpingKernel(
        input.getData(),
        output.getData(),
        width,
        height,
        depth,
        channels,
        transform_matrix,
        interpolation_mode,
        device_caps_,
        warp_params
    );
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

double HighEndGPUAdapter::executeMutualInformation(
    const DeviceImage<scalar_t>& image1,
    const DeviceImage<scalar_t>& image2,
    int num_bins,
    scalar_t& mi_value
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Get dimensions
    int width = image1.getWidth();
    int height = image1.getHeight();
    
    // Get optimal kernel parameters
    auto mi_params = AdaptiveKernelManager::getInstance().getOptimalKernelParams(
        "MutualInformation", width, height, 1
    );
    
    // Launch mutual information kernel
    float max_val = 1.0f;  // Assuming normalized images
    launchMutualInformationKernel(
        image1.getData(),
        image2.getData(),
        width,
        height,
        num_bins,
        max_val,
        mi_value,
        device_caps_,
        mi_params
    );
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

//-----------------------------------
// T4Adapter implementation
//-----------------------------------

bool T4Adapter::initialize(int device_id) {
    device_id_ = device_id;
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Get device properties
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Determine device type
    DeviceType device_type = DeviceType::OtherGPU;
    
    if (props.major == 7 && props.minor == 5) {
        if (std::string(props.name).find("T4") != std::string::npos) {
            device_type = DeviceType::T4;
        }
    }
    
    // Store capabilities
    device_caps_.device_type = device_type;
    device_caps_.compute_capability_major = props.major;
    device_caps_.compute_capability_minor = props.minor;
    
    return true;
}

bool T4Adapter::isCompatible() const {
    // Check if this adapter is compatible with the current device
    return device_caps_.device_type == DeviceType::T4;
}

// Note: Implementations for T4Adapter methods would be similar to HighEndGPUAdapter
// with optimizations specific to the T4 GPU architecture. For brevity, we'll omit them here.

//-----------------------------------
// JetsonOrinAdapter implementation
//-----------------------------------

bool JetsonOrinAdapter::initialize(int device_id) {
    device_id_ = device_id;
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Get device properties
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Determine device type
    DeviceType device_type = DeviceType::OtherGPU;
    
    if (props.major == 8 && props.minor >= 7) {
        if (std::string(props.name).find("Orin") != std::string::npos) {
            device_type = DeviceType::JetsonOrinNX;
        }
    }
    
    // Store capabilities
    device_caps_.device_type = device_type;
    device_caps_.compute_capability_major = props.major;
    device_caps_.compute_capability_minor = props.minor;
    
    return true;
}

bool JetsonOrinAdapter::isCompatible() const {
    // Check if this adapter is compatible with the current device
    return device_caps_.device_type == DeviceType::JetsonOrinNX;
}

// Note: Implementations for JetsonOrinAdapter methods would be similar to HighEndGPUAdapter
// with optimizations specific to the Jetson Orin NX architecture. For brevity, we'll omit them here.

//-----------------------------------
// GenericCUDAAdapter implementation
//-----------------------------------

bool GenericCUDAAdapter::initialize(int device_id) {
    device_id_ = device_id;
    
    // Initialize CUDA
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Get device properties
    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Store capabilities
    device_caps_.device_type = DeviceType::OtherGPU;
    device_caps_.compute_capability_major = props.major;
    device_caps_.compute_capability_minor = props.minor;
    
    return true;
}

bool GenericCUDAAdapter::isCompatible() const {
    // This adapter is compatible with any CUDA device
    return device_caps_.compute_capability_major > 0;
}

// Note: Implementations for GenericCUDAAdapter methods would be similar to HighEndGPUAdapter
// but with more conservative optimizations. For brevity, we'll omit them here.

//-----------------------------------
// CPUAdapter implementation
//-----------------------------------

bool CPUAdapter::initialize(int device_id) {
    // Determine number of CPU threads to use
    num_threads_ = std::thread::hardware_concurrency();
    if (num_threads_ == 0) {
        num_threads_ = 4;  // Default to 4 threads if detection fails
    }
    
    std::cout << "CPU adapter initialized with " << num_threads_ << " threads." << std::endl;
    
    return true;
}

// Note: CPU implementations would be included here for each method.
// These would use standard CPU code optimized with multithreading.
// For brevity, we'll omit them here.

//-----------------------------------
// HybridExecutionManager implementation
//-----------------------------------

HybridExecutionManager& HybridExecutionManager::getInstance() {
    static HybridExecutionManager instance;
    return instance;
}

HybridExecutionManager::HybridExecutionManager()
    : gpu_workload_ratio_(0.9f), initialized_(false) {}

bool HybridExecutionManager::initialize(int device_id, int num_cpu_threads) {
    if (initialized_) {
        return true;
    }
    
    // Get the best GPU adapter
    gpu_adapter_ = KernelAdapterFactory::getInstance().getBestAdapter(device_id);
    
    // Initialize CPU adapter
    cpu_adapter_ = std::make_shared<CPUAdapter>();
    cpu_adapter_->initialize(device_id);
    
    initialized_ = (gpu_adapter_ != nullptr && cpu_adapter_ != nullptr);
    
    return initialized_;
}

void HybridExecutionManager::adjustWorkloadRatio(double gpu_time_ms, double cpu_time_ms) {
    if (gpu_time_ms <= 0.0 || cpu_time_ms <= 0.0) {
        return;
    }
    
    // Calculate relative performance
    double perf_ratio = cpu_time_ms / gpu_time_ms;
    
    // Adjust GPU workload ratio based on performance
    // We want the distribution to be proportional to the relative performance
    float new_ratio = static_cast<float>(perf_ratio / (perf_ratio + 1.0));
    
    // Apply smoothing
    constexpr float alpha = 0.3f;  // Smoothing factor
    gpu_workload_ratio_ = alpha * new_ratio + (1.0f - alpha) * gpu_workload_ratio_;
    
    // Clamp to valid range
    gpu_workload_ratio_ = std::max(0.1f, std::min(0.9f, gpu_workload_ratio_));
}

void HybridExecutionManager::splitWorkload(int total_work, int& gpu_work, int& cpu_work) const {
    gpu_work = static_cast<int>(total_work * gpu_workload_ratio_);
    cpu_work = total_work - gpu_work;
}

double HybridExecutionManager::executeHybridFilteredBackProjection(
    const DeviceImage<scalar_t>& projections,
    const std::vector<scalar_t>& angles,
    DeviceImage<scalar_t>& output,
    int filter_type
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!initialized_ || !gpu_adapter_ || !cpu_adapter_) {
        return 0.0;
    }
    
    // Split the workload
    int num_angles = static_cast<int>(angles.size());
    int gpu_angles, cpu_angles;
    splitWorkload(num_angles, gpu_angles, cpu_angles);
    
    // TODO: Implement hybrid execution
    // For now, just use the GPU for everything
    gpu_adapter_->executeFilteredBackProjection(projections, angles, output, filter_type);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

double HybridExecutionManager::executeHybridBilateralFilter(
    const DeviceImage<scalar_t>& input,
    DeviceImage<scalar_t>& output,
    scalar_t spatial_sigma,
    scalar_t range_sigma,
    int radius
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!initialized_ || !gpu_adapter_ || !cpu_adapter_) {
        return 0.0;
    }
    
    // TODO: Implement hybrid execution
    // For now, just use the GPU for everything
    gpu_adapter_->executeBilateralFilter(input, output, spatial_sigma, range_sigma, radius);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

double HybridExecutionManager::executeHybridNLMFilter(
    const DeviceImage<scalar_t>& input,
    DeviceImage<scalar_t>& output,
    int search_radius,
    int patch_radius,
    scalar_t h
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!initialized_ || !gpu_adapter_ || !cpu_adapter_) {
        return 0.0;
    }
    
    // TODO: Implement hybrid execution
    // For now, just use the GPU for everything
    gpu_adapter_->executeNLMFilter(input, output, search_radius, patch_radius, h);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    return execution_time;
}

} // namespace medical_imaging