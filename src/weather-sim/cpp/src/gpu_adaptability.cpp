/**
 * @file gpu_adaptability.cpp
 * @brief Implementation of GPU adaptability pattern for weather simulation.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#include "../include/weather_sim/gpu_adaptability.hpp"
#include <cmath>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <thread>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace weather_sim {

// AdaptiveKernelManager implementation
AdaptiveKernelManager& AdaptiveKernelManager::getInstance() {
    static AdaptiveKernelManager instance;
    return instance;
}

AdaptiveKernelManager::AdaptiveKernelManager() {
    // Default initialization with CPU-only capabilities
    device_caps_.device_type = DeviceType::CPU;
    device_caps_.compute_capability_major = 0;
    device_caps_.compute_capability_minor = 0;
    device_caps_.device_name = "CPU";
    device_caps_.compute_power_ratio = 1.0f;
    
    // Initialize default parameters for different operations
    initializeDefaultParams();
}

bool AdaptiveKernelManager::initialize(int device_id) {
#ifdef __CUDACC__
    // Try to detect and initialize CUDA device
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA not available or no CUDA devices found: " 
                  << cudaGetErrorString(err) << std::endl;
        cuda_available_ = false;
        return false;
    }
    
    cuda_available_ = true;
    
    // Validate and set device
    if (device_id >= device_count) {
        std::cerr << "Invalid device ID " << device_id 
                  << ", using default device 0 instead" << std::endl;
        device_id = 0;
    }
    
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " 
                  << cudaGetErrorString(err) << std::endl;
        cuda_available_ = false;
        return false;
    }
    
    // Detect device capabilities
    detectDeviceCapabilities(device_id);
    
    // Tune parameters based on device type
    tuneParameters();
    
    return cuda_available_;
#else
    // CUDA not available at compile time
    cuda_available_ = false;
    return false;
#endif
}

bool AdaptiveKernelManager::isCudaAvailable() const {
    return cuda_available_;
}

const DeviceCapabilities& AdaptiveKernelManager::getDeviceCapabilities() const {
    return device_caps_;
}

void AdaptiveKernelManager::detectDeviceCapabilities(int device_id) {
#ifdef __CUDACC__
    cudaDeviceProp props;
    cudaError_t err = cudaGetDeviceProperties(&props, device_id);
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties: " 
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Set basic device properties
    device_caps_.compute_capability_major = props.major;
    device_caps_.compute_capability_minor = props.minor;
    device_caps_.device_name = props.name;
    device_caps_.cuda_cores = props.multiProcessorCount * 
                              (props.major == 8 ? 128 : 64);  // Approximation
    device_caps_.multiprocessors = props.multiProcessorCount;
    device_caps_.global_memory = props.totalGlobalMem;
    device_caps_.shared_memory_per_block = props.sharedMemPerBlock;
    device_caps_.max_threads_per_block = props.maxThreadsPerBlock;
    device_caps_.max_threads_per_multiprocessor = props.maxThreadsPerMultiProcessor;
    device_caps_.clock_rate_khz = props.clockRate;
    device_caps_.memory_clock_rate_khz = props.memoryClockRate;
    device_caps_.memory_bus_width = props.memoryBusWidth;
    
    // Determine device type based on compute capability and name
    if (props.major == 8 && props.minor >= 7) {
        if (std::string(props.name).find("Orin") != std::string::npos) {
            device_caps_.device_type = DeviceType::JetsonOrinNX;
            device_caps_.compute_power_ratio = 8.0f;
        } else {
            device_caps_.device_type = DeviceType::HighEndGPU;
            device_caps_.compute_power_ratio = 20.0f;
        }
    } else if (props.major >= 8) {
        device_caps_.device_type = DeviceType::HighEndGPU;
        device_caps_.compute_power_ratio = 15.0f;
    } else if (props.major == 7 && props.minor >= 5) {
        if (std::string(props.name).find("T4") != std::string::npos) {
            device_caps_.device_type = DeviceType::T4;
            device_caps_.compute_power_ratio = 10.0f;
        } else {
            device_caps_.device_type = DeviceType::OtherGPU;
            device_caps_.compute_power_ratio = 8.0f;
        }
    } else {
        device_caps_.device_type = DeviceType::OtherGPU;
        device_caps_.compute_power_ratio = 5.0f;
    }
#else
    // Set CPU device properties
    device_caps_.device_type = DeviceType::CPU;
    device_caps_.compute_capability_major = 0;
    device_caps_.compute_capability_minor = 0;
    device_caps_.device_name = "CPU";
    device_caps_.cuda_cores = 0;
    device_caps_.multiprocessors = std::thread::hardware_concurrency();
    device_caps_.global_memory = 0;
    device_caps_.compute_power_ratio = 1.0f;
#endif
}

void AdaptiveKernelManager::initializeDefaultParams() {
    // Default parameters for shallow water simulation
    KernelLaunchParams shallowWaterParams;
    shallowWaterParams.block_size_x = 16;
    shallowWaterParams.block_size_y = 16;
    
    // Use a tuple of 0,0,0 to represent default parameters
    ProblemSize defaultSize(0, 0, 0);
    optimal_params_["shallow_water"][defaultSize] = shallowWaterParams;
    
    // Default parameters for barotropic vorticity equation
    KernelLaunchParams barotropicParams;
    barotropicParams.block_size_x = 16;
    barotropicParams.block_size_y = 16;
    optimal_params_["barotropic"][defaultSize] = barotropicParams;
    
    // Default parameters for primitive equations
    KernelLaunchParams primitiveParams;
    primitiveParams.block_size_x = 32;
    primitiveParams.block_size_y = 8;
    optimal_params_["primitive_equations"][defaultSize] = primitiveParams;
    
    // Default parameters for diagnostic calculations
    KernelLaunchParams diagnosticParams;
    diagnosticParams.block_size_x = 32;
    diagnosticParams.block_size_y = 8;
    optimal_params_["diagnostics"][defaultSize] = diagnosticParams;
    
    // Default CPU/GPU workload ratios (1.0 = all GPU, 0.0 = all CPU)
    gpu_workload_ratios_["shallow_water"] = 0.9f;
    gpu_workload_ratios_["barotropic"] = 0.9f;
    gpu_workload_ratios_["primitive_equations"] = 0.9f;
    gpu_workload_ratios_["general_circulation"] = 0.9f;
}

void AdaptiveKernelManager::tuneParameters() {
    // Tune parameters based on detected device type
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
        case DeviceType::OtherGPU:
            optimizeForGenericGPU();
            break;
        default:
            // For CPU or unknown, keep defaults
            break;
    }
}

void AdaptiveKernelManager::optimizeForJetsonOrin() {
    ProblemSize defaultSize(0, 0, 0);
    
    // Optimized parameters for Jetson Orin NX
    KernelLaunchParams shallowWaterParams;
    shallowWaterParams.block_size_x = 16;
    shallowWaterParams.block_size_y = 16;
    shallowWaterParams.shared_memory_bytes = 8192; // 8KB shared memory
    optimal_params_["shallow_water"][defaultSize] = shallowWaterParams;
    
    // For different problem sizes
    ProblemSize smallSize(256, 256, 1);
    KernelLaunchParams smallParams;
    smallParams.block_size_x = 16;
    smallParams.block_size_y = 16;
    smallParams.shared_memory_bytes = 8192;
    optimal_params_["shallow_water"][smallSize] = smallParams;
    
    ProblemSize mediumSize(512, 512, 1);
    KernelLaunchParams mediumParams;
    mediumParams.block_size_x = 32;
    mediumParams.block_size_y = 8;
    mediumParams.shared_memory_bytes = 12288;
    optimal_params_["shallow_water"][mediumSize] = mediumParams;
    
    ProblemSize largeSize(1024, 1024, 1);
    KernelLaunchParams largeParams;
    largeParams.block_size_x = 32;
    largeParams.block_size_y = 8;
    largeParams.shared_memory_bytes = 16384;
    optimal_params_["shallow_water"][largeSize] = largeParams;
    
    // Adjust CPU/GPU workload ratios for Jetson Orin
    // Jetson Orin NX has a powerful GPU relative to its CPU
    gpu_workload_ratios_["shallow_water"] = 0.85f;
    gpu_workload_ratios_["barotropic"] = 0.85f;
    gpu_workload_ratios_["primitive_equations"] = 0.90f;
    gpu_workload_ratios_["general_circulation"] = 0.90f;
}

void AdaptiveKernelManager::optimizeForT4() {
    ProblemSize defaultSize(0, 0, 0);
    
    // Optimized parameters for T4 GPU
    KernelLaunchParams shallowWaterParams;
    shallowWaterParams.block_size_x = 32;
    shallowWaterParams.block_size_y = 8;
    shallowWaterParams.shared_memory_bytes = 12288; // 12KB shared memory
    optimal_params_["shallow_water"][defaultSize] = shallowWaterParams;
    
    // For different problem sizes
    ProblemSize smallSize(256, 256, 1);
    KernelLaunchParams smallParams;
    smallParams.block_size_x = 16;
    smallParams.block_size_y = 16;
    smallParams.shared_memory_bytes = 8192;
    optimal_params_["shallow_water"][smallSize] = smallParams;
    
    ProblemSize mediumSize(512, 512, 1);
    KernelLaunchParams mediumParams;
    mediumParams.block_size_x = 32;
    mediumParams.block_size_y = 8;
    mediumParams.shared_memory_bytes = 12288;
    optimal_params_["shallow_water"][mediumSize] = mediumParams;
    
    ProblemSize largeSize(1024, 1024, 1);
    KernelLaunchParams largeParams;
    largeParams.block_size_x = 32;
    largeParams.block_size_y = 8;
    largeParams.shared_memory_bytes = 16384;
    optimal_params_["shallow_water"][largeSize] = largeParams;
    
    // Adjust CPU/GPU workload ratios for T4
    // T4 is typically used in systems with powerful CPUs (AWS Graviton)
    gpu_workload_ratios_["shallow_water"] = 0.75f;
    gpu_workload_ratios_["barotropic"] = 0.75f;
    gpu_workload_ratios_["primitive_equations"] = 0.80f;
    gpu_workload_ratios_["general_circulation"] = 0.80f;
}

void AdaptiveKernelManager::optimizeForHighEndGPU() {
    ProblemSize defaultSize(0, 0, 0);
    
    // Optimized parameters for high-end GPUs
    KernelLaunchParams shallowWaterParams;
    shallowWaterParams.block_size_x = 32;
    shallowWaterParams.block_size_y = 8;
    shallowWaterParams.shared_memory_bytes = 16384; // 16KB shared memory
    optimal_params_["shallow_water"][defaultSize] = shallowWaterParams;
    
    // For different problem sizes
    ProblemSize smallSize(256, 256, 1);
    KernelLaunchParams smallParams;
    smallParams.block_size_x = 16;
    smallParams.block_size_y = 16;
    smallParams.shared_memory_bytes = 8192;
    optimal_params_["shallow_water"][smallSize] = smallParams;
    
    ProblemSize mediumSize(512, 512, 1);
    KernelLaunchParams mediumParams;
    mediumParams.block_size_x = 32;
    mediumParams.block_size_y = 8;
    mediumParams.shared_memory_bytes = 16384;
    optimal_params_["shallow_water"][mediumSize] = mediumParams;
    
    ProblemSize largeSize(1024, 1024, 1);
    KernelLaunchParams largeParams;
    largeParams.block_size_x = 32;
    largeParams.block_size_y = 16;
    largeParams.shared_memory_bytes = 32768;
    optimal_params_["shallow_water"][largeSize] = largeParams;
    
    // Adjust CPU/GPU workload ratios for high-end GPUs
    // High-end systems typically have very powerful GPUs
    gpu_workload_ratios_["shallow_water"] = 0.95f;
    gpu_workload_ratios_["barotropic"] = 0.95f;
    gpu_workload_ratios_["primitive_equations"] = 0.95f;
    gpu_workload_ratios_["general_circulation"] = 0.95f;
}

void AdaptiveKernelManager::optimizeForGenericGPU() {
    ProblemSize defaultSize(0, 0, 0);
    
    // Conservative parameters for generic GPUs
    KernelLaunchParams shallowWaterParams;
    shallowWaterParams.block_size_x = 16;
    shallowWaterParams.block_size_y = 16;
    shallowWaterParams.shared_memory_bytes = 8192; // 8KB shared memory
    optimal_params_["shallow_water"][defaultSize] = shallowWaterParams;
    
    // Adjust CPU/GPU workload ratios for generic GPUs
    // Be more conservative with unknown GPUs
    gpu_workload_ratios_["shallow_water"] = 0.7f;
    gpu_workload_ratios_["barotropic"] = 0.7f;
    gpu_workload_ratios_["primitive_equations"] = 0.7f;
    gpu_workload_ratios_["general_circulation"] = 0.7f;
}

KernelLaunchParams AdaptiveKernelManager::getOptimalKernelParams(
    const std::string& operation_name, int width, int height, int depth
) {
    // Find parameters for specific problem size
    auto& op_params = optimal_params_[operation_name];
    
    // Check for exact size match
    ProblemSize size(width, height, depth);
    auto it = op_params.find(size);
    if (it != op_params.end()) {
        auto params = it->second;
        params.computeGridDimensions(width, height, depth);
        return params;
    }
    
    // Find closest size match
    ProblemSize closest_size(0, 0, 0);
    int min_diff = std::numeric_limits<int>::max();
    
    for (const auto& entry : op_params) {
        const auto& curr_size = entry.first;
        int curr_width = std::get<0>(curr_size);
        int curr_height = std::get<1>(curr_size);
        int curr_depth = std::get<2>(curr_size);
        
        // Skip default parameters (0,0,0)
        if (curr_width == 0 && curr_height == 0 && curr_depth == 0) {
            continue;
        }
        
        // Compute difference metric (Manhattan distance in log space)
        int diff = std::abs(std::log2(curr_width + 1) - std::log2(width + 1)) +
                   std::abs(std::log2(curr_height + 1) - std::log2(height + 1)) +
                   std::abs(std::log2(curr_depth + 1) - std::log2(depth + 1));
        
        if (diff < min_diff) {
            min_diff = diff;
            closest_size = curr_size;
        }
    }
    
    // Use closest size if found, otherwise default
    if (min_diff < std::numeric_limits<int>::max()) {
        auto params = op_params[closest_size];
        params.computeGridDimensions(width, height, depth);
        return params;
    }
    
    // Fall back to default parameters
    auto params = op_params[ProblemSize(0, 0, 0)];
    params.computeGridDimensions(width, height, depth);
    return params;
}

float AdaptiveKernelManager::getGpuWorkloadRatio(const std::string& operation_name) {
    auto it = gpu_workload_ratios_.find(operation_name);
    if (it != gpu_workload_ratios_.end()) {
        return it->second;
    }
    
    // Default to 90% GPU if not specified
    return 0.9f;
}

void AdaptiveKernelManager::updatePerformanceMetrics(
    const std::string& operation_name,
    const KernelLaunchParams& params,
    double execution_time_ms
) {
    // This could be used for future auto-tuning
    // For now, just store the data
    
    // Determine problem size from params
    ProblemSize size(params.grid_size_x * params.block_size_x,
                    params.grid_size_y * params.block_size_y,
                    params.grid_size_z * params.block_size_z);
    
    // Create or update performance record
    auto& records = performance_history_[operation_name][size];
    
    // Check if we already have a record with these exact parameters
    auto it = std::find_if(
        records.begin(), records.end(),
        [&params](const PerformanceRecord& record) {
            return record.params.block_size_x == params.block_size_x &&
                   record.params.block_size_y == params.block_size_y &&
                   record.params.block_size_z == params.block_size_z &&
                   record.params.shared_memory_bytes == params.shared_memory_bytes;
        }
    );
    
    if (it != records.end()) {
        // Update existing record with moving average
        it->execution_time_ms = (it->execution_time_ms * it->sample_count + execution_time_ms) / 
                                (it->sample_count + 1);
        it->sample_count++;
    } else {
        // Create new record
        PerformanceRecord record;
        record.params = params;
        record.execution_time_ms = execution_time_ms;
        record.sample_count = 1;
        records.push_back(record);
    }
}

ComputeBackend AdaptiveKernelManager::determineOptimalBackend(
    int width, int height, const std::string& operation_name
) {
    // If CUDA is not available, use CPU
    if (!cuda_available_) {
        return ComputeBackend::CPU;
    }
    
    // For very small problems, CPU might be faster due to overhead
    const int small_threshold = 64;
    if (width <= small_threshold && height <= small_threshold) {
        return ComputeBackend::CPU;
    }
    
    // Get the GPU workload ratio for this operation
    float gpu_ratio = getGpuWorkloadRatio(operation_name);
    
    // If GPU ratio is close to 1, use GPU only
    if (gpu_ratio > 0.95f) {
        return ComputeBackend::CUDA;
    }
    
    // If GPU ratio is very low, use CPU only
    if (gpu_ratio < 0.2f) {
        return ComputeBackend::CPU;
    }
    
    // For medium-sized problems, use hybrid approach
    const int medium_threshold = 256;
    if (width <= medium_threshold && height <= medium_threshold) {
        return ComputeBackend::Hybrid;
    }
    
    // For large problems, use adaptive hybrid
    return ComputeBackend::AdaptiveHybrid;
}

// KernelLaunchParams implementation
void KernelLaunchParams::computeGridDimensions(int width, int height, int depth) {
    // Compute grid dimensions based on block size and problem size
    grid_size_x = (width + block_size_x - 1) / block_size_x;
    grid_size_y = (height + block_size_y - 1) / block_size_y;
    grid_size_z = (depth + block_size_z - 1) / block_size_z;
}

// DeviceCapabilities implementation
std::string DeviceCapabilities::getSummary() const {
    std::stringstream ss;
    ss << "Device: " << device_name << std::endl;
    ss << "Device Type: ";
    
    switch (device_type) {
        case DeviceType::JetsonOrinNX:
            ss << "Jetson Orin NX";
            break;
        case DeviceType::T4:
            ss << "NVIDIA T4";
            break;
        case DeviceType::HighEndGPU:
            ss << "High-End GPU";
            break;
        case DeviceType::OtherGPU:
            ss << "Other GPU";
            break;
        case DeviceType::CPU:
            ss << "CPU";
            break;
        default:
            ss << "Unknown";
            break;
    }
    ss << std::endl;
    
    if (device_type != DeviceType::CPU) {
        ss << "Compute Capability: " << compute_capability_major << "." 
           << compute_capability_minor << std::endl;
        ss << "CUDA Cores: " << cuda_cores << std::endl;
        ss << "Multiprocessors: " << multiprocessors << std::endl;
        ss << "Global Memory: " << (global_memory / (1024 * 1024)) << " MB" << std::endl;
        ss << "Shared Memory Per Block: " << (shared_memory_per_block / 1024) << " KB" << std::endl;
        ss << "Max Threads Per Block: " << max_threads_per_block << std::endl;
        ss << "Clock Rate: " << (clock_rate_khz / 1000) << " MHz" << std::endl;
    }
    
    ss << "Compute Power Ratio: " << compute_power_ratio << "x" << std::endl;
    
    return ss.str();
}

// KernelAdapterFactory implementation
KernelAdapterFactory& KernelAdapterFactory::getInstance() {
    static KernelAdapterFactory instance;
    return instance;
}

void KernelAdapterFactory::registerAdapter(std::shared_ptr<KernelAdapter> adapter) {
    adapters_.push_back(adapter);
}

std::shared_ptr<KernelAdapter> KernelAdapterFactory::getBestAdapter(int device_id) {
    // Initialize all adapters
    for (auto& adapter : adapters_) {
        adapter->initialize(device_id);
    }
    
    // Find best compatible adapter
    std::shared_ptr<KernelAdapter> best_adapter = nullptr;
    int best_priority = -1;
    
    for (auto& adapter : adapters_) {
        if (adapter->isCompatible() && adapter->getPriority() > best_priority) {
            best_adapter = adapter;
            best_priority = adapter->getPriority();
        }
    }
    
    return best_adapter;
}

std::shared_ptr<KernelAdapter> KernelAdapterFactory::getAdapter(
    const std::string& name, int device_id
) {
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

// HybridExecutionManager implementation
HybridExecutionManager& HybridExecutionManager::getInstance() {
    static HybridExecutionManager instance;
    return instance;
}

HybridExecutionManager::HybridExecutionManager()
    : gpu_adapter_(nullptr), cpu_adapter_(nullptr), gpu_workload_ratio_(0.9f), initialized_(false) {
}

bool HybridExecutionManager::initialize(int device_id, int num_cpu_threads) {
    auto& factory = KernelAdapterFactory::getInstance();
    
    // Get GPU adapter
    gpu_adapter_ = factory.getBestAdapter(device_id);
    
    // Get CPU adapter
    cpu_adapter_ = std::dynamic_pointer_cast<CPUAdapter>(factory.getAdapter("CPUAdapter"));
    
    if (!cpu_adapter_) {
        std::cerr << "CPU adapter not found or not registered" << std::endl;
        return false;
    }
    
    initialized_ = (gpu_adapter_ != nullptr) && (cpu_adapter_ != nullptr);
    return initialized_;
}

double HybridExecutionManager::executeHybridStep(
    const WeatherGrid& in_grid,
    WeatherGrid& out_grid,
    scalar_t dt,
    SimulationModel simulation_model
) {
    if (!initialized_) {
        return -1.0;
    }
    
    // Determine workload split
    int total_work = in_grid.getWidth() * in_grid.getHeight();
    int gpu_work = 0;
    int cpu_work = 0;
    
    splitWorkload(total_work, gpu_work, cpu_work);
    
    // Create temporary grids for split workloads
    // This is simplified; in practice you'd need to split the grid data
    
    double gpu_time = 0.0;
    double cpu_time = 0.0;
    
    // Execute on GPU
    if (gpu_work > 0 && gpu_adapter_) {
        switch (simulation_model) {
            case SimulationModel::ShallowWater:
                gpu_time = gpu_adapter_->executeShallowWaterStep(in_grid, out_grid, dt);
                break;
            case SimulationModel::Barotropic:
                gpu_time = gpu_adapter_->executeBarotropicStep(in_grid, out_grid, dt);
                break;
            case SimulationModel::PrimitiveEquations:
                gpu_time = gpu_adapter_->executePrimitiveEquationsStep(in_grid, out_grid, dt);
                break;
            case SimulationModel::General:
                gpu_time = gpu_adapter_->executeGCMStep(in_grid, out_grid, dt);
                break;
        }
    }
    
    // Execute on CPU
    if (cpu_work > 0 && cpu_adapter_) {
        // For demonstration purposes, we're just reusing the same interface
        // In practice, you'd process different portions of the grid
        switch (simulation_model) {
            case SimulationModel::ShallowWater:
                cpu_time = cpu_adapter_->executeShallowWaterStep(in_grid, out_grid, dt);
                break;
            case SimulationModel::Barotropic:
                cpu_time = cpu_adapter_->executeBarotropicStep(in_grid, out_grid, dt);
                break;
            case SimulationModel::PrimitiveEquations:
                cpu_time = cpu_adapter_->executePrimitiveEquationsStep(in_grid, out_grid, dt);
                break;
            case SimulationModel::General:
                cpu_time = cpu_adapter_->executeGCMStep(in_grid, out_grid, dt);
                break;
        }
    }
    
    // Adjust workload ratio based on performance
    if (gpu_work > 0 && cpu_work > 0) {
        adjustWorkloadRatio(gpu_time, cpu_time);
    }
    
    // Return total execution time
    return gpu_time + cpu_time;
}

void HybridExecutionManager::splitWorkload(int total_work, int& gpu_work, int& cpu_work) const {
    gpu_work = static_cast<int>(total_work * gpu_workload_ratio_);
    cpu_work = total_work - gpu_work;
}

void HybridExecutionManager::adjustWorkloadRatio(double gpu_time_ms, double cpu_time_ms) {
    if (gpu_time_ms <= 0.0 || cpu_time_ms <= 0.0) {
        return; // Invalid times
    }
    
    // Calculate normalized work per unit time
    double gpu_work_per_ms = gpu_workload_ratio_ / gpu_time_ms;
    double cpu_work_per_ms = (1.0f - gpu_workload_ratio_) / cpu_time_ms;
    double total_work_per_ms = gpu_work_per_ms + cpu_work_per_ms;
    
    // Calculate optimal ratio
    float optimal_ratio = static_cast<float>(gpu_work_per_ms / total_work_per_ms);
    
    // Apply smoothing to avoid radical changes
    constexpr float alpha = 0.1f; // Smoothing factor
    gpu_workload_ratio_ = gpu_workload_ratio_ * (1.0f - alpha) + optimal_ratio * alpha;
    
    // Ensure ratio stays in valid range
    gpu_workload_ratio_ = std::max(0.0f, std::min(1.0f, gpu_workload_ratio_));
}

} // namespace weather_sim