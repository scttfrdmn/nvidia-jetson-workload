/**
 * @file gpu_adaptability.hpp
 * @brief GPU adaptability pattern for weather simulation.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <tuple>

#include "weather_sim.hpp"

namespace weather_sim {

/**
 * @brief Enumeration for device types.
 */
enum class DeviceType {
    Unknown,
    CPU,
    JetsonOrinNX,  // Jetson Orin NX (SM 8.7)
    T4,            // NVIDIA T4 GPU (SM 7.5)
    HighEndGPU,    // High-end NVIDIA GPU (SM >= 8.0)
    OtherGPU       // Other GPU types
};

/**
 * @brief Structure to hold device capabilities.
 */
struct DeviceCapabilities {
    DeviceType device_type = DeviceType::Unknown;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int cuda_cores = 0;
    int multiprocessors = 0;
    size_t global_memory = 0;
    size_t shared_memory_per_block = 0;
    int max_threads_per_block = 0;
    int max_threads_per_multiprocessor = 0;
    int clock_rate_khz = 0;
    int memory_clock_rate_khz = 0;
    int memory_bus_width = 0;
    float compute_power_ratio = 0.0f;  // Relative to CPU (higher is better)
    
    std::string device_name;
    
    // Default constructor
    DeviceCapabilities() = default;
    
    // Constructor with device details
    DeviceCapabilities(
        DeviceType type,
        const std::string& name,
        int cc_major,
        int cc_minor,
        int cores,
        int sms,
        size_t mem,
        size_t shared_mem,
        int max_threads,
        int max_threads_sm,
        int clock,
        int mem_clock,
        int bus_width,
        float power_ratio
    ) : device_type(type),
        compute_capability_major(cc_major),
        compute_capability_minor(cc_minor),
        cuda_cores(cores),
        multiprocessors(sms),
        global_memory(mem),
        shared_memory_per_block(shared_mem),
        max_threads_per_block(max_threads),
        max_threads_per_multiprocessor(max_threads_sm),
        clock_rate_khz(clock),
        memory_clock_rate_khz(mem_clock),
        memory_bus_width(bus_width),
        compute_power_ratio(power_ratio),
        device_name(name) {}
    
    // Human-readable summary
    std::string getSummary() const;
};

/**
 * @brief Structure to hold kernel launch parameters.
 */
struct KernelLaunchParams {
    int block_size_x = 16;
    int block_size_y = 16;
    int block_size_z = 1;
    int grid_size_x = 0;  // Computed based on problem size
    int grid_size_y = 0;  // Computed based on problem size
    int grid_size_z = 1;
    size_t shared_memory_bytes = 0;
    int stream_id = 0;
    
    // Constructor with common defaults
    KernelLaunchParams() = default;
    
    // Constructor with specific values
    KernelLaunchParams(
        int bx, int by, int bz,
        int gx, int gy, int gz,
        size_t shared_mem,
        int stream
    ) : block_size_x(bx),
        block_size_y(by),
        block_size_z(bz),
        grid_size_x(gx),
        grid_size_y(gy),
        grid_size_z(gz),
        shared_memory_bytes(shared_mem),
        stream_id(stream) {}
    
    // Compute grid dimensions based on problem size
    void computeGridDimensions(int width, int height, int depth = 1);
};

/**
 * @brief Class for adaptive kernel selection and optimization.
 */
class AdaptiveKernelManager {
public:
    /**
     * @brief Get the singleton instance.
     * @return Reference to the singleton instance
     */
    static AdaptiveKernelManager& getInstance();
    
    /**
     * @brief Initialize the manager.
     * @param device_id CUDA device ID to use
     * @return True if initialization succeeded
     */
    bool initialize(int device_id = 0);
    
    /**
     * @brief Check if CUDA is available.
     * @return True if CUDA is available
     */
    bool isCudaAvailable() const;
    
    /**
     * @brief Get device capabilities.
     * @return Device capabilities
     */
    const DeviceCapabilities& getDeviceCapabilities() const;
    
    /**
     * @brief Get optimal kernel parameters for a specific operation.
     * @param operation_name The name of the operation
     * @param width Grid width
     * @param height Grid height
     * @param depth Grid depth (default: 1)
     * @return Optimal kernel launch parameters
     */
    KernelLaunchParams getOptimalKernelParams(
        const std::string& operation_name,
        int width, int height, int depth = 1
    );
    
    /**
     * @brief Get the distribution ratio between CPU and GPU.
     * @param operation_name The name of the operation
     * @return Ratio of work to be done on GPU (0.0-1.0, where 1.0 means all on GPU)
     */
    float getGpuWorkloadRatio(const std::string& operation_name);
    
    /**
     * @brief Update performance metrics for auto-tuning.
     * @param operation_name The name of the operation
     * @param params The parameters used
     * @param execution_time_ms Execution time in milliseconds
     */
    void updatePerformanceMetrics(
        const std::string& operation_name,
        const KernelLaunchParams& params,
        double execution_time_ms
    );
    
    /**
     * @brief Determine the optimal compute backend for a given workload.
     * @param width Grid width
     * @param height Grid height
     * @param operation_name The name of the operation
     * @return Optimal compute backend
     */
    ComputeBackend determineOptimalBackend(
        int width, int height,
        const std::string& operation_name
    );
    
private:
    // Private constructor for singleton
    AdaptiveKernelManager();
    
    // No copy or move
    AdaptiveKernelManager(const AdaptiveKernelManager&) = delete;
    AdaptiveKernelManager& operator=(const AdaptiveKernelManager&) = delete;
    
    // Helper methods
    void detectDeviceCapabilities(int device_id);
    void initializeDefaultParams();
    void tuneParameters();
    
    // Specialized optimization for different architectures
    void optimizeForJetsonOrin();
    void optimizeForT4();
    void optimizeForHighEndGPU();
    void optimizeForGenericGPU();
    
    // Member variables
    bool cuda_available_ = false;
    DeviceCapabilities device_caps_;
    
    // Map of operation names to optimal parameters for different problem sizes
    using ProblemSize = std::tuple<int, int, int>; // width, height, depth
    std::map<std::string, std::map<ProblemSize, KernelLaunchParams>> optimal_params_;
    
    // Performance history for auto-tuning
    struct PerformanceRecord {
        KernelLaunchParams params;
        double execution_time_ms;
        int sample_count;
    };
    
    std::map<std::string, std::map<ProblemSize, std::vector<PerformanceRecord>>> performance_history_;
    
    // CPU/GPU workload distribution ratios
    std::map<std::string, float> gpu_workload_ratios_;
};

/**
 * @brief Interface for kernel adapters.
 */
class KernelAdapter {
public:
    virtual ~KernelAdapter() = default;
    
    /**
     * @brief Initialize the kernel adapter.
     * @param device_id CUDA device ID to use
     * @return True if initialization succeeded
     */
    virtual bool initialize(int device_id = 0) = 0;
    
    /**
     * @brief Check if the adapter is compatible with the current device.
     * @return True if compatible
     */
    virtual bool isCompatible() const = 0;
    
    /**
     * @brief Get the name of the adapter.
     * @return Name of the adapter
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get the priority of the adapter (higher is better).
     * @return Priority value
     */
    virtual int getPriority() const = 0;
    
    /**
     * @brief Execute the shallow water step kernel.
     * @param in_grid Input grid
     * @param out_grid Output grid
     * @param dt Time step
     * @return Execution time in milliseconds
     */
    virtual double executeShallowWaterStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) = 0;
    
    /**
     * @brief Execute the barotropic vorticity step kernel.
     * @param in_grid Input grid
     * @param out_grid Output grid
     * @param dt Time step
     * @return Execution time in milliseconds
     */
    virtual double executeBarotropicStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) = 0;
    
    /**
     * @brief Execute the primitive equations step kernel.
     * @param in_grid Input grid
     * @param out_grid Output grid
     * @param dt Time step
     * @return Execution time in milliseconds
     */
    virtual double executePrimitiveEquationsStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) = 0;
    
    /**
     * @brief Execute the general circulation model step kernel.
     * @param in_grid Input grid
     * @param out_grid Output grid
     * @param dt Time step
     * @return Execution time in milliseconds
     */
    virtual double executeGCMStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) = 0;
    
    /**
     * @brief Calculate diagnostic fields like vorticity, divergence, etc.
     * @param grid The grid to compute diagnostics for
     * @return Execution time in milliseconds
     */
    virtual double calculateDiagnostics(WeatherGrid& grid) = 0;
};

/**
 * @brief Factory for creating and managing kernel adapters.
 */
class KernelAdapterFactory {
public:
    /**
     * @brief Get the singleton instance.
     * @return Reference to the singleton instance
     */
    static KernelAdapterFactory& getInstance();
    
    /**
     * @brief Register a kernel adapter.
     * @param adapter Shared pointer to the adapter
     */
    void registerAdapter(std::shared_ptr<KernelAdapter> adapter);
    
    /**
     * @brief Get the best compatible adapter.
     * @param device_id CUDA device ID to use
     * @return Shared pointer to the best adapter, or nullptr if none compatible
     */
    std::shared_ptr<KernelAdapter> getBestAdapter(int device_id = 0);
    
    /**
     * @brief Get a specific adapter by name.
     * @param name Name of the adapter
     * @param device_id CUDA device ID to use
     * @return Shared pointer to the adapter, or nullptr if not found
     */
    std::shared_ptr<KernelAdapter> getAdapter(const std::string& name, int device_id = 0);
    
    /**
     * @brief Get a list of all available adapters.
     * @return Vector of adapter names
     */
    std::vector<std::string> getAvailableAdapters() const;
    
private:
    // Private constructor for singleton
    KernelAdapterFactory() = default;
    
    // No copy or move
    KernelAdapterFactory(const KernelAdapterFactory&) = delete;
    KernelAdapterFactory& operator=(const KernelAdapterFactory&) = delete;
    
    // Member variables
    std::vector<std::shared_ptr<KernelAdapter>> adapters_;
};

// Adapter for high-end GPUs with Compute Capability >= 8.0
class HighEndGPUAdapter : public KernelAdapter {
public:
    bool initialize(int device_id = 0) override;
    bool isCompatible() const override;
    std::string getName() const override { return "HighEndGPUAdapter"; }
    int getPriority() const override { return 100; }
    
    double executeShallowWaterStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executeBarotropicStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executePrimitiveEquationsStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executeGCMStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double calculateDiagnostics(WeatherGrid& grid) override;
    
private:
    int device_id_ = 0;
    DeviceCapabilities device_caps_;
};

// Adapter for NVIDIA T4 GPUs (SM 7.5)
class T4Adapter : public KernelAdapter {
public:
    bool initialize(int device_id = 0) override;
    bool isCompatible() const override;
    std::string getName() const override { return "T4Adapter"; }
    int getPriority() const override { return 90; }
    
    double executeShallowWaterStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executeBarotropicStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executePrimitiveEquationsStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executeGCMStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double calculateDiagnostics(WeatherGrid& grid) override;
    
private:
    int device_id_ = 0;
    DeviceCapabilities device_caps_;
};

// Adapter for Jetson Orin NX (SM 8.7)
class JetsonOrinAdapter : public KernelAdapter {
public:
    bool initialize(int device_id = 0) override;
    bool isCompatible() const override;
    std::string getName() const override { return "JetsonOrinAdapter"; }
    int getPriority() const override { return 95; }
    
    double executeShallowWaterStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executeBarotropicStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executePrimitiveEquationsStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executeGCMStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double calculateDiagnostics(WeatherGrid& grid) override;
    
private:
    int device_id_ = 0;
    DeviceCapabilities device_caps_;
};

// Generic CUDA adapter for other GPUs
class GenericCUDAAdapter : public KernelAdapter {
public:
    bool initialize(int device_id = 0) override;
    bool isCompatible() const override;
    std::string getName() const override { return "GenericCUDAAdapter"; }
    int getPriority() const override { return 50; }
    
    double executeShallowWaterStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executeBarotropicStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executePrimitiveEquationsStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executeGCMStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double calculateDiagnostics(WeatherGrid& grid) override;
    
private:
    int device_id_ = 0;
    DeviceCapabilities device_caps_;
};

// CPU adapter for fallback
class CPUAdapter : public KernelAdapter {
public:
    bool initialize(int device_id = 0) override;
    bool isCompatible() const override { return true; }  // Always compatible
    std::string getName() const override { return "CPUAdapter"; }
    int getPriority() const override { return 10; }  // Lowest priority
    
    double executeShallowWaterStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executeBarotropicStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executePrimitiveEquationsStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double executeGCMStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt
    ) override;
    
    double calculateDiagnostics(WeatherGrid& grid) override;
    
private:
    int num_threads_ = 0;  // 0 means use all available
};

// Helper class to manage hybrid CPU-GPU workload distribution
class HybridExecutionManager {
public:
    /**
     * @brief Get the singleton instance.
     * @return Reference to the singleton instance
     */
    static HybridExecutionManager& getInstance();
    
    /**
     * @brief Initialize the manager.
     * @param device_id CUDA device ID to use
     * @param num_cpu_threads Number of CPU threads to use (0 = auto)
     * @return True if initialization succeeded
     */
    bool initialize(int device_id = 0, int num_cpu_threads = 0);
    
    /**
     * @brief Execute a step using hybrid CPU-GPU execution.
     * @param in_grid Input grid
     * @param out_grid Output grid
     * @param dt Time step
     * @param simulation_model The simulation model to use
     * @return Execution time in milliseconds
     */
    double executeHybridStep(
        const WeatherGrid& in_grid,
        WeatherGrid& out_grid,
        scalar_t dt,
        SimulationModel simulation_model
    );
    
    /**
     * @brief Set the GPU workload ratio.
     * @param ratio Ratio of work to be done on GPU (0.0-1.0)
     */
    void setGpuWorkloadRatio(float ratio) {
        gpu_workload_ratio_ = std::max(0.0f, std::min(1.0f, ratio));
    }
    
    /**
     * @brief Get the current GPU workload ratio.
     * @return Current ratio
     */
    float getGpuWorkloadRatio() const { return gpu_workload_ratio_; }
    
    /**
     * @brief Dynamically adjust the workload ratio based on performance.
     * @param gpu_time_ms GPU execution time in milliseconds
     * @param cpu_time_ms CPU execution time in milliseconds
     */
    void adjustWorkloadRatio(double gpu_time_ms, double cpu_time_ms);
    
private:
    // Private constructor for singleton
    HybridExecutionManager();
    
    // No copy or move
    HybridExecutionManager(const HybridExecutionManager&) = delete;
    HybridExecutionManager& operator=(const HybridExecutionManager&) = delete;
    
    // Helper methods
    void splitWorkload(int total_work, int& gpu_work, int& cpu_work) const;
    
    // Member variables
    std::shared_ptr<KernelAdapter> gpu_adapter_;
    std::shared_ptr<CPUAdapter> cpu_adapter_;
    float gpu_workload_ratio_ = 0.9f;  // Default 90% on GPU
    bool initialized_ = false;
};

} // namespace weather_sim