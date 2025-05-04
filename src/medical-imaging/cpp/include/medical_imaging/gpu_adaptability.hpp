/**
 * @file gpu_adaptability.hpp
 * @brief GPU adaptability pattern for medical imaging.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <unordered_map>
#include <functional>

#include "medical_imaging.hpp"

namespace medical_imaging {

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
     * @param width Width of the problem
     * @param height Height of the problem
     * @param depth Depth of the problem (default: 1)
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
     * @param width Problem width
     * @param height Problem height
     * @param depth Problem depth
     * @param operation_name The name of the operation
     * @return Optimal compute backend
     */
    ComputeBackend determineOptimalBackend(
        int width, int height, int depth,
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
 * @brief Interface for memory management.
 */
class MemoryManager {
public:
    /**
     * @brief Get the singleton instance.
     * @return Reference to the singleton instance
     */
    static MemoryManager& getInstance();
    
    /**
     * @brief Initialize the memory manager.
     * @param device_id CUDA device ID to use
     * @return True if initialization succeeds
     */
    bool initialize(int device_id = 0);
    
    /**
     * @brief Allocate device memory.
     * @param size Size in bytes
     * @return Pointer to allocated memory, or nullptr on failure
     */
    void* allocateDevice(size_t size);
    
    /**
     * @brief Free device memory.
     * @param ptr Pointer to memory to free
     */
    void freeDevice(void* ptr);
    
    /**
     * @brief Copy host to device.
     * @param dst Destination pointer (device)
     * @param src Source pointer (host)
     * @param size Size in bytes
     * @return True if copy succeeds
     */
    bool copyHostToDevice(void* dst, const void* src, size_t size);
    
    /**
     * @brief Copy device to host.
     * @param dst Destination pointer (host)
     * @param src Source pointer (device)
     * @param size Size in bytes
     * @return True if copy succeeds
     */
    bool copyDeviceToHost(void* dst, const void* src, size_t size);
    
    /**
     * @brief Copy device to device.
     * @param dst Destination pointer (device)
     * @param src Source pointer (device)
     * @param size Size in bytes
     * @return True if copy succeeds
     */
    bool copyDeviceToDevice(void* dst, const void* src, size_t size);
    
    /**
     * @brief Create a CUDA stream.
     * @return Stream ID, or -1 on failure
     */
    int createStream();
    
    /**
     * @brief Destroy a CUDA stream.
     * @param stream_id Stream ID
     */
    void destroyStream(int stream_id);
    
    /**
     * @brief Synchronize a stream.
     * @param stream_id Stream ID
     */
    void synchronizeStream(int stream_id);
    
    /**
     * @brief Synchronize device.
     */
    void synchronizeDevice();
    
private:
    // Private constructor for singleton
    MemoryManager();
    
    // No copy or move
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    
    // Member variables
    bool initialized_ = false;
    int device_id_ = 0;
    
    // Stream management
    std::vector<void*> streams_;
    std::vector<int> free_stream_ids_;
};

/**
 * @brief Template for device image memory management.
 */
template <typename T>
class DeviceImage {
public:
    /**
     * @brief Constructor with size.
     * @param width Image width
     * @param height Image height
     * @param depth Image depth (default: 1)
     * @param channels Number of channels (default: 1)
     */
    DeviceImage(int width, int height, int depth = 1, int channels = 1);
    
    /**
     * @brief Construct from host image.
     * @param host_image Host image
     */
    explicit DeviceImage(const MedicalImage& host_image);
    
    /**
     * @brief Destructor.
     */
    ~DeviceImage();
    
    /**
     * @brief Copy data from host to device.
     * @param host_data Host data pointer
     * @return True if copy succeeds
     */
    bool copyFromHost(const T* host_data);
    
    /**
     * @brief Copy data from host image to device.
     * @param host_image Host image
     * @return True if copy succeeds
     */
    bool copyFromHostImage(const MedicalImage& host_image);
    
    /**
     * @brief Copy data from device to host.
     * @param host_data Host data pointer
     * @return True if copy succeeds
     */
    bool copyToHost(T* host_data) const;
    
    /**
     * @brief Copy data from device to host image.
     * @param host_image Host image
     * @return True if copy succeeds
     */
    bool copyToHostImage(MedicalImage& host_image) const;
    
    /**
     * @brief Create a host image from device data.
     * @return Host image
     */
    MedicalImage toHostImage() const;
    
    /**
     * @brief Get device data pointer.
     * @return Device data pointer
     */
    T* getData() { return data_; }
    
    /**
     * @brief Get device data pointer (const).
     * @return Const device data pointer
     */
    const T* getData() const { return data_; }
    
    /**
     * @brief Get image width.
     * @return Image width
     */
    int getWidth() const { return width_; }
    
    /**
     * @brief Get image height.
     * @return Image height
     */
    int getHeight() const { return height_; }
    
    /**
     * @brief Get image depth.
     * @return Image depth
     */
    int getDepth() const { return depth_; }
    
    /**
     * @brief Get number of channels.
     * @return Number of channels
     */
    int getChannels() const { return channels_; }
    
    /**
     * @brief Get total elements.
     * @return Total elements (width * height * depth * channels)
     */
    size_t getNumElements() const { return width_ * height_ * depth_ * channels_; }
    
private:
    T* data_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    int depth_ = 0;
    int channels_ = 0;
    
    // Free memory if allocated
    void freeMemory();
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
    
    // --- CT Reconstruction Kernels ---
    
    /**
     * @brief Execute filtered backprojection kernel.
     * @param projections Input projections
     * @param angles Projection angles
     * @param output Output image
     * @param filter_type Filter type (0: Ram-Lak, 1: Shepp-Logan, 2: Cosine, 3: Hamming)
     * @return Execution time in milliseconds
     */
    virtual double executeFilteredBackProjection(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int filter_type = 0
    ) = 0;
    
    /**
     * @brief Execute iterative reconstruction kernel (SIRT).
     * @param projections Input projections
     * @param angles Projection angles
     * @param output Output image
     * @param num_iterations Number of iterations
     * @return Execution time in milliseconds
     */
    virtual double executeIterativeReconstruction(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int num_iterations
    ) = 0;
    
    // --- MRI Reconstruction Kernels ---
    
    /**
     * @brief Execute FFT kernel.
     * @param input Input image (complex)
     * @param output Output image (complex)
     * @param inverse Whether to perform inverse FFT
     * @return Execution time in milliseconds
     */
    virtual double executeFFT(
        const DeviceImage<complex_t>& input,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) = 0;
    
    /**
     * @brief Execute non-Cartesian FFT kernel.
     * @param input Input k-space data (complex)
     * @param trajectories K-space trajectories
     * @param output Output image (complex)
     * @param inverse Whether to perform inverse operation
     * @return Execution time in milliseconds
     */
    virtual double executeNonCartesianFFT(
        const DeviceImage<complex_t>& input,
        const std::vector<std::pair<scalar_t, scalar_t>>& trajectories,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) = 0;
    
    // --- Image Processing Kernels ---
    
    /**
     * @brief Execute convolution kernel.
     * @param input Input image
     * @param kernel Convolution kernel
     * @param output Output image
     * @return Execution time in milliseconds
     */
    virtual double executeConvolution(
        const DeviceImage<scalar_t>& input,
        const std::vector<scalar_t>& kernel,
        DeviceImage<scalar_t>& output
    ) = 0;
    
    /**
     * @brief Execute median filter kernel.
     * @param input Input image
     * @param output Output image
     * @param radius Filter radius
     * @return Execution time in milliseconds
     */
    virtual double executeMedianFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int radius
    ) = 0;
    
    /**
     * @brief Execute bilateral filter kernel.
     * @param input Input image
     * @param output Output image
     * @param spatial_sigma Spatial sigma
     * @param range_sigma Range sigma
     * @param radius Filter radius
     * @return Execution time in milliseconds
     */
    virtual double executeBilateralFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t spatial_sigma,
        scalar_t range_sigma,
        int radius
    ) = 0;
    
    /**
     * @brief Execute non-local means filter kernel.
     * @param input Input image
     * @param output Output image
     * @param search_radius Search window radius
     * @param patch_radius Patch radius
     * @param h Filter parameter
     * @return Execution time in milliseconds
     */
    virtual double executeNLMFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int search_radius,
        int patch_radius,
        scalar_t h
    ) = 0;
    
    // --- Segmentation Kernels ---
    
    /**
     * @brief Execute thresholding kernel.
     * @param input Input image
     * @param output Output image
     * @param threshold Threshold value
     * @param max_value Maximum value for thresholding
     * @return Execution time in milliseconds
     */
    virtual double executeThresholding(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t threshold,
        scalar_t max_value = 1.0f
    ) = 0;
    
    /**
     * @brief Execute region growing kernel.
     * @param input Input image
     * @param output Output image
     * @param seeds Seed points
     * @param threshold Threshold for region growing
     * @return Execution time in milliseconds
     */
    virtual double executeRegionGrowing(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<std::vector<index_t>>& seeds,
        scalar_t threshold
    ) = 0;
    
    // --- Registration Kernels ---
    
    /**
     * @brief Execute image warping kernel.
     * @param input Input image
     * @param output Output image
     * @param transform_matrix Transformation matrix (row-major, 4x4 or 3x3)
     * @param interpolation_mode Interpolation mode (0: nearest, 1: linear, 2: cubic)
     * @return Execution time in milliseconds
     */
    virtual double executeImageWarping(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<scalar_t>& transform_matrix,
        int interpolation_mode = 1
    ) = 0;
    
    /**
     * @brief Execute mutual information kernel.
     * @param image1 First image
     * @param image2 Second image
     * @param num_bins Number of histogram bins
     * @param mi_value Output mutual information value
     * @return Execution time in milliseconds
     */
    virtual double executeMutualInformation(
        const DeviceImage<scalar_t>& image1,
        const DeviceImage<scalar_t>& image2,
        int num_bins,
        scalar_t& mi_value
    ) = 0;
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
    
    // CT Reconstruction
    double executeFilteredBackProjection(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int filter_type = 0
    ) override;
    
    double executeIterativeReconstruction(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int num_iterations
    ) override;
    
    // MRI Reconstruction
    double executeFFT(
        const DeviceImage<complex_t>& input,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) override;
    
    double executeNonCartesianFFT(
        const DeviceImage<complex_t>& input,
        const std::vector<std::pair<scalar_t, scalar_t>>& trajectories,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) override;
    
    // Image Processing
    double executeConvolution(
        const DeviceImage<scalar_t>& input,
        const std::vector<scalar_t>& kernel,
        DeviceImage<scalar_t>& output
    ) override;
    
    double executeMedianFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int radius
    ) override;
    
    double executeBilateralFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t spatial_sigma,
        scalar_t range_sigma,
        int radius
    ) override;
    
    double executeNLMFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int search_radius,
        int patch_radius,
        scalar_t h
    ) override;
    
    // Segmentation
    double executeThresholding(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t threshold,
        scalar_t max_value = 1.0f
    ) override;
    
    double executeRegionGrowing(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<std::vector<index_t>>& seeds,
        scalar_t threshold
    ) override;
    
    // Registration
    double executeImageWarping(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<scalar_t>& transform_matrix,
        int interpolation_mode = 1
    ) override;
    
    double executeMutualInformation(
        const DeviceImage<scalar_t>& image1,
        const DeviceImage<scalar_t>& image2,
        int num_bins,
        scalar_t& mi_value
    ) override;
    
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
    
    // CT Reconstruction
    double executeFilteredBackProjection(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int filter_type = 0
    ) override;
    
    double executeIterativeReconstruction(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int num_iterations
    ) override;
    
    // MRI Reconstruction
    double executeFFT(
        const DeviceImage<complex_t>& input,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) override;
    
    double executeNonCartesianFFT(
        const DeviceImage<complex_t>& input,
        const std::vector<std::pair<scalar_t, scalar_t>>& trajectories,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) override;
    
    // Image Processing
    double executeConvolution(
        const DeviceImage<scalar_t>& input,
        const std::vector<scalar_t>& kernel,
        DeviceImage<scalar_t>& output
    ) override;
    
    double executeMedianFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int radius
    ) override;
    
    double executeBilateralFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t spatial_sigma,
        scalar_t range_sigma,
        int radius
    ) override;
    
    double executeNLMFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int search_radius,
        int patch_radius,
        scalar_t h
    ) override;
    
    // Segmentation
    double executeThresholding(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t threshold,
        scalar_t max_value = 1.0f
    ) override;
    
    double executeRegionGrowing(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<std::vector<index_t>>& seeds,
        scalar_t threshold
    ) override;
    
    // Registration
    double executeImageWarping(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<scalar_t>& transform_matrix,
        int interpolation_mode = 1
    ) override;
    
    double executeMutualInformation(
        const DeviceImage<scalar_t>& image1,
        const DeviceImage<scalar_t>& image2,
        int num_bins,
        scalar_t& mi_value
    ) override;
    
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
    
    // CT Reconstruction
    double executeFilteredBackProjection(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int filter_type = 0
    ) override;
    
    double executeIterativeReconstruction(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int num_iterations
    ) override;
    
    // MRI Reconstruction
    double executeFFT(
        const DeviceImage<complex_t>& input,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) override;
    
    double executeNonCartesianFFT(
        const DeviceImage<complex_t>& input,
        const std::vector<std::pair<scalar_t, scalar_t>>& trajectories,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) override;
    
    // Image Processing
    double executeConvolution(
        const DeviceImage<scalar_t>& input,
        const std::vector<scalar_t>& kernel,
        DeviceImage<scalar_t>& output
    ) override;
    
    double executeMedianFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int radius
    ) override;
    
    double executeBilateralFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t spatial_sigma,
        scalar_t range_sigma,
        int radius
    ) override;
    
    double executeNLMFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int search_radius,
        int patch_radius,
        scalar_t h
    ) override;
    
    // Segmentation
    double executeThresholding(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t threshold,
        scalar_t max_value = 1.0f
    ) override;
    
    double executeRegionGrowing(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<std::vector<index_t>>& seeds,
        scalar_t threshold
    ) override;
    
    // Registration
    double executeImageWarping(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<scalar_t>& transform_matrix,
        int interpolation_mode = 1
    ) override;
    
    double executeMutualInformation(
        const DeviceImage<scalar_t>& image1,
        const DeviceImage<scalar_t>& image2,
        int num_bins,
        scalar_t& mi_value
    ) override;
    
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
    
    // CT Reconstruction
    double executeFilteredBackProjection(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int filter_type = 0
    ) override;
    
    double executeIterativeReconstruction(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int num_iterations
    ) override;
    
    // MRI Reconstruction
    double executeFFT(
        const DeviceImage<complex_t>& input,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) override;
    
    double executeNonCartesianFFT(
        const DeviceImage<complex_t>& input,
        const std::vector<std::pair<scalar_t, scalar_t>>& trajectories,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) override;
    
    // Image Processing
    double executeConvolution(
        const DeviceImage<scalar_t>& input,
        const std::vector<scalar_t>& kernel,
        DeviceImage<scalar_t>& output
    ) override;
    
    double executeMedianFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int radius
    ) override;
    
    double executeBilateralFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t spatial_sigma,
        scalar_t range_sigma,
        int radius
    ) override;
    
    double executeNLMFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int search_radius,
        int patch_radius,
        scalar_t h
    ) override;
    
    // Segmentation
    double executeThresholding(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t threshold,
        scalar_t max_value = 1.0f
    ) override;
    
    double executeRegionGrowing(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<std::vector<index_t>>& seeds,
        scalar_t threshold
    ) override;
    
    // Registration
    double executeImageWarping(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<scalar_t>& transform_matrix,
        int interpolation_mode = 1
    ) override;
    
    double executeMutualInformation(
        const DeviceImage<scalar_t>& image1,
        const DeviceImage<scalar_t>& image2,
        int num_bins,
        scalar_t& mi_value
    ) override;
    
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
    
    // CT Reconstruction
    double executeFilteredBackProjection(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int filter_type = 0
    ) override;
    
    double executeIterativeReconstruction(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int num_iterations
    ) override;
    
    // MRI Reconstruction
    double executeFFT(
        const DeviceImage<complex_t>& input,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) override;
    
    double executeNonCartesianFFT(
        const DeviceImage<complex_t>& input,
        const std::vector<std::pair<scalar_t, scalar_t>>& trajectories,
        DeviceImage<complex_t>& output,
        bool inverse = false
    ) override;
    
    // Image Processing
    double executeConvolution(
        const DeviceImage<scalar_t>& input,
        const std::vector<scalar_t>& kernel,
        DeviceImage<scalar_t>& output
    ) override;
    
    double executeMedianFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int radius
    ) override;
    
    double executeBilateralFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t spatial_sigma,
        scalar_t range_sigma,
        int radius
    ) override;
    
    double executeNLMFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int search_radius,
        int patch_radius,
        scalar_t h
    ) override;
    
    // Segmentation
    double executeThresholding(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t threshold,
        scalar_t max_value = 1.0f
    ) override;
    
    double executeRegionGrowing(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<std::vector<index_t>>& seeds,
        scalar_t threshold
    ) override;
    
    // Registration
    double executeImageWarping(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        const std::vector<scalar_t>& transform_matrix,
        int interpolation_mode = 1
    ) override;
    
    double executeMutualInformation(
        const DeviceImage<scalar_t>& image1,
        const DeviceImage<scalar_t>& image2,
        int num_bins,
        scalar_t& mi_value
    ) override;
    
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
    
    /**
     * @brief Get the best kernel adapter.
     * @return Shared pointer to the best kernel adapter
     */
    std::shared_ptr<KernelAdapter> getBestAdapter() { return gpu_adapter_; }
    
    /**
     * @brief Get the CPU adapter.
     * @return Shared pointer to the CPU adapter
     */
    std::shared_ptr<CPUAdapter> getCPUAdapter() { return cpu_adapter_; }
    
    // --- Hybrid execution methods ---
    
    /**
     * @brief Execute filtered backprojection using hybrid CPU-GPU approach.
     * @param projections Input projections
     * @param angles Projection angles
     * @param output Output image
     * @param filter_type Filter type
     * @return Execution time in milliseconds
     */
    double executeHybridFilteredBackProjection(
        const DeviceImage<scalar_t>& projections,
        const std::vector<scalar_t>& angles,
        DeviceImage<scalar_t>& output,
        int filter_type = 0
    );
    
    /**
     * @brief Execute bilateral filter using hybrid CPU-GPU approach.
     * @param input Input image
     * @param output Output image
     * @param spatial_sigma Spatial sigma
     * @param range_sigma Range sigma
     * @param radius Filter radius
     * @return Execution time in milliseconds
     */
    double executeHybridBilateralFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        scalar_t spatial_sigma,
        scalar_t range_sigma,
        int radius
    );
    
    /**
     * @brief Execute non-local means filter using hybrid CPU-GPU approach.
     * @param input Input image
     * @param output Output image
     * @param search_radius Search window radius
     * @param patch_radius Patch radius
     * @param h Filter parameter
     * @return Execution time in milliseconds
     */
    double executeHybridNLMFilter(
        const DeviceImage<scalar_t>& input,
        DeviceImage<scalar_t>& output,
        int search_radius,
        int patch_radius,
        scalar_t h
    );
    
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

} // namespace medical_imaging