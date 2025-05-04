/**
 * @file medical_imaging.hpp
 * @brief Medical imaging workload for GPU-accelerated image processing and analysis.
 * @author Scott Friedman
 * @copyright 2025 Scott Friedman. All rights reserved.
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <random>
#include <chrono>
#include <iostream>
#include <array>
#include <complex>
#include <unordered_map>

namespace medical_imaging {

/**
 * @brief Common types and utilities for medical imaging.
 */
using scalar_t = float; // Use float for better GPU performance
using complex_t = std::complex<scalar_t>;
using index_t = int32_t; // Use 32-bit integers for indexing

/**
 * @brief Enumeration for image dimensions.
 */
enum class ImageDimension {
    D2,    // 2D image
    D3,    // 3D volume
    D4     // 4D volume (e.g., time series)
};

/**
 * @brief Enumeration for image types.
 */
enum class ImageType {
    Grayscale,    // Single channel grayscale
    RGB,          // 3-channel RGB
    RGBA,         // 4-channel RGBA
    Complex,      // Complex valued (for frequency domain)
    MultiChannel  // N-channel (for general case)
};

/**
 * @brief Enumeration for reconstruction methods.
 */
enum class ReconstructionMethod {
    FilteredBackProjection,  // Filtered backprojection (CT)
    IterativePrimalDual,     // Primal-dual iterative reconstruction
    CompressedSensing,       // Compressed sensing reconstruction
    DeepLearning             // Deep learning based reconstruction
};

/**
 * @brief Enumeration for image filtering methods.
 */
enum class FilterMethod {
    Gaussian,           // Gaussian smoothing
    Median,             // Median filter
    Bilateral,          // Bilateral filter
    Anisotropic,        // Anisotropic diffusion
    NonLocalMeans,      // Non-local means
    BM3D,               // Block-matching 3D
    DeepDenoise         // Deep learning based denoising
};

/**
 * @brief Enumeration for image segmentation methods.
 */
enum class SegmentationMethod {
    Thresholding,       // Simple thresholding
    RegionGrowing,      // Region growing
    Watershed,          // Watershed segmentation
    ActiveContour,      // Active contour / Level set
    GraphCut,           // Graph cut
    DeepSegmentation    // Deep learning based segmentation
};

/**
 * @brief Enumeration for computational backends.
 */
enum class ComputeBackend {
    CUDA,          // CUDA GPU backend
    CPU,           // CPU backend
    Hybrid,        // Hybrid CPU-GPU
    AdaptiveHybrid // Adaptive hybrid (dynamically balances workload)
};

/**
 * @brief Configuration for a medical imaging processing task.
 */
struct ProcessingConfig {
    // Image parameters
    ImageDimension dimension = ImageDimension::D2;
    ImageType image_type = ImageType::Grayscale;
    std::vector<index_t> image_size = {512, 512}; // Width, height, (depth, time)
    index_t channels = 1;
    
    // Computational parameters
    ComputeBackend compute_backend = ComputeBackend::CUDA;
    bool double_precision = false; // Use double precision (affects accuracy vs. performance)
    int device_id = 0;           // GPU device ID
    int num_threads = 0;         // Number of CPU threads (0 = auto)
    
    // Task-specific parameters
    std::unordered_map<std::string, scalar_t> scalar_params;
    std::unordered_map<std::string, std::string> string_params;
    std::unordered_map<std::string, std::vector<scalar_t>> vector_params;
    
    // Output control
    bool save_intermediate = false;
    std::string output_path = "./output";
    
    // Random seed for reproducibility
    unsigned int random_seed = std::random_device{}();
};

/**
 * @brief Performance metrics for the processing task.
 */
struct PerformanceMetrics {
    double total_time_ms = 0.0;
    double compute_time_ms = 0.0;
    double memory_transfer_time_ms = 0.0;
    double io_time_ms = 0.0;
    double preprocessing_time_ms = 0.0;
    double processing_time_ms = 0.0;
    double postprocessing_time_ms = 0.0;
    int num_iterations = 0;
    
    void reset() {
        total_time_ms = 0.0;
        compute_time_ms = 0.0;
        memory_transfer_time_ms = 0.0;
        io_time_ms = 0.0;
        preprocessing_time_ms = 0.0;
        processing_time_ms = 0.0;
        postprocessing_time_ms = 0.0;
        num_iterations = 0;
    }
    
    void print() const {
        std::cout << "Performance Metrics:" << std::endl;
        std::cout << "  Total time: " << total_time_ms << " ms" << std::endl;
        
        if (total_time_ms > 0) {
            auto percentage = [this](double time) { return (time / total_time_ms * 100.0); };
            
            std::cout << "  Compute time: " << compute_time_ms << " ms (" 
                      << percentage(compute_time_ms) << "%)" << std::endl;
            std::cout << "  Memory transfer time: " << memory_transfer_time_ms << " ms ("
                      << percentage(memory_transfer_time_ms) << "%)" << std::endl;
            std::cout << "  I/O time: " << io_time_ms << " ms ("
                      << percentage(io_time_ms) << "%)" << std::endl;
            std::cout << "  Preprocessing time: " << preprocessing_time_ms << " ms ("
                      << percentage(preprocessing_time_ms) << "%)" << std::endl;
            std::cout << "  Processing time: " << processing_time_ms << " ms ("
                      << percentage(processing_time_ms) << "%)" << std::endl;
            std::cout << "  Postprocessing time: " << postprocessing_time_ms << " ms ("
                      << percentage(postprocessing_time_ms) << "%)" << std::endl;
        }
        
        if (num_iterations > 0) {
            std::cout << "  Iterations: " << num_iterations << std::endl;
            std::cout << "  Time per iteration: " << (total_time_ms / num_iterations) << " ms" << std::endl;
        }
    }
};

// Forward declarations
class MedicalImage;
class ImageProcessor;
class CTReconstructor;
class MRIReconstructor;
class ImageFilter;
class ImageSegmenter;
class ImageRegistration;
class Visualization;

/**
 * @brief Class representing a medical image.
 * 
 * This class can handle 2D, 3D, and 4D images with various types
 * (grayscale, RGB, complex, etc.)
 */
class MedicalImage {
public:
    /**
     * @brief Default constructor.
     */
    MedicalImage() = default;
    
    /**
     * @brief Construct a new Medical Image.
     * @param size Image dimensions (width, height, depth, time)
     * @param dimension Dimensionality (2D, 3D, 4D)
     * @param type Image type
     * @param channels Number of channels
     */
    MedicalImage(
        const std::vector<index_t>& size,
        ImageDimension dimension = ImageDimension::D2,
        ImageType type = ImageType::Grayscale,
        index_t channels = 1
    );
    
    /**
     * @brief Construct an image from raw data.
     * @param data Raw image data
     * @param size Image dimensions
     * @param dimension Dimensionality
     * @param type Image type
     * @param channels Number of channels
     */
    MedicalImage(
        const std::vector<scalar_t>& data,
        const std::vector<index_t>& size,
        ImageDimension dimension = ImageDimension::D2,
        ImageType type = ImageType::Grayscale,
        index_t channels = 1
    );
    
    /**
     * @brief Construct an image from file.
     * @param filename Path to image file
     */
    explicit MedicalImage(const std::string& filename);
    
    /**
     * @brief Destroy the Medical Image object.
     */
    ~MedicalImage() = default;
    
    /**
     * @brief Load image from file.
     * @param filename Path to image file
     * @return True if successful
     */
    bool load(const std::string& filename);
    
    /**
     * @brief Save image to file.
     * @param filename Path to save to
     * @param format File format (e.g., "png", "jpg", "nii", "dicom")
     * @return True if successful
     */
    bool save(const std::string& filename, const std::string& format = "") const;
    
    /**
     * @brief Get image dimensions.
     * @return Vector of dimensions (width, height, depth, time)
     */
    const std::vector<index_t>& getSize() const { return size_; }
    
    /**
     * @brief Get image dimensionality.
     * @return Image dimension (2D, 3D, 4D)
     */
    ImageDimension getDimension() const { return dimension_; }
    
    /**
     * @brief Get image type.
     * @return Image type
     */
    ImageType getType() const { return type_; }
    
    /**
     * @brief Get number of channels.
     * @return Number of channels
     */
    index_t getChannels() const { return channels_; }
    
    /**
     * @brief Get total number of elements.
     * @return Total elements (width * height * depth * time * channels)
     */
    size_t getNumElements() const { return data_.size(); }
    
    /**
     * @brief Check if the image is empty.
     * @return True if empty
     */
    bool isEmpty() const { return data_.empty(); }
    
    /**
     * @brief Get raw data pointer.
     * @return Pointer to raw data
     */
    scalar_t* getData() { return data_.data(); }
    
    /**
     * @brief Get raw data pointer (const).
     * @return Const pointer to raw data
     */
    const scalar_t* getData() const { return data_.data(); }
    
    /**
     * @brief Get raw data vector.
     * @return Reference to raw data vector
     */
    std::vector<scalar_t>& getDataVector() { return data_; }
    
    /**
     * @brief Get raw data vector (const).
     * @return Const reference to raw data vector
     */
    const std::vector<scalar_t>& getDataVector() const { return data_; }
    
    /**
     * @brief Get pixel/voxel value at specified indices.
     * @param indices Indices (x, y, z, t)
     * @param channel Channel index
     * @return Pixel/voxel value
     */
    scalar_t getValue(const std::vector<index_t>& indices, index_t channel = 0) const;
    
    /**
     * @brief Set pixel/voxel value at specified indices.
     * @param indices Indices (x, y, z, t)
     * @param value New value
     * @param channel Channel index
     */
    void setValue(const std::vector<index_t>& indices, scalar_t value, index_t channel = 0);
    
    /**
     * @brief Get pixel value at 2D coordinates.
     * @param x X coordinate
     * @param y Y coordinate
     * @param channel Channel index
     * @return Pixel value
     */
    scalar_t getPixel(index_t x, index_t y, index_t channel = 0) const;
    
    /**
     * @brief Set pixel value at 2D coordinates.
     * @param x X coordinate
     * @param y Y coordinate
     * @param value New value
     * @param channel Channel index
     */
    void setPixel(index_t x, index_t y, scalar_t value, index_t channel = 0);
    
    /**
     * @brief Get voxel value at 3D coordinates.
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @param channel Channel index
     * @return Voxel value
     */
    scalar_t getVoxel(index_t x, index_t y, index_t z, index_t channel = 0) const;
    
    /**
     * @brief Set voxel value at 3D coordinates.
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @param value New value
     * @param channel Channel index
     */
    void setVoxel(index_t x, index_t y, index_t z, scalar_t value, index_t channel = 0);
    
    /**
     * @brief Get 4D voxel value.
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @param t Time point
     * @param channel Channel index
     * @return 4D voxel value
     */
    scalar_t getVoxel4D(index_t x, index_t y, index_t z, index_t t, index_t channel = 0) const;
    
    /**
     * @brief Set 4D voxel value.
     * @param x X coordinate
     * @param y Y coordinate
     * @param z Z coordinate
     * @param t Time point
     * @param value New value
     * @param channel Channel index
     */
    void setVoxel4D(index_t x, index_t y, index_t z, index_t t, scalar_t value, index_t channel = 0);
    
    /**
     * @brief Get image statistics.
     * @param channel Channel index
     * @return Tuple of (min, max, mean, stddev)
     */
    std::tuple<scalar_t, scalar_t, scalar_t, scalar_t> getStatistics(index_t channel = 0) const;
    
    /**
     * @brief Normalize image to [0, 1] range.
     * @param channel Channel to normalize (-1 for all channels)
     */
    void normalize(index_t channel = -1);
    
    /**
     * @brief Resize image.
     * @param new_size New dimensions
     * @param interpolation Interpolation method (0: nearest, 1: linear, 2: cubic)
     * @return Resized image
     */
    MedicalImage resize(const std::vector<index_t>& new_size, int interpolation = 1) const;
    
    /**
     * @brief Extract a slice from 3D/4D image.
     * @param dimension Dimension to slice along (0: x, 1: y, 2: z, 3: t)
     * @param index Index along the dimension
     * @return 2D/3D slice
     */
    MedicalImage extractSlice(index_t dimension, index_t index) const;
    
    /**
     * @brief Convert to different type.
     * @param new_type Target image type
     * @return Converted image
     */
    MedicalImage convertType(ImageType new_type) const;
    
    /**
     * @brief Apply window/level adjustment.
     * @param window Window width
     * @param level Window center
     */
    void applyWindowLevel(scalar_t window, scalar_t level);
    
    /**
     * @brief Create a deep copy of the image.
     * @return Copied image
     */
    MedicalImage clone() const;
    
    /**
     * @brief Get metadata value.
     * @param key Metadata key
     * @param default_value Default value if key not found
     * @return Metadata value
     */
    template <typename T>
    T getMetadata(const std::string& key, const T& default_value) const;
    
    /**
     * @brief Set metadata value.
     * @param key Metadata key
     * @param value Metadata value
     */
    template <typename T>
    void setMetadata(const std::string& key, const T& value);
    
private:
    std::vector<scalar_t> data_;       // Raw image data
    std::vector<index_t> size_;        // Image dimensions
    ImageDimension dimension_;         // Image dimension (2D, 3D, 4D)
    ImageType type_;                   // Image type
    index_t channels_;                 // Number of channels
    
    // Metadata
    std::unordered_map<std::string, std::string> metadata_;
    
    // Helper methods
    size_t calculateIndex(const std::vector<index_t>& indices, index_t channel) const;
    bool isValidIndex(const std::vector<index_t>& indices) const;
    void initializeImage(const std::vector<index_t>& size, ImageDimension dimension, 
                         ImageType type, index_t channels);
};

/**
 * @brief Base class for medical image processors.
 */
class ImageProcessor {
public:
    /**
     * @brief Constructor with configuration.
     * @param config Processing configuration
     */
    explicit ImageProcessor(const ProcessingConfig& config);
    
    /**
     * @brief Virtual destructor.
     */
    virtual ~ImageProcessor() = default;
    
    /**
     * @brief Initialize the processor.
     * @return True if initialization succeeds
     */
    virtual bool initialize() = 0;
    
    /**
     * @brief Process an image.
     * @param input Input image
     * @return Processed image
     */
    virtual MedicalImage process(const MedicalImage& input) = 0;
    
    /**
     * @brief Get the current configuration.
     * @return Current configuration
     */
    const ProcessingConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set a new configuration.
     * @param config New configuration
     */
    void setConfig(const ProcessingConfig& config) { config_ = config; }
    
    /**
     * @brief Get performance metrics.
     * @return Performance metrics
     */
    const PerformanceMetrics& getPerformanceMetrics() const { return metrics_; }
    
    /**
     * @brief Reset performance metrics.
     */
    void resetPerformanceMetrics() { metrics_.reset(); }
    
    /**
     * @brief Set a string parameter.
     * @param name Parameter name
     * @param value Parameter value
     */
    void setStringParameter(const std::string& name, const std::string& value) {
        config_.string_params[name] = value;
    }
    
    /**
     * @brief Set a scalar parameter.
     * @param name Parameter name
     * @param value Parameter value
     */
    void setScalarParameter(const std::string& name, scalar_t value) {
        config_.scalar_params[name] = value;
    }
    
    /**
     * @brief Set a vector parameter.
     * @param name Parameter name
     * @param value Parameter value
     */
    void setVectorParameter(const std::string& name, const std::vector<scalar_t>& value) {
        config_.vector_params[name] = value;
    }
    
    /**
     * @brief Get a string parameter.
     * @param name Parameter name
     * @param default_value Default value if parameter not found
     * @return Parameter value
     */
    std::string getStringParameter(const std::string& name, const std::string& default_value) const {
        auto it = config_.string_params.find(name);
        return (it != config_.string_params.end()) ? it->second : default_value;
    }
    
    /**
     * @brief Get a scalar parameter.
     * @param name Parameter name
     * @param default_value Default value if parameter not found
     * @return Parameter value
     */
    scalar_t getScalarParameter(const std::string& name, scalar_t default_value) const {
        auto it = config_.scalar_params.find(name);
        return (it != config_.scalar_params.end()) ? it->second : default_value;
    }
    
    /**
     * @brief Get a vector parameter.
     * @param name Parameter name
     * @param default_value Default value if parameter not found
     * @return Parameter value
     */
    std::vector<scalar_t> getVectorParameter(
        const std::string& name, 
        const std::vector<scalar_t>& default_value
    ) const {
        auto it = config_.vector_params.find(name);
        return (it != config_.vector_params.end()) ? it->second : default_value;
    }
    
protected:
    ProcessingConfig config_;    // Processing configuration
    PerformanceMetrics metrics_; // Performance metrics
    
    // Utility method to measure execution time
    template <typename Func>
    double measureExecutionTime(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        std::forward<Func>(func)();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Helper to check if CUDA is available
    bool isCudaAvailable() const;
};

/**
 * @brief Class for CT image reconstruction.
 */
class CTReconstructor : public ImageProcessor {
public:
    /**
     * @brief Constructor with configuration.
     * @param config Processing configuration
     */
    explicit CTReconstructor(const ProcessingConfig& config);
    
    /**
     * @brief Initialize the reconstructor.
     * @return True if initialization succeeds
     */
    bool initialize() override;
    
    /**
     * @brief Reconstruct CT image from projections.
     * @param projections Projection data
     * @return Reconstructed image
     */
    MedicalImage process(const MedicalImage& projections) override;
    
    /**
     * @brief Set reconstruction method.
     * @param method Reconstruction method
     */
    void setMethod(ReconstructionMethod method) {
        method_ = method;
    }
    
    /**
     * @brief Get reconstruction method.
     * @return Current reconstruction method
     */
    ReconstructionMethod getMethod() const { return method_; }
    
    /**
     * @brief Set projection angles.
     * @param angles Vector of projection angles (in radians)
     */
    void setProjectionAngles(const std::vector<scalar_t>& angles) {
        projection_angles_ = angles;
    }
    
    /**
     * @brief Get projection angles.
     * @return Vector of projection angles
     */
    const std::vector<scalar_t>& getProjectionAngles() const {
        return projection_angles_;
    }
    
    /**
     * @brief Set the number of iterations for iterative methods.
     * @param iterations Number of iterations
     */
    void setNumIterations(int iterations) {
        num_iterations_ = iterations;
    }
    
    /**
     * @brief Get the number of iterations.
     * @return Number of iterations
     */
    int getNumIterations() const { return num_iterations_; }
    
private:
    ReconstructionMethod method_ = ReconstructionMethod::FilteredBackProjection;
    std::vector<scalar_t> projection_angles_; // Projection angles in radians
    std::vector<scalar_t> filter_coeffs_;     // Filter coefficients for FBP
    int num_iterations_ = 10;                 // Number of iterations for iterative methods
    
    // Implementation of different reconstruction methods
    MedicalImage reconstructFBP(const MedicalImage& projections);
    MedicalImage reconstructIterativePrimalDual(const MedicalImage& projections);
    MedicalImage reconstructCompressedSensing(const MedicalImage& projections);
    MedicalImage reconstructDeepLearning(const MedicalImage& projections);
    
    // Helper methods
    void computeFilterCoefficients();
    std::vector<scalar_t> applyFilter(const std::vector<scalar_t>& input);
    std::vector<scalar_t> backproject(const std::vector<scalar_t>& filtered, scalar_t angle);
};

/**
 * @brief Class for MRI image reconstruction.
 */
class MRIReconstructor : public ImageProcessor {
public:
    /**
     * @brief Constructor with configuration.
     * @param config Processing configuration
     */
    explicit MRIReconstructor(const ProcessingConfig& config);
    
    /**
     * @brief Initialize the reconstructor.
     * @return True if initialization succeeds
     */
    bool initialize() override;
    
    /**
     * @brief Reconstruct MRI image from k-space data.
     * @param kspace K-space data
     * @return Reconstructed image
     */
    MedicalImage process(const MedicalImage& kspace) override;
    
    /**
     * @brief Set reconstruction method.
     * @param method Reconstruction method
     */
    void setMethod(ReconstructionMethod method) {
        method_ = method;
    }
    
    /**
     * @brief Get reconstruction method.
     * @return Current reconstruction method
     */
    ReconstructionMethod getMethod() const { return method_; }
    
    /**
     * @brief Set the number of iterations for iterative methods.
     * @param iterations Number of iterations
     */
    void setNumIterations(int iterations) {
        num_iterations_ = iterations;
    }
    
    /**
     * @brief Get the number of iterations.
     * @return Number of iterations
     */
    int getNumIterations() const { return num_iterations_; }
    
    /**
     * @brief Set the acceleration factor for parallel imaging.
     * @param factor Acceleration factor
     */
    void setAccelerationFactor(int factor) {
        acceleration_factor_ = factor;
    }
    
    /**
     * @brief Get the acceleration factor.
     * @return Acceleration factor
     */
    int getAccelerationFactor() const { return acceleration_factor_; }
    
    /**
     * @brief Set the sensitivity maps for parallel imaging.
     * @param maps Sensitivity maps
     */
    void setSensitivityMaps(const std::vector<MedicalImage>& maps) {
        sensitivity_maps_ = maps;
    }
    
private:
    ReconstructionMethod method_ = ReconstructionMethod::IterativePrimalDual;
    int num_iterations_ = 10;                  // Number of iterations for iterative methods
    int acceleration_factor_ = 1;              // Acceleration factor for parallel imaging
    std::vector<MedicalImage> sensitivity_maps_; // Sensitivity maps for parallel imaging
    
    // Implementation of different reconstruction methods
    MedicalImage reconstructFFT(const MedicalImage& kspace);
    MedicalImage reconstructCompressedSensing(const MedicalImage& kspace);
    MedicalImage reconstructIterativePrimalDual(const MedicalImage& kspace);
    MedicalImage reconstructDeepLearning(const MedicalImage& kspace);
    
    // Helper methods
    std::vector<complex_t> applyFFT(const std::vector<complex_t>& input, bool inverse);
    std::vector<complex_t> applyIFFT(const std::vector<complex_t>& input) {
        return applyFFT(input, true);
    }
};

/**
 * @brief Class for image filtering.
 */
class ImageFilter : public ImageProcessor {
public:
    /**
     * @brief Constructor with configuration.
     * @param config Processing configuration
     */
    explicit ImageFilter(const ProcessingConfig& config);
    
    /**
     * @brief Initialize the filter.
     * @return True if initialization succeeds
     */
    bool initialize() override;
    
    /**
     * @brief Apply filter to an image.
     * @param input Input image
     * @return Filtered image
     */
    MedicalImage process(const MedicalImage& input) override;
    
    /**
     * @brief Set filter method.
     * @param method Filter method
     */
    void setMethod(FilterMethod method) {
        method_ = method;
    }
    
    /**
     * @brief Get filter method.
     * @return Current filter method
     */
    FilterMethod getMethod() const { return method_; }
    
    /**
     * @brief Set filter parameters.
     * @param params Filter parameters
     */
    void setFilterParameters(const std::unordered_map<std::string, scalar_t>& params) {
        filter_params_ = params;
    }
    
private:
    FilterMethod method_ = FilterMethod::Gaussian;
    std::unordered_map<std::string, scalar_t> filter_params_;
    
    // Implementation of different filter methods
    MedicalImage applyGaussianFilter(const MedicalImage& input);
    MedicalImage applyMedianFilter(const MedicalImage& input);
    MedicalImage applyBilateralFilter(const MedicalImage& input);
    MedicalImage applyAnisotropicFilter(const MedicalImage& input);
    MedicalImage applyNLMFilter(const MedicalImage& input);
    MedicalImage applyBM3DFilter(const MedicalImage& input);
    MedicalImage applyDeepDenoise(const MedicalImage& input);
};

/**
 * @brief Class for image segmentation.
 */
class ImageSegmenter : public ImageProcessor {
public:
    /**
     * @brief Constructor with configuration.
     * @param config Processing configuration
     */
    explicit ImageSegmenter(const ProcessingConfig& config);
    
    /**
     * @brief Initialize the segmenter.
     * @return True if initialization succeeds
     */
    bool initialize() override;
    
    /**
     * @brief Segment an image.
     * @param input Input image
     * @return Segmented image
     */
    MedicalImage process(const MedicalImage& input) override;
    
    /**
     * @brief Set segmentation method.
     * @param method Segmentation method
     */
    void setMethod(SegmentationMethod method) {
        method_ = method;
    }
    
    /**
     * @brief Get segmentation method.
     * @return Current segmentation method
     */
    SegmentationMethod getMethod() const { return method_; }
    
    /**
     * @brief Set the number of segments to extract.
     * @param num_segments Number of segments
     */
    void setNumSegments(int num_segments) {
        num_segments_ = num_segments;
    }
    
    /**
     * @brief Get the number of segments.
     * @return Number of segments
     */
    int getNumSegments() const { return num_segments_; }
    
    /**
     * @brief Set seed points for region growing.
     * @param seeds Seed points (x, y, z)
     */
    void setSeedPoints(const std::vector<std::vector<index_t>>& seeds) {
        seed_points_ = seeds;
    }
    
private:
    SegmentationMethod method_ = SegmentationMethod::Thresholding;
    int num_segments_ = 2;
    std::vector<std::vector<index_t>> seed_points_;
    
    // Implementation of different segmentation methods
    MedicalImage applyThresholding(const MedicalImage& input);
    MedicalImage applyRegionGrowing(const MedicalImage& input);
    MedicalImage applyWatershed(const MedicalImage& input);
    MedicalImage applyActiveContour(const MedicalImage& input);
    MedicalImage applyGraphCut(const MedicalImage& input);
    MedicalImage applyDeepSegmentation(const MedicalImage& input);
};

/**
 * @brief Class for image registration.
 */
class ImageRegistration : public ImageProcessor {
public:
    /**
     * @brief Constructor with configuration.
     * @param config Processing configuration
     */
    explicit ImageRegistration(const ProcessingConfig& config);
    
    /**
     * @brief Initialize the registration.
     * @return True if initialization succeeds
     */
    bool initialize() override;
    
    /**
     * @brief Register moving image to fixed image.
     * @param fixed Fixed image
     * @param moving Moving image
     * @return Registered moving image
     */
    MedicalImage registerImages(const MedicalImage& fixed, const MedicalImage& moving);
    
    /**
     * @brief Override process method to throw an error.
     * @param input Input image
     * @return Processed image
     */
    MedicalImage process(const MedicalImage& input) override {
        throw std::runtime_error("ImageRegistration requires both fixed and moving images. Use registerImages instead.");
    }
    
    /**
     * @brief Get transformation parameters.
     * @return Transformation parameters
     */
    std::vector<scalar_t> getTransformParameters() const {
        return transform_params_;
    }
    
    /**
     * @brief Get transformation matrix.
     * @return 4x4 transformation matrix
     */
    std::vector<std::vector<scalar_t>> getTransformMatrix() const {
        return transform_matrix_;
    }
    
private:
    enum class RegistrationMethod {
        Rigid,               // Rigid registration (rotation, translation)
        Affine,              // Affine registration (+ scaling, shearing)
        DeformableElastic,   // Elastic deformation
        DeformableBSpline,   // B-spline deformation
        DeepLearning         // Deep learning based registration
    };
    
    RegistrationMethod method_ = RegistrationMethod::Rigid;
    std::vector<scalar_t> transform_params_;
    std::vector<std::vector<scalar_t>> transform_matrix_;
    
    // Implementation of different registration methods
    MedicalImage applyRigidRegistration(const MedicalImage& fixed, const MedicalImage& moving);
    MedicalImage applyAffineRegistration(const MedicalImage& fixed, const MedicalImage& moving);
    MedicalImage applyElasticRegistration(const MedicalImage& fixed, const MedicalImage& moving);
    MedicalImage applyBSplineRegistration(const MedicalImage& fixed, const MedicalImage& moving);
    MedicalImage applyDeepRegistration(const MedicalImage& fixed, const MedicalImage& moving);
    
    // Helper methods
    scalar_t calculateSimilarity(const MedicalImage& img1, const MedicalImage& img2);
    MedicalImage applyTransform(const MedicalImage& image, const std::vector<scalar_t>& params);
};

/**
 * @brief Factory for creating image processors.
 */
class ProcessorFactory {
public:
    /**
     * @brief Get the singleton instance.
     * @return Reference to the singleton instance
     */
    static ProcessorFactory& getInstance();
    
    /**
     * @brief Create a CT reconstructor.
     * @param config Processing configuration
     * @return Shared pointer to CT reconstructor
     */
    std::shared_ptr<CTReconstructor> createCTReconstructor(const ProcessingConfig& config);
    
    /**
     * @brief Create an MRI reconstructor.
     * @param config Processing configuration
     * @return Shared pointer to MRI reconstructor
     */
    std::shared_ptr<MRIReconstructor> createMRIReconstructor(const ProcessingConfig& config);
    
    /**
     * @brief Create an image filter.
     * @param config Processing configuration
     * @return Shared pointer to image filter
     */
    std::shared_ptr<ImageFilter> createImageFilter(const ProcessingConfig& config);
    
    /**
     * @brief Create an image segmenter.
     * @param config Processing configuration
     * @return Shared pointer to image segmenter
     */
    std::shared_ptr<ImageSegmenter> createImageSegmenter(const ProcessingConfig& config);
    
    /**
     * @brief Create an image registration.
     * @param config Processing configuration
     * @return Shared pointer to image registration
     */
    std::shared_ptr<ImageRegistration> createImageRegistration(const ProcessingConfig& config);
    
private:
    // Private constructor for singleton
    ProcessorFactory() = default;
    
    // No copy or move
    ProcessorFactory(const ProcessorFactory&) = delete;
    ProcessorFactory& operator=(const ProcessorFactory&) = delete;
};

} // namespace medical_imaging