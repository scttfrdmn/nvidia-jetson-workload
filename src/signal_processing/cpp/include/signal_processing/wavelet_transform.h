/**
 * @file wavelet_transform.h
 * @brief Wavelet transform functionality for signal processing
 * 
 * This header provides classes and functions for performing discrete wavelet transforms,
 * continuous wavelet transforms, and wavelet packet decomposition with GPU acceleration.
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#ifndef SIGNAL_PROCESSING_WAVELET_TRANSFORM_H
#define SIGNAL_PROCESSING_WAVELET_TRANSFORM_H

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <complex>

namespace signal_processing {

// Forward declarations
class WaveletTransform;
class DiscreteWaveletTransform;
class ContinuousWaveletTransform;
class WaveletPacketTransform;
class MaximalOverlapDWT;

/**
 * @enum WaveletFamily
 * @brief Enumeration of supported wavelet families
 */
enum class WaveletFamily {
    HAAR,           ///< Haar wavelet
    DAUBECHIES,     ///< Daubechies wavelets
    SYMLET,         ///< Symlets
    COIFLET,        ///< Coiflets
    BIORTHOGONAL,   ///< Biorthogonal wavelets
    REVERSE_BIORT,  ///< Reverse biorthogonal wavelets
    DMEY,           ///< Discrete Meyer wavelet
    GAUSSIAN,       ///< Gaussian wavelets
    MEXICAN_HAT,    ///< Mexican hat wavelet (only for CWT)
    MORLET,         ///< Morlet wavelet (only for CWT)
    RICKER,         ///< Ricker wavelet
    SHANNON,        ///< Shannon wavelets
    FREQUENCY_B_SPLINE ///< Frequency B-spline wavelets
};

/**
 * @enum WaveletTransformType
 * @brief Enumeration of wavelet transform types
 */
enum class WaveletTransformType {
    DWT,      ///< Discrete Wavelet Transform
    MODWT,    ///< Maximal Overlap Discrete Wavelet Transform
    CWT,      ///< Continuous Wavelet Transform
    WPT       ///< Wavelet Packet Transform
};

/**
 * @enum BoundaryMode
 * @brief Enumeration of boundary handling modes for wavelet transforms
 */
enum class BoundaryMode {
    ZERO_PADDING,   ///< Zero padding
    SYMMETRIC,      ///< Symmetric padding
    PERIODIC,       ///< Periodic padding
    REFLECT         ///< Reflection of samples
};

/**
 * @enum WaveletBoundaryMode
 * @brief Enumeration of boundary handling modes for wavelet transforms
 * @deprecated Use BoundaryMode instead
 */
enum class WaveletBoundaryMode {
    ZERO,           ///< Zero padding
    SYMMETRIC,      ///< Symmetric padding
    PERIODIC,       ///< Periodic padding
    REFLECT,        ///< Reflection of samples
    EXTEND,         ///< Constant extension
    SMOOTH          ///< Linear extrapolation
};

/**
 * @class WaveletTransform
 * @brief Base class for wavelet transforms
 */
class WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param device_id CUDA device ID (-1 for CPU)
     */
    WaveletTransform(int device_id = 0);
    
    /**
     * @brief Destructor
     */
    virtual ~WaveletTransform();
    
    /**
     * @brief Get transform type
     * @return Transform type
     */
    virtual WaveletTransformType getTransformType() const = 0;
    
    /**
     * @brief Get device ID
     * @return Device ID
     */
    int getDeviceID() const;
    
    /**
     * @brief Set device ID
     * @param device_id Device ID
     */
    void setDeviceID(int device_id);
    
protected:
    int device_id_;          ///< CUDA device ID
    bool has_cuda_device_;   ///< Whether CUDA device is available
};

/**
 * @class DiscreteWaveletTransform
 * @brief Performs discrete wavelet transforms (DWT)
 */
class DiscreteWaveletTransform : public WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param wavelet_family Wavelet family
     * @param vanishing_moments Number of vanishing moments (for applicable wavelets)
     * @param boundary_mode Boundary handling mode
     * @param device_id CUDA device ID (-1 for CPU)
     */
    DiscreteWaveletTransform(
        WaveletFamily wavelet_family = WaveletFamily::DAUBECHIES,
        int vanishing_moments = 4,
        WaveletBoundaryMode boundary_mode = WaveletBoundaryMode::SYMMETRIC,
        int device_id = 0
    );
    
    /**
     * @brief Perform forward wavelet transform
     * @param input Input signal
     * @param levels Number of decomposition levels
     * @return Vector of coefficients (approximation and details)
     */
    std::vector<std::vector<float>> forward(
        const std::vector<float>& input,
        int levels = 1
    );
    
    /**
     * @brief Perform inverse wavelet transform
     * @param coeffs Vector of wavelet coefficients
     * @return Reconstructed signal
     */
    std::vector<float> inverse(
        const std::vector<std::vector<float>>& coeffs
    );
    
    /**
     * @brief Get transform type
     * @return Transform type
     */
    WaveletTransformType getTransformType() const override;
    
    /**
     * @brief Get wavelet family
     * @return Wavelet family
     */
    WaveletFamily getWaveletFamily() const;
    
    /**
     * @brief Get number of vanishing moments
     * @return Number of vanishing moments
     */
    int getVanishingMoments() const;
    
    /**
     * @brief Get boundary mode
     * @return Boundary mode
     */
    WaveletBoundaryMode getBoundaryMode() const;
    
    /**
     * @brief Set wavelet family
     * @param family Wavelet family
     */
    void setWaveletFamily(WaveletFamily family);
    
    /**
     * @brief Set number of vanishing moments
     * @param moments Number of vanishing moments
     */
    void setVanishingMoments(int moments);
    
    /**
     * @brief Set boundary mode
     * @param mode Boundary mode
     */
    void setBoundaryMode(WaveletBoundaryMode mode);
    
    /**
     * @brief Get decomposition filter (low-pass)
     * @return Filter coefficients
     */
    const std::vector<float>& getDecompositionLowPass() const;
    
    /**
     * @brief Get decomposition filter (high-pass)
     * @return Filter coefficients
     */
    const std::vector<float>& getDecompositionHighPass() const;
    
    /**
     * @brief Get reconstruction filter (low-pass)
     * @return Filter coefficients
     */
    const std::vector<float>& getReconstructionLowPass() const;
    
    /**
     * @brief Get reconstruction filter (high-pass)
     * @return Filter coefficients
     */
    const std::vector<float>& getReconstructionHighPass() const;
    
private:
    WaveletFamily wavelet_family_;           ///< Wavelet family
    int vanishing_moments_;                  ///< Number of vanishing moments
    WaveletBoundaryMode boundary_mode_;      ///< Boundary handling mode
    std::vector<float> decomp_low_pass_;     ///< Decomposition filter (low-pass)
    std::vector<float> decomp_high_pass_;    ///< Decomposition filter (high-pass)
    std::vector<float> recon_low_pass_;      ///< Reconstruction filter (low-pass)
    std::vector<float> recon_high_pass_;     ///< Reconstruction filter (high-pass)
    
    /**
     * @brief Initialize wavelet filters
     */
    void initializeFilters();
    
    /**
     * @brief Apply filter to signal (convolution with subsampling)
     * @param input Input signal
     * @param filter Filter coefficients
     * @param boundary Boundary mode
     * @return Filtered signal
     */
    std::vector<float> applyFilter(
        const std::vector<float>& input,
        const std::vector<float>& filter,
        WaveletBoundaryMode boundary
    );
    
    /**
     * @brief Apply filter to signal (convolution with upsampling)
     * @param input Input signal
     * @param filter Filter coefficients
     * @param output_size Expected output size
     * @return Filtered signal
     */
    std::vector<float> applyFilterWithUpsampling(
        const std::vector<float>& input,
        const std::vector<float>& filter,
        int output_size
    );
};

/**
 * @class ContinuousWaveletTransform
 * @brief Performs continuous wavelet transforms (CWT)
 */
class ContinuousWaveletTransform : public WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param wavelet_family Wavelet family
     * @param scales Scales for wavelet transform
     * @param parameters Additional wavelet parameters (e.g., for Morlet)
     * @param device_id CUDA device ID (-1 for CPU)
     */
    ContinuousWaveletTransform(
        WaveletFamily wavelet_family = WaveletFamily::MORLET,
        const std::vector<float>& scales = std::vector<float>(),
        const std::vector<float>& parameters = std::vector<float>(),
        int device_id = 0
    );
    
    /**
     * @brief Perform continuous wavelet transform
     * @param input Input signal
     * @param scales Scales for transform (overrides constructor scales if provided)
     * @return 2D array of complex coefficients [scale][time]
     */
    std::vector<std::vector<std::complex<float>>> transform(
        const std::vector<float>& input,
        const std::vector<float>& scales = std::vector<float>()
    );
    
    /**
     * @brief Get transform type
     * @return Transform type
     */
    WaveletTransformType getTransformType() const override;
    
    /**
     * @brief Get wavelet family
     * @return Wavelet family
     */
    WaveletFamily getWaveletFamily() const;
    
    /**
     * @brief Get scales
     * @return Vector of scales
     */
    const std::vector<float>& getScales() const;
    
    /**
     * @brief Set wavelet family
     * @param family Wavelet family
     */
    void setWaveletFamily(WaveletFamily family);
    
    /**
     * @brief Set scales
     * @param scales Vector of scales
     */
    void setScales(const std::vector<float>& scales);
    
    /**
     * @brief Generate logarithmic scales
     * @param min_scale Minimum scale
     * @param max_scale Maximum scale
     * @param num_scales Number of scales
     * @return Vector of scales
     */
    static std::vector<float> generateLogScales(
        float min_scale, 
        float max_scale, 
        int num_scales
    );
    
    /**
     * @brief Generate linear scales
     * @param min_scale Minimum scale
     * @param max_scale Maximum scale
     * @param num_scales Number of scales
     * @return Vector of scales
     */
    static std::vector<float> generateLinearScales(
        float min_scale, 
        float max_scale, 
        int num_scales
    );
    
    /**
     * @brief Generate scales based on frequencies
     * @param min_freq Minimum frequency (Hz)
     * @param max_freq Maximum frequency (Hz)
     * @param num_scales Number of scales
     * @param sample_rate Sample rate (Hz)
     * @return Vector of scales
     */
    static std::vector<float> generateFrequencyScales(
        float min_freq, 
        float max_freq, 
        int num_scales, 
        float sample_rate
    );
    
private:
    WaveletFamily wavelet_family_;        ///< Wavelet family
    std::vector<float> scales_;           ///< Scales for wavelet transform
    std::vector<float> parameters_;       ///< Additional wavelet parameters
    
    /**
     * @brief Generate wavelet frequency function
     * @param scale Scale for wavelet
     * @param num_points Number of points
     * @return Complex wavelet function
     */
    std::vector<std::complex<float>> generateWavelet(
        float scale,
        int num_points
    );
};

/**
 * @class WaveletPacketTransform
 * @brief Performs wavelet packet transforms (WPT)
 */
class WaveletPacketTransform : public WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param wavelet_family Wavelet family
     * @param vanishing_moments Number of vanishing moments (for applicable wavelets)
     * @param boundary_mode Boundary handling mode
     * @param device_id CUDA device ID (-1 for CPU)
     */
    WaveletPacketTransform(
        WaveletFamily wavelet_family = WaveletFamily::DAUBECHIES,
        int vanishing_moments = 4,
        WaveletBoundaryMode boundary_mode = WaveletBoundaryMode::SYMMETRIC,
        int device_id = 0
    );
    
    /**
     * @brief Perform forward wavelet packet transform
     * @param input Input signal
     * @param max_level Maximum decomposition level
     * @return Wavelet packet coefficients (tree structure)
     */
    std::vector<std::vector<std::vector<float>>> forward(
        const std::vector<float>& input,
        int max_level = 3
    );
    
    /**
     * @brief Perform inverse wavelet packet transform
     * @param coeffs Wavelet packet coefficients
     * @param nodes_to_use Nodes to use for reconstruction (empty means all)
     * @return Reconstructed signal
     */
    std::vector<float> inverse(
        const std::vector<std::vector<std::vector<float>>>& coeffs,
        const std::vector<std::pair<int, int>>& nodes_to_use = std::vector<std::pair<int, int>>()
    );
    
    /**
     * @brief Get transform type
     * @return Transform type
     */
    WaveletTransformType getTransformType() const override;
    
    /**
     * @brief Get wavelet family
     * @return Wavelet family
     */
    WaveletFamily getWaveletFamily() const;
    
    /**
     * @brief Get number of vanishing moments
     * @return Number of vanishing moments
     */
    int getVanishingMoments() const;
    
    /**
     * @brief Get boundary mode
     * @return Boundary mode
     */
    WaveletBoundaryMode getBoundaryMode() const;
    
    /**
     * @brief Set wavelet family
     * @param family Wavelet family
     */
    void setWaveletFamily(WaveletFamily family);
    
    /**
     * @brief Set number of vanishing moments
     * @param moments Number of vanishing moments
     */
    void setVanishingMoments(int moments);
    
    /**
     * @brief Set boundary mode
     * @param mode Boundary mode
     */
    void setBoundaryMode(WaveletBoundaryMode mode);
    
private:
    WaveletFamily wavelet_family_;           ///< Wavelet family
    int vanishing_moments_;                  ///< Number of vanishing moments
    WaveletBoundaryMode boundary_mode_;      ///< Boundary handling mode
    std::vector<float> decomp_low_pass_;     ///< Decomposition filter (low-pass)
    std::vector<float> decomp_high_pass_;    ///< Decomposition filter (high-pass)
    std::vector<float> recon_low_pass_;      ///< Reconstruction filter (low-pass)
    std::vector<float> recon_high_pass_;     ///< Reconstruction filter (high-pass)
    
    /**
     * @brief Initialize wavelet filters
     */
    void initializeFilters();
    
    /**
     * @brief Decompose signal into wavelet packets
     * @param input Input signal
     * @param level Current decomposition level
     * @param max_level Maximum decomposition level
     * @param node_index Node index
     * @param result Result container
     */
    void decomposePacket(
        const std::vector<float>& input,
        int level,
        int max_level,
        int node_index,
        std::vector<std::vector<std::vector<float>>>& result
    );
    
    /**
     * @brief Reconstruct signal from wavelet packets
     * @param coeffs Wavelet packet coefficients
     * @param level Current reconstruction level
     * @param max_level Maximum reconstruction level
     * @param node_index Node index
     * @param nodes_to_use Nodes to use for reconstruction
     * @return Reconstructed signal
     */
    std::vector<float> reconstructPacket(
        const std::vector<std::vector<std::vector<float>>>& coeffs,
        int level,
        int max_level,
        int node_index,
        const std::vector<std::pair<int, int>>& nodes_to_use
    );
};

/**
 * @class MaximalOverlapDWT
 * @brief Performs maximal overlap discrete wavelet transform (MODWT)
 */
class MaximalOverlapDWT : public WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param wavelet_family Wavelet family
     * @param vanishing_moments Number of vanishing moments (for applicable wavelets)
     * @param boundary_mode Boundary handling mode
     * @param device_id CUDA device ID (-1 for CPU)
     */
    MaximalOverlapDWT(
        WaveletFamily wavelet_family = WaveletFamily::DAUBECHIES,
        int vanishing_moments = 4,
        WaveletBoundaryMode boundary_mode = WaveletBoundaryMode::PERIODIC,
        int device_id = 0
    );
    
    /**
     * @brief Perform forward maximal overlap wavelet transform
     * @param input Input signal
     * @param levels Number of decomposition levels
     * @return Vector of coefficients (approximation and details)
     */
    std::vector<std::vector<float>> forward(
        const std::vector<float>& input,
        int levels = 1
    );
    
    /**
     * @brief Perform inverse maximal overlap wavelet transform
     * @param coeffs Vector of wavelet coefficients
     * @return Reconstructed signal
     */
    std::vector<float> inverse(
        const std::vector<std::vector<float>>& coeffs
    );
    
    /**
     * @brief Get transform type
     * @return Transform type
     */
    WaveletTransformType getTransformType() const override;
    
    /**
     * @brief Get wavelet family
     * @return Wavelet family
     */
    WaveletFamily getWaveletFamily() const;
    
    /**
     * @brief Get number of vanishing moments
     * @return Number of vanishing moments
     */
    int getVanishingMoments() const;
    
    /**
     * @brief Get boundary mode
     * @return Boundary mode
     */
    WaveletBoundaryMode getBoundaryMode() const;
    
    /**
     * @brief Set wavelet family
     * @param family Wavelet family
     */
    void setWaveletFamily(WaveletFamily family);
    
    /**
     * @brief Set number of vanishing moments
     * @param moments Number of vanishing moments
     */
    void setVanishingMoments(int moments);
    
    /**
     * @brief Set boundary mode
     * @param mode Boundary mode
     */
    void setBoundaryMode(WaveletBoundaryMode mode);
    
private:
    WaveletFamily wavelet_family_;           ///< Wavelet family
    int vanishing_moments_;                  ///< Number of vanishing moments
    WaveletBoundaryMode boundary_mode_;      ///< Boundary handling mode
    std::vector<float> decomp_low_pass_;     ///< Decomposition filter (low-pass)
    std::vector<float> decomp_high_pass_;    ///< Decomposition filter (high-pass)
    std::vector<float> recon_low_pass_;      ///< Reconstruction filter (low-pass)
    std::vector<float> recon_high_pass_;     ///< Reconstruction filter (high-pass)
    
    /**
     * @brief Initialize wavelet filters
     */
    void initializeFilters();
    
    /**
     * @brief Apply filter to signal (convolution without downsampling)
     * @param input Input signal
     * @param filter Filter coefficients
     * @param boundary Boundary mode
     * @return Filtered signal
     */
    std::vector<float> applyFilter(
        const std::vector<float>& input,
        const std::vector<float>& filter,
        WaveletBoundaryMode boundary
    );
};

/**
 * @brief Generate wavelet filter coefficients
 * @param family Wavelet family
 * @param vanishing_moments Number of vanishing moments
 * @return Tuple of (decomp_low, decomp_high, recon_low, recon_high)
 */
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
generateWaveletFilters(WaveletFamily family, int vanishing_moments);

/**
 * @brief Get string representation of wavelet family
 * @param family Wavelet family
 * @return String representation
 */
std::string waveletFamilyToString(WaveletFamily family);

/**
 * @brief Get string representation of wavelet transform type
 * @param type Wavelet transform type
 * @return String representation
 */
std::string waveletTransformTypeToString(WaveletTransformType type);

/**
 * @brief Get string representation of wavelet boundary mode
 * @param mode Wavelet boundary mode
 * @return String representation
 */
std::string waveletBoundaryModeToString(WaveletBoundaryMode mode);

/**
 * @struct WaveletTransformResult
 * @brief Structure to hold the results of a wavelet transform
 */
struct WaveletTransformResult {
    std::vector<std::vector<float>> approximation_coefficients; ///< Approximation coefficients for each level
    std::vector<std::vector<float>> detail_coefficients;        ///< Detail coefficients for each level
};

/**
 * @struct WaveletPacketResult
 * @brief Structure to hold the results of a wavelet packet transform
 */
struct WaveletPacketResult {
    std::vector<std::vector<std::vector<float>>> coefficients; ///< Coefficients organized by level and node
};

/**
 * @class WaveletTransform
 * @brief Base class for wavelet transforms in the new API
 */
class WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param family Wavelet family
     * @param vanishing_moments Number of vanishing moments
     */
    WaveletTransform(WaveletFamily family, int vanishing_moments);
    
    /**
     * @brief Virtual destructor
     */
    virtual ~WaveletTransform() = default;
    
    /**
     * @brief Get decomposition low pass filter
     * @return Filter coefficients
     */
    const std::vector<float>& getDecompositionLowPassFilter() const {
        return decomposition_low_pass_;
    }
    
    /**
     * @brief Get decomposition high pass filter
     * @return Filter coefficients
     */
    const std::vector<float>& getDecompositionHighPassFilter() const {
        return decomposition_high_pass_;
    }
    
    /**
     * @brief Get reconstruction low pass filter
     * @return Filter coefficients
     */
    const std::vector<float>& getReconstructionLowPassFilter() const {
        return reconstruction_low_pass_;
    }
    
    /**
     * @brief Get reconstruction high pass filter
     * @return Filter coefficients
     */
    const std::vector<float>& getReconstructionHighPassFilter() const {
        return reconstruction_high_pass_;
    }
    
protected:
    WaveletFamily family_; ///< Wavelet family
    int vanishing_moments_; ///< Number of vanishing moments
    
    std::vector<float> decomposition_low_pass_; ///< Decomposition low pass filter
    std::vector<float> decomposition_high_pass_; ///< Decomposition high pass filter
    std::vector<float> reconstruction_low_pass_; ///< Reconstruction low pass filter
    std::vector<float> reconstruction_high_pass_; ///< Reconstruction high pass filter
    
    /**
     * @brief Generate filters based on wavelet family and vanishing moments
     */
    void generateFilters();
    
    /**
     * @brief Generate Haar filters
     */
    void generateHaarFilters();
    
    /**
     * @brief Generate Daubechies filters
     */
    void generateDaubechiesFilters();
    
    /**
     * @brief Generate Symlet filters
     */
    void generateSymletFilters();
    
    /**
     * @brief Generate Coiflet filters
     */
    void generateCoifletFilters();
    
    /**
     * @brief Generate Biorthogonal filters
     */
    void generateBiorthogonalFilters();
    
    /**
     * @brief Generate Meyer filters
     */
    void generateMeyerFilters();
    
    /**
     * @brief Generate Morlet filters
     */
    void generateMorletFilters();
    
    /**
     * @brief Generate Mexican Hat filters
     */
    void generateMexicanHatFilters();
    
    /**
     * @brief Convolve signal with filter
     * @param signal Input signal
     * @param filter Filter coefficients
     * @return Convolved signal
     */
    std::vector<float> convolve(const std::vector<float>& signal, 
                             const std::vector<float>& filter);
    
    /**
     * @brief Downsample signal by factor of 2
     * @param signal Input signal
     * @return Downsampled signal
     */
    std::vector<float> downsample(const std::vector<float>& signal);
    
    /**
     * @brief Upsample signal by factor of 2
     * @param signal Input signal
     * @return Upsampled signal
     */
    std::vector<float> upsample(const std::vector<float>& signal);
    
    /**
     * @brief Extend signal to handle boundary effects
     * @param signal Input signal
     * @param filter_length Filter length
     * @param mode Boundary mode
     * @return Extended signal
     */
    std::vector<float> extendSignal(const std::vector<float>& signal, 
                                  int filter_length, 
                                  BoundaryMode mode);
};

/**
 * @class DiscreteWaveletTransform
 * @brief Implements the Discrete Wavelet Transform (DWT)
 */
class DiscreteWaveletTransform : public WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param family Wavelet family
     * @param vanishing_moments Number of vanishing moments
     */
    DiscreteWaveletTransform(WaveletFamily family = WaveletFamily::DAUBECHIES, 
                           int vanishing_moments = 4);
    
    /**
     * @brief Perform forward DWT
     * @param signal Input signal
     * @param levels Number of decomposition levels
     * @param mode Boundary mode
     * @return WaveletTransformResult with approximation and detail coefficients
     */
    WaveletTransformResult forward(const std::vector<float>& signal, 
                                 int levels = 1, 
                                 BoundaryMode mode = BoundaryMode::SYMMETRIC);
    
    /**
     * @brief Perform inverse DWT
     * @param transform_result WaveletTransformResult from forward transform
     * @param mode Boundary mode
     * @return Reconstructed signal
     */
    std::vector<float> inverse(const WaveletTransformResult& transform_result, 
                            BoundaryMode mode = BoundaryMode::SYMMETRIC);
};

/**
 * @class ContinuousWaveletTransform
 * @brief Implements the Continuous Wavelet Transform (CWT)
 */
class ContinuousWaveletTransform : public WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param family Wavelet family
     * @param vanishing_moments Number of vanishing moments
     */
    ContinuousWaveletTransform(WaveletFamily family = WaveletFamily::MORLET, 
                             int vanishing_moments = 1);
    
    /**
     * @brief Perform forward CWT
     * @param signal Input signal
     * @param scales Scales for wavelet transform
     * @return 2D array of complex coefficients [scale][time]
     */
    std::vector<std::vector<std::complex<float>>> forward(
        const std::vector<float>& signal, 
        const std::vector<float>& scales);
    
    /**
     * @brief Perform inverse CWT
     * @param coefficients 2D array of complex coefficients [scale][time]
     * @param scales Scales used in forward transform
     * @return Reconstructed signal
     */
    std::vector<float> inverse(
        const std::vector<std::vector<std::complex<float>>>& coefficients,
        const std::vector<float>& scales);
    
    /**
     * @brief Generate logarithmically spaced scales
     * @param num_scales Number of scales
     * @param min_scale Minimum scale
     * @param max_scale Maximum scale
     * @return Vector of scales
     */
    std::vector<float> generateScales(int num_scales, 
                                   float min_scale = 1.0f, 
                                   float max_scale = 32.0f);
};

/**
 * @class WaveletPacketTransform
 * @brief Implements the Wavelet Packet Transform (WPT)
 */
class WaveletPacketTransform : public WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param family Wavelet family
     * @param vanishing_moments Number of vanishing moments
     */
    WaveletPacketTransform(WaveletFamily family = WaveletFamily::DAUBECHIES, 
                         int vanishing_moments = 4);
    
    /**
     * @brief Perform forward WPT
     * @param signal Input signal
     * @param levels Number of decomposition levels
     * @param mode Boundary mode
     * @return WaveletPacketResult with coefficients
     */
    WaveletPacketResult forward(const std::vector<float>& signal, 
                              int levels = 1, 
                              BoundaryMode mode = BoundaryMode::SYMMETRIC);
    
    /**
     * @brief Perform inverse WPT
     * @param packet_result WaveletPacketResult from forward transform
     * @param mode Boundary mode
     * @return Reconstructed signal
     */
    std::vector<float> inverse(const WaveletPacketResult& packet_result, 
                            BoundaryMode mode = BoundaryMode::SYMMETRIC);
};

/**
 * @class MaximalOverlapDWT
 * @brief Implements the Maximal Overlap Discrete Wavelet Transform (MODWT)
 */
class MaximalOverlapDWT : public WaveletTransform {
public:
    /**
     * @brief Constructor
     * @param family Wavelet family
     * @param vanishing_moments Number of vanishing moments
     */
    MaximalOverlapDWT(WaveletFamily family = WaveletFamily::DAUBECHIES, 
                    int vanishing_moments = 4);
    
    /**
     * @brief Perform forward MODWT
     * @param signal Input signal
     * @param levels Number of decomposition levels
     * @param mode Boundary mode
     * @return WaveletTransformResult with approximation and detail coefficients
     */
    WaveletTransformResult forward(const std::vector<float>& signal, 
                                 int levels = 1, 
                                 BoundaryMode mode = BoundaryMode::SYMMETRIC);
    
    /**
     * @brief Perform inverse MODWT
     * @param transform_result WaveletTransformResult from forward transform
     * @param mode Boundary mode
     * @return Reconstructed signal
     */
    std::vector<float> inverse(const WaveletTransformResult& transform_result, 
                            BoundaryMode mode = BoundaryMode::SYMMETRIC);
};

#ifdef WITH_CUDA
/**
 * @brief Perform GPU-accelerated discrete wavelet transform
 * @param signal Input signal
 * @param decomp_low_pass Decomposition filter (low-pass)
 * @param decomp_high_pass Decomposition filter (high-pass)
 * @param levels Number of decomposition levels
 * @param mode Boundary handling mode
 * @return Transform result
 */
WaveletTransformResult cuda_discrete_wavelet_transform(
    const std::vector<float>& signal,
    const std::vector<float>& decomp_low_pass,
    const std::vector<float>& decomp_high_pass,
    int levels,
    BoundaryMode mode
);

/**
 * @brief Perform GPU-accelerated inverse discrete wavelet transform
 * @param transform_result Transform result
 * @param recon_low_pass Reconstruction filter (low-pass)
 * @param recon_high_pass Reconstruction filter (high-pass)
 * @param mode Boundary handling mode
 * @return Reconstructed signal
 */
std::vector<float> cuda_inverse_discrete_wavelet_transform(
    const WaveletTransformResult& transform_result,
    const std::vector<float>& recon_low_pass,
    const std::vector<float>& recon_high_pass,
    BoundaryMode mode
);

/**
 * @brief Perform GPU-accelerated continuous wavelet transform
 * @param signal Input signal
 * @param scales Scales for transform
 * @param family Wavelet family
 * @return 2D array of complex coefficients [scale][time]
 */
std::vector<std::vector<std::complex<float>>> cuda_continuous_wavelet_transform(
    const std::vector<float>& signal,
    const std::vector<float>& scales,
    WaveletFamily family
);

/**
 * @brief Perform GPU-accelerated wavelet packet transform
 * @param signal Input signal
 * @param decomp_low_pass Decomposition filter (low-pass)
 * @param decomp_high_pass Decomposition filter (high-pass)
 * @param levels Number of decomposition levels
 * @param mode Boundary handling mode
 * @return Wavelet packet transform result
 */
WaveletPacketResult cuda_wavelet_packet_transform(
    const std::vector<float>& signal,
    const std::vector<float>& decomp_low_pass,
    const std::vector<float>& decomp_high_pass,
    int levels,
    BoundaryMode mode
);

/**
 * @brief Perform GPU-accelerated inverse wavelet packet transform
 * @param packet_result Wavelet packet transform result
 * @param recon_low_pass Reconstruction filter (low-pass)
 * @param recon_high_pass Reconstruction filter (high-pass)
 * @param mode Boundary handling mode
 * @return Reconstructed signal
 */
std::vector<float> cuda_inverse_wavelet_packet_transform(
    const WaveletPacketResult& packet_result,
    const std::vector<float>& recon_low_pass,
    const std::vector<float>& recon_high_pass,
    BoundaryMode mode
);
#endif // WITH_CUDA

} // namespace signal_processing

#endif // SIGNAL_PROCESSING_WAVELET_TRANSFORM_H