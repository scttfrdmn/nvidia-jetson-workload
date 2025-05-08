/**
 * @file digital_filtering.h
 * @brief Digital filtering operations optimized for NVIDIA GPUs
 * 
 * This file provides GPU-optimized implementations for:
 * - FIR filters (low-pass, high-pass, band-pass, band-stop)
 * - IIR filters (Butterworth, Chebyshev, Elliptic)
 * - Adaptive filters (LMS, RLS, Kalman)
 * - Median and other non-linear filters
 * - Multi-rate filtering (decimation, interpolation, resampling)
 * 
 * The implementation automatically adapts to the available GPU architecture:
 * - Optimized for Jetson Orin NX (SM 8.7)
 * - Optimized for AWS T4G instances (SM 7.5)
 * - Fallback CPU implementation when GPU is unavailable
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#ifndef SIGNAL_PROCESSING_DIGITAL_FILTERING_H
#define SIGNAL_PROCESSING_DIGITAL_FILTERING_H

#include <vector>
#include <complex>
#include <memory>
#include <functional>
#include <string>
#include <optional>

namespace signal_processing {

// Forward declarations
class FIRFilterImpl;
class IIRFilterImpl;
class AdaptiveFilterImpl;
class MultirateFilterImpl;

/**
 * @brief FIR filter design methods
 */
enum class FIRDesignMethod {
    WINDOW,         // Window-based design
    LEAST_SQUARES,  // Least squares optimization
    PARKS_MCCLELLAN, // Parks-McClellan (Remez) algorithm
    FREQUENCY_SAMPLING // Frequency sampling method
};

/**
 * @brief IIR filter design methods
 */
enum class IIRDesignMethod {
    BUTTERWORTH,   // Butterworth filter
    CHEBYSHEV1,    // Chebyshev Type I filter
    CHEBYSHEV2,    // Chebyshev Type II filter
    ELLIPTIC,      // Elliptic (Cauer) filter
    BESSEL         // Bessel filter
};

/**
 * @brief Window types for FIR filter design
 */
enum class WindowType {
    RECTANGULAR,   // Rectangular window
    TRIANGULAR,    // Triangular window
    HANN,          // Hann window
    HAMMING,       // Hamming window
    BLACKMAN,      // Blackman window
    KAISER         // Kaiser window
};

/**
 * @brief Filter types
 */
enum class FilterType {
    LOWPASS,      // Low-pass filter
    HIGHPASS,     // High-pass filter
    BANDPASS,     // Band-pass filter
    BANDSTOP      // Band-stop filter
};

/**
 * @brief Adaptive filter types
 */
enum class AdaptiveFilterType {
    LMS,          // Least Mean Squares
    NLMS,         // Normalized Least Mean Squares
    RLS,          // Recursive Least Squares
    KALMAN        // Kalman filter
};

/**
 * @brief FIR Filter Parameters
 */
struct FIRFilterParams {
    int num_taps = 0;                         // Number of filter taps
    FilterType filter_type = FilterType::LOWPASS; // Filter type
    std::vector<float> cutoff_freqs;          // Cutoff frequencies [0.0, 1.0] normalized
    FIRDesignMethod design_method = FIRDesignMethod::WINDOW; // Design method
    WindowType window_type = WindowType::HAMMING; // Window type for window method
    float window_param = 0.0f;                // Parameter for parameterized windows
    std::vector<float> gains = {1.0f, 0.0f};  // Passband and stopband gains
};

/**
 * @brief IIR Filter Parameters
 */
struct IIRFilterParams {
    int order = 0;                            // Filter order
    FilterType filter_type = FilterType::LOWPASS; // Filter type
    std::vector<float> cutoff_freqs;          // Cutoff frequencies [0.0, 1.0] normalized
    IIRDesignMethod design_method = IIRDesignMethod::BUTTERWORTH; // Design method
    float ripple_db = 0.5f;                   // Passband ripple in dB (Chebyshev, Elliptic)
    float stopband_atten_db = 40.0f;          // Stopband attenuation in dB (Chebyshev II, Elliptic)
};

/**
 * @brief Adaptive Filter Parameters
 */
struct AdaptiveFilterParams {
    int filter_length = 0;                    // Filter length
    AdaptiveFilterType filter_type = AdaptiveFilterType::LMS; // Adaptive filter type
    float step_size = 0.1f;                   // Step size (mu) for LMS/NLMS
    float forgetting_factor = 0.99f;          // Forgetting factor for RLS
    float regularization = 1e-6f;             // Regularization parameter
};

/**
 * @brief Multirate Filter Parameters
 */
struct MultirateFilterParams {
    int interpolation_factor = 1;             // Interpolation factor
    int decimation_factor = 1;                // Decimation factor
    FIRFilterParams filter_params;            // FIR filter parameters for anti-aliasing
};

/**
 * @brief FIR (Finite Impulse Response) filter with GPU acceleration
 */
class FIRFilter {
public:
    /**
     * @brief Construct a new FIR filter with coefficients
     * 
     * @param coefficients Filter coefficients
     * @param device_id CUDA device ID (-1 for CPU)
     */
    FIRFilter(const std::vector<float>& coefficients, int device_id = 0);
    
    /**
     * @brief Construct a new FIR filter using design parameters
     * 
     * @param params Filter design parameters
     * @param sample_rate Sample rate (Hz)
     * @param device_id CUDA device ID (-1 for CPU)
     */
    FIRFilter(const FIRFilterParams& params, float sample_rate, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~FIRFilter();
    
    // Prevent copying
    FIRFilter(const FIRFilter&) = delete;
    FIRFilter& operator=(const FIRFilter&) = delete;
    
    // Allow moving
    FIRFilter(FIRFilter&&) noexcept;
    FIRFilter& operator=(FIRFilter&&) noexcept;
    
    /**
     * @brief Apply the filter to a signal
     * 
     * @param input Input signal
     * @return std::vector<float> Filtered signal
     */
    std::vector<float> filter(const std::vector<float>& input);
    
    /**
     * @brief Reset the filter state
     */
    void reset();
    
    /**
     * @brief Get the filter coefficients
     * 
     * @return std::vector<float> Filter coefficients
     */
    std::vector<float> get_coefficients() const;
    
    /**
     * @brief Get the filter frequency response
     * 
     * @param num_points Number of frequency points
     * @return std::pair<std::vector<float>, std::vector<float>> 
     *         Frequencies and magnitude response
     */
    std::pair<std::vector<float>, std::vector<float>> get_frequency_response(int num_points = 512) const;
    
    /**
     * @brief Get the filter phase response
     * 
     * @param num_points Number of frequency points
     * @return std::pair<std::vector<float>, std::vector<float>> 
     *         Frequencies and phase response
     */
    std::pair<std::vector<float>, std::vector<float>> get_phase_response(int num_points = 512) const;
    
    /**
     * @brief Get the filter step response
     * 
     * @param num_points Number of time points
     * @return std::vector<float> Step response
     */
    std::vector<float> get_step_response(int num_points = 100) const;
    
    /**
     * @brief Get the filter impulse response
     * 
     * @param num_points Number of time points
     * @return std::vector<float> Impulse response
     */
    std::vector<float> get_impulse_response(int num_points = 100) const;
    
private:
    std::unique_ptr<FIRFilterImpl> impl_;
};

/**
 * @brief IIR (Infinite Impulse Response) filter with GPU acceleration
 */
class IIRFilter {
public:
    /**
     * @brief Construct a new IIR filter with coefficients
     * 
     * @param a Denominator coefficients (a0 = 1.0 assumed)
     * @param b Numerator coefficients
     * @param device_id CUDA device ID (-1 for CPU)
     */
    IIRFilter(const std::vector<float>& a, const std::vector<float>& b, int device_id = 0);
    
    /**
     * @brief Construct a new IIR filter using design parameters
     * 
     * @param params Filter design parameters
     * @param sample_rate Sample rate (Hz)
     * @param device_id CUDA device ID (-1 for CPU)
     */
    IIRFilter(const IIRFilterParams& params, float sample_rate, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~IIRFilter();
    
    // Prevent copying
    IIRFilter(const IIRFilter&) = delete;
    IIRFilter& operator=(const IIRFilter&) = delete;
    
    // Allow moving
    IIRFilter(IIRFilter&&) noexcept;
    IIRFilter& operator=(IIRFilter&&) noexcept;
    
    /**
     * @brief Apply the filter to a signal
     * 
     * @param input Input signal
     * @return std::vector<float> Filtered signal
     */
    std::vector<float> filter(const std::vector<float>& input);
    
    /**
     * @brief Apply the filter to a signal using second-order sections (SOS)
     * 
     * @param input Input signal
     * @return std::vector<float> Filtered signal
     */
    std::vector<float> filter_sos(const std::vector<float>& input);
    
    /**
     * @brief Reset the filter state
     */
    void reset();
    
    /**
     * @brief Get the filter coefficients
     * 
     * @return std::pair<std::vector<float>, std::vector<float>> 
     *         Denominator and numerator coefficients
     */
    std::pair<std::vector<float>, std::vector<float>> get_coefficients() const;
    
    /**
     * @brief Get the filter as second-order sections
     * 
     * @return std::vector<std::array<float, 6>> 
     *         List of second-order sections [b0, b1, b2, a0, a1, a2]
     */
    std::vector<std::array<float, 6>> get_sos() const;
    
    /**
     * @brief Get the filter frequency response
     * 
     * @param num_points Number of frequency points
     * @return std::pair<std::vector<float>, std::vector<float>> 
     *         Frequencies and magnitude response
     */
    std::pair<std::vector<float>, std::vector<float>> get_frequency_response(int num_points = 512) const;
    
    /**
     * @brief Get the filter phase response
     * 
     * @param num_points Number of frequency points
     * @return std::pair<std::vector<float>, std::vector<float>> 
     *         Frequencies and phase response
     */
    std::pair<std::vector<float>, std::vector<float>> get_phase_response(int num_points = 512) const;
    
    /**
     * @brief Check filter stability
     * 
     * @return true If the filter is stable
     */
    bool is_stable() const;
    
private:
    std::unique_ptr<IIRFilterImpl> impl_;
};

/**
 * @brief Adaptive filter with GPU acceleration
 */
class AdaptiveFilter {
public:
    /**
     * @brief Construct a new Adaptive Filter
     * 
     * @param params Adaptive filter parameters
     * @param device_id CUDA device ID (-1 for CPU)
     */
    AdaptiveFilter(const AdaptiveFilterParams& params, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~AdaptiveFilter();
    
    // Prevent copying
    AdaptiveFilter(const AdaptiveFilter&) = delete;
    AdaptiveFilter& operator=(const AdaptiveFilter&) = delete;
    
    // Allow moving
    AdaptiveFilter(AdaptiveFilter&&) noexcept;
    AdaptiveFilter& operator=(AdaptiveFilter&&) noexcept;
    
    /**
     * @brief Apply the adaptive filter and update coefficients
     * 
     * @param input Input signal
     * @param desired Desired signal (for error calculation)
     * @return std::pair<std::vector<float>, std::vector<float>> 
     *         Filtered signal and error signal
     */
    std::pair<std::vector<float>, std::vector<float>> filter(
        const std::vector<float>& input,
        const std::vector<float>& desired);
    
    /**
     * @brief Get the current filter coefficients
     * 
     * @return std::vector<float> Current filter coefficients
     */
    std::vector<float> get_coefficients() const;
    
    /**
     * @brief Get the learning curve (error vs iteration)
     * 
     * @return std::vector<float> Learning curve
     */
    std::vector<float> get_learning_curve() const;
    
    /**
     * @brief Reset the filter state and coefficients
     */
    void reset();
    
private:
    std::unique_ptr<AdaptiveFilterImpl> impl_;
};

/**
 * @brief Multirate filter for resampling operations
 */
class MultirateFilter {
public:
    /**
     * @brief Construct a new Multirate Filter
     * 
     * @param params Multirate filter parameters
     * @param device_id CUDA device ID (-1 for CPU)
     */
    MultirateFilter(const MultirateFilterParams& params, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~MultirateFilter();
    
    // Prevent copying
    MultirateFilter(const MultirateFilter&) = delete;
    MultirateFilter& operator=(const MultirateFilter&) = delete;
    
    // Allow moving
    MultirateFilter(MultirateFilter&&) noexcept;
    MultirateFilter& operator=(MultirateFilter&&) noexcept;
    
    /**
     * @brief Apply upsampling (interpolation)
     * 
     * @param input Input signal
     * @return std::vector<float> Upsampled signal
     */
    std::vector<float> upsample(const std::vector<float>& input);
    
    /**
     * @brief Apply downsampling (decimation)
     * 
     * @param input Input signal
     * @return std::vector<float> Downsampled signal
     */
    std::vector<float> downsample(const std::vector<float>& input);
    
    /**
     * @brief Apply resampling (rational rate conversion)
     * 
     * @param input Input signal
     * @return std::vector<float> Resampled signal
     */
    std::vector<float> resample(const std::vector<float>& input);
    
    /**
     * @brief Reset the filter state
     */
    void reset();
    
    /**
     * @brief Get the effective filter coefficients
     * 
     * @return std::vector<float> Filter coefficients
     */
    std::vector<float> get_coefficients() const;
    
private:
    std::unique_ptr<MultirateFilterImpl> impl_;
};

/**
 * @brief Static filtering functions with GPU acceleration
 */
namespace filters {

/**
 * @brief Apply median filtering to a signal
 * 
 * @param input Input signal
 * @param kernel_size Size of the median filter kernel
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::vector<float> Filtered signal
 */
std::vector<float> median_filter(
    const std::vector<float>& input,
    int kernel_size,
    int device_id = 0);

/**
 * @brief Apply one-dimensional convolution
 * 
 * @param input Input signal
 * @param kernel Convolution kernel
 * @param mode Boundary handling mode ('full', 'same', 'valid')
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::vector<float> Convolved signal
 */
std::vector<float> convolve(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    const std::string& mode = "same",
    int device_id = 0);

/**
 * @brief Apply Savitzky-Golay smoothing filter
 * 
 * @param input Input signal
 * @param window_length Window length (must be odd)
 * @param poly_order Polynomial order
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::vector<float> Smoothed signal
 */
std::vector<float> savitzky_golay(
    const std::vector<float>& input,
    int window_length,
    int poly_order,
    int device_id = 0);

/**
 * @brief Apply Wiener filter for noise reduction
 * 
 * @param input Input signal
 * @param noise_power Estimated noise power (variance)
 * @param kernel_size Size of the local variance estimation window
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::vector<float> Filtered signal
 */
std::vector<float> wiener_filter(
    const std::vector<float>& input,
    float noise_power,
    int kernel_size = 5,
    int device_id = 0);

/**
 * @brief Apply Kalman filter to a signal
 * 
 * @param input Input signal
 * @param process_variance Process variance (Q)
 * @param measurement_variance Measurement variance (R)
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::vector<float> Filtered signal
 */
std::vector<float> kalman_filter(
    const std::vector<float>& input,
    float process_variance,
    float measurement_variance,
    int device_id = 0);

/**
 * @brief Apply bilateral filter for edge-preserving smoothing
 * 
 * @param input Input signal
 * @param spatial_sigma Spatial sigma
 * @param range_sigma Range sigma
 * @param kernel_size Size of the filter kernel
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::vector<float> Filtered signal
 */
std::vector<float> bilateral_filter(
    const std::vector<float>& input,
    float spatial_sigma,
    float range_sigma,
    int kernel_size = 5,
    int device_id = 0);

/**
 * @brief Apply a custom filter function to the signal
 * 
 * @param input Input signal
 * @param filter_func Custom filter function
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::vector<float> Filtered signal
 */
std::vector<float> custom_filter(
    const std::vector<float>& input,
    const std::function<float(const std::vector<float>&, int)>& filter_func,
    int device_id = 0);

} // namespace filters

} // namespace signal_processing

#endif // SIGNAL_PROCESSING_DIGITAL_FILTERING_H