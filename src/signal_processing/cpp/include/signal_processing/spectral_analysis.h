/**
 * @file spectral_analysis.h
 * @brief FFT and spectral analysis functions optimized for NVIDIA GPUs
 * 
 * This file provides GPU-optimized implementations for:
 * - Fast Fourier Transform (FFT) and inverse FFT
 * - Power Spectral Density estimation
 * - Spectral correlation
 * - Harmonic analysis
 * - Cross-spectral analysis
 * 
 * The implementation automatically adapts to the available GPU architecture:
 * - Optimized for Jetson Orin NX (SM 8.7)
 * - Optimized for AWS T4G instances (SM 7.5)
 * - Fallback CPU implementation when GPU is unavailable
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#ifndef SIGNAL_PROCESSING_SPECTRAL_ANALYSIS_H
#define SIGNAL_PROCESSING_SPECTRAL_ANALYSIS_H

#include <complex>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <optional>
#include <array>

namespace signal_processing {

// Forward declarations
class FFTImpl;
class SpectralAnalyzerImpl;

/**
 * Window function types for spectral analysis
 */
enum class WindowType {
    RECTANGULAR,  // No windowing
    HANN,         // Hann window
    HAMMING,      // Hamming window
    BLACKMAN,     // Blackman window
    FLATTOP,      // Flat top window
    KAISER,       // Kaiser window
    TUKEY,        // Tukey window
    GAUSSIAN      // Gaussian window
};

/**
 * @brief Fast Fourier Transform (FFT) implementation with GPU acceleration
 * 
 * This class provides an optimized implementation of 1D and 2D FFT for signal processing
 * using GPU acceleration when available and CPU fallback otherwise.
 */
class FFT {
public:
    /**
     * @brief Construct a new FFT object
     * 
     * @param device_id CUDA device ID to use (-1 for CPU)
     */
    explicit FFT(int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~FFT();
    
    // Prevent copying
    FFT(const FFT&) = delete;
    FFT& operator=(const FFT&) = delete;
    
    // Allow moving
    FFT(FFT&&) noexcept;
    FFT& operator=(FFT&&) noexcept;
    
    /**
     * @brief Check if CUDA is available
     * 
     * @return true if CUDA is available
     */
    bool has_cuda() const;
    
    /**
     * @brief Get the CUDA device ID being used
     * 
     * @return int Device ID (-1 if using CPU)
     */
    int get_device_id() const;
    
    /**
     * @brief Compute 1D forward FFT of real input
     * 
     * @param input Real-valued input signal
     * @param normalize Whether to normalize the result
     * @return std::vector<std::complex<float>> Complex FFT result
     */
    std::vector<std::complex<float>> forward_1d_real(
        const std::vector<float>& input,
        bool normalize = false);
    
    /**
     * @brief Compute 1D forward FFT of complex input
     * 
     * @param input Complex-valued input signal
     * @param normalize Whether to normalize the result
     * @return std::vector<std::complex<float>> Complex FFT result
     */
    std::vector<std::complex<float>> forward_1d_complex(
        const std::vector<std::complex<float>>& input,
        bool normalize = false);
    
    /**
     * @brief Compute 1D inverse FFT to real output
     * 
     * @param input Complex-valued input frequency domain data
     * @param normalize Whether to normalize the result
     * @return std::vector<float> Real FFT result
     */
    std::vector<float> inverse_1d_real(
        const std::vector<std::complex<float>>& input,
        bool normalize = true);
    
    /**
     * @brief Compute 1D inverse FFT to complex output
     * 
     * @param input Complex-valued input frequency domain data
     * @param normalize Whether to normalize the result
     * @return std::vector<std::complex<float>> Complex FFT result
     */
    std::vector<std::complex<float>> inverse_1d_complex(
        const std::vector<std::complex<float>>& input,
        bool normalize = true);
    
    /**
     * @brief Compute 2D forward FFT of real input
     * 
     * @param input Real-valued input signal (row-major order)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param normalize Whether to normalize the result
     * @return std::vector<std::complex<float>> Complex FFT result (row-major order)
     */
    std::vector<std::complex<float>> forward_2d_real(
        const std::vector<float>& input,
        int rows,
        int cols,
        bool normalize = false);
    
    /**
     * @brief Compute 2D forward FFT of complex input
     * 
     * @param input Complex-valued input signal (row-major order)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param normalize Whether to normalize the result
     * @return std::vector<std::complex<float>> Complex FFT result (row-major order)
     */
    std::vector<std::complex<float>> forward_2d_complex(
        const std::vector<std::complex<float>>& input,
        int rows,
        int cols,
        bool normalize = false);
    
    /**
     * @brief Compute 2D inverse FFT to real output
     * 
     * @param input Complex-valued input frequency domain data (row-major order)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param normalize Whether to normalize the result
     * @return std::vector<float> Real FFT result (row-major order)
     */
    std::vector<float> inverse_2d_real(
        const std::vector<std::complex<float>>& input,
        int rows,
        int cols,
        bool normalize = true);
    
    /**
     * @brief Compute 2D inverse FFT to complex output
     * 
     * @param input Complex-valued input frequency domain data (row-major order)
     * @param rows Number of rows
     * @param cols Number of columns
     * @param normalize Whether to normalize the result
     * @return std::vector<std::complex<float>> Complex FFT result (row-major order)
     */
    std::vector<std::complex<float>> inverse_2d_complex(
        const std::vector<std::complex<float>>& input,
        int rows,
        int cols,
        bool normalize = true);
    
private:
    std::unique_ptr<FFTImpl> impl_;
};

/**
 * @brief Parameters for spectral analysis
 */
struct SpectralParams {
    WindowType window_type = WindowType::HANN;      // Window function type
    float window_param = 0.0f;                      // Parameter for parameterized windows
    int nfft = 0;                                   // FFT size (0 = auto)
    int overlap = 0;                                // Overlap between segments (0 = auto)
    float sample_rate = 1.0f;                       // Sample rate in Hz
    std::string detrend = "constant";              // Detrend method: "none", "constant", "linear"
    bool scaling = true;                            // Apply scaling
    bool return_onesided = true;                    // Return one-sided spectrum
};

/**
 * @brief Result of Power Spectral Density estimation
 */
struct PSDResult {
    std::vector<float> frequencies;                // Frequency bins
    std::vector<float> psd;                        // Power spectral density
    std::vector<float> coherence;                  // Optional coherence
};

/**
 * @brief Result of Cross Spectral Density estimation
 */
struct CSDResult {
    std::vector<float> frequencies;                // Frequency bins
    std::vector<std::complex<float>> csd;          // Cross spectral density
    std::vector<float> coherence;                  // Magnitude-squared coherence
    std::vector<float> phase;                      // Phase spectrum
};

/**
 * @brief Result of Spectrogram computation
 */
struct SpectrogramResult {
    std::vector<float> times;                      // Time bins (center of each segment)
    std::vector<float> frequencies;                // Frequency bins
    std::vector<std::vector<float>> spectrogram;   // Spectrogram values [time][frequency]
};

/**
 * @brief Spectral analysis class for advanced spectral processing
 * 
 * This class provides GPU-accelerated spectral analysis functions including:
 * - Power Spectral Density (PSD) estimation
 * - Cross Spectral Density (CSD) estimation
 * - Spectrogram computation
 * - Coherence estimation
 * - Periodogram computation
 */
class SpectralAnalyzer {
public:
    /**
     * @brief Construct a new SpectralAnalyzer object
     * 
     * @param device_id CUDA device ID to use (-1 for CPU)
     */
    explicit SpectralAnalyzer(int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~SpectralAnalyzer();
    
    // Prevent copying
    SpectralAnalyzer(const SpectralAnalyzer&) = delete;
    SpectralAnalyzer& operator=(const SpectralAnalyzer&) = delete;
    
    // Allow moving
    SpectralAnalyzer(SpectralAnalyzer&&) noexcept;
    SpectralAnalyzer& operator=(SpectralAnalyzer&&) noexcept;
    
    /**
     * @brief Check if CUDA is available
     * 
     * @return true if CUDA is available
     */
    bool has_cuda() const;
    
    /**
     * @brief Get the CUDA device ID being used
     * 
     * @return int Device ID (-1 if using CPU)
     */
    int get_device_id() const;
    
    /**
     * @brief Compute power spectral density (Welch's method)
     * 
     * @param signal Input signal
     * @param params Spectral analysis parameters
     * @return PSDResult Power spectral density result
     */
    PSDResult compute_psd(
        const std::vector<float>& signal,
        const SpectralParams& params = {});
    
    /**
     * @brief Compute cross spectral density between two signals
     * 
     * @param signal1 First input signal
     * @param signal2 Second input signal
     * @param params Spectral analysis parameters
     * @return CSDResult Cross spectral density result
     */
    CSDResult compute_csd(
        const std::vector<float>& signal1,
        const std::vector<float>& signal2,
        const SpectralParams& params = {});
    
    /**
     * @brief Compute spectrogram of a signal
     * 
     * @param signal Input signal
     * @param params Spectral analysis parameters
     * @return SpectrogramResult Spectrogram result
     */
    SpectrogramResult compute_spectrogram(
        const std::vector<float>& signal,
        const SpectralParams& params = {});
    
    /**
     * @brief Compute coherence between two signals
     * 
     * @param signal1 First input signal
     * @param signal2 Second input signal
     * @param params Spectral analysis parameters
     * @return PSDResult Coherence result (stored in coherence field)
     */
    PSDResult compute_coherence(
        const std::vector<float>& signal1,
        const std::vector<float>& signal2,
        const SpectralParams& params = {});
    
    /**
     * @brief Compute periodogram of a signal
     * 
     * @param signal Input signal
     * @param params Spectral analysis parameters
     * @return PSDResult Periodogram result
     */
    PSDResult compute_periodogram(
        const std::vector<float>& signal,
        const SpectralParams& params = {});
    
    /**
     * @brief Detect peaks in a spectrum
     * 
     * @param spectrum Input spectrum
     * @param frequencies Corresponding frequencies
     * @param threshold Peak detection threshold
     * @param min_distance Minimum distance between peaks
     * @return std::vector<std::pair<float, float>> Vector of (frequency, magnitude) pairs
     */
    std::vector<std::pair<float, float>> detect_peaks(
        const std::vector<float>& spectrum,
        const std::vector<float>& frequencies,
        float threshold = 0.5f,
        int min_distance = 1);
    
    /**
     * @brief Compute harmonic distortion
     * 
     * @param signal Input signal
     * @param fundamental_freq Fundamental frequency
     * @param num_harmonics Number of harmonics to analyze
     * @param params Spectral analysis parameters
     * @return std::vector<float> Harmonic distortion for each harmonic
     */
    std::vector<float> compute_harmonic_distortion(
        const std::vector<float>& signal,
        float fundamental_freq,
        int num_harmonics = 5,
        const SpectralParams& params = {});
    
private:
    std::unique_ptr<SpectralAnalyzerImpl> impl_;
};

} // namespace signal_processing

#endif // SIGNAL_PROCESSING_SPECTRAL_ANALYSIS_H