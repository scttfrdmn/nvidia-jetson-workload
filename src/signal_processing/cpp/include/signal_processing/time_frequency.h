/**
 * @file time_frequency.h
 * @brief Time-frequency analysis operations optimized for NVIDIA GPUs
 * 
 * This file provides GPU-optimized implementations for:
 * - Short-Time Fourier Transform (STFT)
 * - Continuous Wavelet Transform (CWT)
 * - Discrete Wavelet Transform (DWT)
 * - Wigner-Ville Distribution
 * - Empirical Mode Decomposition (EMD)
 * - Hilbert-Huang Transform
 * 
 * The implementation automatically adapts to the available GPU architecture:
 * - Optimized for Jetson Orin NX (SM 8.7)
 * - Optimized for AWS T4G instances (SM 7.5)
 * - Fallback CPU implementation when GPU is unavailable
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#ifndef SIGNAL_PROCESSING_TIME_FREQUENCY_H
#define SIGNAL_PROCESSING_TIME_FREQUENCY_H

#include <vector>
#include <complex>
#include <memory>
#include <string>
#include <optional>

namespace signal_processing {

// Forward declarations
class STFTImpl;
class CWTImpl;
class DWTImpl;
class WignerVilleImpl;
class EMDImpl;

/**
 * @brief Window types for STFT
 */
enum class WindowType {
    RECTANGULAR,  // Rectangular window
    HANN,         // Hann window
    HAMMING,      // Hamming window
    BLACKMAN,     // Blackman window
    KAISER        // Kaiser window
};

/**
 * @brief Wavelet types for CWT
 */
enum class WaveletType {
    MORLET,      // Morlet wavelet
    MEXICAN_HAT, // Mexican hat (Ricker) wavelet
    PAUL,        // Paul wavelet
    DOG,         // Derivative of Gaussian wavelet
    HAAR,        // Haar wavelet
    DB4          // Daubechies 4 wavelet
};

/**
 * @brief STFT Parameters
 */
struct STFTParams {
    int window_size = 1024;                     // Size of the window
    int hop_size = 256;                         // Hop size between windows
    WindowType window_type = WindowType::HANN;  // Window type
    float window_param = 0.0f;                  // Parameter for parameterized windows
    int fft_size = 0;                           // FFT size (0 = window_size)
    bool center = true;                         // Center the windows
    bool pad_mode = true;                       // Pad the signal
    std::string pad_mode_str = "reflect";       // Padding mode (reflect, constant, edge)
};

/**
 * @brief CWT Parameters
 */
struct CWTParams {
    WaveletType wavelet_type = WaveletType::MORLET; // Wavelet type
    float wavelet_param = 6.0f;                   // Wavelet-specific parameter
    int num_scales = 32;                          // Number of scales
    float min_scale = 1.0f;                       // Minimum scale
    float max_scale = 0.0f;                       // Maximum scale (0 = auto)
    bool normalize_scales = true;                 // Normalize the scales
};

/**
 * @brief DWT Parameters
 */
struct DWTParams {
    WaveletType wavelet_type = WaveletType::DB4;  // Wavelet type
    int levels = 0;                               // Number of decomposition levels (0 = auto)
    std::string mode = "reflect";                 // Border extension mode
    bool use_swt = false;                         // Use stationary wavelet transform
};

/**
 * @brief Short-Time Fourier Transform (STFT) result
 */
struct STFTResult {
    std::vector<std::vector<std::complex<float>>> spectrogram; // Time-frequency representation
    std::vector<float> times;                                // Time bins
    std::vector<float> frequencies;                          // Frequency bins
    float sample_rate;                                       // Sample rate
};

/**
 * @brief Continuous Wavelet Transform (CWT) result
 */
struct CWTResult {
    std::vector<std::vector<std::complex<float>>> scalogram;   // Time-scale representation
    std::vector<float> times;                                // Time bins
    std::vector<float> scales;                               // Scale bins
    std::vector<float> frequencies;                          // Corresponding frequencies
    float sample_rate;                                       // Sample rate
};

/**
 * @brief Discrete Wavelet Transform (DWT) result
 */
struct DWTResult {
    std::vector<std::vector<float>> coeffs;                  // Wavelet coefficients
    std::vector<int> lengths;                                // Coefficient lengths
    int levels;                                              // Number of decomposition levels
};

/**
 * @brief Intrinsic Mode Function (IMF) for EMD
 */
struct IMF {
    std::vector<float> signal;                               // The IMF signal
    float instantaneous_frequency;                           // Mean instantaneous frequency
    std::vector<float> envelope;                             // Envelope of the IMF
};

/**
 * @brief Empirical Mode Decomposition (EMD) result
 */
struct EMDResult {
    std::vector<IMF> imfs;                                   // Intrinsic mode functions
    std::vector<float> residue;                              // Residue signal
    int num_imfs;                                            // Number of IMFs
};

/**
 * @brief Short-Time Fourier Transform (STFT) with GPU acceleration
 */
class STFT {
public:
    /**
     * @brief Construct a new STFT object
     * 
     * @param params STFT parameters
     * @param device_id CUDA device ID (-1 for CPU)
     */
    STFT(const STFTParams& params = {}, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~STFT();
    
    // Prevent copying
    STFT(const STFT&) = delete;
    STFT& operator=(const STFT&) = delete;
    
    // Allow moving
    STFT(STFT&&) noexcept;
    STFT& operator=(STFT&&) noexcept;
    
    /**
     * @brief Compute STFT of a signal
     * 
     * @param signal Input signal
     * @param sample_rate Sample rate
     * @return STFTResult STFT result
     */
    STFTResult transform(const std::vector<float>& signal, float sample_rate);
    
    /**
     * @brief Compute inverse STFT
     * 
     * @param stft_result STFT result
     * @return std::vector<float> Reconstructed signal
     */
    std::vector<float> inverse_transform(const STFTResult& stft_result);
    
    /**
     * @brief Get magnitude spectrogram from STFT result
     * 
     * @param stft_result STFT result
     * @param log_scale Apply log scaling to magnitudes
     * @return std::vector<std::vector<float>> Magnitude spectrogram
     */
    std::vector<std::vector<float>> get_magnitude(
        const STFTResult& stft_result,
        bool log_scale = false);
    
    /**
     * @brief Get phase spectrogram from STFT result
     * 
     * @param stft_result STFT result
     * @return std::vector<std::vector<float>> Phase spectrogram
     */
    std::vector<std::vector<float>> get_phase(const STFTResult& stft_result);
    
    /**
     * @brief Get power spectrogram from STFT result
     * 
     * @param stft_result STFT result
     * @param log_scale Apply log scaling to power values
     * @return std::vector<std::vector<float>> Power spectrogram
     */
    std::vector<std::vector<float>> get_power(
        const STFTResult& stft_result,
        bool log_scale = false);
    
private:
    std::unique_ptr<STFTImpl> impl_;
};

/**
 * @brief Continuous Wavelet Transform (CWT) with GPU acceleration
 */
class CWT {
public:
    /**
     * @brief Construct a new CWT object
     * 
     * @param params CWT parameters
     * @param device_id CUDA device ID (-1 for CPU)
     */
    CWT(const CWTParams& params = {}, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~CWT();
    
    // Prevent copying
    CWT(const CWT&) = delete;
    CWT& operator=(const CWT&) = delete;
    
    // Allow moving
    CWT(CWT&&) noexcept;
    CWT& operator=(CWT&&) noexcept;
    
    /**
     * @brief Compute CWT of a signal
     * 
     * @param signal Input signal
     * @param sample_rate Sample rate
     * @return CWTResult CWT result
     */
    CWTResult transform(const std::vector<float>& signal, float sample_rate);
    
    /**
     * @brief Get magnitude scalogram from CWT result
     * 
     * @param cwt_result CWT result
     * @param log_scale Apply log scaling to magnitudes
     * @return std::vector<std::vector<float>> Magnitude scalogram
     */
    std::vector<std::vector<float>> get_magnitude(
        const CWTResult& cwt_result,
        bool log_scale = false);
    
    /**
     * @brief Get phase scalogram from CWT result
     * 
     * @param cwt_result CWT result
     * @return std::vector<std::vector<float>> Phase scalogram
     */
    std::vector<std::vector<float>> get_phase(const CWTResult& cwt_result);
    
    /**
     * @brief Get power scalogram from CWT result
     * 
     * @param cwt_result CWT result
     * @param log_scale Apply log scaling to power values
     * @return std::vector<std::vector<float>> Power scalogram
     */
    std::vector<std::vector<float>> get_power(
        const CWTResult& cwt_result,
        bool log_scale = false);
    
private:
    std::unique_ptr<CWTImpl> impl_;
};

/**
 * @brief Discrete Wavelet Transform (DWT) with GPU acceleration
 */
class DWT {
public:
    /**
     * @brief Construct a new DWT object
     * 
     * @param params DWT parameters
     * @param device_id CUDA device ID (-1 for CPU)
     */
    DWT(const DWTParams& params = {}, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~DWT();
    
    // Prevent copying
    DWT(const DWT&) = delete;
    DWT& operator=(const DWT&) = delete;
    
    // Allow moving
    DWT(DWT&&) noexcept;
    DWT& operator=(DWT&&) noexcept;
    
    /**
     * @brief Compute DWT of a signal
     * 
     * @param signal Input signal
     * @return DWTResult DWT result
     */
    DWTResult transform(const std::vector<float>& signal);
    
    /**
     * @brief Compute inverse DWT
     * 
     * @param dwt_result DWT result
     * @return std::vector<float> Reconstructed signal
     */
    std::vector<float> inverse_transform(const DWTResult& dwt_result);
    
    /**
     * @brief Denoise a signal using wavelet thresholding
     * 
     * @param signal Input signal
     * @param threshold Threshold value
     * @param threshold_mode Threshold mode ("soft" or "hard")
     * @return std::vector<float> Denoised signal
     */
    std::vector<float> denoise(
        const std::vector<float>& signal,
        float threshold,
        const std::string& threshold_mode = "soft");
    
private:
    std::unique_ptr<DWTImpl> impl_;
};

/**
 * @brief Wigner-Ville Distribution with GPU acceleration
 */
class WignerVille {
public:
    /**
     * @brief Construct a new WignerVille object
     * 
     * @param device_id CUDA device ID (-1 for CPU)
     */
    explicit WignerVille(int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~WignerVille();
    
    // Prevent copying
    WignerVille(const WignerVille&) = delete;
    WignerVille& operator=(const WignerVille&) = delete;
    
    // Allow moving
    WignerVille(WignerVille&&) noexcept;
    WignerVille& operator=(WignerVille&&) noexcept;
    
    /**
     * @brief Compute Wigner-Ville distribution of a signal
     * 
     * @param signal Input signal
     * @param sample_rate Sample rate
     * @return std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
     *         Distribution and time-frequency axes
     */
    std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
    transform(const std::vector<float>& signal, float sample_rate);
    
    /**
     * @brief Compute pseudo Wigner-Ville distribution (smoothed)
     * 
     * @param signal Input signal
     * @param sample_rate Sample rate
     * @param window_size Window size for time smoothing
     * @return std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
     *         Distribution and time-frequency axes
     */
    std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
    transform_pseudo(
        const std::vector<float>& signal,
        float sample_rate,
        int window_size = 127);
    
private:
    std::unique_ptr<WignerVilleImpl> impl_;
};

/**
 * @brief Empirical Mode Decomposition (EMD) with GPU acceleration
 */
class EMD {
public:
    /**
     * @brief Construct a new EMD object
     * 
     * @param max_imfs Maximum number of IMFs to extract (0 = automatic)
     * @param device_id CUDA device ID (-1 for CPU)
     */
    explicit EMD(int max_imfs = 0, int device_id = 0);
    
    /**
     * @brief Destructor
     */
    ~EMD();
    
    // Prevent copying
    EMD(const EMD&) = delete;
    EMD& operator=(const EMD&) = delete;
    
    // Allow moving
    EMD(EMD&&) noexcept;
    EMD& operator=(EMD&&) noexcept;
    
    /**
     * @brief Decompose a signal into IMFs
     * 
     * @param signal Input signal
     * @param sample_rate Sample rate
     * @return EMDResult EMD result
     */
    EMDResult decompose(const std::vector<float>& signal, float sample_rate);
    
    /**
     * @brief Compute Hilbert-Huang spectrum
     * 
     * @param emd_result EMD result
     * @param sample_rate Sample rate
     * @param num_freqs Number of frequency bins
     * @return std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
     *         Spectrum and time-frequency axes
     */
    std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
    hilbert_huang_spectrum(
        const EMDResult& emd_result,
        float sample_rate,
        int num_freqs = 256);
    
    /**
     * @brief Reconstruct signal from IMFs
     * 
     * @param emd_result EMD result
     * @param imf_indices Indices of IMFs to include (empty = all)
     * @return std::vector<float> Reconstructed signal
     */
    std::vector<float> reconstruct(
        const EMDResult& emd_result,
        const std::vector<int>& imf_indices = {});
    
private:
    std::unique_ptr<EMDImpl> impl_;
};

/**
 * @brief Static time-frequency analysis functions
 */
namespace time_frequency {

/**
 * @brief Compute spectrogram of a signal
 * 
 * @param signal Input signal
 * @param sample_rate Sample rate
 * @param window_size Window size
 * @param hop_size Hop size
 * @param window_type Window type
 * @param log_scale Apply log scaling
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
 *         Spectrogram and time-frequency axes
 */
std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
spectrogram(
    const std::vector<float>& signal,
    float sample_rate,
    int window_size = 1024,
    int hop_size = 256,
    WindowType window_type = WindowType::HANN,
    bool log_scale = true,
    int device_id = 0);

/**
 * @brief Compute scalogram of a signal
 * 
 * @param signal Input signal
 * @param sample_rate Sample rate
 * @param wavelet_type Wavelet type
 * @param num_scales Number of scales
 * @param log_scale Apply log scaling
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
 *         Scalogram and time-scale axes
 */
std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
scalogram(
    const std::vector<float>& signal,
    float sample_rate,
    WaveletType wavelet_type = WaveletType::MORLET,
    int num_scales = 32,
    bool log_scale = true,
    int device_id = 0);

/**
 * @brief Compute Mel spectrogram of a signal
 * 
 * @param signal Input signal
 * @param sample_rate Sample rate
 * @param n_fft FFT size
 * @param hop_size Hop size
 * @param n_mels Number of Mel bands
 * @param fmin Minimum frequency
 * @param fmax Maximum frequency
 * @param log_scale Apply log scaling
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
 *         Mel spectrogram and time-frequency axes
 */
std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
mel_spectrogram(
    const std::vector<float>& signal,
    float sample_rate,
    int n_fft = 2048,
    int hop_size = 512,
    int n_mels = 128,
    float fmin = 0.0f,
    float fmax = 0.0f,
    bool log_scale = true,
    int device_id = 0);

/**
 * @brief Compute MFCC (Mel-frequency cepstral coefficients)
 * 
 * @param signal Input signal
 * @param sample_rate Sample rate
 * @param n_mfcc Number of MFCCs
 * @param n_fft FFT size
 * @param hop_size Hop size
 * @param n_mels Number of Mel bands
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::pair<std::vector<std::vector<float>>, std::vector<float>>
 *         MFCCs and time axis
 */
std::pair<std::vector<std::vector<float>>, std::vector<float>>
mfcc(
    const std::vector<float>& signal,
    float sample_rate,
    int n_mfcc = 13,
    int n_fft = 2048,
    int hop_size = 512,
    int n_mels = 128,
    int device_id = 0);

/**
 * @brief Compute chroma feature
 * 
 * @param signal Input signal
 * @param sample_rate Sample rate
 * @param n_fft FFT size
 * @param hop_size Hop size
 * @param n_chroma Number of chroma bins
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::pair<std::vector<std::vector<float>>, std::vector<float>>
 *         Chroma features and time axis
 */
std::pair<std::vector<std::vector<float>>, std::vector<float>>
chroma(
    const std::vector<float>& signal,
    float sample_rate,
    int n_fft = 2048,
    int hop_size = 512,
    int n_chroma = 12,
    int device_id = 0);

/**
 * @brief Compute Hilbert transform of a signal
 * 
 * @param signal Input signal
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::vector<std::complex<float>> Analytic signal
 */
std::vector<std::complex<float>> hilbert_transform(
    const std::vector<float>& signal,
    int device_id = 0);

/**
 * @brief Compute instantaneous frequency of a signal
 * 
 * @param signal Input signal
 * @param sample_rate Sample rate
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::vector<float> Instantaneous frequency
 */
std::vector<float> instantaneous_frequency(
    const std::vector<float>& signal,
    float sample_rate,
    int device_id = 0);

/**
 * @brief Compute reassigned spectrogram
 * 
 * @param signal Input signal
 * @param sample_rate Sample rate
 * @param window_size Window size
 * @param hop_size Hop size
 * @param device_id CUDA device ID (-1 for CPU)
 * @return std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
 *         Reassigned spectrogram and time-frequency axes
 */
std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
reassigned_spectrogram(
    const std::vector<float>& signal,
    float sample_rate,
    int window_size = 1024,
    int hop_size = 256,
    int device_id = 0);

} // namespace time_frequency

} // namespace signal_processing

#endif // SIGNAL_PROCESSING_TIME_FREQUENCY_H