/**
 * @file time_frequency_kernels.cu
 * @brief CUDA kernels for time-frequency analysis operations
 * 
 * This file contains CUDA kernels and device functions for:
 * - Short-Time Fourier Transform (STFT)
 * - Continuous Wavelet Transform (CWT)
 * - Discrete Wavelet Transform (DWT)
 * - Wigner-Ville Distribution
 * - Mel-frequency operations
 * 
 * Kernels are optimized for different GPU architectures:
 * - SM 8.7 for Jetson Orin NX
 * - SM 7.5 for AWS T4G
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

#include <cmath>
#include <complex>

namespace cg = cooperative_groups;

namespace signal_processing {
namespace kernels {

// Constants
constexpr float PI = 3.14159265358979323846f;

//------------------------------------------------------------------------------
// GPU Architecture Adaptation
//------------------------------------------------------------------------------

// Choose optimal block size based on SM architecture
inline int get_optimal_block_size(cudaDeviceProp& props) {
    if (props.major == 8 && props.minor == 7) {
        // Jetson Orin NX (SM 8.7)
        return 128;
    } else if (props.major == 7 && props.minor == 5) {
        // AWS T4G with Tesla T4 (SM 7.5)
        return 256;
    } else {
        // Default for other architectures
        return 256;
    }
}

//------------------------------------------------------------------------------
// STFT Kernels
//------------------------------------------------------------------------------

/**
 * @brief Apply window function to signal frame
 */
template<typename T>
__global__ void stft_window_kernel(
    const T* __restrict__ signal,
    T* __restrict__ windowed_signal,
    const float* __restrict__ window,
    int frame_size,
    int hop_size,
    int n_frames,
    int signal_length) {
    
    extern __shared__ float shared_window[];
    
    // Load window into shared memory
    if (threadIdx.x < frame_size) {
        shared_window[threadIdx.x] = window[threadIdx.x];
    }
    
    __syncthreads();
    
    int frame_idx = blockIdx.x;
    int sample_idx = threadIdx.x;
    
    if (frame_idx < n_frames && sample_idx < frame_size) {
        int signal_idx = frame_idx * hop_size + sample_idx;
        
        // Apply window
        if (signal_idx < signal_length) {
            windowed_signal[frame_idx * frame_size + sample_idx] = signal[signal_idx] * shared_window[sample_idx];
        } else {
            windowed_signal[frame_idx * frame_size + sample_idx] = 0.0f;
        }
    }
}

/**
 * @brief Convert complex FFT results to magnitude/power spectrogram
 */
__global__ void stft_magnitude_kernel(
    const cufftComplex* __restrict__ stft_data,
    float* __restrict__ magnitude,
    int n_frames,
    int n_bins,
    float scale,
    bool log_scale,
    float min_value = 1e-10f) {
    
    int frame_idx = blockIdx.x;
    int bin_idx = threadIdx.x;
    
    if (frame_idx < n_frames && bin_idx < n_bins) {
        int idx = frame_idx * n_bins + bin_idx;
        float real = stft_data[idx].x;
        float imag = stft_data[idx].y;
        float mag = sqrt(real * real + imag * imag) * scale;
        
        if (log_scale) {
            magnitude[idx] = 10.0f * log10f(max(mag * mag, min_value));
        } else {
            magnitude[idx] = mag;
        }
    }
}

/**
 * @brief Extract phase from complex STFT data
 */
__global__ void stft_phase_kernel(
    const cufftComplex* __restrict__ stft_data,
    float* __restrict__ phase,
    int n_frames,
    int n_bins) {
    
    int frame_idx = blockIdx.x;
    int bin_idx = threadIdx.x;
    
    if (frame_idx < n_frames && bin_idx < n_bins) {
        int idx = frame_idx * n_bins + bin_idx;
        float real = stft_data[idx].x;
        float imag = stft_data[idx].y;
        phase[idx] = atan2f(imag, real);
    }
}

/**
 * @brief Optimized STFT kernel for SM 8.7 architecture (Jetson Orin NX)
 */
__global__ void __launch_bounds__(128, 8)
stft_sm87_kernel(
    const float* __restrict__ signal,
    cufftComplex* __restrict__ stft_data,
    const float* __restrict__ window,
    int signal_length,
    int frame_size,
    int hop_size,
    int n_frames,
    int fft_size) {
    
    extern __shared__ float shared_mem[];
    
    float* shared_window = shared_mem;
    float* shared_signal = shared_mem + frame_size;
    
    // Each thread block processes one frame
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (frame_idx < n_frames) {
        // Load window into shared memory
        if (tid < frame_size) {
            shared_window[tid] = window[tid];
        }
        
        __syncthreads();
        
        // Load signal segment into shared memory
        int signal_start = frame_idx * hop_size;
        
        // Each thread loads multiple samples
        const int samples_per_thread = (frame_size + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < samples_per_thread; ++i) {
            int sample_idx = tid * samples_per_thread + i;
            if (sample_idx < frame_size) {
                int signal_idx = signal_start + sample_idx;
                if (signal_idx < signal_length) {
                    shared_signal[sample_idx] = signal[signal_idx] * shared_window[sample_idx];
                } else {
                    shared_signal[sample_idx] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Copy windowed signal to output (will be processed by cuFFT)
        // Pad with zeros if frame_size < fft_size
        for (int i = tid; i < fft_size; i += blockDim.x) {
            if (i < frame_size) {
                stft_data[frame_idx * fft_size + i].x = shared_signal[i];
                stft_data[frame_idx * fft_size + i].y = 0.0f;
            } else {
                stft_data[frame_idx * fft_size + i].x = 0.0f;
                stft_data[frame_idx * fft_size + i].y = 0.0f;
            }
        }
    }
}

/**
 * @brief Optimized STFT kernel for SM 7.5 architecture (AWS T4G)
 */
__global__ void __launch_bounds__(256, 4)
stft_sm75_kernel(
    const float* __restrict__ signal,
    cufftComplex* __restrict__ stft_data,
    const float* __restrict__ window,
    int signal_length,
    int frame_size,
    int hop_size,
    int n_frames,
    int fft_size) {
    
    extern __shared__ float shared_mem[];
    
    float* shared_window = shared_mem;
    float* shared_signal = shared_mem + frame_size;
    
    // Each thread block processes one frame
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (frame_idx < n_frames) {
        // Load window into shared memory
        if (tid < frame_size) {
            shared_window[tid] = window[tid];
        }
        
        __syncthreads();
        
        // Load signal segment into shared memory and apply window
        int signal_start = frame_idx * hop_size;
        for (int i = tid; i < frame_size; i += blockDim.x) {
            int signal_idx = signal_start + i;
            if (signal_idx < signal_length) {
                shared_signal[i] = signal[signal_idx] * shared_window[i];
            } else {
                shared_signal[i] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Copy windowed signal to output (will be processed by cuFFT)
        // Pad with zeros if frame_size < fft_size
        for (int i = tid; i < fft_size; i += blockDim.x) {
            if (i < frame_size) {
                stft_data[frame_idx * fft_size + i].x = shared_signal[i];
                stft_data[frame_idx * fft_size + i].y = 0.0f;
            } else {
                stft_data[frame_idx * fft_size + i].x = 0.0f;
                stft_data[frame_idx * fft_size + i].y = 0.0f;
            }
        }
    }
}

/**
 * @brief Inverse STFT kernel for overlap-add synthesis
 */
__global__ void istft_overlap_add_kernel(
    const cufftComplex* __restrict__ stft_data,
    float* __restrict__ output,
    const float* __restrict__ window,
    int signal_length,
    int frame_size,
    int hop_size,
    int n_frames,
    int fft_size) {
    
    extern __shared__ float shared_mem[];
    
    float* shared_window = shared_mem;
    float* shared_frame = shared_mem + frame_size;
    
    // Load window into shared memory
    if (threadIdx.x < frame_size) {
        shared_window[threadIdx.x] = window[threadIdx.x];
    }
    
    __syncthreads();
    
    // Each thread processes one sample across all frames
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample_idx < signal_length) {
        float sum = 0.0f;
        float window_sum = 0.0f;
        
        // Find all frames that contribute to this sample
        for (int frame_idx = 0; frame_idx < n_frames; ++frame_idx) {
            int frame_start = frame_idx * hop_size;
            int frame_offset = sample_idx - frame_start;
            
            if (frame_offset >= 0 && frame_offset < frame_size) {
                float window_val = shared_window[frame_offset];
                sum += stft_data[frame_idx * fft_size + frame_offset].x * window_val;
                window_sum += window_val * window_val;
            }
        }
        
        // Normalize by window sum to ensure perfect reconstruction
        if (window_sum > 1e-10f) {
            output[sample_idx] = sum / window_sum;
        } else {
            output[sample_idx] = sum;
        }
    }
}

//------------------------------------------------------------------------------
// CWT Kernels
//------------------------------------------------------------------------------

/**
 * @brief Generate Morlet wavelet kernel
 */
__device__ cufftComplex morlet_wavelet(float t, float scale, float wavelet_param) {
    float t_scaled = t / scale;
    float arg = -0.5f * t_scaled * t_scaled;
    float envelope = expf(arg);
    float cos_term = cosf(wavelet_param * t_scaled);
    float sin_term = sinf(wavelet_param * t_scaled);
    
    // Normalization factor
    float norm = 1.0f / sqrtf(scale);
    
    return make_cuFloatComplex(norm * envelope * cos_term, norm * envelope * sin_term);
}

/**
 * @brief Generate Mexican hat wavelet kernel
 */
__device__ cufftComplex mexican_hat_wavelet(float t, float scale, float wavelet_param) {
    float t_scaled = t / scale;
    float arg = -0.5f * t_scaled * t_scaled;
    float term = (1.0f - t_scaled * t_scaled) * expf(arg);
    
    // Normalization factor
    float norm = 1.0f / sqrtf(scale);
    
    return make_cuFloatComplex(norm * term, 0.0f);
}

/**
 * @brief CWT kernel using direct convolution
 */
__global__ void cwt_direct_kernel(
    const float* __restrict__ signal,
    cufftComplex* __restrict__ cwt_data,
    int signal_length,
    int n_scales,
    const float* __restrict__ scales,
    int wavelet_type,
    float wavelet_param,
    int wavelet_width) {
    
    int scale_idx = blockIdx.x;
    int time_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (scale_idx < n_scales && time_idx < signal_length) {
        float scale = scales[scale_idx];
        float half_wavelet = wavelet_width * scale * 0.5f;
        
        cufftComplex sum = make_cuFloatComplex(0.0f, 0.0f);
        
        // Convolve signal with wavelet
        for (int i = -wavelet_width; i <= wavelet_width; ++i) {
            float t = i * 0.5f;  // Discretized time
            int signal_idx = time_idx + i;
            
            // Handle boundary conditions
            if (signal_idx < 0) {
                signal_idx = 0;
            } else if (signal_idx >= signal_length) {
                signal_idx = signal_length - 1;
            }
            
            // Generate wavelet sample
            cufftComplex wavelet;
            if (wavelet_type == 0) {  // Morlet
                wavelet = morlet_wavelet(t, scale, wavelet_param);
            } else if (wavelet_type == 1) {  // Mexican hat
                wavelet = mexican_hat_wavelet(t, scale, wavelet_param);
            } else {
                wavelet = make_cuFloatComplex(0.0f, 0.0f);
            }
            
            // Multiply signal with wavelet
            float signal_val = signal[signal_idx];
            sum.x += signal_val * wavelet.x;
            sum.y += signal_val * wavelet.y;
        }
        
        // Store result
        cwt_data[scale_idx * signal_length + time_idx] = sum;
    }
}

/**
 * @brief Generate wavelet filter kernel for frequency domain CWT
 */
__global__ void cwt_generate_filter_kernel(
    cufftComplex* __restrict__ wavelet_filters,
    const float* __restrict__ frequencies,
    const float* __restrict__ scales,
    int n_frequencies,
    int n_scales,
    int wavelet_type,
    float wavelet_param) {
    
    int scale_idx = blockIdx.x;
    int freq_idx = threadIdx.x;
    
    if (scale_idx < n_scales && freq_idx < n_frequencies) {
        float scale = scales[scale_idx];
        float freq = frequencies[freq_idx];
        
        cufftComplex filter;
        
        // Generate wavelet filter in frequency domain
        if (wavelet_type == 0) {  // Morlet
            float freq_scaled = scale * freq;
            float term = -0.5f * (freq_scaled - wavelet_param) * (freq_scaled - wavelet_param);
            filter.x = expf(term);
            filter.y = 0.0f;
        } else if (wavelet_type == 1) {  // Mexican hat
            float freq_scaled = scale * freq;
            float term = -0.5f * freq_scaled * freq_scaled;
            filter.x = freq_scaled * freq_scaled * expf(term);
            filter.y = 0.0f;
        } else {
            filter.x = 0.0f;
            filter.y = 0.0f;
        }
        
        // Normalization
        float norm = sqrtf(scale);
        filter.x *= norm;
        filter.y *= norm;
        
        // Store filter
        wavelet_filters[scale_idx * n_frequencies + freq_idx] = filter;
    }
}

/**
 * @brief CWT kernel using frequency domain multiplication
 */
__global__ void cwt_frequency_kernel(
    const cufftComplex* __restrict__ signal_fft,
    const cufftComplex* __restrict__ wavelet_filters,
    cufftComplex* __restrict__ cwt_data,
    int n_frequencies,
    int n_scales) {
    
    int scale_idx = blockIdx.x;
    int freq_idx = threadIdx.x;
    
    if (scale_idx < n_scales && freq_idx < n_frequencies) {
        // Get signal and filter components
        cufftComplex signal_val = signal_fft[freq_idx];
        cufftComplex filter_val = wavelet_filters[scale_idx * n_frequencies + freq_idx];
        
        // Complex multiplication
        float a = signal_val.x;
        float b = signal_val.y;
        float c = filter_val.x;
        float d = filter_val.y;
        
        cufftComplex result;
        result.x = a * c - b * d;
        result.y = a * d + b * c;
        
        // Store result for inverse FFT
        cwt_data[scale_idx * n_frequencies + freq_idx] = result;
    }
}

/**
 * @brief CWT scalogram computation kernel
 */
__global__ void cwt_scalogram_kernel(
    const cufftComplex* __restrict__ cwt_data,
    float* __restrict__ scalogram,
    int signal_length,
    int n_scales,
    bool log_scale,
    float min_value = 1e-10f) {
    
    int scale_idx = blockIdx.x;
    int time_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (scale_idx < n_scales && time_idx < signal_length) {
        int idx = scale_idx * signal_length + time_idx;
        float real = cwt_data[idx].x;
        float imag = cwt_data[idx].y;
        float power = real * real + imag * imag;
        
        if (log_scale) {
            scalogram[idx] = 10.0f * log10f(max(power, min_value));
        } else {
            scalogram[idx] = power;
        }
    }
}

//------------------------------------------------------------------------------
// DWT Kernels
//------------------------------------------------------------------------------

/**
 * @brief Haar wavelet transform forward kernel
 */
__global__ void dwt_haar_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ approx,
    float* __restrict__ detail,
    int length) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_idx = idx;
    int input_idx = idx * 2;
    
    if (output_idx < length / 2) {
        // Scaling coefficients (approximation)
        approx[output_idx] = (input[input_idx] + input[input_idx + 1]) * 0.7071067811865475f;  // 1/sqrt(2)
        
        // Wavelet coefficients (detail)
        detail[output_idx] = (input[input_idx] - input[input_idx + 1]) * 0.7071067811865475f;  // 1/sqrt(2)
    }
}

/**
 * @brief Haar wavelet transform inverse kernel
 */
__global__ void dwt_haar_inverse_kernel(
    const float* __restrict__ approx,
    const float* __restrict__ detail,
    float* __restrict__ output,
    int length) {
    
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int input_idx = output_idx / 2;
    bool is_even = (output_idx % 2) == 0;
    
    if (output_idx < length) {
        float a = approx[input_idx] * 0.7071067811865475f;  // 1/sqrt(2)
        float d = detail[input_idx] * 0.7071067811865475f;  // 1/sqrt(2)
        
        if (is_even) {
            output[output_idx] = a + d;
        } else {
            output[output_idx] = a - d;
        }
    }
}

/**
 * @brief Daubechies 4 wavelet transform forward kernel
 */
__global__ void dwt_db4_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ approx,
    float* __restrict__ detail,
    int length) {
    
    extern __shared__ float shared_mem[];
    
    // Daubechies 4 filter coefficients
    const float h0 = 0.482962913144534f;
    const float h1 = 0.836516303737808f;
    const float h2 = 0.224143868042013f;
    const float h3 = -0.129409522551260f;
    
    const float g0 = h3;
    const float g1 = -h2;
    const float g2 = h1;
    const float g3 = -h0;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_idx = idx;
    
    if (output_idx < length / 2) {
        int i = output_idx * 2;
        
        // Load input into shared memory with padding
        // Need 4 samples for each output
        int shared_idx = threadIdx.x * 4;
        for (int j = 0; j < 4; ++j) {
            int input_idx = i + j - 1;
            
            // Handle boundary conditions
            if (input_idx < 0) {
                input_idx = -input_idx;
            } else if (input_idx >= length) {
                input_idx = 2 * length - input_idx - 2;
            }
            
            shared_mem[shared_idx + j] = input[input_idx];
        }
        
        __syncthreads();
        
        // Compute approximation coefficient
        approx[output_idx] = h0 * shared_mem[shared_idx] +
                             h1 * shared_mem[shared_idx + 1] +
                             h2 * shared_mem[shared_idx + 2] +
                             h3 * shared_mem[shared_idx + 3];
        
        // Compute detail coefficient
        detail[output_idx] = g0 * shared_mem[shared_idx] +
                             g1 * shared_mem[shared_idx + 1] +
                             g2 * shared_mem[shared_idx + 2] +
                             g3 * shared_mem[shared_idx + 3];
    }
}

//------------------------------------------------------------------------------
// Wigner-Ville Distribution Kernels
//------------------------------------------------------------------------------

/**
 * @brief Generate analytic signal using Hilbert transform
 */
__global__ void analytic_signal_kernel(
    const cufftComplex* __restrict__ fft_data,
    cufftComplex* __restrict__ analytic_fft,
    int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (idx == 0) {
            // DC component: no change
            analytic_fft[idx] = fft_data[idx];
        } else if (idx < n / 2) {
            // Positive frequencies: multiply by 2
            analytic_fft[idx].x = 2.0f * fft_data[idx].x;
            analytic_fft[idx].y = 2.0f * fft_data[idx].y;
        } else {
            // Negative frequencies: set to zero
            analytic_fft[idx].x = 0.0f;
            analytic_fft[idx].y = 0.0f;
        }
    }
}

/**
 * @brief Compute Wigner-Ville distribution
 */
__global__ void wigner_ville_kernel(
    const cufftComplex* __restrict__ analytic_signal,
    float* __restrict__ wvd,
    int signal_length,
    int n_freqs) {
    
    int time_idx = blockIdx.x;
    int freq_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (time_idx < signal_length && freq_idx < n_freqs) {
        float sum_real = 0.0f;
        float sum_imag = 0.0f;
        
        // Maximum lag is half the signal length
        int max_lag = min(time_idx, signal_length - time_idx - 1);
        
        // Compute auto-correlation and FFT for each lag
        for (int lag = -max_lag; lag <= max_lag; ++lag) {
            // Get values at t+lag/2 and t-lag/2
            int idx_plus = time_idx + lag / 2;
            int idx_minus = time_idx - lag / 2;
            
            // Ensure indices are valid
            if (idx_plus >= 0 && idx_plus < signal_length &&
                idx_minus >= 0 && idx_minus < signal_length) {
                
                // Compute auto-correlation
                float z1_real = analytic_signal[idx_plus].x;
                float z1_imag = analytic_signal[idx_plus].y;
                float z2_real = analytic_signal[idx_minus].x;
                float z2_imag = -analytic_signal[idx_minus].y;  // Complex conjugate
                
                float corr_real = z1_real * z2_real - z1_imag * z2_imag;
                float corr_imag = z1_real * z2_imag + z1_imag * z2_real;
                
                // Apply FFT for lag (direct computation for simplicity)
                float angle = -2.0f * PI * freq_idx * lag / n_freqs;
                float cos_term = cosf(angle);
                float sin_term = sinf(angle);
                
                sum_real += corr_real * cos_term - corr_imag * sin_term;
                sum_imag += corr_real * sin_term + corr_imag * cos_term;
            }
        }
        
        // Take magnitude squared and store
        wvd[time_idx * n_freqs + freq_idx] = sum_real * sum_real + sum_imag * sum_imag;
    }
}

//------------------------------------------------------------------------------
// Mel-Spectrogram and MFCC Kernels
//------------------------------------------------------------------------------

/**
 * @brief Apply Mel filterbank to spectrogram
 */
__global__ void mel_filterbank_kernel(
    const float* __restrict__ spectrogram,
    float* __restrict__ mel_spectrogram,
    const float* __restrict__ mel_filters,
    int n_frames,
    int n_freqs,
    int n_mels) {
    
    int frame_idx = blockIdx.x;
    int mel_idx = threadIdx.x;
    
    if (frame_idx < n_frames && mel_idx < n_mels) {
        float sum = 0.0f;
        
        // Apply Mel filter
        for (int freq_idx = 0; freq_idx < n_freqs; ++freq_idx) {
            sum += spectrogram[frame_idx * n_freqs + freq_idx] * 
                   mel_filters[mel_idx * n_freqs + freq_idx];
        }
        
        mel_spectrogram[frame_idx * n_mels + mel_idx] = sum;
    }
}

/**
 * @brief Apply DCT to compute MFCCs from Mel spectrogram
 */
__global__ void mfcc_dct_kernel(
    const float* __restrict__ mel_spectrogram,
    float* __restrict__ mfcc,
    int n_frames,
    int n_mels,
    int n_mfcc) {
    
    int frame_idx = blockIdx.x;
    int mfcc_idx = threadIdx.x;
    
    if (frame_idx < n_frames && mfcc_idx < n_mfcc) {
        float sum = 0.0f;
        
        // Apply DCT-II
        for (int mel_idx = 0; mel_idx < n_mels; ++mel_idx) {
            float mel_val = logf(max(mel_spectrogram[frame_idx * n_mels + mel_idx], 1e-10f));
            float angle = (PI / n_mels) * (mel_idx + 0.5f) * mfcc_idx;
            sum += mel_val * cosf(angle);
        }
        
        // Scale
        if (mfcc_idx == 0) {
            sum *= sqrtf(1.0f / n_mels);
        } else {
            sum *= sqrtf(2.0f / n_mels);
        }
        
        mfcc[frame_idx * n_mfcc + mfcc_idx] = sum;
    }
}

/**
 * @brief Generate Mel filterbank
 */
__global__ void generate_mel_filters_kernel(
    float* __restrict__ mel_filters,
    const float* __restrict__ fft_freqs,
    const float* __restrict__ mel_freqs,
    int n_freqs,
    int n_mels) {
    
    int mel_idx = blockIdx.x;
    
    if (mel_idx < n_mels) {
        float lower_mel = mel_freqs[mel_idx];
        float center_mel = mel_freqs[mel_idx + 1];
        float upper_mel = mel_freqs[mel_idx + 2];
        
        for (int freq_idx = 0; freq_idx < n_freqs; ++freq_idx) {
            float freq = fft_freqs[freq_idx];
            float weight = 0.0f;
            
            if (freq >= lower_mel && freq <= center_mel) {
                weight = (freq - lower_mel) / (center_mel - lower_mel);
            } else if (freq > center_mel && freq <= upper_mel) {
                weight = (upper_mel - freq) / (upper_mel - center_mel);
            }
            
            mel_filters[mel_idx * n_freqs + freq_idx] = weight;
        }
    }
}

//------------------------------------------------------------------------------
// Helper Functions
//------------------------------------------------------------------------------

/**
 * @brief Convert frequency to Mel scale
 */
__device__ float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

/**
 * @brief Convert Mel to frequency scale
 */
__device__ float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

} // namespace kernels
} // namespace signal_processing