/**
 * @file fft_kernels.cu
 * @brief CUDA kernels for FFT and spectral analysis operations
 * 
 * This file contains CUDA kernels and device functions for:
 * - Custom FFT operations
 * - Window functions
 * - Spectral computation acceleration
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
// Window Function Kernels
//------------------------------------------------------------------------------

/**
 * @brief Apply window function to signal in-place
 */
template<typename T>
__global__ void apply_window_kernel(T* __restrict__ signal, const float* __restrict__ window, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        signal[idx] *= window[idx];
    }
}

/**
 * @brief Generate Hann window on GPU
 */
__global__ void generate_hann_window_kernel(float* __restrict__ window, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        window[idx] = 0.5f * (1.0f - cosf(2.0f * PI * idx / (size - 1)));
    }
}

/**
 * @brief Generate Hamming window on GPU
 */
__global__ void generate_hamming_window_kernel(float* __restrict__ window, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        window[idx] = 0.54f - 0.46f * cosf(2.0f * PI * idx / (size - 1));
    }
}

/**
 * @brief Generate Blackman window on GPU
 */
__global__ void generate_blackman_window_kernel(float* __restrict__ window, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = 2.0f * PI * idx / (size - 1);
        window[idx] = 0.42f - 0.5f * cosf(x) + 0.08f * cosf(2.0f * x);
    }
}

/**
 * @brief Generate Flat Top window on GPU
 */
__global__ void generate_flattop_window_kernel(float* __restrict__ window, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = 2.0f * PI * idx / (size - 1);
        window[idx] = 0.21557895f - 0.41663158f * cosf(x) + 
                      0.277263158f * cosf(2.0f * x) - 
                      0.083578947f * cosf(3.0f * x) + 
                      0.006947368f * cosf(4.0f * x);
    }
}

/**
 * @brief Generate Tukey window on GPU
 */
__global__ void generate_tukey_window_kernel(float* __restrict__ window, int size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = static_cast<float>(idx) / (size - 1);
        if (x < alpha / 2.0f) {
            window[idx] = 0.5f * (1.0f - cosf(2.0f * PI * x / alpha));
        } else if (x > 1.0f - alpha / 2.0f) {
            window[idx] = 0.5f * (1.0f - cosf(2.0f * PI * (1.0f - x) / alpha));
        } else {
            window[idx] = 1.0f;
        }
    }
}

/**
 * @brief Generate Gaussian window on GPU
 */
__global__ void generate_gaussian_window_kernel(float* __restrict__ window, int size, float sigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = (idx - (size - 1) / 2.0f) / ((size - 1) / 2.0f);
        window[idx] = expf(-0.5f * x * x / (sigma * sigma));
    }
}

/**
 * @brief Generate Kaiser window on GPU
 * 
 * An approximation of the Kaiser window using a Chebyshev polynomial approximation
 * of the modified Bessel function of the first kind I0(x).
 */
__global__ void generate_kaiser_window_kernel(float* __restrict__ window, int size, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = 2.0f * idx / (size - 1) - 1.0f;  // Map to [-1, 1]
        float m2 = 1.0f - x * x;
        float arg = beta * sqrtf(m2 >= 0.0f ? m2 : 0.0f);
        
        // Polynomial approximation of I0(x) for small arguments
        float value = 1.0f;
        if (arg <= 15.0f) {
            float y = arg * arg;
            float sum = 1.0f + y * (0.25f + y * (0.015625f + y * 0.000434028f));
            // Approximate the normalization factor
            value = sum / 1.0f;
        } else {
            value = expf(arg) / sqrtf(2.0f * PI * arg);
        }
        
        window[idx] = value;
    }
}

//------------------------------------------------------------------------------
// FFT Helper Kernels
//------------------------------------------------------------------------------

/**
 * @brief Normalize FFT result
 */
__global__ void normalize_fft_kernel(
    cufftComplex* __restrict__ data,
    int size,
    float scale) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

/**
 * @brief Convert real data to complex
 */
__global__ void real_to_complex_kernel(
    const float* __restrict__ real_data,
    cufftComplex* __restrict__ complex_data,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        complex_data[idx].x = real_data[idx];
        complex_data[idx].y = 0.0f;
    }
}

/**
 * @brief Convert complex data to magnitude squared
 */
__global__ void complex_to_magnitude_squared_kernel(
    const cufftComplex* __restrict__ complex_data,
    float* __restrict__ magnitude_squared,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float real = complex_data[idx].x;
        float imag = complex_data[idx].y;
        magnitude_squared[idx] = real * real + imag * imag;
    }
}

/**
 * @brief Convert complex data to magnitude
 */
__global__ void complex_to_magnitude_kernel(
    const cufftComplex* __restrict__ complex_data,
    float* __restrict__ magnitude,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float real = complex_data[idx].x;
        float imag = complex_data[idx].y;
        magnitude[idx] = sqrtf(real * real + imag * imag);
    }
}

/**
 * @brief Convert complex data to phase
 */
__global__ void complex_to_phase_kernel(
    const cufftComplex* __restrict__ complex_data,
    float* __restrict__ phase,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float real = complex_data[idx].x;
        float imag = complex_data[idx].y;
        phase[idx] = atan2f(imag, real);
    }
}

/**
 * @brief Apply log scaling to data
 */
__global__ void apply_log_scaling_kernel(
    float* __restrict__ data,
    int size,
    float min_value) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        data[idx] = data[idx] > min_value ? 10.0f * log10f(data[idx]) : -100.0f;
    }
}

//------------------------------------------------------------------------------
// Spectral Analysis Kernels
//------------------------------------------------------------------------------

/**
 * @brief Compute power spectrum from FFT result
 */
__global__ void compute_power_spectrum_kernel(
    const cufftComplex* __restrict__ fft_result,
    float* __restrict__ power_spectrum,
    int size,
    float scale,
    bool is_even,
    bool scale_ends) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float real = fft_result[idx].x;
        float imag = fft_result[idx].y;
        float value = (real * real + imag * imag) / scale;
        
        // Scale DC and Nyquist components
        if (scale_ends && (idx == 0 || (is_even && idx == size - 1))) {
            value *= 0.5f;
        }
        
        power_spectrum[idx] = value;
    }
}

/**
 * @brief Compute cross-spectrum from two FFT results
 */
__global__ void compute_cross_spectrum_kernel(
    const cufftComplex* __restrict__ fft1,
    const cufftComplex* __restrict__ fft2,
    cufftComplex* __restrict__ cross_spectrum,
    int size,
    float scale,
    bool is_even,
    bool scale_ends) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float real1 = fft1[idx].x;
        float imag1 = fft1[idx].y;
        float real2 = fft2[idx].x;
        float imag2 = fft2[idx].y;
        
        // Cross-spectrum = FFT1 * conj(FFT2)
        float real = (real1 * real2 + imag1 * imag2) / scale;
        float imag = (imag1 * real2 - real1 * imag2) / scale;
        
        // Scale DC and Nyquist components
        if (scale_ends && (idx == 0 || (is_even && idx == size - 1))) {
            real *= 0.5f;
            imag *= 0.5f;
        }
        
        cross_spectrum[idx].x = real;
        cross_spectrum[idx].y = imag;
    }
}

/**
 * @brief Compute coherence from cross-spectrum and auto-spectra
 */
__global__ void compute_coherence_kernel(
    const cufftComplex* __restrict__ cross_spectrum,
    const float* __restrict__ psd1,
    const float* __restrict__ psd2,
    float* __restrict__ coherence,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float cross_mag_squared = cross_spectrum[idx].x * cross_spectrum[idx].x + 
                                cross_spectrum[idx].y * cross_spectrum[idx].y;
        float denominator = psd1[idx] * psd2[idx];
        
        coherence[idx] = denominator > 0.0f ? cross_mag_squared / denominator : 0.0f;
    }
}

/**
 * @brief Compute phase spectrum from cross-spectrum
 */
__global__ void compute_phase_kernel(
    const cufftComplex* __restrict__ cross_spectrum,
    float* __restrict__ phase,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float real = cross_spectrum[idx].x;
        float imag = cross_spectrum[idx].y;
        
        phase[idx] = atan2f(imag, real);
    }
}

/**
 * @brief Add segment power to accumulated spectrum
 */
__global__ void accumulate_spectrum_kernel(
    const cufftComplex* __restrict__ fft_result,
    float* __restrict__ accumulated_spectrum,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float real = fft_result[idx].x;
        float imag = fft_result[idx].y;
        float power = real * real + imag * imag;
        
        atomicAdd(&accumulated_spectrum[idx], power);
    }
}

/**
 * @brief Detrend signal by removing mean
 */
__global__ void detrend_constant_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size,
    float mean) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = input[idx] - mean;
    }
}

/**
 * @brief Detrend signal by removing linear trend
 */
__global__ void detrend_linear_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size,
    float slope,
    float intercept) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = input[idx] - (slope * idx + intercept);
    }
}

/**
 * @brief Optimized peak detection kernel
 */
__global__ void detect_peaks_kernel(
    const float* __restrict__ spectrum,
    int* __restrict__ peak_indices,
    float* __restrict__ peak_values,
    int* __restrict__ peak_count,
    int size,
    float threshold) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx > 0 && idx < size - 1) {
        float value = spectrum[idx];
        if (value > spectrum[idx - 1] && value > spectrum[idx + 1]) {
            if (value >= threshold) {
                int peak_idx = atomicAdd(peak_count, 1);
                peak_indices[peak_idx] = idx;
                peak_values[peak_idx] = value;
            }
        }
    }
}

/**
 * @brief Find harmonic indices in a spectrum
 */
__global__ void find_harmonics_kernel(
    int* __restrict__ harmonic_indices,
    const float* __restrict__ spectrum,
    int size,
    int fundamental_idx,
    int num_harmonics,
    int freq_tolerance) {
    
    int harmonic_num = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (harmonic_num < num_harmonics) {
        // Target harmonic index
        int target_idx = (harmonic_num + 2) * fundamental_idx;  // +2 means first harmonic is h2
        
        // Check if it's within range
        if (target_idx < size - freq_tolerance) {
            // Find local maximum around target frequency
            int max_idx = target_idx;
            float max_val = spectrum[target_idx];
            
            for (int i = -freq_tolerance; i <= freq_tolerance; ++i) {
                int idx = target_idx + i;
                if (idx > 0 && idx < size) {
                    if (spectrum[idx] > max_val) {
                        max_val = spectrum[idx];
                        max_idx = idx;
                    }
                }
            }
            
            harmonic_indices[harmonic_num] = max_idx;
        } else {
            // Out of range
            harmonic_indices[harmonic_num] = -1;
        }
    }
}

/**
 * @brief Compute harmonic distortion ratios
 */
__global__ void compute_harmonic_distortion_kernel(
    float* __restrict__ distortion,
    const float* __restrict__ spectrum,
    const int* __restrict__ harmonic_indices,
    int fundamental_idx,
    int num_harmonics) {
    
    int harmonic_num = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (harmonic_num < num_harmonics) {
        int harmonic_idx = harmonic_indices[harmonic_num];
        
        if (harmonic_idx > 0) {
            float fund_power = spectrum[fundamental_idx];
            float harmonic_power = spectrum[harmonic_idx];
            
            // Harmonic distortion ratio
            distortion[harmonic_num] = sqrtf(harmonic_power / fund_power);
        } else {
            distortion[harmonic_num] = 0.0f;
        }
    }
}

//------------------------------------------------------------------------------
// Spectrogram Kernels
//------------------------------------------------------------------------------

/**
 * @brief Compute spectrogram from segmented signal
 */
__global__ void compute_spectrogram_kernel(
    const cufftComplex* __restrict__ segment_ffts,
    float* __restrict__ spectrogram,
    int num_segments,
    int num_frequencies,
    float scale,
    bool scale_ends) {
    
    int segment_idx = blockIdx.x;
    int freq_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (segment_idx < num_segments && freq_idx < num_frequencies) {
        const cufftComplex& fft_val = segment_ffts[segment_idx * num_frequencies + freq_idx];
        float power = (fft_val.x * fft_val.x + fft_val.y * fft_val.y) / scale;
        
        // Scale DC and Nyquist components
        if (scale_ends && (freq_idx == 0 || freq_idx == num_frequencies - 1)) {
            power *= 0.5f;
        }
        
        spectrogram[segment_idx * num_frequencies + freq_idx] = power;
    }
}

/**
 * @brief Apply log scaling to spectrogram
 */
__global__ void log_scale_spectrogram_kernel(
    float* __restrict__ spectrogram,
    int num_segments,
    int num_frequencies,
    float min_value) {
    
    int segment_idx = blockIdx.x;
    int freq_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (segment_idx < num_segments && freq_idx < num_frequencies) {
        int idx = segment_idx * num_frequencies + freq_idx;
        spectrogram[idx] = spectrogram[idx] > min_value ? 10.0f * log10f(spectrogram[idx]) : -100.0f;
    }
}

/**
 * @brief Prepare segments for FFT processing
 * 
 * Extracts segment, applies window function, and places in output buffer ready for FFT
 */
__global__ void prepare_segments_kernel(
    const float* __restrict__ signal,
    float* __restrict__ segments,
    const float* __restrict__ window,
    int signal_length,
    int segment_size,
    int step_size,
    int num_segments) {
    
    int segment_idx = blockIdx.x;
    int sample_idx = threadIdx.x;
    
    if (segment_idx < num_segments && sample_idx < segment_size) {
        int signal_pos = segment_idx * step_size + sample_idx;
        float value = 0.0f;
        
        // Get sample from signal if in range
        if (signal_pos < signal_length) {
            value = signal[signal_pos];
        }
        
        // Apply window and store
        segments[segment_idx * segment_size + sample_idx] = value * window[sample_idx];
    }
}

//------------------------------------------------------------------------------
// Advanced FFT Kernels (Optimized for different architectures)
//------------------------------------------------------------------------------

/**
 * @brief Optimized 1D FFT kernel for small sizes (power of 2)
 * Specialized for SM 8.7 architecture
 */
template<int LOG2_FFT_SIZE>
__global__ void __launch_bounds__(128, 8)
fft_sm87_kernel(const cufftComplex* __restrict__ input, 
                cufftComplex* __restrict__ output,
                int batch_size) {
    constexpr int FFT_SIZE = 1 << LOG2_FFT_SIZE;
    extern __shared__ float shared_memory[];
    
    cufftComplex* shared_in = reinterpret_cast<cufftComplex*>(shared_memory);
    
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Load data into shared memory
    if (thread_idx < FFT_SIZE) {
        shared_in[thread_idx] = input[batch_idx * FFT_SIZE + thread_idx];
    }
    
    __syncthreads();
    
    // Perform in-place FFT in shared memory
    // Using Cooley-Tukey algorithm with bit reversal
    int j = 0;
    for (int i = 0; i < FFT_SIZE - 1; i++) {
        if (i < j) {
            cufftComplex temp = shared_in[i];
            shared_in[i] = shared_in[j];
            shared_in[j] = temp;
        }
        int k = FFT_SIZE / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
    
    __syncthreads();
    
    // Compute FFT
    for (int step = 1; step < FFT_SIZE; step *= 2) {
        float theta = -PI / step;
        
        for (int m = thread_idx; m < FFT_SIZE; m += blockDim.x) {
            if ((m % (2 * step)) < step) {
                int k = m + step;
                float tr = shared_in[k].x * cosf(theta * (m % step)) - 
                           shared_in[k].y * sinf(theta * (m % step));
                float ti = shared_in[k].x * sinf(theta * (m % step)) + 
                           shared_in[k].y * cosf(theta * (m % step));
                
                cufftComplex temp;
                temp.x = shared_in[m].x - tr;
                temp.y = shared_in[m].y - ti;
                
                shared_in[k] = temp;
                
                shared_in[m].x += tr;
                shared_in[m].y += ti;
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (thread_idx < FFT_SIZE) {
        output[batch_idx * FFT_SIZE + thread_idx] = shared_in[thread_idx];
    }
}

/**
 * @brief Optimized 1D FFT kernel for small sizes (power of 2)
 * Specialized for SM 7.5 architecture
 */
template<int LOG2_FFT_SIZE>
__global__ void __launch_bounds__(256, 4)
fft_sm75_kernel(const cufftComplex* __restrict__ input, 
                cufftComplex* __restrict__ output,
                int batch_size) {
    constexpr int FFT_SIZE = 1 << LOG2_FFT_SIZE;
    extern __shared__ float shared_memory[];
    
    cufftComplex* shared_in = reinterpret_cast<cufftComplex*>(shared_memory);
    
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Load data into shared memory
    if (thread_idx < FFT_SIZE) {
        shared_in[thread_idx] = input[batch_idx * FFT_SIZE + thread_idx];
    }
    
    __syncthreads();
    
    // Perform in-place FFT in shared memory
    // Using Cooley-Tukey algorithm with bit reversal
    int j = 0;
    for (int i = 0; i < FFT_SIZE - 1; i++) {
        if (i < j) {
            cufftComplex temp = shared_in[i];
            shared_in[i] = shared_in[j];
            shared_in[j] = temp;
        }
        int k = FFT_SIZE / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
    
    __syncthreads();
    
    // Compute FFT
    for (int step = 1; step < FFT_SIZE; step *= 2) {
        float theta = -PI / step;
        
        for (int m = thread_idx; m < FFT_SIZE; m += blockDim.x) {
            if ((m % (2 * step)) < step) {
                int k = m + step;
                float tr = shared_in[k].x * cosf(theta * (m % step)) - 
                           shared_in[k].y * sinf(theta * (m % step));
                float ti = shared_in[k].x * sinf(theta * (m % step)) + 
                           shared_in[k].y * cosf(theta * (m % step));
                
                cufftComplex temp;
                temp.x = shared_in[m].x - tr;
                temp.y = shared_in[m].y - ti;
                
                shared_in[k] = temp;
                
                shared_in[m].x += tr;
                shared_in[m].y += ti;
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (thread_idx < FFT_SIZE) {
        output[batch_idx * FFT_SIZE + thread_idx] = shared_in[thread_idx];
    }
}

//------------------------------------------------------------------------------
// Enhanced FFT Kernels for 2D Transforms
//------------------------------------------------------------------------------

/**
 * @brief Row-wise FFT kernel for 2D FFT
 * 
 * Performs the first step in a 2D FFT by transforming each row
 */
__global__ void row_wise_fft_kernel(
    const cufftComplex* __restrict__ input, 
    cufftComplex* __restrict__ output,
    int rows,
    int cols) {
    
    int row_idx = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (row_idx < rows) {
        // Get pointers to input and output rows
        const cufftComplex* row_in = input + row_idx * cols;
        cufftComplex* row_out = output + row_idx * cols;
        
        // Cooley-Tukey FFT algorithm
        // This is a simplified version - in practice would call cufftExec* or use shared memory
        
        for (int k = 0; k < cols; ++k) {
            cufftComplex sum = {0.0f, 0.0f};
            
            for (int n = 0; n < cols; ++n) {
                float angle = -2.0f * PI * k * n / cols;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);
                
                sum.x += row_in[n].x * cos_val - row_in[n].y * sin_val;
                sum.y += row_in[n].x * sin_val + row_in[n].y * cos_val;
            }
            
            row_out[k] = sum;
        }
    }
}

/**
 * @brief Column-wise FFT kernel for 2D FFT
 * 
 * Performs the second step in a 2D FFT by transforming each column
 */
__global__ void column_wise_fft_kernel(
    const cufftComplex* __restrict__ input, 
    cufftComplex* __restrict__ output,
    int rows,
    int cols) {
    
    int col_idx = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (col_idx < cols) {
        // Cooley-Tukey FFT algorithm for columns
        // This is a simplified version - in practice would call cufftExec* or use shared memory
        
        for (int k = 0; k < rows; ++k) {
            cufftComplex sum = {0.0f, 0.0f};
            
            for (int n = 0; n < rows; ++n) {
                float angle = -2.0f * PI * k * n / rows;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);
                
                const cufftComplex& val = input[n * cols + col_idx];
                
                sum.x += val.x * cos_val - val.y * sin_val;
                sum.y += val.x * sin_val + val.y * cos_val;
            }
            
            output[k * cols + col_idx] = sum;
        }
    }
}

/**
 * @brief Transpose matrix kernel for better memory access patterns in 2D FFT
 */
__global__ void transpose_kernel(
    const cufftComplex* __restrict__ input,
    cufftComplex* __restrict__ output,
    int rows,
    int cols) {
    
    // Use shared memory for coalesced memory access
    __shared__ cufftComplex shared_tile[32][32+1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Load data into shared memory
    if (x < cols && y < rows) {
        shared_tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    // Calculate transposed coordinates
    int x_out = blockIdx.y * 32 + threadIdx.x;
    int y_out = blockIdx.x * 32 + threadIdx.y;
    
    // Write transposed data
    if (x_out < rows && y_out < cols) {
        output[y_out * rows + x_out] = shared_tile[threadIdx.x][threadIdx.y];
    }
}

/**
 * @brief Sophisticated 2D FFT kernel using row-column algorithm with transpose
 * 
 * This approach has better memory access patterns than naive implementations
 */
template<int BLOCK_SIZE>
__global__ void fft_2d_kernel(
    const cufftComplex* __restrict__ input,
    cufftComplex* __restrict__ output,
    int rows,
    int cols) {
    
    extern __shared__ cufftComplex shared_mem[];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Load row into shared memory
        for (int col = threadIdx.x; col < cols; col += blockDim.x) {
            if (col < cols) {
                shared_mem[tid + (col - threadIdx.x) / blockDim.x * BLOCK_SIZE * BLOCK_SIZE] = 
                    input[row * cols + col];
            }
        }
        
        __syncthreads();
        
        // Perform row-wise FFT using shared memory
        // In a real implementation, would call device functions or use cuFFT batched transforms
        
        // Write results back
        for (int col = threadIdx.x; col < cols; col += blockDim.x) {
            if (col < cols) {
                output[row * cols + col] = 
                    shared_mem[tid + (col - threadIdx.x) / blockDim.x * BLOCK_SIZE * BLOCK_SIZE];
            }
        }
    }
}

//------------------------------------------------------------------------------
// New Functionality: Radix-2 FFT Implementation
//------------------------------------------------------------------------------

/**
 * @brief Bit-reversal permutation for radix-2 FFT
 */
__global__ void bit_reversal_permutation_kernel(
    const cufftComplex* __restrict__ input,
    cufftComplex* __restrict__ output,
    int size,
    int log2_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Compute bit-reversed index
        int reversed_idx = 0;
        for (int i = 0; i < log2_size; ++i) {
            reversed_idx = (reversed_idx << 1) | ((idx >> i) & 1);
        }
        
        // Copy data
        output[reversed_idx] = input[idx];
    }
}

/**
 * @brief Optimized radix-2 FFT butterfly operation kernel
 * 
 * Performs a single stage of the FFT butterfly operations
 */
__global__ void fft_butterfly_kernel(
    cufftComplex* __restrict__ data,
    int size,
    int stage,
    int direction) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int step = 1 << stage;
    int half_step = step >> 1;
    
    if (idx < size && (idx & half_step) == 0) {
        int k = idx & ~(step - 1);
        int j = idx & (half_step - 1);
        int butterfly_idx = k | j;
        int butterfly_partner = butterfly_idx | half_step;
        
        // Load data
        cufftComplex a = data[butterfly_idx];
        cufftComplex b = data[butterfly_partner];
        
        // Compute twiddle factor
        float angle = direction * 2.0f * PI * j / step;
        float sin_val = sinf(angle);
        float cos_val = cosf(angle);
        
        // Apply twiddle factor to second point
        float temp_real = b.x * cos_val - b.y * sin_val;
        float temp_imag = b.x * sin_val + b.y * cos_val;
        
        // Butterfly computation
        data[butterfly_idx].x = a.x + temp_real;
        data[butterfly_idx].y = a.y + temp_imag;
        data[butterfly_partner].x = a.x - temp_real;
        data[butterfly_partner].y = a.y - temp_imag;
    }
}

/**
 * @brief Perform all stages of radix-2 FFT in a single kernel
 * 
 * More efficient for small-to-medium size FFTs
 */
template<int LOG2_SIZE>
__global__ void radix2_fft_kernel(
    cufftComplex* __restrict__ data,
    int direction) {
    
    constexpr int SIZE = 1 << LOG2_SIZE;
    extern __shared__ cufftComplex shared_data[];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * SIZE + tid;
    
    // Load data into shared memory
    if (tid < SIZE) {
        shared_data[tid] = data[global_idx];
    }
    
    __syncthreads();
    
    // Perform bit-reversal permutation in shared memory
    int reversed_tid = 0;
    for (int i = 0; i < LOG2_SIZE; ++i) {
        reversed_tid = (reversed_tid << 1) | ((tid >> i) & 1);
    }
    
    cufftComplex temp = shared_data[tid];
    __syncthreads();
    
    if (tid < SIZE) {
        shared_data[reversed_tid] = temp;
    }
    
    __syncthreads();
    
    // Perform butterfly operations
    for (int stage = 0; stage < LOG2_SIZE; ++stage) {
        int step = 1 << stage;
        int half_step = step >> 1;
        int group_size = step << 1;
        
        // Determine butterfly indices
        int group = tid / group_size;
        int butterfly_id = tid % group_size;
        
        if (butterfly_id < step) {
            int butterfly_idx = group * group_size + butterfly_id;
            int butterfly_partner = butterfly_idx + step;
            
            // Load data
            cufftComplex a = shared_data[butterfly_idx];
            cufftComplex b = shared_data[butterfly_partner];
            
            // Compute twiddle factor
            float angle = direction * 2.0f * PI * butterfly_id / (step << 1);
            float sin_val = sinf(angle);
            float cos_val = cosf(angle);
            
            // Apply twiddle factor to second point
            float temp_real = b.x * cos_val - b.y * sin_val;
            float temp_imag = b.x * sin_val + b.y * cos_val;
            
            // Butterfly computation
            shared_data[butterfly_idx].x = a.x + temp_real;
            shared_data[butterfly_idx].y = a.y + temp_imag;
            shared_data[butterfly_partner].x = a.x - temp_real;
            shared_data[butterfly_partner].y = a.y - temp_imag;
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    if (tid < SIZE) {
        data[global_idx] = shared_data[tid];
    }
}

//------------------------------------------------------------------------------
// New: Specialized CUDA FFT Launch Functions (for C++ wrapper)
//------------------------------------------------------------------------------

/**
 * @brief Template function to select the most appropriate FFT kernel based on size
 * 
 * @param d_input Device pointer to input data
 * @param d_output Device pointer to output buffer
 * @param size Size of the FFT (must be power of 2)
 * @param batch_size Number of FFTs to perform in batch
 * @param props CUDA device properties
 * @param stream CUDA stream to use
 * @return cudaError_t
 */
template<typename T>
cudaError_t launch_optimized_fft(
    const T* d_input,
    T* d_output,
    int size,
    int batch_size,
    const cudaDeviceProp& props,
    cudaStream_t stream = nullptr) {
    
    // Compute log2 of size
    int log2_size = 0;
    int temp_size = size;
    while (temp_size >>= 1) {
        ++log2_size;
    }
    
    // Choose proper kernel based on architecture
    if (props.major == 8 && props.minor == 7) {  // Jetson Orin NX
        // Choose block size based on FFT size
        int block_size = size < 256 ? 64 : 128;
        
        // Launch kernel
        switch (log2_size) {
            case 5: // 32
                fft_sm87_kernel<5><<<batch_size, block_size, size * sizeof(cufftComplex), stream>>>(
                    reinterpret_cast<const cufftComplex*>(d_input),
                    reinterpret_cast<cufftComplex*>(d_output),
                    batch_size);
                break;
            case 6: // 64
                fft_sm87_kernel<6><<<batch_size, block_size, size * sizeof(cufftComplex), stream>>>(
                    reinterpret_cast<const cufftComplex*>(d_input),
                    reinterpret_cast<cufftComplex*>(d_output),
                    batch_size);
                break;
            case 7: // 128
                fft_sm87_kernel<7><<<batch_size, block_size, size * sizeof(cufftComplex), stream>>>(
                    reinterpret_cast<const cufftComplex*>(d_input),
                    reinterpret_cast<cufftComplex*>(d_output),
                    batch_size);
                break;
            case 8: // 256
                fft_sm87_kernel<8><<<batch_size, block_size, size * sizeof(cufftComplex), stream>>>(
                    reinterpret_cast<const cufftComplex*>(d_input),
                    reinterpret_cast<cufftComplex*>(d_output),
                    batch_size);
                break;
            default:
                // For larger sizes, use cuFFT directly
                return cudaErrorInvalidValue;
        }
    } else {  // Fallback for other architectures including AWS T4G
        // Choose block size based on FFT size
        int block_size = size < 512 ? 128 : 256;
        
        // Launch kernel
        switch (log2_size) {
            case 5: // 32
                fft_sm75_kernel<5><<<batch_size, block_size, size * sizeof(cufftComplex), stream>>>(
                    reinterpret_cast<const cufftComplex*>(d_input),
                    reinterpret_cast<cufftComplex*>(d_output),
                    batch_size);
                break;
            case 6: // 64
                fft_sm75_kernel<6><<<batch_size, block_size, size * sizeof(cufftComplex), stream>>>(
                    reinterpret_cast<const cufftComplex*>(d_input),
                    reinterpret_cast<cufftComplex*>(d_output),
                    batch_size);
                break;
            case 7: // 128
                fft_sm75_kernel<7><<<batch_size, block_size, size * sizeof(cufftComplex), stream>>>(
                    reinterpret_cast<const cufftComplex*>(d_input),
                    reinterpret_cast<cufftComplex*>(d_output),
                    batch_size);
                break;
            case 8: // 256
                fft_sm75_kernel<8><<<batch_size, block_size, size * sizeof(cufftComplex), stream>>>(
                    reinterpret_cast<const cufftComplex*>(d_input),
                    reinterpret_cast<cufftComplex*>(d_output),
                    batch_size);
                break;
            default:
                // For larger sizes, use cuFFT directly
                return cudaErrorInvalidValue;
        }
    }
    
    return cudaGetLastError();
}

/**
 * @brief Advanced radix-2 FFT implementation with GPU adaptability
 * 
 * @param d_input Device pointer to input data
 * @param d_output Device pointer to output buffer
 * @param size Size of the FFT (must be power of 2)
 * @param props CUDA device properties
 * @param direction Direction of transform (-1 for forward, 1 for inverse)
 * @param stream CUDA stream to use
 * @return cudaError_t
 */
cudaError_t launch_radix2_fft(
    cufftComplex* d_input,
    cufftComplex* d_output,
    int size,
    const cudaDeviceProp& props,
    int direction = -1,
    cudaStream_t stream = nullptr) {
    
    // Compute log2 of size
    int log2_size = 0;
    int temp_size = size;
    while (temp_size >>= 1) {
        ++log2_size;
    }
    
    // For larger FFTs, perform it in stages
    if (log2_size > 10) {  // Threshold for switching to multi-stage approach
        // Bit-reversal permutation
        int block_size = props.major >= 8 ? 256 : 512;
        int grid_size = (size + block_size - 1) / block_size;
        
        bit_reversal_permutation_kernel<<<grid_size, block_size, 0, stream>>>(
            d_input, d_output, size, log2_size);
        
        // Swap pointers
        std::swap(d_input, d_output);
        
        // Butterfly stages
        for (int stage = 0; stage < log2_size; ++stage) {
            fft_butterfly_kernel<<<grid_size, block_size, 0, stream>>>(
                d_input, size, stage, direction);
        }
    } else {
        // For smaller FFTs, use the combined kernel
        int threads_per_block = props.major >= 8 ? 128 : 256;
        int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
        
        // Launch the appropriate kernel based on FFT size
        switch (log2_size) {
            case 5:  // 32-point FFT
                radix2_fft_kernel<5><<<blocks_per_grid, threads_per_block, 32 * sizeof(cufftComplex), stream>>>(
                    d_input, direction);
                break;
            case 6:  // 64-point FFT
                radix2_fft_kernel<6><<<blocks_per_grid, threads_per_block, 64 * sizeof(cufftComplex), stream>>>(
                    d_input, direction);
                break;
            case 7:  // 128-point FFT
                radix2_fft_kernel<7><<<blocks_per_grid, threads_per_block, 128 * sizeof(cufftComplex), stream>>>(
                    d_input, direction);
                break;
            case 8:  // 256-point FFT
                radix2_fft_kernel<8><<<blocks_per_grid, threads_per_block, 256 * sizeof(cufftComplex), stream>>>(
                    d_input, direction);
                break;
            case 9:  // 512-point FFT
                radix2_fft_kernel<9><<<blocks_per_grid, threads_per_block, 512 * sizeof(cufftComplex), stream>>>(
                    d_input, direction);
                break;
            case 10:  // 1024-point FFT
                radix2_fft_kernel<10><<<blocks_per_grid, threads_per_block, 1024 * sizeof(cufftComplex), stream>>>(
                    d_input, direction);
                break;
            default:
                return cudaErrorInvalidValue;
        }
    }
    
    return cudaGetLastError();
}

//------------------------------------------------------------------------------
// New: Advanced Signal Processing Analysis Kernels
//------------------------------------------------------------------------------

/**
 * @brief Compute Short-Time Fourier Transform (STFT) from windowed segments
 */
__global__ void stft_kernel(
    const float* __restrict__ signal,
    cufftComplex* __restrict__ stft_output,
    const float* __restrict__ window,
    int signal_length,
    int window_size,
    int hop_size,
    int num_frames,
    int fft_size) {
    
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (frame_idx < num_frames) {
        extern __shared__ float shared_mem[];
        float* segment = shared_mem;
        cufftComplex* fft_result = reinterpret_cast<cufftComplex*>(segment + fft_size);
        
        // Initialize segment with zeros
        for (int i = tid; i < fft_size; i += blockDim.x) {
            segment[i] = 0.0f;
        }
        
        __syncthreads();
        
        // Load and window signal
        int frame_start = frame_idx * hop_size;
        for (int i = tid; i < window_size; i += blockDim.x) {
            int signal_idx = frame_start + i;
            if (signal_idx < signal_length) {
                segment[i] = signal[signal_idx] * window[i];
            }
        }
        
        __syncthreads();
        
        // Compute FFT in shared memory (simplified for illustration)
        // In practice, you would call a custom device function for the FFT
        // or use a specialized kernel via kernel fusion
        
        // Store result
        for (int i = tid; i < fft_size / 2 + 1; i += blockDim.x) {
            stft_output[frame_idx * (fft_size / 2 + 1) + i] = fft_result[i];
        }
    }
}

/**
 * @brief Find fundamental frequency using autocorrelation method
 */
__global__ void find_fundamental_freq_kernel(
    const float* __restrict__ signal,
    float* __restrict__ autocorrelation,
    float* __restrict__ peak_lags,
    float* __restrict__ peak_values,
    int* __restrict__ num_peaks,
    int signal_length,
    int max_lag,
    float sample_rate,
    float min_freq,
    float max_freq) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int min_lag = static_cast<int>(sample_rate / max_freq);
    int max_lag_idx = static_cast<int>(sample_rate / min_freq);
    
    if (max_lag_idx > max_lag) max_lag_idx = max_lag;
    
    // Compute autocorrelation
    if (tid < max_lag) {
        float sum = 0.0f;
        int count = 0;
        
        for (int i = 0; i < signal_length - tid; ++i) {
            sum += signal[i] * signal[i + tid];
            count++;
        }
        
        autocorrelation[tid] = count > 0 ? sum / count : 0.0f;
    }
    
    __syncthreads();
    
    // Find peaks in autocorrelation
    if (tid >= min_lag && tid < max_lag_idx - 1) {
        if (autocorrelation[tid] > autocorrelation[tid - 1] && 
            autocorrelation[tid] > autocorrelation[tid + 1]) {
            
            // Simple peak: confirm it's above the threshold
            float threshold = 0.5f * autocorrelation[0];  // 50% of zero-lag autocorrelation
            
            if (autocorrelation[tid] > threshold) {
                int peak_idx = atomicAdd(num_peaks, 1);
                peak_lags[peak_idx] = static_cast<float>(tid);
                peak_values[peak_idx] = autocorrelation[tid];
            }
        }
    }
}

/**
 * @brief Compute cepstrum (inverse FFT of log magnitude spectrum)
 */
__global__ void compute_cepstrum_kernel(
    const cufftComplex* __restrict__ fft_result,
    cufftComplex* __restrict__ cepstrum_fft,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Compute log magnitude spectrum
        float magnitude = sqrtf(fft_result[idx].x * fft_result[idx].x + 
                               fft_result[idx].y * fft_result[idx].y);
        
        // Avoid log(0)
        float log_mag = magnitude > 1e-10f ? logf(magnitude) : -23.0f;  // -23 is ~log(1e-10)
        
        // Store for inverse FFT
        cepstrum_fft[idx].x = log_mag;
        cepstrum_fft[idx].y = 0.0f;
    }
}

/**
 * @brief Extract pitch using cepstral analysis
 */
__global__ void extract_pitch_cepstrum_kernel(
    const float* __restrict__ cepstrum,
    float* __restrict__ peak_quefrency,
    float* __restrict__ peak_value,
    int size,
    float sample_rate,
    float min_freq,
    float max_freq) {
    
    int tid = threadIdx.x;
    
    // Shared memory for reduction
    __shared__ float shared_max_value;
    __shared__ float shared_max_quefrency;
    
    if (tid == 0) {
        shared_max_value = -1.0f;
        shared_max_quefrency = 0.0f;
    }
    
    __syncthreads();
    
    // Convert frequency bounds to quefrency bounds
    int min_quefrency = static_cast<int>(sample_rate / max_freq);
    int max_quefrency = static_cast<int>(sample_rate / min_freq);
    
    // Limit to valid range
    if (max_quefrency >= size) max_quefrency = size - 1;
    
    // Each thread finds its local maximum in assigned region
    float thread_max_value = -1.0f;
    float thread_max_quefrency = 0.0f;
    
    for (int i = min_quefrency + tid; i < max_quefrency; i += blockDim.x) {
        if (cepstrum[i] > thread_max_value) {
            thread_max_value = cepstrum[i];
            thread_max_quefrency = static_cast<float>(i);
        }
    }
    
    // Reduce to find global maximum
    atomicMax((int*)&shared_max_value, __float_as_int(thread_max_value));
    
    __syncthreads();
    
    // Thread with the max value writes its quefrency
    if (thread_max_value == __int_as_float(*(int*)&shared_max_value)) {
        atomicExch((int*)&shared_max_quefrency, __float_as_int(thread_max_quefrency));
    }
    
    __syncthreads();
    
    // Write result
    if (tid == 0) {
        *peak_quefrency = shared_max_quefrency;
        *peak_value = shared_max_value;
    }
}

//------------------------------------------------------------------------------
// New: Batch Processing Kernels
//------------------------------------------------------------------------------

/**
 * @brief Batch FFT filter for efficient processing of multiple signals
 */
template<int FILTER_TYPE>
__global__ void batch_fft_filter_kernel(
    const cufftComplex* __restrict__ fft_data,
    cufftComplex* __restrict__ filtered_fft,
    const float* __restrict__ filter_coeffs,
    int num_batches,
    int fft_size) {
    
    int batch_idx = blockIdx.y;
    int freq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < num_batches && freq_idx < fft_size) {
        int idx = batch_idx * fft_size + freq_idx;
        float filter_val = filter_coeffs[freq_idx];
        
        // Apply filter based on filter type
        if (FILTER_TYPE == 0) {  // Low-pass
            filtered_fft[idx].x = fft_data[idx].x * filter_val;
            filtered_fft[idx].y = fft_data[idx].y * filter_val;
        } else if (FILTER_TYPE == 1) {  // High-pass
            float high_pass_val = 1.0f - filter_val;
            filtered_fft[idx].x = fft_data[idx].x * high_pass_val;
            filtered_fft[idx].y = fft_data[idx].y * high_pass_val;
        } else if (FILTER_TYPE == 2) {  // Band-pass
            // Assuming filter_coeffs represents the bandpass response
            filtered_fft[idx].x = fft_data[idx].x * filter_val;
            filtered_fft[idx].y = fft_data[idx].y * filter_val;
        } else if (FILTER_TYPE == 3) {  // Band-stop
            float band_stop_val = 1.0f - filter_val;
            filtered_fft[idx].x = fft_data[idx].x * band_stop_val;
            filtered_fft[idx].y = fft_data[idx].y * band_stop_val;
        }
    }
}

/**
 * @brief Batch FFT-based convolution for efficient filtering
 */
__global__ void batch_fft_convolution_kernel(
    const cufftComplex* __restrict__ signal_fft,
    const cufftComplex* __restrict__ kernel_fft,
    cufftComplex* __restrict__ result_fft,
    int num_batches,
    int fft_size) {
    
    int batch_idx = blockIdx.y;
    int freq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < num_batches && freq_idx < fft_size) {
        int idx = batch_idx * fft_size + freq_idx;
        
        // Complex multiplication
        float signal_real = signal_fft[idx].x;
        float signal_imag = signal_fft[idx].y;
        float kernel_real = kernel_fft[freq_idx].x;  // Kernel is shared across batches
        float kernel_imag = kernel_fft[freq_idx].y;
        
        result_fft[idx].x = signal_real * kernel_real - signal_imag * kernel_imag;
        result_fft[idx].y = signal_real * kernel_imag + signal_imag * kernel_real;
    }
}

/**
 * @brief Batch PSD computation for multiple signals
 */
__global__ void batch_psd_compute_kernel(
    const cufftComplex* __restrict__ fft_results,
    float* __restrict__ psd_results,
    int num_batches,
    int fft_size,
    float scale,
    bool scale_ends) {
    
    int batch_idx = blockIdx.y;
    int freq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < num_batches && freq_idx < fft_size) {
        int idx = batch_idx * fft_size + freq_idx;
        
        // Compute PSD value
        float real = fft_results[idx].x;
        float imag = fft_results[idx].y;
        float psd = (real * real + imag * imag) / scale;
        
        // Scale DC and Nyquist components
        if (scale_ends && (freq_idx == 0 || freq_idx == fft_size - 1)) {
            psd *= 0.5f;
        }
        
        psd_results[idx] = psd;
    }
}

} // namespace kernels
} // namespace signal_processing