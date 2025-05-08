/**
 * @file time_frequency.cpp
 * @brief Implementation of time-frequency analysis operations
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include "signal_processing/time_frequency.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <deque>
#include <limits>
#include <complex>
#include <vector>

// Check for CUDA availability
#if defined(WITH_CUDA)
#include <cuda_runtime.h>
#include <cufft.h>
// Include CUDA kernel definitions
#include "../src/kernels/time_frequency_kernels.cu"
#endif

namespace signal_processing {

// Constants
constexpr float PI = 3.14159265358979323846f;

namespace {

// Generate window function
std::vector<float> generate_window(int window_size, WindowType window_type, float window_param = 0.0f) {
    std::vector<float> window(window_size);
    
    switch (window_type) {
        case WindowType::RECTANGULAR:
            std::fill(window.begin(), window.end(), 1.0f);
            break;
            
        case WindowType::HANN:
            for (int i = 0; i < window_size; ++i) {
                window[i] = 0.5f * (1.0f - std::cos(2 * PI * i / (window_size - 1)));
            }
            break;
            
        case WindowType::HAMMING:
            for (int i = 0; i < window_size; ++i) {
                window[i] = 0.54f - 0.46f * std::cos(2 * PI * i / (window_size - 1));
            }
            break;
            
        case WindowType::BLACKMAN:
            for (int i = 0; i < window_size; ++i) {
                float x = 2 * PI * i / (window_size - 1);
                window[i] = 0.42f - 0.5f * std::cos(x) + 0.08f * std::cos(2 * x);
            }
            break;
            
        case WindowType::KAISER:
            {
                // Default beta parameter
                float beta = window_param > 0.0f ? window_param : 4.0f;
                
                // Compute I0(beta) denominator
                float i0_beta = 1.0f;
                float term = 1.0f;
                for (int i = 1; i <= 20; ++i) {
                    term *= (beta / 2.0f) * (beta / 2.0f) / (i * i);
                    i0_beta += term;
                    if (term < 1e-7f * i0_beta) {
                        break;
                    }
                }
                
                for (int i = 0; i < window_size; ++i) {
                    float x = 2.0f * i / (window_size - 1) - 1.0f;
                    
                    if (std::abs(x) < 1.0f) {
                        float arg = beta * std::sqrt(1.0f - x * x);
                        
                        // Compute modified Bessel function of the first kind, order 0
                        float i0_val = 1.0f;
                        term = 1.0f;
                        for (int j = 1; j <= 20; ++j) {
                            term *= (arg / 2.0f) * (arg / 2.0f) / (j * j);
                            i0_val += term;
                            if (term < 1e-7f * i0_val) {
                                break;
                            }
                        }
                        
                        window[i] = i0_val / i0_beta;
                    } else {
                        window[i] = 0.0f;
                    }
                }
            }
            break;
            
        default:
            std::fill(window.begin(), window.end(), 1.0f);
            break;
    }
    
    // Normalize window to preserve energy
    float sum = std::accumulate(window.begin(), window.end(), 0.0f);
    if (sum > 1e-10f) {
        for (int i = 0; i < window_size; ++i) {
            window[i] /= std::sqrt(sum);
        }
    }
    
    return window;
}

// Pad signal for STFT
std::vector<float> pad_signal(const std::vector<float>& signal, int frame_size, int hop_size, bool center, const std::string& mode) {
    if (!center) {
        return signal;
    }
    
    int pad_size = frame_size / 2;
    std::vector<float> padded(signal.size() + 2 * pad_size);
    
    // Copy original signal to center of padded signal
    std::copy(signal.begin(), signal.end(), padded.begin() + pad_size);
    
    // Apply padding mode
    if (mode == "reflect") {
        // Reflect padding
        for (int i = 0; i < pad_size; ++i) {
            padded[i] = signal[pad_size - i - 1];
            padded[padded.size() - i - 1] = signal[signal.size() - pad_size + i];
        }
    } else if (mode == "constant") {
        // Constant padding (zeros)
        std::fill(padded.begin(), padded.begin() + pad_size, 0.0f);
        std::fill(padded.end() - pad_size, padded.end(), 0.0f);
    } else if (mode == "edge") {
        // Edge padding
        std::fill(padded.begin(), padded.begin() + pad_size, signal.front());
        std::fill(padded.end() - pad_size, padded.end(), signal.back());
    }
    
    return padded;
}

// Generate scales for CWT
std::vector<float> generate_scales(int num_scales, float min_scale, float max_scale, bool normalize) {
    std::vector<float> scales(num_scales);
    
    if (num_scales == 1) {
        scales[0] = min_scale;
        return scales;
    }
    
    if (max_scale <= 0.0f) {
        max_scale = min_scale * std::pow(2.0f, num_scales - 1);
    }
    
    // Generate scales using geometric spacing
    float scale_step = std::pow(max_scale / min_scale, 1.0f / (num_scales - 1));
    
    for (int i = 0; i < num_scales; ++i) {
        scales[i] = min_scale * std::pow(scale_step, i);
    }
    
    // Normalize scales
    if (normalize) {
        for (int i = 0; i < num_scales; ++i) {
            scales[i] /= min_scale;
        }
    }
    
    return scales;
}

// Convert scales to frequencies for CWT
std::vector<float> scales_to_frequencies(const std::vector<float>& scales, float wavelet_param, float sample_rate) {
    std::vector<float> frequencies(scales.size());
    
    for (size_t i = 0; i < scales.size(); ++i) {
        frequencies[i] = wavelet_param / (2 * PI * scales[i]) * sample_rate;
    }
    
    return frequencies;
}

} // anonymous namespace

//------------------------------------------------------------------------------
// STFT Implementation
//------------------------------------------------------------------------------

class STFTImpl {
public:
    STFTImpl(const STFTParams& params, int device_id)
        : params_(params), device_id_(device_id), has_cuda_(false) {
        initialize();
    }
    
    ~STFTImpl() {
        cleanup();
    }
    
    STFTResult transform(const std::vector<float>& signal, float sample_rate) {
        if (signal.empty()) {
            return {};
        }
        
        // Determine FFT size if not specified
        int fft_size = params_.fft_size > 0 ? params_.fft_size : params_.window_size;
        
        // Generate window function if not already done
        if (window_.empty() || window_.size() != static_cast<size_t>(params_.window_size)) {
            window_ = generate_window(params_.window_size, params_.window_type, params_.window_param);
        }
        
        // Pad signal if requested
        std::vector<float> padded_signal;
        if (params_.center) {
            padded_signal = pad_signal(signal, params_.window_size, params_.hop_size, true, params_.pad_mode_str);
        } else {
            padded_signal = signal;
        }
        
        // Calculate number of frames
        int n_frames = 1 + (padded_signal.size() - params_.window_size) / params_.hop_size;
        
        // Prepare result structure
        STFTResult result;
        result.spectrogram.resize(n_frames, std::vector<std::complex<float>>(fft_size / 2 + 1));
        result.sample_rate = sample_rate;
        
        // Calculate time points
        result.times.resize(n_frames);
        for (int i = 0; i < n_frames; ++i) {
            if (params_.center) {
                result.times[i] = static_cast<float>(i * params_.hop_size) / sample_rate;
            } else {
                result.times[i] = static_cast<float>(i * params_.hop_size + params_.window_size / 2) / sample_rate;
            }
        }
        
        // Calculate frequency points
        result.frequencies.resize(fft_size / 2 + 1);
        for (int i = 0; i <= fft_size / 2; ++i) {
            result.frequencies[i] = static_cast<float>(i * sample_rate) / fft_size;
        }
        
        // Perform STFT
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                float* d_signal = nullptr;
                float* d_window = nullptr;
                cufftComplex* d_stft_data = nullptr;
                cufftComplex* d_stft_result = nullptr;
                
                cudaMalloc(&d_signal, padded_signal.size() * sizeof(float));
                cudaMalloc(&d_window, window_.size() * sizeof(float));
                cudaMalloc(&d_stft_data, n_frames * fft_size * sizeof(cufftComplex));
                cudaMalloc(&d_stft_result, n_frames * (fft_size / 2 + 1) * sizeof(cufftComplex));
                
                // Copy data to device
                cudaMemcpy(d_signal, padded_signal.data(), padded_signal.size() * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_window, window_.data(), window_.size() * sizeof(float), cudaMemcpyHostToDevice);
                
                // Zero-initialize STFT data
                cudaMemset(d_stft_data, 0, n_frames * fft_size * sizeof(cufftComplex));
                
                // Get device properties
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, device_id_);
                
                // Choose optimal kernel based on architecture
                int block_size = 256;
                size_t shared_mem_size = params_.window_size * 2 * sizeof(float);  // window + signal segment
                
                if (props.major == 8 && props.minor == 7) {
                    // Jetson Orin NX (SM 8.7)
                    block_size = 128;
                    // Launch Jetson-optimized kernel
                    kernels::stft_sm87_kernel<<<n_frames, block_size, shared_mem_size>>>(
                        d_signal, d_stft_data, d_window, padded_signal.size(),
                        params_.window_size, params_.hop_size, n_frames, fft_size);
                } else if (props.major == 7 && props.minor == 5) {
                    // AWS T4G (SM 7.5)
                    kernels::stft_sm75_kernel<<<n_frames, block_size, shared_mem_size>>>(
                        d_signal, d_stft_data, d_window, padded_signal.size(),
                        params_.window_size, params_.hop_size, n_frames, fft_size);
                } else {
                    // Default kernel
                    // Launch window kernel
                    block_size = std::min(256, params_.window_size);
                    shared_mem_size = params_.window_size * sizeof(float);
                    
                    kernels::stft_window_kernel<<<n_frames, block_size, shared_mem_size>>>(
                        d_signal, (float*)d_stft_data, d_window, params_.window_size,
                        params_.hop_size, n_frames, padded_signal.size());
                }
                
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                }
                
                // Create cuFFT plan
                cufftHandle plan;
                cufftPlan1d(&plan, fft_size, CUFFT_C2C, n_frames);
                
                // Execute FFT
                cufftExecC2C(plan, d_stft_data, d_stft_data, CUFFT_FORWARD);
                
                // Extract non-redundant half of the spectrum
                block_size = std::min(256, fft_size / 2 + 1);
                for (int i = 0; i < n_frames; ++i) {
                    cudaMemcpy(d_stft_result + i * (fft_size / 2 + 1),
                               d_stft_data + i * fft_size,
                               (fft_size / 2 + 1) * sizeof(cufftComplex),
                               cudaMemcpyDeviceToDevice);
                }
                
                // Allocate host memory for result
                std::vector<std::complex<float>> host_stft(n_frames * (fft_size / 2 + 1));
                
                // Copy result back to host
                cudaMemcpy(host_stft.data(), d_stft_result,
                          n_frames * (fft_size / 2 + 1) * sizeof(cufftComplex),
                          cudaMemcpyDeviceToHost);
                
                // Cleanup
                cufftDestroy(plan);
                cudaFree(d_signal);
                cudaFree(d_window);
                cudaFree(d_stft_data);
                cudaFree(d_stft_result);
                
                // Reshape result
                for (int i = 0; i < n_frames; ++i) {
                    for (int j = 0; j <= fft_size / 2; ++j) {
                        result.spectrogram[i][j] = host_stft[i * (fft_size / 2 + 1) + j];
                    }
                }
                
                return result;
            } catch (const std::exception& e) {
                std::cerr << "CUDA STFT failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation of STFT
        std::vector<std::complex<float>> fft_buffer(fft_size);
        
        // Process each frame
        for (int i = 0; i < n_frames; ++i) {
            int frame_start = i * params_.hop_size;
            
            // Apply window function
            for (int j = 0; j < params_.window_size; ++j) {
                if (frame_start + j < static_cast<int>(padded_signal.size())) {
                    fft_buffer[j] = std::complex<float>(padded_signal[frame_start + j] * window_[j], 0.0f);
                } else {
                    fft_buffer[j] = std::complex<float>(0.0f, 0.0f);
                }
            }
            
            // Zero-pad if necessary
            for (int j = params_.window_size; j < fft_size; ++j) {
                fft_buffer[j] = std::complex<float>(0.0f, 0.0f);
            }
            
            // Compute FFT (simplified implementation using DFT for clarity)
            // In practice, would use a more efficient FFT library
            for (int k = 0; k <= fft_size / 2; ++k) {
                std::complex<float> sum(0.0f, 0.0f);
                
                for (int n = 0; n < fft_size; ++n) {
                    float angle = -2.0f * PI * k * n / fft_size;
                    std::complex<float> factor(std::cos(angle), std::sin(angle));
                    sum += fft_buffer[n] * factor;
                }
                
                result.spectrogram[i][k] = sum;
            }
        }
        
        return result;
    }
    
    std::vector<float> inverse_transform(const STFTResult& stft_result) {
        int n_frames = stft_result.spectrogram.size();
        int n_freqs = stft_result.spectrogram[0].size();
        int fft_size = (n_freqs - 1) * 2;
        
        // Generate window function if not already done
        if (window_.empty() || window_.size() != static_cast<size_t>(params_.window_size)) {
            window_ = generate_window(params_.window_size, params_.window_type, params_.window_param);
        }
        
        // Estimate signal length from time values
        float signal_duration = stft_result.times.back() + static_cast<float>(params_.window_size) / (2 * stft_result.sample_rate);
        int signal_length = static_cast<int>(signal_duration * stft_result.sample_rate) + 1;
        
        // Account for padding
        if (params_.center) {
            signal_length -= params_.window_size;
        }
        
        std::vector<float> output(signal_length, 0.0f);
        std::vector<float> window_sum(signal_length, 0.0f);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                cufftComplex* d_stft_data = nullptr;
                cufftComplex* d_full_fft = nullptr;
                float* d_output = nullptr;
                float* d_window = nullptr;
                
                cudaMalloc(&d_stft_data, n_frames * n_freqs * sizeof(cufftComplex));
                cudaMalloc(&d_full_fft, n_frames * fft_size * sizeof(cufftComplex));
                cudaMalloc(&d_output, signal_length * sizeof(float));
                cudaMalloc(&d_window, window_.size() * sizeof(float));
                
                // Zero-initialize output
                cudaMemset(d_output, 0, signal_length * sizeof(float));
                
                // Copy data to device
                // Flatten STFT result
                std::vector<std::complex<float>> flattened_stft;
                flattened_stft.reserve(n_frames * n_freqs);
                for (const auto& frame : stft_result.spectrogram) {
                    flattened_stft.insert(flattened_stft.end(), frame.begin(), frame.end());
                }
                
                cudaMemcpy(d_stft_data, flattened_stft.data(), flattened_stft.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);
                cudaMemcpy(d_window, window_.data(), window_.size() * sizeof(float), cudaMemcpyHostToDevice);
                
                // Create cuFFT plan
                cufftHandle plan;
                cufftPlan1d(&plan, fft_size, CUFFT_C2C, n_frames);
                
                // Reconstruct full spectrum by symmetry
                int block_size = 256;
                for (int i = 0; i < n_frames; ++i) {
                    // Copy positive frequencies
                    cudaMemcpy(d_full_fft + i * fft_size, 
                              d_stft_data + i * n_freqs, 
                              n_freqs * sizeof(cufftComplex), 
                              cudaMemcpyDeviceToDevice);
                    
                    // Fill negative frequencies by conjugate symmetry
                    // Skip DC and Nyquist
                    for (int j = 1; j < n_freqs - 1; ++j) {
                        cufftComplex conj;
                        cudaMemcpy(&conj, d_stft_data + i * n_freqs + j, sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                        conj.y = -conj.y;  // Conjugate
                        cudaMemcpy(d_full_fft + i * fft_size + fft_size - j, &conj, sizeof(cufftComplex), cudaMemcpyHostToDevice);
                    }
                }
                
                // Execute IFFT
                cufftExecC2C(plan, d_full_fft, d_full_fft, CUFFT_INVERSE);
                
                // Scale to normalize IFFT
                float scale = 1.0f / fft_size;
                for (int i = 0; i < n_frames * fft_size; ++i) {
                    cufftComplex scaled;
                    cudaMemcpy(&scaled, d_full_fft + i, sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                    scaled.x *= scale;
                    scaled.y *= scale;
                    cudaMemcpy(d_full_fft + i, &scaled, sizeof(cufftComplex), cudaMemcpyHostToDevice);
                }
                
                // Overlap-add synthesis
                int shared_mem_size = (params_.window_size + fft_size) * sizeof(float);
                int grid_size = (signal_length + block_size - 1) / block_size;
                
                kernels::istft_overlap_add_kernel<<<grid_size, block_size, shared_mem_size>>>(
                    d_full_fft, d_output, d_window, signal_length, params_.window_size,
                    params_.hop_size, n_frames, fft_size);
                
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                }
                
                // Copy output back to host
                cudaMemcpy(output.data(), d_output, signal_length * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Cleanup
                cufftDestroy(plan);
                cudaFree(d_stft_data);
                cudaFree(d_full_fft);
                cudaFree(d_output);
                cudaFree(d_window);
                
                // If center padding was used, trim output
                if (params_.center) {
                    int pad_size = params_.window_size / 2;
                    output.erase(output.begin(), output.begin() + pad_size);
                    output.resize(signal_length);
                }
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA inverse STFT failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation of inverse STFT
        std::vector<std::complex<float>> ifft_buffer(fft_size);
        
        // Process each frame
        for (int i = 0; i < n_frames; ++i) {
            int frame_start = i * params_.hop_size;
            
            // Reconstruct full FFT result using conjugate symmetry
            for (int j = 0; j <= fft_size / 2; ++j) {
                ifft_buffer[j] = stft_result.spectrogram[i][j];
            }
            
            for (int j = 1; j < fft_size / 2; ++j) {
                ifft_buffer[fft_size - j] = std::conj(stft_result.spectrogram[i][j]);
            }
            
            // Compute IFFT (simplified implementation using DFT for clarity)
            std::vector<std::complex<float>> time_frame(fft_size);
            
            for (int n = 0; n < fft_size; ++n) {
                std::complex<float> sum(0.0f, 0.0f);
                
                for (int k = 0; k < fft_size; ++k) {
                    float angle = 2.0f * PI * k * n / fft_size;
                    std::complex<float> factor(std::cos(angle), std::sin(angle));
                    sum += ifft_buffer[k] * factor;
                }
                
                time_frame[n] = sum / static_cast<float>(fft_size);
            }
            
            // Overlap-add
            for (int j = 0; j < params_.window_size; ++j) {
                int output_idx = frame_start + j;
                
                if (output_idx < signal_length) {
                    output[output_idx] += time_frame[j].real() * window_[j];
                    window_sum[output_idx] += window_[j] * window_[j];
                }
            }
        }
        
        // Normalize by window overlap
        for (int i = 0; i < signal_length; ++i) {
            if (window_sum[i] > 1e-10f) {
                output[i] /= window_sum[i];
            }
        }
        
        // If center padding was used, trim output
        if (params_.center) {
            int pad_size = params_.window_size / 2;
            output.erase(output.begin(), output.begin() + pad_size);
            output.resize(signal_length - params_.window_size);
        }
        
        return output;
    }
    
    std::vector<std::vector<float>> get_magnitude(const STFTResult& stft_result, bool log_scale) {
        int n_frames = stft_result.spectrogram.size();
        int n_freqs = stft_result.spectrogram[0].size();
        
        std::vector<std::vector<float>> magnitude(n_frames, std::vector<float>(n_freqs));
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                cufftComplex* d_stft_data = nullptr;
                float* d_magnitude = nullptr;
                
                cudaMalloc(&d_stft_data, n_frames * n_freqs * sizeof(cufftComplex));
                cudaMalloc(&d_magnitude, n_frames * n_freqs * sizeof(float));
                
                // Flatten STFT data
                std::vector<std::complex<float>> flattened_stft;
                flattened_stft.reserve(n_frames * n_freqs);
                for (const auto& frame : stft_result.spectrogram) {
                    flattened_stft.insert(flattened_stft.end(), frame.begin(), frame.end());
                }
                
                // Copy data to device
                cudaMemcpy(d_stft_data, flattened_stft.data(), flattened_stft.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);
                
                // Launch kernel
                int block_size = std::min(256, n_freqs);
                
                kernels::stft_magnitude_kernel<<<n_frames, block_size>>>(
                    d_stft_data, d_magnitude, n_frames, n_freqs, 1.0f, log_scale);
                
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                }
                
                // Copy result back to host
                std::vector<float> flattened_magnitude(n_frames * n_freqs);
                cudaMemcpy(flattened_magnitude.data(), d_magnitude, flattened_magnitude.size() * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Reshape result
                for (int i = 0; i < n_frames; ++i) {
                    for (int j = 0; j < n_freqs; ++j) {
                        magnitude[i][j] = flattened_magnitude[i * n_freqs + j];
                    }
                }
                
                // Cleanup
                cudaFree(d_stft_data);
                cudaFree(d_magnitude);
                
                return magnitude;
            } catch (const std::exception& e) {
                std::cerr << "CUDA magnitude computation failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation
        const float min_value = 1e-10f;
        
        for (int i = 0; i < n_frames; ++i) {
            for (int j = 0; j < n_freqs; ++j) {
                float mag = std::abs(stft_result.spectrogram[i][j]);
                
                if (log_scale) {
                    magnitude[i][j] = 10.0f * std::log10(std::max(mag * mag, min_value));
                } else {
                    magnitude[i][j] = mag;
                }
            }
        }
        
        return magnitude;
    }
    
    std::vector<std::vector<float>> get_phase(const STFTResult& stft_result) {
        int n_frames = stft_result.spectrogram.size();
        int n_freqs = stft_result.spectrogram[0].size();
        
        std::vector<std::vector<float>> phase(n_frames, std::vector<float>(n_freqs));
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                cufftComplex* d_stft_data = nullptr;
                float* d_phase = nullptr;
                
                cudaMalloc(&d_stft_data, n_frames * n_freqs * sizeof(cufftComplex));
                cudaMalloc(&d_phase, n_frames * n_freqs * sizeof(float));
                
                // Flatten STFT data
                std::vector<std::complex<float>> flattened_stft;
                flattened_stft.reserve(n_frames * n_freqs);
                for (const auto& frame : stft_result.spectrogram) {
                    flattened_stft.insert(flattened_stft.end(), frame.begin(), frame.end());
                }
                
                // Copy data to device
                cudaMemcpy(d_stft_data, flattened_stft.data(), flattened_stft.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);
                
                // Launch kernel
                int block_size = std::min(256, n_freqs);
                
                kernels::stft_phase_kernel<<<n_frames, block_size>>>(
                    d_stft_data, d_phase, n_frames, n_freqs);
                
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                }
                
                // Copy result back to host
                std::vector<float> flattened_phase(n_frames * n_freqs);
                cudaMemcpy(flattened_phase.data(), d_phase, flattened_phase.size() * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Reshape result
                for (int i = 0; i < n_frames; ++i) {
                    for (int j = 0; j < n_freqs; ++j) {
                        phase[i][j] = flattened_phase[i * n_freqs + j];
                    }
                }
                
                // Cleanup
                cudaFree(d_stft_data);
                cudaFree(d_phase);
                
                return phase;
            } catch (const std::exception& e) {
                std::cerr << "CUDA phase computation failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation
        for (int i = 0; i < n_frames; ++i) {
            for (int j = 0; j < n_freqs; ++j) {
                phase[i][j] = std::arg(stft_result.spectrogram[i][j]);
            }
        }
        
        return phase;
    }
    
    std::vector<std::vector<float>> get_power(const STFTResult& stft_result, bool log_scale) {
        int n_frames = stft_result.spectrogram.size();
        int n_freqs = stft_result.spectrogram[0].size();
        
        std::vector<std::vector<float>> power(n_frames, std::vector<float>(n_freqs));
        
        // Get magnitude (reuse existing code)
        auto magnitude = get_magnitude(stft_result, false);
        
        // Convert to power
        const float min_value = 1e-10f;
        
        for (int i = 0; i < n_frames; ++i) {
            for (int j = 0; j < n_freqs; ++j) {
                float pow_val = magnitude[i][j] * magnitude[i][j];
                
                if (log_scale) {
                    power[i][j] = 10.0f * std::log10(std::max(pow_val, min_value));
                } else {
                    power[i][j] = pow_val;
                }
            }
        }
        
        return power;
    }
    
private:
    STFTParams params_;
    std::vector<float> window_;
    int device_id_;
    bool has_cuda_;
    
#if defined(WITH_CUDA)
    // CUDA-specific variables would go here
#endif
    
    void initialize() {
        // Generate window function
        window_ = generate_window(params_.window_size, params_.window_type, params_.window_param);
        
#if defined(WITH_CUDA)
        // Check for CUDA
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error == cudaSuccess && device_count > 0) {
            // Check if requested device is valid
            if (device_id_ >= 0 && device_id_ < device_count) {
                has_cuda_ = true;
                
                // Set device
                cudaSetDevice(device_id_);
                
                // Get device properties
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, device_id_);
                
                std::cout << "Using CUDA device " << device_id_ << ": " << props.name << std::endl;
                std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
            } else {
                std::cerr << "Invalid CUDA device ID: " << device_id_ << std::endl;
                has_cuda_ = false;
                device_id_ = -1;
            }
        } else {
            std::cerr << "CUDA not available, using CPU implementation" << std::endl;
            has_cuda_ = false;
            device_id_ = -1;
        }
#else
        has_cuda_ = false;
        device_id_ = -1;
#endif
    }
    
    void cleanup() {
#if defined(WITH_CUDA)
        if (has_cuda_) {
            // Reset CUDA device
            cudaDeviceSynchronize();
        }
#endif
    }
};

//------------------------------------------------------------------------------
// CWT Implementation
//------------------------------------------------------------------------------

class CWTImpl {
public:
    CWTImpl(const CWTParams& params, int device_id)
        : params_(params), device_id_(device_id), has_cuda_(false) {
        initialize();
    }
    
    ~CWTImpl() {
        cleanup();
    }
    
    CWTResult transform(const std::vector<float>& signal, float sample_rate) {
        if (signal.empty()) {
            return {};
        }
        
        // Generate scales
        auto scales = generate_scales(params_.num_scales, params_.min_scale, params_.max_scale, params_.normalize_scales);
        
        // Prepare result structure
        CWTResult result;
        result.scalogram.resize(scales.size(), std::vector<std::complex<float>>(signal.size()));
        result.scales = scales;
        result.times.resize(signal.size());
        result.sample_rate = sample_rate;
        
        // Calculate time points
        for (size_t i = 0; i < signal.size(); ++i) {
            result.times[i] = static_cast<float>(i) / sample_rate;
        }
        
        // Calculate corresponding frequencies
        result.frequencies = scales_to_frequencies(scales, params_.wavelet_param, sample_rate);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Get device properties
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, device_id_);
                
                // Choose implementation strategy based on signal size
                if (signal.size() > 1024) {
                    // For large signals, use frequency domain method (FFT convolution)
                    
                    // Allocate device memory
                    float* d_signal = nullptr;
                    cufftComplex* d_signal_fft = nullptr;
                    cufftComplex* d_wavelet_filters = nullptr;
                    cufftComplex* d_cwt_data = nullptr;
                    float* d_frequencies = nullptr;
                    float* d_scales = nullptr;
                    
                    cudaMalloc(&d_signal, signal.size() * sizeof(float));
                    cudaMalloc(&d_signal_fft, signal.size() * sizeof(cufftComplex));
                    cudaMalloc(&d_wavelet_filters, scales.size() * signal.size() * sizeof(cufftComplex));
                    cudaMalloc(&d_cwt_data, scales.size() * signal.size() * sizeof(cufftComplex));
                    cudaMalloc(&d_frequencies, signal.size() * sizeof(float));
                    cudaMalloc(&d_scales, scales.size() * sizeof(float));
                    
                    // Copy data to device
                    cudaMemcpy(d_signal, signal.data(), signal.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice);
                    
                    // Create FFT plan
                    cufftHandle plan;
                    cufftPlan1d(&plan, signal.size(), CUFFT_R2C, 1);
                    
                    // Compute FFT of signal
                    cufftExecR2C(plan, d_signal, d_signal_fft);
                    
                    // Calculate frequencies for FFT bins
                    std::vector<float> frequencies(signal.size());
                    for (size_t i = 0; i < signal.size(); ++i) {
                        frequencies[i] = static_cast<float>(i) * sample_rate / signal.size();
                    }
                    
                    // Copy frequencies to device
                    cudaMemcpy(d_frequencies, frequencies.data(), frequencies.size() * sizeof(float), cudaMemcpyHostToDevice);
                    
                    // Generate wavelet filters in frequency domain
                    int wavelet_type = static_cast<int>(params_.wavelet_type);
                    int block_size = std::min(256, static_cast<int>(signal.size()));
                    
                    kernels::cwt_generate_filter_kernel<<<scales.size(), block_size>>>(
                        d_wavelet_filters, d_frequencies, d_scales, signal.size(), scales.size(),
                        wavelet_type, params_.wavelet_param);
                    
                    // Multiply signal FFT with wavelet filters
                    kernels::cwt_frequency_kernel<<<scales.size(), block_size>>>(
                        d_signal_fft, d_wavelet_filters, d_cwt_data, signal.size(), scales.size());
                    
                    // Perform inverse FFT for each scale
                    cufftHandle ifft_plan;
                    cufftPlan1d(&ifft_plan, signal.size(), CUFFT_C2C, scales.size());
                    cufftExecC2C(ifft_plan, d_cwt_data, d_cwt_data, CUFFT_INVERSE);
                    
                    // Normalize IFFT
                    float scale = 1.0f / signal.size();
                    for (int i = 0; i < scales.size() * signal.size(); ++i) {
                        cufftComplex scaled;
                        cudaMemcpy(&scaled, d_cwt_data + i, sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                        scaled.x *= scale;
                        scaled.y *= scale;
                        cudaMemcpy(d_cwt_data + i, &scaled, sizeof(cufftComplex), cudaMemcpyHostToDevice);
                    }
                    
                    // Copy result back to host
                    std::vector<std::complex<float>> cwt_result(scales.size() * signal.size());
                    cudaMemcpy(cwt_result.data(), d_cwt_data, cwt_result.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                    
                    // Reshape result
                    for (size_t i = 0; i < scales.size(); ++i) {
                        for (size_t j = 0; j < signal.size(); ++j) {
                            result.scalogram[i][j] = cwt_result[i * signal.size() + j];
                        }
                    }
                    
                    // Cleanup
                    cufftDestroy(plan);
                    cufftDestroy(ifft_plan);
                    cudaFree(d_signal);
                    cudaFree(d_signal_fft);
                    cudaFree(d_wavelet_filters);
                    cudaFree(d_cwt_data);
                    cudaFree(d_frequencies);
                    cudaFree(d_scales);
                } else {
                    // For small signals, use direct convolution method
                    
                    // Allocate device memory
                    float* d_signal = nullptr;
                    cufftComplex* d_cwt_data = nullptr;
                    float* d_scales = nullptr;
                    
                    cudaMalloc(&d_signal, signal.size() * sizeof(float));
                    cudaMalloc(&d_cwt_data, scales.size() * signal.size() * sizeof(cufftComplex));
                    cudaMalloc(&d_scales, scales.size() * sizeof(float));
                    
                    // Copy data to device
                    cudaMemcpy(d_signal, signal.data(), signal.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice);
                    
                    // Determine wavelet width in samples (larger for lower frequencies)
                    int wavelet_width = 10;  // Default width in wavelets
                    
                    // Launch kernel
                    int wavelet_type = static_cast<int>(params_.wavelet_type);
                    dim3 block_size(256);
                    dim3 grid_size(scales.size(), (signal.size() + block_size.x - 1) / block_size.x);
                    
                    kernels::cwt_direct_kernel<<<grid_size, block_size>>>(
                        d_signal, d_cwt_data, signal.size(), scales.size(), d_scales,
                        wavelet_type, params_.wavelet_param, wavelet_width);
                    
                    // Check for kernel launch errors
                    cudaError_t err = cudaGetLastError();
                    if (err != cudaSuccess) {
                        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                    }
                    
                    // Copy result back to host
                    std::vector<std::complex<float>> cwt_result(scales.size() * signal.size());
                    cudaMemcpy(cwt_result.data(), d_cwt_data, cwt_result.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                    
                    // Reshape result
                    for (size_t i = 0; i < scales.size(); ++i) {
                        for (size_t j = 0; j < signal.size(); ++j) {
                            result.scalogram[i][j] = cwt_result[i * signal.size() + j];
                        }
                    }
                    
                    // Cleanup
                    cudaFree(d_signal);
                    cudaFree(d_cwt_data);
                    cudaFree(d_scales);
                }
                
                return result;
            } catch (const std::exception& e) {
                std::cerr << "CUDA CWT failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation of CWT (direct method for clarity)
        // In practice, would use FFT convolution for efficiency
        
        // Define wavelet function based on type
        auto generateWavelet = [this](float t, float scale) -> std::complex<float> {
            float t_scaled = t / scale;
            
            switch (params_.wavelet_type) {
                case WaveletType::MORLET: {
                    // Morlet wavelet
                    float arg = -0.5f * t_scaled * t_scaled;
                    float envelope = std::exp(arg);
                    float cos_term = std::cos(params_.wavelet_param * t_scaled);
                    float sin_term = std::sin(params_.wavelet_param * t_scaled);
                    
                    // Normalization factor
                    float norm = 1.0f / std::sqrt(scale);
                    
                    return std::complex<float>(norm * envelope * cos_term, norm * envelope * sin_term);
                }
                
                case WaveletType::MEXICAN_HAT: {
                    // Mexican hat (Ricker) wavelet
                    float arg = -0.5f * t_scaled * t_scaled;
                    float term = (1.0f - t_scaled * t_scaled) * std::exp(arg);
                    
                    // Normalization factor
                    float norm = 1.0f / std::sqrt(scale);
                    
                    return std::complex<float>(norm * term, 0.0f);
                }
                
                default:
                    // Default to Morlet
                    float arg = -0.5f * t_scaled * t_scaled;
                    float envelope = std::exp(arg);
                    float cos_term = std::cos(params_.wavelet_param * t_scaled);
                    float sin_term = std::sin(params_.wavelet_param * t_scaled);
                    
                    // Normalization factor
                    float norm = 1.0f / std::sqrt(scale);
                    
                    return std::complex<float>(norm * envelope * cos_term, norm * envelope * sin_term);
            }
        };
        
        // Compute CWT for each scale
        for (size_t scale_idx = 0; scale_idx < scales.size(); ++scale_idx) {
            float scale = scales[scale_idx];
            
            // Width of wavelet (in samples) depends on scale
            int wavelet_width = static_cast<int>(10 * scale);
            
            // Convolve signal with wavelet
            for (size_t time_idx = 0; time_idx < signal.size(); ++time_idx) {
                std::complex<float> sum(0.0f, 0.0f);
                
                // Apply wavelet centered at current time
                for (int i = -wavelet_width; i <= wavelet_width; ++i) {
                    float t = i / sample_rate;
                    int signal_idx = static_cast<int>(time_idx) + i;
                    
                    // Handle boundary conditions
                    if (signal_idx >= 0 && signal_idx < static_cast<int>(signal.size())) {
                        std::complex<float> wavelet = generateWavelet(t, scale);
                        sum += signal[signal_idx] * wavelet;
                    }
                }
                
                result.scalogram[scale_idx][time_idx] = sum;
            }
        }
        
        return result;
    }
    
    std::vector<std::vector<float>> get_magnitude(const CWTResult& cwt_result, bool log_scale) {
        int n_scales = cwt_result.scalogram.size();
        int n_times = cwt_result.scalogram[0].size();
        
        std::vector<std::vector<float>> magnitude(n_scales, std::vector<float>(n_times));
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                cufftComplex* d_cwt_data = nullptr;
                float* d_magnitude = nullptr;
                
                cudaMalloc(&d_cwt_data, n_scales * n_times * sizeof(cufftComplex));
                cudaMalloc(&d_magnitude, n_scales * n_times * sizeof(float));
                
                // Flatten CWT data
                std::vector<std::complex<float>> flattened_cwt;
                flattened_cwt.reserve(n_scales * n_times);
                for (const auto& scale : cwt_result.scalogram) {
                    flattened_cwt.insert(flattened_cwt.end(), scale.begin(), scale.end());
                }
                
                // Copy data to device
                cudaMemcpy(d_cwt_data, flattened_cwt.data(), flattened_cwt.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);
                
                // Launch kernel
                int block_size = std::min(256, n_times);
                dim3 grid_size(n_scales, (n_times + block_size - 1) / block_size);
                
                kernels::cwt_scalogram_kernel<<<n_scales, block_size>>>(
                    d_cwt_data, d_magnitude, n_times, n_scales, log_scale);
                
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                }
                
                // Copy result back to host
                std::vector<float> flattened_magnitude(n_scales * n_times);
                cudaMemcpy(flattened_magnitude.data(), d_magnitude, flattened_magnitude.size() * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Reshape result
                for (int i = 0; i < n_scales; ++i) {
                    for (int j = 0; j < n_times; ++j) {
                        magnitude[i][j] = flattened_magnitude[i * n_times + j];
                    }
                }
                
                // Cleanup
                cudaFree(d_cwt_data);
                cudaFree(d_magnitude);
                
                return magnitude;
            } catch (const std::exception& e) {
                std::cerr << "CUDA magnitude computation failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation
        const float min_value = 1e-10f;
        
        for (int i = 0; i < n_scales; ++i) {
            for (int j = 0; j < n_times; ++j) {
                float mag = std::abs(cwt_result.scalogram[i][j]);
                
                if (log_scale) {
                    magnitude[i][j] = 10.0f * std::log10(std::max(mag * mag, min_value));
                } else {
                    magnitude[i][j] = mag;
                }
            }
        }
        
        return magnitude;
    }
    
    std::vector<std::vector<float>> get_phase(const CWTResult& cwt_result) {
        int n_scales = cwt_result.scalogram.size();
        int n_times = cwt_result.scalogram[0].size();
        
        std::vector<std::vector<float>> phase(n_scales, std::vector<float>(n_times));
        
        // CPU implementation (CUDA implementation would be similar to STFT phase)
        for (int i = 0; i < n_scales; ++i) {
            for (int j = 0; j < n_times; ++j) {
                phase[i][j] = std::arg(cwt_result.scalogram[i][j]);
            }
        }
        
        return phase;
    }
    
    std::vector<std::vector<float>> get_power(const CWTResult& cwt_result, bool log_scale) {
        // Get magnitude (reuse existing code)
        auto magnitude = get_magnitude(cwt_result, false);
        
        // Convert to power
        int n_scales = magnitude.size();
        int n_times = magnitude[0].size();
        
        std::vector<std::vector<float>> power(n_scales, std::vector<float>(n_times));
        const float min_value = 1e-10f;
        
        for (int i = 0; i < n_scales; ++i) {
            for (int j = 0; j < n_times; ++j) {
                float pow_val = magnitude[i][j] * magnitude[i][j];
                
                if (log_scale) {
                    power[i][j] = 10.0f * std::log10(std::max(pow_val, min_value));
                } else {
                    power[i][j] = pow_val;
                }
            }
        }
        
        return power;
    }
    
private:
    CWTParams params_;
    int device_id_;
    bool has_cuda_;
    
#if defined(WITH_CUDA)
    // CUDA-specific variables would go here
#endif
    
    void initialize() {
#if defined(WITH_CUDA)
        // Check for CUDA
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error == cudaSuccess && device_count > 0) {
            // Check if requested device is valid
            if (device_id_ >= 0 && device_id_ < device_count) {
                has_cuda_ = true;
                
                // Set device
                cudaSetDevice(device_id_);
                
                // Get device properties
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, device_id_);
                
                std::cout << "Using CUDA device " << device_id_ << ": " << props.name << std::endl;
                std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
            } else {
                std::cerr << "Invalid CUDA device ID: " << device_id_ << std::endl;
                has_cuda_ = false;
                device_id_ = -1;
            }
        } else {
            std::cerr << "CUDA not available, using CPU implementation" << std::endl;
            has_cuda_ = false;
            device_id_ = -1;
        }
#else
        has_cuda_ = false;
        device_id_ = -1;
#endif
    }
    
    void cleanup() {
#if defined(WITH_CUDA)
        if (has_cuda_) {
            // Reset CUDA device
            cudaDeviceSynchronize();
        }
#endif
    }
};

//------------------------------------------------------------------------------
// STFT Class Implementation
//------------------------------------------------------------------------------

STFT::STFT(const STFTParams& params, int device_id)
    : impl_(std::make_unique<STFTImpl>(params, device_id)) {}

STFT::~STFT() = default;

STFT::STFT(STFT&&) noexcept = default;
STFT& STFT::operator=(STFT&&) noexcept = default;

STFTResult STFT::transform(const std::vector<float>& signal, float sample_rate) {
    return impl_->transform(signal, sample_rate);
}

std::vector<float> STFT::inverse_transform(const STFTResult& stft_result) {
    return impl_->inverse_transform(stft_result);
}

std::vector<std::vector<float>> STFT::get_magnitude(const STFTResult& stft_result, bool log_scale) {
    return impl_->get_magnitude(stft_result, log_scale);
}

std::vector<std::vector<float>> STFT::get_phase(const STFTResult& stft_result) {
    return impl_->get_phase(stft_result);
}

std::vector<std::vector<float>> STFT::get_power(const STFTResult& stft_result, bool log_scale) {
    return impl_->get_power(stft_result, log_scale);
}

//------------------------------------------------------------------------------
// CWT Class Implementation
//------------------------------------------------------------------------------

CWT::CWT(const CWTParams& params, int device_id)
    : impl_(std::make_unique<CWTImpl>(params, device_id)) {}

CWT::~CWT() = default;

CWT::CWT(CWT&&) noexcept = default;
CWT& CWT::operator=(CWT&&) noexcept = default;

CWTResult CWT::transform(const std::vector<float>& signal, float sample_rate) {
    return impl_->transform(signal, sample_rate);
}

std::vector<std::vector<float>> CWT::get_magnitude(const CWTResult& cwt_result, bool log_scale) {
    return impl_->get_magnitude(cwt_result, log_scale);
}

std::vector<std::vector<float>> CWT::get_phase(const CWTResult& cwt_result) {
    return impl_->get_phase(cwt_result);
}

std::vector<std::vector<float>> CWT::get_power(const CWTResult& cwt_result, bool log_scale) {
    return impl_->get_power(cwt_result, log_scale);
}

//------------------------------------------------------------------------------
// Static time-frequency functions
//------------------------------------------------------------------------------

namespace time_frequency {

std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
spectrogram(
    const std::vector<float>& signal,
    float sample_rate,
    int window_size,
    int hop_size,
    WindowType window_type,
    bool log_scale,
    int device_id) {
    
    // Create STFT parameters
    STFTParams params;
    params.window_size = window_size;
    params.hop_size = hop_size;
    params.window_type = window_type;
    
    // Create STFT object
    STFT stft(params, device_id);
    
    // Compute STFT
    auto stft_result = stft.transform(signal, sample_rate);
    
    // Compute magnitude spectrogram
    auto magnitude = stft.get_magnitude(stft_result, log_scale);
    
    return {magnitude, {stft_result.times, stft_result.frequencies}};
}

std::pair<std::vector<std::vector<float>>, std::pair<std::vector<float>, std::vector<float>>>
scalogram(
    const std::vector<float>& signal,
    float sample_rate,
    WaveletType wavelet_type,
    int num_scales,
    bool log_scale,
    int device_id) {
    
    // Create CWT parameters
    CWTParams params;
    params.wavelet_type = wavelet_type;
    params.num_scales = num_scales;
    
    // Create CWT object
    CWT cwt(params, device_id);
    
    // Compute CWT
    auto cwt_result = cwt.transform(signal, sample_rate);
    
    // Compute magnitude scalogram
    auto magnitude = cwt.get_magnitude(cwt_result, log_scale);
    
    return {magnitude, {cwt_result.times, cwt_result.frequencies}};
}

std::vector<std::complex<float>> hilbert_transform(
    const std::vector<float>& signal,
    int device_id) {
    
    if (signal.empty()) {
        return {};
    }
    
    std::vector<std::complex<float>> analytic_signal(signal.size());
    
#if defined(WITH_CUDA)
    bool has_cuda = false;
    
    // Check for CUDA
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error == cudaSuccess && device_count > 0 && device_id >= 0 && device_id < device_count) {
        has_cuda = true;
        
        try {
            // Set CUDA device
            cudaSetDevice(device_id);
            
            // Allocate device memory
            float* d_signal = nullptr;
            cufftComplex* d_fft = nullptr;
            cufftComplex* d_analytic_fft = nullptr;
            
            cudaMalloc(&d_signal, signal.size() * sizeof(float));
            cudaMalloc(&d_fft, signal.size() * sizeof(cufftComplex));
            cudaMalloc(&d_analytic_fft, signal.size() * sizeof(cufftComplex));
            
            // Copy data to device
            cudaMemcpy(d_signal, signal.data(), signal.size() * sizeof(float), cudaMemcpyHostToDevice);
            
            // Create FFT plan
            cufftHandle plan;
            cufftPlan1d(&plan, signal.size(), CUFFT_R2C, 1);
            
            // Execute FFT
            cufftExecR2C(plan, d_signal, d_fft);
            
            // Generate analytic signal
            int block_size = 256;
            int grid_size = (signal.size() + block_size - 1) / block_size;
            
            kernels::analytic_signal_kernel<<<grid_size, block_size>>>(
                d_fft, d_analytic_fft, signal.size());
            
            // Execute inverse FFT
            cufftHandle ifft_plan;
            cufftPlan1d(&ifft_plan, signal.size(), CUFFT_C2C, 1);
            cufftExecC2C(ifft_plan, d_analytic_fft, d_analytic_fft, CUFFT_INVERSE);
            
            // Normalize result
            float scale = 1.0f / signal.size();
            for (int i = 0; i < signal.size(); ++i) {
                cufftComplex scaled;
                cudaMemcpy(&scaled, d_analytic_fft + i, sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                scaled.x *= scale;
                scaled.y *= scale;
                cudaMemcpy(d_analytic_fft + i, &scaled, sizeof(cufftComplex), cudaMemcpyHostToDevice);
            }
            
            // Copy result back to host
            cudaMemcpy(analytic_signal.data(), d_analytic_fft, analytic_signal.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
            
            // Cleanup
            cufftDestroy(plan);
            cufftDestroy(ifft_plan);
            cudaFree(d_signal);
            cudaFree(d_fft);
            cudaFree(d_analytic_fft);
            
            return analytic_signal;
        } catch (const std::exception& e) {
            std::cerr << "CUDA Hilbert transform failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU implementation" << std::endl;
            // Fall back to CPU implementation
        }
    }
#endif
    
    // CPU implementation
    // Perform FFT
    std::vector<std::complex<float>> fft_result(signal.size());
    
    // Simple DFT implementation (in practice, use a fast FFT library)
    for (size_t k = 0; k < signal.size(); ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        
        for (size_t n = 0; n < signal.size(); ++n) {
            float angle = -2.0f * PI * k * n / signal.size();
            std::complex<float> factor(std::cos(angle), std::sin(angle));
            sum += signal[n] * factor;
        }
        
        fft_result[k] = sum;
    }
    
    // Generate analytic signal
    std::vector<std::complex<float>> analytic_fft(signal.size());
    
    // DC component remains unchanged
    analytic_fft[0] = fft_result[0];
    
    // Positive frequencies: multiply by 2
    for (size_t i = 1; i < signal.size() / 2; ++i) {
        analytic_fft[i] = 2.0f * fft_result[i];
    }
    
    // Nyquist frequency (if even-sized signal)
    if (signal.size() % 2 == 0) {
        analytic_fft[signal.size() / 2] = fft_result[signal.size() / 2];
    }
    
    // Negative frequencies: set to zero
    for (size_t i = signal.size() / 2 + 1; i < signal.size(); ++i) {
        analytic_fft[i] = std::complex<float>(0.0f, 0.0f);
    }
    
    // Perform inverse FFT
    for (size_t n = 0; n < signal.size(); ++n) {
        std::complex<float> sum(0.0f, 0.0f);
        
        for (size_t k = 0; k < signal.size(); ++k) {
            float angle = 2.0f * PI * k * n / signal.size();
            std::complex<float> factor(std::cos(angle), std::sin(angle));
            sum += analytic_fft[k] * factor;
        }
        
        analytic_signal[n] = sum / static_cast<float>(signal.size());
    }
    
    return analytic_signal;
}

std::vector<float> instantaneous_frequency(
    const std::vector<float>& signal,
    float sample_rate,
    int device_id) {
    
    if (signal.empty()) {
        return {};
    }
    
    // Compute analytic signal using Hilbert transform
    auto analytic_signal = hilbert_transform(signal, device_id);
    
    // Compute instantaneous frequency from phase of analytic signal
    std::vector<float> inst_freq(signal.size());
    
    // Skip first and last samples where derivative might be unreliable
    for (size_t i = 1; i < signal.size() - 1; ++i) {
        // Compute phase of current and previous samples
        float phase_prev = std::arg(analytic_signal[i - 1]);
        float phase_curr = std::arg(analytic_signal[i]);
        
        // Compute wrapped phase difference
        float phase_diff = phase_curr - phase_prev;
        
        // Unwrap phase difference
        if (phase_diff > PI) {
            phase_diff -= 2 * PI;
        } else if (phase_diff < -PI) {
            phase_diff += 2 * PI;
        }
        
        // Convert to frequency
        inst_freq[i] = phase_diff * sample_rate / (2 * PI);
    }
    
    // Fill first and last samples with neighbor values
    if (signal.size() > 1) {
        inst_freq[0] = inst_freq[1];
        inst_freq[signal.size() - 1] = inst_freq[signal.size() - 2];
    }
    
    return inst_freq;
}

} // namespace time_frequency

} // namespace signal_processing