/**
 * @file spectral_analysis.cpp
 * @brief Implementation of FFT and spectral analysis functions
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include "signal_processing/spectral_analysis.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <unordered_map>
#include <mutex>

// Check for CUDA availability
#if defined(WITH_CUDA)
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#endif

namespace signal_processing {

namespace {
// Constants
constexpr float PI = 3.14159265358979323846f;

// Helper functions
template<typename T>
std::vector<T> pad_to_power_of_two(const std::vector<T>& input) {
    size_t size = input.size();
    size_t padded_size = 1;
    while (padded_size < size) {
        padded_size *= 2;
    }
    
    std::vector<T> padded(padded_size);
    std::copy(input.begin(), input.end(), padded.begin());
    return padded;
}

// Generate window function
std::vector<float> generate_window(WindowType type, int size, float param = 0.0f) {
    std::vector<float> window(size);
    
    switch (type) {
        case WindowType::RECTANGULAR:
            std::fill(window.begin(), window.end(), 1.0f);
            break;
        
        case WindowType::HANN:
            for (int i = 0; i < size; ++i) {
                window[i] = 0.5f * (1.0f - std::cos(2.0f * PI * i / (size - 1)));
            }
            break;
        
        case WindowType::HAMMING:
            for (int i = 0; i < size; ++i) {
                window[i] = 0.54f - 0.46f * std::cos(2.0f * PI * i / (size - 1));
            }
            break;
        
        case WindowType::BLACKMAN:
            for (int i = 0; i < size; ++i) {
                float x = 2.0f * PI * i / (size - 1);
                window[i] = 0.42f - 0.5f * std::cos(x) + 0.08f * std::cos(2.0f * x);
            }
            break;
        
        case WindowType::FLATTOP:
            for (int i = 0; i < size; ++i) {
                float x = 2.0f * PI * i / (size - 1);
                window[i] = 0.21557895f - 0.41663158f * std::cos(x) + 
                          0.277263158f * std::cos(2.0f * x) - 
                          0.083578947f * std::cos(3.0f * x) + 
                          0.006947368f * std::cos(4.0f * x);
            }
            break;
        
        case WindowType::KAISER:
            {
                float beta = param > 0.0f ? param : 3.0f;  // Default beta = 3.0
                
                // For Kaiser window, we need to compute the modified Bessel function of order 0
                auto bessel_i0 = [](float x) {
                    float sum = 1.0f;
                    float term = 1.0f;
                    
                    for (int k = 1; k <= 20; ++k) {
                        term *= (x * x) / (4.0f * k * k);
                        sum += term;
                        
                        if (term < 1e-7f * sum) {
                            break;  // Convergence reached
                        }
                    }
                    
                    return sum;
                };
                
                float denom = bessel_i0(beta);
                for (int i = 0; i < size; ++i) {
                    float x = 2.0f * i / (size - 1) - 1.0f;  // Map to [-1, 1]
                    window[i] = bessel_i0(beta * std::sqrt(1.0f - x * x)) / denom;
                }
            }
            break;
        
        case WindowType::TUKEY:
            {
                float alpha = param > 0.0f ? param : 0.5f;  // Default alpha = 0.5
                for (int i = 0; i < size; ++i) {
                    float x = static_cast<float>(i) / (size - 1);
                    if (x < alpha / 2.0f) {
                        window[i] = 0.5f * (1.0f - std::cos(2.0f * PI * x / alpha));
                    } else if (x > 1.0f - alpha / 2.0f) {
                        window[i] = 0.5f * (1.0f - std::cos(2.0f * PI * (1.0f - x) / alpha));
                    } else {
                        window[i] = 1.0f;
                    }
                }
            }
            break;
        
        case WindowType::GAUSSIAN:
            {
                float sigma = param > 0.0f ? param : 0.5f;  // Default sigma = 0.5
                for (int i = 0; i < size; ++i) {
                    float x = (i - (size - 1) / 2.0f) / ((size - 1) / 2.0f);
                    window[i] = std::exp(-0.5f * std::pow(x / sigma, 2.0f));
                }
            }
            break;
        
        default:
            std::fill(window.begin(), window.end(), 1.0f);
            break;
    }
    
    return window;
}

// Apply window function to a signal
template<typename T>
std::vector<T> apply_window(const std::vector<T>& signal, const std::vector<float>& window) {
    if (signal.size() != window.size()) {
        throw std::invalid_argument("Signal and window sizes must match");
    }
    
    std::vector<T> windowed(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        windowed[i] = signal[i] * window[i];
    }
    
    return windowed;
}

// Detrend a signal
std::vector<float> detrend(const std::vector<float>& signal, const std::string& method) {
    if (method == "none") {
        return signal;
    } else if (method == "constant") {
        // Remove mean
        float mean = std::accumulate(signal.begin(), signal.end(), 0.0f) / signal.size();
        std::vector<float> detrended(signal.size());
        for (size_t i = 0; i < signal.size(); ++i) {
            detrended[i] = signal[i] - mean;
        }
        return detrended;
    } else if (method == "linear") {
        // Remove linear trend
        std::vector<float> x(signal.size());
        std::iota(x.begin(), x.end(), 0);
        
        // Calculate slope and intercept using least squares
        float sum_x = std::accumulate(x.begin(), x.end(), 0.0f);
        float sum_y = std::accumulate(signal.begin(), signal.end(), 0.0f);
        float sum_xx = 0.0f, sum_xy = 0.0f;
        
        for (size_t i = 0; i < signal.size(); ++i) {
            sum_xx += x[i] * x[i];
            sum_xy += x[i] * signal[i];
        }
        
        float n = static_cast<float>(signal.size());
        float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        float intercept = (sum_y - slope * sum_x) / n;
        
        // Remove trend
        std::vector<float> detrended(signal.size());
        for (size_t i = 0; i < signal.size(); ++i) {
            detrended[i] = signal[i] - (slope * x[i] + intercept);
        }
        
        return detrended;
    } else {
        throw std::invalid_argument("Unknown detrend method: " + method);
    }
}

// Segment a signal for Welch's method
std::vector<std::vector<float>> segment_signal(
    const std::vector<float>& signal,
    int segment_size,
    int overlap) {
    
    if (segment_size >= static_cast<int>(signal.size())) {
        return {signal};
    }
    
    int step = segment_size - overlap;
    if (step <= 0) {
        throw std::invalid_argument("Overlap must be less than segment size");
    }
    
    int num_segments = (signal.size() - overlap) / step;
    std::vector<std::vector<float>> segments;
    segments.reserve(num_segments);
    
    for (int i = 0; i < num_segments; ++i) {
        int start = i * step;
        std::vector<float> segment(segment_size);
        std::copy(signal.begin() + start, signal.begin() + start + segment_size, segment.begin());
        segments.push_back(std::move(segment));
    }
    
    return segments;
}

// Modified Bessel function of first kind, order 0
// Used for Kaiser window calculation
float bessel_i0(float x) {
    float sum = 1.0f;
    float term = 1.0f;
    
    for (int k = 1; k <= 30; ++k) {
        term *= (x * x) / (4.0f * k * k);
        sum += term;
        
        if (term < 1e-7f * sum) {
            break;  // Convergence reached
        }
    }
    
    return sum;
}

} // anonymous namespace

//------------------------------------------------------------------------------
// FFT Implementation
//------------------------------------------------------------------------------

class FFTImpl {
public:
    explicit FFTImpl(int device_id)
        : device_id_(device_id), has_cuda_(false) {
        initialize();
    }
    
    ~FFTImpl() {
        cleanup();
    }
    
    bool has_cuda() const {
        return has_cuda_;
    }
    
    int get_device_id() const {
        return device_id_;
    }
    
    std::vector<std::complex<float>> forward_1d_real(
        const std::vector<float>& input,
        bool normalize) {
        
        // Pad input to power of 2 for efficiency
        auto padded_input = pad_to_power_of_two(input);
        int n = static_cast<int>(padded_input.size());
        
        std::vector<std::complex<float>> output(n / 2 + 1);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                float* d_input = nullptr;
                cufftComplex* d_output = nullptr;
                
                cudaMalloc(&d_input, n * sizeof(float));
                cudaMalloc(&d_output, (n / 2 + 1) * sizeof(cufftComplex));
                
                // Copy input to device
                cudaMemcpy(d_input, padded_input.data(), n * sizeof(float), cudaMemcpyHostToDevice);
                
                // Create plan
                cufftHandle plan;
                cufftPlan1d(&plan, n, CUFFT_R2C, 1);
                
                // Execute plan
                cufftExecR2C(plan, d_input, d_output);
                
                // Copy output to host
                std::vector<cufftComplex> temp((n / 2 + 1));
                cudaMemcpy(temp.data(), d_output, (n / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                
                // Convert to std::complex
                for (int i = 0; i < n / 2 + 1; ++i) {
                    output[i] = std::complex<float>(temp[i].x, temp[i].y);
                    if (normalize) {
                        output[i] /= std::sqrt(static_cast<float>(n));
                    }
                }
                
                // Cleanup
                cufftDestroy(plan);
                cudaFree(d_input);
                cudaFree(d_output);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA FFT forward_1d_real failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation using Cooley-Tukey FFT algorithm
        return cpu_forward_1d_real(padded_input, normalize);
    }
    
    std::vector<std::complex<float>> forward_1d_complex(
        const std::vector<std::complex<float>>& input,
        bool normalize) {
        
        // Pad input to power of 2 for efficiency
        auto padded_input = pad_to_power_of_two(input);
        int n = static_cast<int>(padded_input.size());
        
        std::vector<std::complex<float>> output(n);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                cufftComplex* d_input = nullptr;
                cufftComplex* d_output = nullptr;
                
                cudaMalloc(&d_input, n * sizeof(cufftComplex));
                cudaMalloc(&d_output, n * sizeof(cufftComplex));
                
                // Copy input to device
                cudaMemcpy(d_input, padded_input.data(), n * sizeof(cufftComplex), cudaMemcpyHostToDevice);
                
                // Create plan
                cufftHandle plan;
                cufftPlan1d(&plan, n, CUFFT_C2C, 1);
                
                // Execute plan
                cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
                
                // Copy output to host
                std::vector<cufftComplex> temp(n);
                cudaMemcpy(temp.data(), d_output, n * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                
                // Convert to std::complex
                for (int i = 0; i < n; ++i) {
                    output[i] = std::complex<float>(temp[i].x, temp[i].y);
                    if (normalize) {
                        output[i] /= std::sqrt(static_cast<float>(n));
                    }
                }
                
                // Cleanup
                cufftDestroy(plan);
                cudaFree(d_input);
                cudaFree(d_output);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA FFT forward_1d_complex failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation using Cooley-Tukey FFT algorithm
        return cpu_forward_1d_complex(padded_input, normalize);
    }
    
    std::vector<float> inverse_1d_real(
        const std::vector<std::complex<float>>& input,
        bool normalize) {
        
        int n = (input.size() - 1) * 2;  // For real FFT, input size is n/2+1
        
        std::vector<float> output(n);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                cufftComplex* d_input = nullptr;
                float* d_output = nullptr;
                
                cudaMalloc(&d_input, input.size() * sizeof(cufftComplex));
                cudaMalloc(&d_output, n * sizeof(float));
                
                // Copy input to device
                std::vector<cufftComplex> temp(input.size());
                for (size_t i = 0; i < input.size(); ++i) {
                    temp[i].x = input[i].real();
                    temp[i].y = input[i].imag();
                }
                cudaMemcpy(d_input, temp.data(), input.size() * sizeof(cufftComplex), cudaMemcpyHostToDevice);
                
                // Create plan
                cufftHandle plan;
                cufftPlan1d(&plan, n, CUFFT_C2R, 1);
                
                // Execute plan
                cufftExecC2R(plan, d_input, d_output);
                
                // Copy output to host
                cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Normalize
                if (normalize) {
                    float norm_factor = 1.0f / n;
                    for (int i = 0; i < n; ++i) {
                        output[i] *= norm_factor;
                    }
                }
                
                // Cleanup
                cufftDestroy(plan);
                cudaFree(d_input);
                cudaFree(d_output);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA FFT inverse_1d_real failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation
        return cpu_inverse_1d_real(input, normalize);
    }
    
    std::vector<std::complex<float>> inverse_1d_complex(
        const std::vector<std::complex<float>>& input,
        bool normalize) {
        
        // Pad input to power of 2 for efficiency
        auto padded_input = pad_to_power_of_two(input);
        int n = static_cast<int>(padded_input.size());
        
        std::vector<std::complex<float>> output(n);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                cufftComplex* d_input = nullptr;
                cufftComplex* d_output = nullptr;
                
                cudaMalloc(&d_input, n * sizeof(cufftComplex));
                cudaMalloc(&d_output, n * sizeof(cufftComplex));
                
                // Copy input to device
                std::vector<cufftComplex> temp(n);
                for (int i = 0; i < n; ++i) {
                    temp[i].x = padded_input[i].real();
                    temp[i].y = padded_input[i].imag();
                }
                cudaMemcpy(d_input, temp.data(), n * sizeof(cufftComplex), cudaMemcpyHostToDevice);
                
                // Create plan
                cufftHandle plan;
                cufftPlan1d(&plan, n, CUFFT_C2C, 1);
                
                // Execute plan
                cufftExecC2C(plan, d_input, d_output, CUFFT_INVERSE);
                
                // Copy output to host
                cudaMemcpy(temp.data(), d_output, n * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                
                // Convert to std::complex
                for (int i = 0; i < n; ++i) {
                    output[i] = std::complex<float>(temp[i].x, temp[i].y);
                    if (normalize) {
                        output[i] /= static_cast<float>(n);
                    }
                }
                
                // Cleanup
                cufftDestroy(plan);
                cudaFree(d_input);
                cudaFree(d_output);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA FFT inverse_1d_complex failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation using Cooley-Tukey FFT algorithm
        return cpu_inverse_1d_complex(padded_input, normalize);
    }
    
    std::vector<std::complex<float>> forward_2d_real(
        const std::vector<float>& input,
        int rows,
        int cols,
        bool normalize) {
        
        // Check dimensions
        if (static_cast<int>(input.size()) != rows * cols) {
            throw std::invalid_argument("Input size does not match rows*cols");
        }
        
        // Output size for real-to-complex 2D FFT
        std::vector<std::complex<float>> output(rows * (cols / 2 + 1));
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                float* d_input = nullptr;
                cufftComplex* d_output = nullptr;
                
                cudaMalloc(&d_input, rows * cols * sizeof(float));
                cudaMalloc(&d_output, rows * (cols / 2 + 1) * sizeof(cufftComplex));
                
                // Copy input to device
                cudaMemcpy(d_input, input.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
                
                // Create plan
                cufftHandle plan;
                int dims[2] = {rows, cols};
                cufftPlanMany(&plan, 2, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_R2C, 1);
                
                // Execute plan
                cufftExecR2C(plan, d_input, d_output);
                
                // Copy output to host
                std::vector<cufftComplex> temp(rows * (cols / 2 + 1));
                cudaMemcpy(temp.data(), d_output, rows * (cols / 2 + 1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                
                // Convert to std::complex
                for (int i = 0; i < rows * (cols / 2 + 1); ++i) {
                    output[i] = std::complex<float>(temp[i].x, temp[i].y);
                    if (normalize) {
                        output[i] /= std::sqrt(static_cast<float>(rows * cols));
                    }
                }
                
                // Cleanup
                cufftDestroy(plan);
                cudaFree(d_input);
                cudaFree(d_output);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA FFT forward_2d_real failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation
        return cpu_forward_2d_real(input, rows, cols, normalize);
    }
    
    std::vector<std::complex<float>> forward_2d_complex(
        const std::vector<std::complex<float>>& input,
        int rows,
        int cols,
        bool normalize) {
        
        // Check dimensions
        if (static_cast<int>(input.size()) != rows * cols) {
            throw std::invalid_argument("Input size does not match rows*cols");
        }
        
        std::vector<std::complex<float>> output(rows * cols);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                cufftComplex* d_input = nullptr;
                cufftComplex* d_output = nullptr;
                
                cudaMalloc(&d_input, rows * cols * sizeof(cufftComplex));
                cudaMalloc(&d_output, rows * cols * sizeof(cufftComplex));
                
                // Copy input to device
                std::vector<cufftComplex> temp(rows * cols);
                for (int i = 0; i < rows * cols; ++i) {
                    temp[i].x = input[i].real();
                    temp[i].y = input[i].imag();
                }
                cudaMemcpy(d_input, temp.data(), rows * cols * sizeof(cufftComplex), cudaMemcpyHostToDevice);
                
                // Create plan
                cufftHandle plan;
                int dims[2] = {rows, cols};
                cufftPlanMany(&plan, 2, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 1);
                
                // Execute plan
                cufftExecC2C(plan, d_input, d_output, CUFFT_FORWARD);
                
                // Copy output to host
                cudaMemcpy(temp.data(), d_output, rows * cols * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                
                // Convert to std::complex
                for (int i = 0; i < rows * cols; ++i) {
                    output[i] = std::complex<float>(temp[i].x, temp[i].y);
                    if (normalize) {
                        output[i] /= std::sqrt(static_cast<float>(rows * cols));
                    }
                }
                
                // Cleanup
                cufftDestroy(plan);
                cudaFree(d_input);
                cudaFree(d_output);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA FFT forward_2d_complex failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation
        return cpu_forward_2d_complex(input, rows, cols, normalize);
    }
    
    std::vector<float> inverse_2d_real(
        const std::vector<std::complex<float>>& input,
        int rows,
        int cols,
        bool normalize) {
        
        // Check dimensions for real-to-complex FFT
        int output_cols = (cols - 1) * 2;
        if (static_cast<int>(input.size()) != rows * cols) {
            throw std::invalid_argument("Input size does not match rows*cols");
        }
        
        std::vector<float> output(rows * output_cols);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                cufftComplex* d_input = nullptr;
                float* d_output = nullptr;
                
                cudaMalloc(&d_input, rows * cols * sizeof(cufftComplex));
                cudaMalloc(&d_output, rows * output_cols * sizeof(float));
                
                // Copy input to device
                std::vector<cufftComplex> temp(rows * cols);
                for (int i = 0; i < rows * cols; ++i) {
                    temp[i].x = input[i].real();
                    temp[i].y = input[i].imag();
                }
                cudaMemcpy(d_input, temp.data(), rows * cols * sizeof(cufftComplex), cudaMemcpyHostToDevice);
                
                // Create plan
                cufftHandle plan;
                int dims[2] = {rows, output_cols};
                cufftPlanMany(&plan, 2, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2R, 1);
                
                // Execute plan
                cufftExecC2R(plan, d_input, d_output);
                
                // Copy output to host
                cudaMemcpy(output.data(), d_output, rows * output_cols * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Normalize
                if (normalize) {
                    float norm_factor = 1.0f / (rows * output_cols);
                    for (int i = 0; i < rows * output_cols; ++i) {
                        output[i] *= norm_factor;
                    }
                }
                
                // Cleanup
                cufftDestroy(plan);
                cudaFree(d_input);
                cudaFree(d_output);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA FFT inverse_2d_real failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation
        return cpu_inverse_2d_real(input, rows, cols, normalize);
    }
    
    std::vector<std::complex<float>> inverse_2d_complex(
        const std::vector<std::complex<float>>& input,
        int rows,
        int cols,
        bool normalize) {
        
        // Check dimensions
        if (static_cast<int>(input.size()) != rows * cols) {
            throw std::invalid_argument("Input size does not match rows*cols");
        }
        
        std::vector<std::complex<float>> output(rows * cols);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                cufftComplex* d_input = nullptr;
                cufftComplex* d_output = nullptr;
                
                cudaMalloc(&d_input, rows * cols * sizeof(cufftComplex));
                cudaMalloc(&d_output, rows * cols * sizeof(cufftComplex));
                
                // Copy input to device
                std::vector<cufftComplex> temp(rows * cols);
                for (int i = 0; i < rows * cols; ++i) {
                    temp[i].x = input[i].real();
                    temp[i].y = input[i].imag();
                }
                cudaMemcpy(d_input, temp.data(), rows * cols * sizeof(cufftComplex), cudaMemcpyHostToDevice);
                
                // Create plan
                cufftHandle plan;
                int dims[2] = {rows, cols};
                cufftPlanMany(&plan, 2, dims, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2C, 1);
                
                // Execute plan
                cufftExecC2C(plan, d_input, d_output, CUFFT_INVERSE);
                
                // Copy output to host
                cudaMemcpy(temp.data(), d_output, rows * cols * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
                
                // Convert to std::complex
                for (int i = 0; i < rows * cols; ++i) {
                    output[i] = std::complex<float>(temp[i].x, temp[i].y);
                    if (normalize) {
                        output[i] /= static_cast<float>(rows * cols);
                    }
                }
                
                // Cleanup
                cufftDestroy(plan);
                cudaFree(d_input);
                cudaFree(d_output);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA FFT inverse_2d_complex failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation
        return cpu_inverse_2d_complex(input, rows, cols, normalize);
    }
    
private:
    int device_id_;
    bool has_cuda_;
    
    // Initialize CUDA if available
    void initialize() {
#if defined(WITH_CUDA)
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
                std::cerr << "Available devices: " << device_count << std::endl;
                
                if (device_id_ >= device_count && device_count > 0) {
                    // Use first available device
                    device_id_ = 0;
                    has_cuda_ = true;
                    std::cout << "Using CUDA device 0 instead" << std::endl;
                    
                    // Set device
                    cudaSetDevice(device_id_);
                }
            }
        } else {
            std::cerr << "CUDA not available, using CPU implementation" << std::endl;
            has_cuda_ = false;
            device_id_ = -1;
        }
#else
        has_cuda_ = false;
        device_id_ = -1;
        std::cout << "Built without CUDA, using CPU implementation" << std::endl;
#endif
    }
    
    // Cleanup CUDA resources
    void cleanup() {
#if defined(WITH_CUDA)
        if (has_cuda_) {
            cudaDeviceReset();
        }
#endif
    }
    
    // CPU implementations
    std::vector<std::complex<float>> cpu_forward_1d_real(
        const std::vector<float>& input,
        bool normalize) {
        
        int n = static_cast<int>(input.size());
        
        // Convert real input to complex
        std::vector<std::complex<float>> complex_input(n);
        for (int i = 0; i < n; ++i) {
            complex_input[i] = std::complex<float>(input[i], 0.0f);
        }
        
        // Use complex FFT and take first n/2+1 elements for real input
        auto result = cpu_forward_1d_complex(complex_input, normalize);
        result.resize(n / 2 + 1);
        
        return result;
    }
    
    std::vector<std::complex<float>> cpu_forward_1d_complex(
        const std::vector<std::complex<float>>& input,
        bool normalize) {
        
        int n = static_cast<int>(input.size());
        
        // Check if n is a power of 2
        if ((n & (n - 1)) != 0) {
            throw std::invalid_argument("Input size must be a power of 2");
        }
        
        std::vector<std::complex<float>> output = input;
        
        // Bit reversal permutation
        int j = 0;
        for (int i = 0; i < n - 1; ++i) {
            if (i < j) {
                std::swap(output[i], output[j]);
            }
            int k = n / 2;
            while (k <= j) {
                j -= k;
                k /= 2;
            }
            j += k;
        }
        
        // Cooley-Tukey decimation-in-time algorithm
        for (int step = 1; step < n; step *= 2) {
            float theta = -PI / step;
            std::complex<float> wm(std::cos(theta), std::sin(theta));
            
            for (int m = 0; m < n; m += 2 * step) {
                std::complex<float> w(1.0f, 0.0f);
                for (int i = m; i < m + step; ++i) {
                    std::complex<float> t = w * output[i + step];
                    std::complex<float> u = output[i];
                    output[i] = u + t;
                    output[i + step] = u - t;
                    w *= wm;
                }
            }
        }
        
        // Normalize if requested
        if (normalize) {
            float norm_factor = 1.0f / std::sqrt(static_cast<float>(n));
            for (int i = 0; i < n; ++i) {
                output[i] *= norm_factor;
            }
        }
        
        return output;
    }
    
    std::vector<float> cpu_inverse_1d_real(
        const std::vector<std::complex<float>>& input,
        bool normalize) {
        
        int n = (input.size() - 1) * 2;  // For real FFT, input size is n/2+1
        
        // Reconstruct full complex spectrum for inverse FFT
        std::vector<std::complex<float>> full_spectrum(n);
        for (size_t i = 0; i < input.size(); ++i) {
            full_spectrum[i] = input[i];
        }
        
        // Complex conjugate symmetry for real signal
        for (int i = 1; i < n - static_cast<int>(input.size()) + 1; ++i) {
            full_spectrum[n - i] = std::conj(input[i]);
        }
        
        // Compute inverse complex FFT
        auto complex_result = cpu_inverse_1d_complex(full_spectrum, normalize);
        
        // Extract real part
        std::vector<float> result(n);
        for (int i = 0; i < n; ++i) {
            result[i] = complex_result[i].real();
        }
        
        return result;
    }
    
    std::vector<std::complex<float>> cpu_inverse_1d_complex(
        const std::vector<std::complex<float>>& input,
        bool normalize) {
        
        int n = static_cast<int>(input.size());
        
        // Check if n is a power of 2
        if ((n & (n - 1)) != 0) {
            throw std::invalid_argument("Input size must be a power of 2");
        }
        
        // Take complex conjugate of input
        std::vector<std::complex<float>> conj_input(n);
        for (int i = 0; i < n; ++i) {
            conj_input[i] = std::conj(input[i]);
        }
        
        // Use forward FFT on conjugated input
        auto result = cpu_forward_1d_complex(conj_input, false);
        
        // Take complex conjugate of result and normalize
        float norm_factor = normalize ? 1.0f / static_cast<float>(n) : 1.0f;
        for (int i = 0; i < n; ++i) {
            result[i] = std::conj(result[i]) * norm_factor;
        }
        
        return result;
    }
    
    std::vector<std::complex<float>> cpu_forward_2d_real(
        const std::vector<float>& input,
        int rows,
        int cols,
        bool normalize) {
        
        std::vector<std::complex<float>> output(rows * (cols / 2 + 1));
        
        // Process each row
        std::vector<std::vector<std::complex<float>>> row_results(rows);
        for (int i = 0; i < rows; ++i) {
            std::vector<float> row(cols);
            for (int j = 0; j < cols; ++j) {
                row[j] = input[i * cols + j];
            }
            row_results[i] = cpu_forward_1d_real(row, false);
        }
        
        // Process each column
        for (int j = 0; j < cols / 2 + 1; ++j) {
            std::vector<std::complex<float>> column(rows);
            for (int i = 0; i < rows; ++i) {
                column[i] = row_results[i][j];
            }
            auto col_result = cpu_forward_1d_complex(column, false);
            for (int i = 0; i < rows; ++i) {
                output[i * (cols / 2 + 1) + j] = col_result[i];
            }
        }
        
        // Normalize if requested
        if (normalize) {
            float norm_factor = 1.0f / std::sqrt(static_cast<float>(rows * cols));
            for (int i = 0; i < rows * (cols / 2 + 1); ++i) {
                output[i] *= norm_factor;
            }
        }
        
        return output;
    }
    
    std::vector<std::complex<float>> cpu_forward_2d_complex(
        const std::vector<std::complex<float>>& input,
        int rows,
        int cols,
        bool normalize) {
        
        std::vector<std::complex<float>> output(rows * cols);
        
        // Process each row
        std::vector<std::vector<std::complex<float>>> row_results(rows);
        for (int i = 0; i < rows; ++i) {
            std::vector<std::complex<float>> row(cols);
            for (int j = 0; j < cols; ++j) {
                row[j] = input[i * cols + j];
            }
            row_results[i] = cpu_forward_1d_complex(row, false);
        }
        
        // Process each column
        for (int j = 0; j < cols; ++j) {
            std::vector<std::complex<float>> column(rows);
            for (int i = 0; i < rows; ++i) {
                column[i] = row_results[i][j];
            }
            auto col_result = cpu_forward_1d_complex(column, false);
            for (int i = 0; i < rows; ++i) {
                output[i * cols + j] = col_result[i];
            }
        }
        
        // Normalize if requested
        if (normalize) {
            float norm_factor = 1.0f / std::sqrt(static_cast<float>(rows * cols));
            for (int i = 0; i < rows * cols; ++i) {
                output[i] *= norm_factor;
            }
        }
        
        return output;
    }
    
    std::vector<float> cpu_inverse_2d_real(
        const std::vector<std::complex<float>>& input,
        int rows,
        int cols,
        bool normalize) {
        
        int output_cols = (cols - 1) * 2;
        std::vector<float> output(rows * output_cols);
        
        // Process each column
        std::vector<std::vector<std::complex<float>>> col_results(output_cols);
        for (int j = 0; j < cols; ++j) {
            std::vector<std::complex<float>> column(rows);
            for (int i = 0; i < rows; ++i) {
                column[i] = input[i * cols + j];
            }
            auto col_result = cpu_inverse_1d_complex(column, false);
            col_results[j] = col_result;
        }
        
        // Reconstruct remaining columns due to complex conjugate symmetry
        for (int j = 1; j < cols - 1; ++j) {
            std::vector<std::complex<float>> column(rows);
            for (int i = 0; i < rows; ++i) {
                column[i] = std::conj(input[i * cols + (cols - j)]);
            }
            auto col_result = cpu_inverse_1d_complex(column, false);
            col_results[output_cols - j] = col_result;
        }
        
        // Process each row
        for (int i = 0; i < rows; ++i) {
            std::vector<std::complex<float>> row(output_cols);
            for (int j = 0; j < output_cols; ++j) {
                row[j] = col_results[j][i];
            }
            auto row_result = cpu_inverse_1d_real(row, false);
            for (int j = 0; j < output_cols; ++j) {
                output[i * output_cols + j] = row_result[j];
            }
        }
        
        // Normalize if requested
        if (normalize) {
            float norm_factor = 1.0f / static_cast<float>(rows * output_cols);
            for (int i = 0; i < rows * output_cols; ++i) {
                output[i] *= norm_factor;
            }
        }
        
        return output;
    }
    
    std::vector<std::complex<float>> cpu_inverse_2d_complex(
        const std::vector<std::complex<float>>& input,
        int rows,
        int cols,
        bool normalize) {
        
        std::vector<std::complex<float>> output(rows * cols);
        
        // Process each column
        std::vector<std::vector<std::complex<float>>> col_results(cols);
        for (int j = 0; j < cols; ++j) {
            std::vector<std::complex<float>> column(rows);
            for (int i = 0; i < rows; ++i) {
                column[i] = input[i * cols + j];
            }
            auto col_result = cpu_inverse_1d_complex(column, false);
            col_results[j] = col_result;
        }
        
        // Process each row
        for (int i = 0; i < rows; ++i) {
            std::vector<std::complex<float>> row(cols);
            for (int j = 0; j < cols; ++j) {
                row[j] = col_results[j][i];
            }
            auto row_result = cpu_inverse_1d_complex(row, false);
            for (int j = 0; j < cols; ++j) {
                output[i * cols + j] = row_result[j];
            }
        }
        
        // Normalize if requested
        if (normalize) {
            float norm_factor = 1.0f / static_cast<float>(rows * cols);
            for (int i = 0; i < rows * cols; ++i) {
                output[i] *= norm_factor;
            }
        }
        
        return output;
    }
};

//------------------------------------------------------------------------------
// Spectral Analyzer Implementation
//------------------------------------------------------------------------------

class SpectralAnalyzerImpl {
public:
    explicit SpectralAnalyzerImpl(int device_id)
        : fft_(device_id), device_id_(device_id), has_cuda_(fft_.has_cuda()) {}
    
    bool has_cuda() const {
        return has_cuda_;
    }
    
    int get_device_id() const {
        return device_id_;
    }
    
    PSDResult compute_psd(
        const std::vector<float>& signal,
        const SpectralParams& params) {
        
        // Apply defaults if needed
        SpectralParams p = params;
        if (p.nfft == 0) {
            p.nfft = next_power_of_two(signal.size());
        }
        if (p.overlap == 0) {
            p.overlap = p.nfft / 2;  // 50% overlap by default
        }
        
        // Detrend the signal
        auto detrended = detrend(signal, p.detrend);
        
        // Segment the signal
        auto segments = segment_signal(detrended, p.nfft, p.overlap);
        
        // Generate window function
        auto window = generate_window(p.window_type, p.nfft, p.window_param);
        
        // Calculate window scaling factors
        float scale = 0.0f;
        for (float w : window) {
            scale += w * w;
        }
        scale /= p.nfft;
        
        // Initialize output
        std::vector<float> psd(p.nfft / 2 + 1, 0.0f);
        
        // Compute periodogram for each segment and average
        for (const auto& segment : segments) {
            // Apply window
            auto windowed = apply_window(segment, window);
            
            // Compute FFT
            auto fft_result = fft_.forward_1d_real(windowed, false);
            
            // Compute periodogram
            for (int i = 0; i < p.nfft / 2 + 1; ++i) {
                float magnitude_squared = std::norm(fft_result[i]);
                psd[i] += magnitude_squared;
            }
        }
        
        // Average and scale
        float scaling_factor = 1.0f / (segments.size() * scale);
        if (p.scaling) {
            scaling_factor /= p.sample_rate;
        }
        
        for (int i = 0; i < p.nfft / 2 + 1; ++i) {
            psd[i] *= scaling_factor;
        }
        
        // DC and Nyquist components are scaled differently due to not being double-counted
        if (p.return_onesided && p.nfft % 2 == 0) {
            psd[0] *= 0.5f;
            psd[p.nfft / 2] *= 0.5f;
        }
        
        // Generate frequency array
        std::vector<float> frequencies(p.nfft / 2 + 1);
        for (int i = 0; i < p.nfft / 2 + 1; ++i) {
            frequencies[i] = static_cast<float>(i) * p.sample_rate / p.nfft;
        }
        
        // Return result
        PSDResult result;
        result.frequencies = std::move(frequencies);
        result.psd = std::move(psd);
        
        return result;
    }
    
    CSDResult compute_csd(
        const std::vector<float>& signal1,
        const std::vector<float>& signal2,
        const SpectralParams& params) {
        
        // Check that signals have the same length
        if (signal1.size() != signal2.size()) {
            throw std::invalid_argument("Signals must have the same length");
        }
        
        // Apply defaults if needed
        SpectralParams p = params;
        if (p.nfft == 0) {
            p.nfft = next_power_of_two(signal1.size());
        }
        if (p.overlap == 0) {
            p.overlap = p.nfft / 2;  // 50% overlap by default
        }
        
        // Detrend the signals
        auto detrended1 = detrend(signal1, p.detrend);
        auto detrended2 = detrend(signal2, p.detrend);
        
        // Segment the signals
        auto segments1 = segment_signal(detrended1, p.nfft, p.overlap);
        auto segments2 = segment_signal(detrended2, p.nfft, p.overlap);
        
        // Generate window function
        auto window = generate_window(p.window_type, p.nfft, p.window_param);
        
        // Calculate window scaling factors
        float scale = 0.0f;
        for (float w : window) {
            scale += w * w;
        }
        scale /= p.nfft;
        
        // Initialize outputs
        std::vector<std::complex<float>> csd(p.nfft / 2 + 1, std::complex<float>(0.0f, 0.0f));
        std::vector<float> psd1(p.nfft / 2 + 1, 0.0f);
        std::vector<float> psd2(p.nfft / 2 + 1, 0.0f);
        
        // Compute cross-spectral density and auto-spectral densities
        for (size_t s = 0; s < segments1.size(); ++s) {
            // Apply window
            auto windowed1 = apply_window(segments1[s], window);
            auto windowed2 = apply_window(segments2[s], window);
            
            // Compute FFTs
            auto fft1 = fft_.forward_1d_real(windowed1, false);
            auto fft2 = fft_.forward_1d_real(windowed2, false);
            
            // Compute cross-spectral density and auto-spectral densities
            for (int i = 0; i < p.nfft / 2 + 1; ++i) {
                csd[i] += fft1[i] * std::conj(fft2[i]);
                psd1[i] += std::norm(fft1[i]);
                psd2[i] += std::norm(fft2[i]);
            }
        }
        
        // Average and scale
        float scaling_factor = 1.0f / (segments1.size() * scale);
        if (p.scaling) {
            scaling_factor /= p.sample_rate;
        }
        
        for (int i = 0; i < p.nfft / 2 + 1; ++i) {
            csd[i] *= scaling_factor;
            psd1[i] *= scaling_factor;
            psd2[i] *= scaling_factor;
        }
        
        // DC and Nyquist components are scaled differently due to not being double-counted
        if (p.return_onesided && p.nfft % 2 == 0) {
            csd[0] *= 0.5f;
            csd[p.nfft / 2] *= 0.5f;
            psd1[0] *= 0.5f;
            psd1[p.nfft / 2] *= 0.5f;
            psd2[0] *= 0.5f;
            psd2[p.nfft / 2] *= 0.5f;
        }
        
        // Generate frequency array
        std::vector<float> frequencies(p.nfft / 2 + 1);
        for (int i = 0; i < p.nfft / 2 + 1; ++i) {
            frequencies[i] = static_cast<float>(i) * p.sample_rate / p.nfft;
        }
        
        // Compute coherence and phase
        std::vector<float> coherence(p.nfft / 2 + 1);
        std::vector<float> phase(p.nfft / 2 + 1);
        
        for (int i = 0; i < p.nfft / 2 + 1; ++i) {
            float denominator = psd1[i] * psd2[i];
            coherence[i] = denominator > 0.0f ? std::norm(csd[i]) / denominator : 0.0f;
            phase[i] = std::arg(csd[i]);
        }
        
        // Return result
        CSDResult result;
        result.frequencies = std::move(frequencies);
        result.csd = std::move(csd);
        result.coherence = std::move(coherence);
        result.phase = std::move(phase);
        
        return result;
    }
    
    SpectrogramResult compute_spectrogram(
        const std::vector<float>& signal,
        const SpectralParams& params) {
        
        // Apply defaults if needed
        SpectralParams p = params;
        if (p.nfft == 0) {
            p.nfft = std::min(next_power_of_two(signal.size() / 8), 1024);
        }
        if (p.overlap == 0) {
            p.overlap = p.nfft / 2;  // 50% overlap by default
        }
        
        // Detrend the signal
        auto detrended = detrend(signal, p.detrend);
        
        // Segment the signal
        auto segments = segment_signal(detrended, p.nfft, p.overlap);
        
        // Generate window function
        auto window = generate_window(p.window_type, p.nfft, p.window_param);
        
        // Calculate window scaling factors
        float scale = 0.0f;
        for (float w : window) {
            scale += w * w;
        }
        scale /= p.nfft;
        
        // Initialize output
        int num_segments = static_cast<int>(segments.size());
        int num_frequencies = p.nfft / 2 + 1;
        std::vector<std::vector<float>> spectrogram(num_segments, std::vector<float>(num_frequencies, 0.0f));
        
        // Compute periodogram for each segment
        for (int s = 0; s < num_segments; ++s) {
            // Apply window
            auto windowed = apply_window(segments[s], window);
            
            // Compute FFT
            auto fft_result = fft_.forward_1d_real(windowed, false);
            
            // Compute power spectrum
            for (int i = 0; i < num_frequencies; ++i) {
                float magnitude_squared = std::norm(fft_result[i]);
                spectrogram[s][i] = magnitude_squared / scale;
                
                if (p.scaling) {
                    spectrogram[s][i] /= p.sample_rate;
                }
            }
            
            // DC and Nyquist components are scaled differently due to not being double-counted
            if (p.return_onesided && p.nfft % 2 == 0) {
                spectrogram[s][0] *= 0.5f;
                spectrogram[s][num_frequencies - 1] *= 0.5f;
            }
        }
        
        // Generate frequency array
        std::vector<float> frequencies(num_frequencies);
        for (int i = 0; i < num_frequencies; ++i) {
            frequencies[i] = static_cast<float>(i) * p.sample_rate / p.nfft;
        }
        
        // Generate time array
        std::vector<float> times(num_segments);
        float step = static_cast<float>(p.nfft - p.overlap) / p.sample_rate;
        float offset = static_cast<float>(p.nfft) / (2.0f * p.sample_rate);  // Center of window
        
        for (int i = 0; i < num_segments; ++i) {
            times[i] = i * step + offset;
        }
        
        // Return result
        SpectrogramResult result;
        result.times = std::move(times);
        result.frequencies = std::move(frequencies);
        result.spectrogram = std::move(spectrogram);
        
        return result;
    }
    
    PSDResult compute_coherence(
        const std::vector<float>& signal1,
        const std::vector<float>& signal2,
        const SpectralParams& params) {
        
        // Compute CSD
        auto csd_result = compute_csd(signal1, signal2, params);
        
        // Extract and return coherence
        PSDResult result;
        result.frequencies = csd_result.frequencies;
        result.coherence = csd_result.coherence;
        
        return result;
    }
    
    PSDResult compute_periodogram(
        const std::vector<float>& signal,
        const SpectralParams& params) {
        
        // Apply defaults if needed
        SpectralParams p = params;
        if (p.nfft == 0) {
            p.nfft = next_power_of_two(signal.size());
        }
        
        // Detrend the signal
        auto detrended = detrend(signal, p.detrend);
        
        // Pad or truncate to nfft
        std::vector<float> padded(p.nfft, 0.0f);
        for (int i = 0; i < std::min(static_cast<int>(detrended.size()), p.nfft); ++i) {
            padded[i] = detrended[i];
        }
        
        // Generate window function
        auto window = generate_window(p.window_type, p.nfft, p.window_param);
        
        // Calculate window scaling factors
        float scale = 0.0f;
        for (float w : window) {
            scale += w * w;
        }
        scale /= p.nfft;
        
        // Apply window
        auto windowed = apply_window(padded, window);
        
        // Compute FFT
        auto fft_result = fft_.forward_1d_real(windowed, false);
        
        // Compute periodogram
        std::vector<float> periodogram(p.nfft / 2 + 1, 0.0f);
        for (int i = 0; i < p.nfft / 2 + 1; ++i) {
            float magnitude_squared = std::norm(fft_result[i]);
            periodogram[i] = magnitude_squared / scale;
            
            if (p.scaling) {
                periodogram[i] /= p.sample_rate;
            }
        }
        
        // DC and Nyquist components are scaled differently due to not being double-counted
        if (p.return_onesided && p.nfft % 2 == 0) {
            periodogram[0] *= 0.5f;
            periodogram[p.nfft / 2] *= 0.5f;
        }
        
        // Generate frequency array
        std::vector<float> frequencies(p.nfft / 2 + 1);
        for (int i = 0; i < p.nfft / 2 + 1; ++i) {
            frequencies[i] = static_cast<float>(i) * p.sample_rate / p.nfft;
        }
        
        // Return result
        PSDResult result;
        result.frequencies = std::move(frequencies);
        result.psd = std::move(periodogram);
        
        return result;
    }
    
    std::vector<std::pair<float, float>> detect_peaks(
        const std::vector<float>& spectrum,
        const std::vector<float>& frequencies,
        float threshold,
        int min_distance) {
        
        if (spectrum.size() != frequencies.size()) {
            throw std::invalid_argument("Spectrum and frequencies must have the same size");
        }
        
        // Find local maxima
        std::vector<std::pair<float, float>> peaks;
        
        for (size_t i = 1; i < spectrum.size() - 1; ++i) {
            if (spectrum[i] > spectrum[i - 1] && spectrum[i] > spectrum[i + 1]) {
                float peak_value = spectrum[i];
                float peak_freq = frequencies[i];
                
                // Refine peak position using quadratic interpolation
                float y1 = spectrum[i - 1];
                float y2 = spectrum[i];
                float y3 = spectrum[i + 1];
                float d = 0.5f * (y3 - y1) / (2.0f * y2 - y1 - y3);
                
                // Limit interpolation to avoid extreme values
                if (std::abs(d) < 0.5f) {
                    // Compute interpolated frequency
                    float interp_freq = frequencies[i];
                    if (i + 1 < frequencies.size()) {
                        float freq_step = frequencies[i + 1] - frequencies[i];
                        interp_freq += d * freq_step;
                    }
                    
                    // Compute interpolated peak value
                    float interp_value = y2 - 0.25f * (y1 - y3) * d;
                    
                    if (interp_value > peak_value) {
                        peak_value = interp_value;
                        peak_freq = interp_freq;
                    }
                }
                
                // Check threshold
                float max_value = *std::max_element(spectrum.begin(), spectrum.end());
                if (peak_value >= threshold * max_value) {
                    // Check min distance
                    bool add_peak = true;
                    for (const auto& existing_peak : peaks) {
                        int existing_idx = std::distance(
                            frequencies.begin(),
                            std::lower_bound(frequencies.begin(), frequencies.end(), existing_peak.first));
                        
                        if (std::abs(static_cast<int>(i) - existing_idx) < min_distance) {
                            if (peak_value <= existing_peak.second) {
                                add_peak = false;
                                break;
                            }
                        }
                    }
                    
                    if (add_peak) {
                        peaks.emplace_back(peak_freq, peak_value);
                    }
                }
            }
        }
        
        // Sort by frequency
        std::sort(peaks.begin(), peaks.end(),
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        return peaks;
    }
    
    std::vector<float> compute_harmonic_distortion(
        const std::vector<float>& signal,
        float fundamental_freq,
        int num_harmonics,
        const SpectralParams& params) {
        
        // Compute power spectrum
        auto psd_result = compute_psd(signal, params);
        
        // Find fundamental frequency peak
        float freq_resolution = params.sample_rate / params.nfft;
        int fund_idx = static_cast<int>(std::round(fundamental_freq / freq_resolution));
        
        if (fund_idx >= static_cast<int>(psd_result.frequencies.size()) || fund_idx <= 0) {
            throw std::invalid_argument("Fundamental frequency out of range");
        }
        
        // Find exact peak around the expected fundamental frequency
        int window_size = 5;  // Search window size
        int freq_tolerance = std::max(1, static_cast<int>(0.05f * fund_idx));  // 5% tolerance
        
        // Find peak near expected fundamental
        int peak_idx = fund_idx;
        float peak_power = psd_result.psd[fund_idx];
        
        for (int i = std::max(1, fund_idx - freq_tolerance); 
             i <= std::min(static_cast<int>(psd_result.psd.size()) - 1, fund_idx + freq_tolerance); ++i) {
            if (psd_result.psd[i] > peak_power) {
                peak_idx = i;
                peak_power = psd_result.psd[i];
            }
        }
        
        fund_idx = peak_idx;
        float fund_power = peak_power;
        
        // Measure harmonic powers
        std::vector<float> harmonic_distortion(num_harmonics, 0.0f);
        
        for (int h = 1; h <= num_harmonics; ++h) {
            int harmonic_idx = (h + 1) * fund_idx;  // +1 because h=1 is the 2nd harmonic
            
            if (harmonic_idx < static_cast<int>(psd_result.frequencies.size())) {
                // Find peak near expected harmonic
                int peak_h_idx = harmonic_idx;
                float peak_h_power = psd_result.psd[harmonic_idx];
                
                for (int i = std::max(1, harmonic_idx - freq_tolerance); 
                     i <= std::min(static_cast<int>(psd_result.psd.size()) - 1, harmonic_idx + freq_tolerance); ++i) {
                    if (psd_result.psd[i] > peak_h_power) {
                        peak_h_idx = i;
                        peak_h_power = psd_result.psd[i];
                    }
                }
                
                float harmonic_power = peak_h_power;
                
                // Calculate ratio with fundamental
                harmonic_distortion[h - 1] = std::sqrt(harmonic_power / fund_power);
            }
        }
        
        return harmonic_distortion;
    }
    
private:
    FFT fft_;
    int device_id_;
    bool has_cuda_;
    
    // Helper function to find next power of two
    int next_power_of_two(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }
};

//------------------------------------------------------------------------------
// FFT class implementation
//------------------------------------------------------------------------------

FFT::FFT(int device_id) : impl_(std::make_unique<FFTImpl>(device_id)) {}

FFT::~FFT() = default;

FFT::FFT(FFT&&) noexcept = default;
FFT& FFT::operator=(FFT&&) noexcept = default;

bool FFT::has_cuda() const {
    return impl_->has_cuda();
}

int FFT::get_device_id() const {
    return impl_->get_device_id();
}

std::vector<std::complex<float>> FFT::forward_1d_real(
    const std::vector<float>& input,
    bool normalize) {
    return impl_->forward_1d_real(input, normalize);
}

std::vector<std::complex<float>> FFT::forward_1d_complex(
    const std::vector<std::complex<float>>& input,
    bool normalize) {
    return impl_->forward_1d_complex(input, normalize);
}

std::vector<float> FFT::inverse_1d_real(
    const std::vector<std::complex<float>>& input,
    bool normalize) {
    return impl_->inverse_1d_real(input, normalize);
}

std::vector<std::complex<float>> FFT::inverse_1d_complex(
    const std::vector<std::complex<float>>& input,
    bool normalize) {
    return impl_->inverse_1d_complex(input, normalize);
}

std::vector<std::complex<float>> FFT::forward_2d_real(
    const std::vector<float>& input,
    int rows,
    int cols,
    bool normalize) {
    return impl_->forward_2d_real(input, rows, cols, normalize);
}

std::vector<std::complex<float>> FFT::forward_2d_complex(
    const std::vector<std::complex<float>>& input,
    int rows,
    int cols,
    bool normalize) {
    return impl_->forward_2d_complex(input, rows, cols, normalize);
}

std::vector<float> FFT::inverse_2d_real(
    const std::vector<std::complex<float>>& input,
    int rows,
    int cols,
    bool normalize) {
    return impl_->inverse_2d_real(input, rows, cols, normalize);
}

std::vector<std::complex<float>> FFT::inverse_2d_complex(
    const std::vector<std::complex<float>>& input,
    int rows,
    int cols,
    bool normalize) {
    return impl_->inverse_2d_complex(input, rows, cols, normalize);
}

//------------------------------------------------------------------------------
// SpectralAnalyzer class implementation
//------------------------------------------------------------------------------

SpectralAnalyzer::SpectralAnalyzer(int device_id)
    : impl_(std::make_unique<SpectralAnalyzerImpl>(device_id)) {}

SpectralAnalyzer::~SpectralAnalyzer() = default;

SpectralAnalyzer::SpectralAnalyzer(SpectralAnalyzer&&) noexcept = default;
SpectralAnalyzer& SpectralAnalyzer::operator=(SpectralAnalyzer&&) noexcept = default;

bool SpectralAnalyzer::has_cuda() const {
    return impl_->has_cuda();
}

int SpectralAnalyzer::get_device_id() const {
    return impl_->get_device_id();
}

PSDResult SpectralAnalyzer::compute_psd(
    const std::vector<float>& signal,
    const SpectralParams& params) {
    return impl_->compute_psd(signal, params);
}

CSDResult SpectralAnalyzer::compute_csd(
    const std::vector<float>& signal1,
    const std::vector<float>& signal2,
    const SpectralParams& params) {
    return impl_->compute_csd(signal1, signal2, params);
}

SpectrogramResult SpectralAnalyzer::compute_spectrogram(
    const std::vector<float>& signal,
    const SpectralParams& params) {
    return impl_->compute_spectrogram(signal, params);
}

PSDResult SpectralAnalyzer::compute_coherence(
    const std::vector<float>& signal1,
    const std::vector<float>& signal2,
    const SpectralParams& params) {
    return impl_->compute_coherence(signal1, signal2, params);
}

PSDResult SpectralAnalyzer::compute_periodogram(
    const std::vector<float>& signal,
    const SpectralParams& params) {
    return impl_->compute_periodogram(signal, params);
}

std::vector<std::pair<float, float>> SpectralAnalyzer::detect_peaks(
    const std::vector<float>& spectrum,
    const std::vector<float>& frequencies,
    float threshold,
    int min_distance) {
    return impl_->detect_peaks(spectrum, frequencies, threshold, min_distance);
}

std::vector<float> SpectralAnalyzer::compute_harmonic_distortion(
    const std::vector<float>& signal,
    float fundamental_freq,
    int num_harmonics,
    const SpectralParams& params) {
    return impl_->compute_harmonic_distortion(signal, fundamental_freq, num_harmonics, params);
}

} // namespace signal_processing