/**
 * @file digital_filtering.cpp
 * @brief Implementation of digital filtering operations
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include "signal_processing/digital_filtering.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <deque>
#include <limits>
#include <complex>

// Check for CUDA availability
#if defined(WITH_CUDA)
#include <cuda_runtime.h>
#include <cufft.h>
// Include CUDA kernel definitions
#include "../src/kernels/filter_kernels.cu"
#endif

namespace signal_processing {

// Constants
constexpr float PI = 3.14159265358979323846f;

namespace {

// Helper functions
inline float sinc(float x) {
    if (std::abs(x) < 1e-10f) {
        return 1.0f;
    }
    return std::sin(PI * x) / (PI * x);
}

// FIR filter design functions
std::vector<float> design_fir_window(
    const FIRFilterParams& params,
    float sample_rate) {
    
    // Validate parameters
    if (params.num_taps <= 0) {
        throw std::invalid_argument("Number of taps must be positive");
    }
    
    if (params.cutoff_freqs.empty()) {
        throw std::invalid_argument("Cutoff frequencies must be specified");
    }
    
    // Normalize cutoff frequencies to Nyquist frequency
    std::vector<float> norm_freqs;
    for (float freq : params.cutoff_freqs) {
        norm_freqs.push_back(freq / (sample_rate / 2.0f));
        if (norm_freqs.back() <= 0.0f || norm_freqs.back() >= 1.0f) {
            throw std::invalid_argument("Normalized cutoff frequencies must be in (0, 1)");
        }
    }
    
    // Check filter type requirements
    if ((params.filter_type == FilterType::LOWPASS || params.filter_type == FilterType::HIGHPASS) &&
        norm_freqs.size() != 1) {
        throw std::invalid_argument("Low-pass and high-pass filters require exactly one cutoff frequency");
    }
    
    if ((params.filter_type == FilterType::BANDPASS || params.filter_type == FilterType::BANDSTOP) &&
        norm_freqs.size() != 2) {
        throw std::invalid_argument("Band-pass and band-stop filters require exactly two cutoff frequencies");
    }
    
    if (params.filter_type == FilterType::BANDPASS || params.filter_type == FilterType::BANDSTOP) {
        if (norm_freqs[0] >= norm_freqs[1]) {
            throw std::invalid_argument("For band filters, the first cutoff frequency must be less than the second");
        }
    }
    
    // Generate ideal impulse response based on filter type
    std::vector<float> h_ideal(params.num_taps);
    int half_taps = params.num_taps / 2;
    
    for (int i = 0; i < params.num_taps; ++i) {
        int n = i - half_taps;
        
        if (n == 0) {
            // Handle center point separately
            if (params.filter_type == FilterType::LOWPASS) {
                h_ideal[i] = norm_freqs[0];
            } else if (params.filter_type == FilterType::HIGHPASS) {
                h_ideal[i] = 1.0f - norm_freqs[0];
            } else if (params.filter_type == FilterType::BANDPASS) {
                h_ideal[i] = norm_freqs[1] - norm_freqs[0];
            } else if (params.filter_type == FilterType::BANDSTOP) {
                h_ideal[i] = 1.0f - (norm_freqs[1] - norm_freqs[0]);
            }
        } else {
            // Non-center points
            if (params.filter_type == FilterType::LOWPASS) {
                h_ideal[i] = norm_freqs[0] * sinc(norm_freqs[0] * n);
            } else if (params.filter_type == FilterType::HIGHPASS) {
                h_ideal[i] = -norm_freqs[0] * sinc(norm_freqs[0] * n);
                // Add shifted delta function for highpass
                if (n == 0) {
                    h_ideal[i] += 1.0f;
                }
            } else if (params.filter_type == FilterType::BANDPASS) {
                h_ideal[i] = norm_freqs[1] * sinc(norm_freqs[1] * n) - 
                             norm_freqs[0] * sinc(norm_freqs[0] * n);
            } else if (params.filter_type == FilterType::BANDSTOP) {
                h_ideal[i] = norm_freqs[0] * sinc(norm_freqs[0] * n) - 
                             norm_freqs[1] * sinc(norm_freqs[1] * n);
                // Add shifted delta function for bandstop
                if (n == 0) {
                    h_ideal[i] += 1.0f;
                }
            }
        }
    }
    
    // Apply window function
    std::vector<float> window(params.num_taps);
    
    switch (params.window_type) {
        case WindowType::RECTANGULAR:
            std::fill(window.begin(), window.end(), 1.0f);
            break;
            
        case WindowType::TRIANGULAR:
            {
                for (int i = 0; i < params.num_taps; ++i) {
                    window[i] = 1.0f - std::abs(2.0f * (i - half_taps) / params.num_taps);
                }
            }
            break;
            
        case WindowType::HANN:
            {
                for (int i = 0; i < params.num_taps; ++i) {
                    window[i] = 0.5f * (1.0f - std::cos(2 * PI * i / (params.num_taps - 1)));
                }
            }
            break;
            
        case WindowType::HAMMING:
            {
                for (int i = 0; i < params.num_taps; ++i) {
                    window[i] = 0.54f - 0.46f * std::cos(2 * PI * i / (params.num_taps - 1));
                }
            }
            break;
            
        case WindowType::BLACKMAN:
            {
                for (int i = 0; i < params.num_taps; ++i) {
                    float x = 2 * PI * i / (params.num_taps - 1);
                    window[i] = 0.42f - 0.5f * std::cos(x) + 0.08f * std::cos(2 * x);
                }
            }
            break;
            
        case WindowType::KAISER:
            {
                // Default beta parameter
                float beta = params.window_param > 0.0f ? params.window_param : 4.0f;
                
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
                
                for (int i = 0; i < params.num_taps; ++i) {
                    float x = 2.0f * i / (params.num_taps - 1) - 1.0f;
                    
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
    
    // Apply window to ideal impulse response
    std::vector<float> coeffs(params.num_taps);
    for (int i = 0; i < params.num_taps; ++i) {
        coeffs[i] = h_ideal[i] * window[i];
    }
    
    // Normalize the filter gain
    float total_gain = 0.0f;
    for (float coeff : coeffs) {
        total_gain += coeff;
    }
    
    // Apply desired gain normalization
    float target_gain = 0.0f;
    if (params.filter_type == FilterType::LOWPASS || params.filter_type == FilterType::BANDPASS) {
        target_gain = params.gains[0];  // Passband gain
    } else if (params.filter_type == FilterType::HIGHPASS || params.filter_type == FilterType::BANDSTOP) {
        // For highpass and bandstop, the total gain should equal passband gain
        target_gain = params.gains[0];
    }
    
    if (std::abs(total_gain) > 1e-10f) {
        float scale = target_gain / total_gain;
        for (int i = 0; i < params.num_taps; ++i) {
            coeffs[i] *= scale;
        }
    }
    
    return coeffs;
}

// IIR filter design functions
struct IIRCoefficients {
    std::vector<float> a;  // Denominator coefficients (a0 = 1.0 assumed)
    std::vector<float> b;  // Numerator coefficients
};

// Convert IIR filter to second-order sections
std::vector<std::array<float, 6>> convert_to_sos(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    
    // Simple placeholder implementation - in practice, would compute roots of polynomial
    // and group them into second-order sections
    
    int order = std::max(a.size(), b.size()) - 1;
    int num_sections = (order + 1) / 2;
    
    std::vector<std::array<float, 6>> sos(num_sections);
    
    // Set default SOS sections with just gain
    for (auto& section : sos) {
        section = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    }
    
    // TODO: Implement proper conversion to SOS form
    
    return sos;
}

// Design Butterworth filter
IIRCoefficients design_butterworth(
    const IIRFilterParams& params,
    float sample_rate) {
    
    // Validate parameters
    if (params.order <= 0) {
        throw std::invalid_argument("Filter order must be positive");
    }
    
    if (params.cutoff_freqs.empty()) {
        throw std::invalid_argument("Cutoff frequencies must be specified");
    }
    
    // Normalize cutoff frequencies to Nyquist frequency
    std::vector<float> norm_freqs;
    for (float freq : params.cutoff_freqs) {
        norm_freqs.push_back(freq / (sample_rate / 2.0f));
        if (norm_freqs.back() <= 0.0f || norm_freqs.back() >= 1.0f) {
            throw std::invalid_argument("Normalized cutoff frequencies must be in (0, 1)");
        }
    }
    
    // Check filter type requirements
    if ((params.filter_type == FilterType::LOWPASS || params.filter_type == FilterType::HIGHPASS) &&
        norm_freqs.size() != 1) {
        throw std::invalid_argument("Low-pass and high-pass filters require exactly one cutoff frequency");
    }
    
    if ((params.filter_type == FilterType::BANDPASS || params.filter_type == FilterType::BANDSTOP) &&
        norm_freqs.size() != 2) {
        throw std::invalid_argument("Band-pass and band-stop filters require exactly two cutoff frequencies");
    }
    
    if (params.filter_type == FilterType::BANDPASS || params.filter_type == FilterType::BANDSTOP) {
        if (norm_freqs[0] >= norm_freqs[1]) {
            throw std::invalid_argument("For band filters, the first cutoff frequency must be less than the second");
        }
    }
    
    // TODO: Implement actual Butterworth filter design
    // This is a placeholder implementation
    
    // Create bilinear transformation
    float warped_cutoff = 0.0f;
    if (params.filter_type == FilterType::LOWPASS || params.filter_type == FilterType::HIGHPASS) {
        warped_cutoff = std::tan(PI * norm_freqs[0] / 2.0f);
    } else {
        // For bandpass/bandstop, use geometric mean of cutoffs
        warped_cutoff = std::tan(PI * std::sqrt(norm_freqs[0] * norm_freqs[1]) / 2.0f);
    }
    
    IIRCoefficients coeffs;
    
    // First-order Butterworth as a simple example
    if (params.filter_type == FilterType::LOWPASS) {
        float b0 = warped_cutoff / (1.0f + warped_cutoff);
        float b1 = b0;
        float a1 = (warped_cutoff - 1.0f) / (warped_cutoff + 1.0f);
        
        coeffs.b = {b0, b1};
        coeffs.a = {1.0f, -a1};
    } else if (params.filter_type == FilterType::HIGHPASS) {
        float b0 = 1.0f / (1.0f + warped_cutoff);
        float b1 = -b0;
        float a1 = (warped_cutoff - 1.0f) / (warped_cutoff + 1.0f);
        
        coeffs.b = {b0, b1};
        coeffs.a = {1.0f, -a1};
    }
    
    // Note: This is highly simplified - a real implementation would handle:
    // - Higher order filters by chaining first and second-order sections
    // - Proper conversion to discrete-time using bilinear transform
    // - Bandpass and bandstop designs
    
    return coeffs;
}

// Design Chebyshev Type I filter
IIRCoefficients design_chebyshev1(
    const IIRFilterParams& params,
    float sample_rate) {
    
    // Validate parameters
    if (params.order <= 0) {
        throw std::invalid_argument("Filter order must be positive");
    }
    
    // TODO: Implement Chebyshev Type I filter design
    // This is a placeholder implementation
    
    IIRCoefficients coeffs;
    coeffs.a = {1.0f};
    coeffs.b = {1.0f};
    
    return coeffs;
}

// Design Chebyshev Type II filter
IIRCoefficients design_chebyshev2(
    const IIRFilterParams& params,
    float sample_rate) {
    
    // Validate parameters
    if (params.order <= 0) {
        throw std::invalid_argument("Filter order must be positive");
    }
    
    // TODO: Implement Chebyshev Type II filter design
    // This is a placeholder implementation
    
    IIRCoefficients coeffs;
    coeffs.a = {1.0f};
    coeffs.b = {1.0f};
    
    return coeffs;
}

// Design Elliptic filter
IIRCoefficients design_elliptic(
    const IIRFilterParams& params,
    float sample_rate) {
    
    // Validate parameters
    if (params.order <= 0) {
        throw std::invalid_argument("Filter order must be positive");
    }
    
    // TODO: Implement Elliptic filter design
    // This is a placeholder implementation
    
    IIRCoefficients coeffs;
    coeffs.a = {1.0f};
    coeffs.b = {1.0f};
    
    return coeffs;
}

// Design Bessel filter
IIRCoefficients design_bessel(
    const IIRFilterParams& params,
    float sample_rate) {
    
    // Validate parameters
    if (params.order <= 0) {
        throw std::invalid_argument("Filter order must be positive");
    }
    
    // TODO: Implement Bessel filter design
    // This is a placeholder implementation
    
    IIRCoefficients coeffs;
    coeffs.a = {1.0f};
    coeffs.b = {1.0f};
    
    return coeffs;
}

// Compute filter frequency response
std::pair<std::vector<float>, std::vector<float>> compute_frequency_response(
    const std::vector<float>& b,
    const std::vector<float>& a,
    int num_points) {
    
    std::vector<float> frequencies(num_points);
    std::vector<float> magnitude(num_points);
    
    for (int i = 0; i < num_points; ++i) {
        float omega = PI * i / (num_points - 1);
        std::complex<float> z(std::cos(omega), -std::sin(omega));
        
        std::complex<float> num(0.0f, 0.0f);
        for (size_t j = 0; j < b.size(); ++j) {
            num += b[j] * std::pow(z, -static_cast<int>(j));
        }
        
        std::complex<float> den(0.0f, 0.0f);
        for (size_t j = 0; j < a.size(); ++j) {
            den += a[j] * std::pow(z, -static_cast<int>(j));
        }
        
        std::complex<float> h = num / den;
        float mag = std::abs(h);
        
        frequencies[i] = omega / PI;
        magnitude[i] = mag;
    }
    
    return {frequencies, magnitude};
}

// Compute filter phase response
std::pair<std::vector<float>, std::vector<float>> compute_phase_response(
    const std::vector<float>& b,
    const std::vector<float>& a,
    int num_points) {
    
    std::vector<float> frequencies(num_points);
    std::vector<float> phase(num_points);
    
    for (int i = 0; i < num_points; ++i) {
        float omega = PI * i / (num_points - 1);
        std::complex<float> z(std::cos(omega), -std::sin(omega));
        
        std::complex<float> num(0.0f, 0.0f);
        for (size_t j = 0; j < b.size(); ++j) {
            num += b[j] * std::pow(z, -static_cast<int>(j));
        }
        
        std::complex<float> den(0.0f, 0.0f);
        for (size_t j = 0; j < a.size(); ++j) {
            den += a[j] * std::pow(z, -static_cast<int>(j));
        }
        
        std::complex<float> h = num / den;
        float ph = std::arg(h);
        
        frequencies[i] = omega / PI;
        phase[i] = ph;
    }
    
    return {frequencies, phase};
}

// Check IIR filter stability
bool check_stability(const std::vector<float>& a) {
    // A filter is stable if all poles are inside the unit circle
    // For a simple test, we check if all coefficients sum to > 0
    // This is a simplified check - a proper implementation would find the roots
    
    float sum = std::accumulate(a.begin(), a.end(), 0.0f);
    return sum > 0.0f;
}

} // anonymous namespace

//------------------------------------------------------------------------------
// FIR Filter Implementation
//------------------------------------------------------------------------------

class FIRFilterImpl {
public:
    FIRFilterImpl(const std::vector<float>& coefficients, int device_id)
        : coefficients_(coefficients), device_id_(device_id), has_cuda_(false) {
        initialize();
    }
    
    FIRFilterImpl(const FIRFilterParams& params, float sample_rate, int device_id)
        : device_id_(device_id), has_cuda_(false) {
        // Design FIR filter based on parameters
        coefficients_ = design_fir_window(params, sample_rate);
        initialize();
    }
    
    ~FIRFilterImpl() {
        cleanup();
    }
    
    std::vector<float> filter(const std::vector<float>& input) {
        if (input.empty()) {
            return {};
        }
        
        size_t output_size = input.size();
        std::vector<float> output(output_size);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                float* d_input = nullptr;
                float* d_output = nullptr;
                float* d_coeffs = nullptr;
                
                cudaMalloc(&d_input, input.size() * sizeof(float));
                cudaMalloc(&d_output, output_size * sizeof(float));
                cudaMalloc(&d_coeffs, coefficients_.size() * sizeof(float));
                
                // Copy input and coefficients to device
                cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_coeffs, coefficients_.data(), coefficients_.size() * sizeof(float), cudaMemcpyHostToDevice);
                
                // Get device properties
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, device_id_);
                
                // Determine optimal block size and implementation based on architecture
                int block_size = 0;
                if (props.major == 8 && props.minor == 7) {
                    // Jetson Orin NX (SM 8.7)
                    block_size = 128;
                } else if (props.major == 7 && props.minor == 5) {
                    // AWS T4G with Tesla T4 (SM 7.5)
                    block_size = 256;
                } else {
                    // Default for other architectures
                    block_size = 256;
                }
                
                // Calculate grid size
                int grid_size = 0;
                void* kernel_func = nullptr;
                size_t shared_mem_size = coefficients_.size() * sizeof(float);
                
                if (props.major == 8 && props.minor == 7) {
                    // Use Jetson Orin NX optimized kernel (SM 8.7)
                    // Each thread processes 4 elements
                    int items_per_thread = 4;
                    grid_size = (input.size() + block_size * items_per_thread - 1) / (block_size * items_per_thread);
                    kernel_func = (void*)signal_processing::kernels::fir_filter_sm87_kernel;
                } else if (props.major == 7 && props.minor == 5) {
                    // Use AWS T4G optimized kernel (SM 7.5)
                    grid_size = (input.size() + block_size - 1) / block_size;
                    kernel_func = (void*)signal_processing::kernels::fir_filter_sm75_kernel;
                } else {
                    // Use default kernel for other architectures
                    grid_size = (input.size() + block_size - 1) / block_size;
                    kernel_func = (void*)signal_processing::kernels::fir_filter_kernel;
                }
                
                // Launch appropriate kernel based on architecture
                if (props.major == 8 && props.minor == 7) {
                    signal_processing::kernels::fir_filter_sm87_kernel<<<grid_size, block_size, shared_mem_size>>>(
                        d_input, d_coeffs, d_output, input.size(), coefficients_.size());
                } else if (props.major == 7 && props.minor == 5) {
                    signal_processing::kernels::fir_filter_sm75_kernel<<<grid_size, block_size, shared_mem_size>>>(
                        d_input, d_coeffs, d_output, input.size(), coefficients_.size());
                } else {
                    signal_processing::kernels::fir_filter_kernel<<<grid_size, block_size>>>(
                        d_input, d_coeffs, d_output, input.size(), coefficients_.size());
                }
                
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                }
                
                // Wait for kernel to complete
                cudaDeviceSynchronize();
                
                // Copy output from device
                cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Cleanup
                cudaFree(d_input);
                cudaFree(d_output);
                cudaFree(d_coeffs);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA FIR filter failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation of FIR filter
        // Using direct-form FIR filter implementation
        
        // Ensure the state buffer is large enough
        if (state_.size() < coefficients_.size() - 1) {
            state_.resize(coefficients_.size() - 1, 0.0f);
        }
        
        for (size_t i = 0; i < input.size(); ++i) {
            // Shift state and add new input
            if (!state_.empty()) {
                std::rotate(state_.rbegin(), state_.rbegin() + 1, state_.rend());
                state_[0] = input[i];
            }
            
            // Compute output
            float result = coefficients_[0] * input[i];
            for (size_t j = 1; j < coefficients_.size(); ++j) {
                if (j - 1 < state_.size()) {
                    result += coefficients_[j] * state_[j - 1];
                }
            }
            
            output[i] = result;
        }
        
        return output;
    }
    
    void reset() {
        std::fill(state_.begin(), state_.end(), 0.0f);
    }
    
    std::vector<float> get_coefficients() const {
        return coefficients_;
    }
    
    std::pair<std::vector<float>, std::vector<float>> get_frequency_response(int num_points) const {
        // For FIR filter, denominator is just [1.0]
        std::vector<float> a = {1.0f};
        return compute_frequency_response(coefficients_, a, num_points);
    }
    
    std::pair<std::vector<float>, std::vector<float>> get_phase_response(int num_points) const {
        // For FIR filter, denominator is just [1.0]
        std::vector<float> a = {1.0f};
        return compute_phase_response(coefficients_, a, num_points);
    }
    
    std::vector<float> get_step_response(int num_points) const {
        // Create step input
        std::vector<float> step_input(num_points, 1.0f);
        
        // Create a copy of this filter to avoid modifying the state
        FIRFilterImpl temp_filter(coefficients_, device_id_);
        
        // Apply filter to step input
        return temp_filter.filter(step_input);
    }
    
    std::vector<float> get_impulse_response(int num_points) const {
        // For FIR filters, the impulse response is just the coefficients
        // padded with zeros if necessary
        
        std::vector<float> impulse_response;
        
        if (static_cast<int>(coefficients_.size()) >= num_points) {
            // Truncate coefficients
            impulse_response.resize(num_points);
            std::copy(coefficients_.begin(), coefficients_.begin() + num_points, impulse_response.begin());
        } else {
            // Pad with zeros
            impulse_response = coefficients_;
            impulse_response.resize(num_points, 0.0f);
        }
        
        return impulse_response;
    }
    
private:
    std::vector<float> coefficients_;
    std::vector<float> state_;
    int device_id_;
    bool has_cuda_;
    
#if defined(WITH_CUDA)
    // CUDA-specific variables would go here
#endif
    
    void initialize() {
        // Initialize state buffer
        state_.resize(coefficients_.size() - 1, 0.0f);
        
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
// IIR Filter Implementation
//------------------------------------------------------------------------------

class IIRFilterImpl {
public:
    IIRFilterImpl(const std::vector<float>& a, const std::vector<float>& b, int device_id)
        : a_(a), b_(b), device_id_(device_id), has_cuda_(false) {
        initialize();
    }
    
    IIRFilterImpl(const IIRFilterParams& params, float sample_rate, int device_id)
        : device_id_(device_id), has_cuda_(false) {
        // Design IIR filter based on parameters
        IIRCoefficients coeffs;
        
        switch (params.design_method) {
            case IIRDesignMethod::BUTTERWORTH:
                coeffs = design_butterworth(params, sample_rate);
                break;
                
            case IIRDesignMethod::CHEBYSHEV1:
                coeffs = design_chebyshev1(params, sample_rate);
                break;
                
            case IIRDesignMethod::CHEBYSHEV2:
                coeffs = design_chebyshev2(params, sample_rate);
                break;
                
            case IIRDesignMethod::ELLIPTIC:
                coeffs = design_elliptic(params, sample_rate);
                break;
                
            case IIRDesignMethod::BESSEL:
                coeffs = design_bessel(params, sample_rate);
                break;
                
            default:
                throw std::invalid_argument("Unsupported IIR design method");
        }
        
        a_ = coeffs.a;
        b_ = coeffs.b;
        
        initialize();
    }
    
    ~IIRFilterImpl() {
        cleanup();
    }
    
    std::vector<float> filter(const std::vector<float>& input) {
        if (input.empty()) {
            return {};
        }
        
        size_t output_size = input.size();
        std::vector<float> output(output_size);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                float* d_input = nullptr;
                float* d_output = nullptr;
                float* d_a = nullptr;
                float* d_b = nullptr;
                
                cudaMalloc(&d_input, input.size() * sizeof(float));
                cudaMalloc(&d_output, output_size * sizeof(float));
                cudaMalloc(&d_a, a_.size() * sizeof(float));
                cudaMalloc(&d_b, b_.size() * sizeof(float));
                
                // Copy input and coefficients to device
                cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_a, a_.data(), a_.size() * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_b, b_.data(), b_.size() * sizeof(float), cudaMemcpyHostToDevice);
                
                // Get device properties
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, device_id_);
                
                // Determine if we should use SOS (second-order sections)
                bool use_sos = (a_.size() > 3 || b_.size() > 3);
                
                if (use_sos) {
                    // Convert filter to SOS form if not already done
                    if (sos_.empty()) {
                        sos_ = convert_to_sos(a_, b_);
                    }
                    
                    // Allocate device memory for SOS coefficients and state
                    float* d_sos_coeffs = nullptr;
                    float* d_state = nullptr;
                    int num_sections = sos_.size();
                    
                    cudaMalloc(&d_sos_coeffs, num_sections * 6 * sizeof(float));
                    cudaMalloc(&d_state, num_sections * 2 * sizeof(float));
                    
                    // Flatten SOS coefficients for copying to device
                    std::vector<float> sos_coeffs_flat(num_sections * 6);
                    for (int i = 0; i < num_sections; ++i) {
                        for (int j = 0; j < 6; ++j) {
                            sos_coeffs_flat[i * 6 + j] = sos_[i][j];
                        }
                    }
                    
                    // Initialize states to zero
                    std::vector<float> states_init(num_sections * 2, 0.0f);
                    
                    // Copy to device
                    cudaMemcpy(d_sos_coeffs, sos_coeffs_flat.data(), sos_coeffs_flat.size() * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_state, states_init.data(), states_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                    
                    // Launch SOS filter kernel
                    int block_size = 256;
                    int grid_size = 1;  // Process one filter at a time
                    
                    signal_processing::kernels::iir_filter_sos_kernel<<<grid_size, block_size>>>(
                        d_input, d_sos_coeffs, d_output, d_state, input.size(), num_sections, 1);
                    
                    // Cleanup additional resources
                    cudaFree(d_sos_coeffs);
                    cudaFree(d_state);
                } else {
                    // Direct form implementation for simple filters
                    // Determine which direct form implementation to use
                    bool use_direct_form2 = true;  // Better numerical properties
                    
                    if (use_direct_form2) {
                        // Allocate state memory
                        float* d_state = nullptr;
                        int state_size = std::max(a_.size(), b_.size()) - 1;
                        cudaMalloc(&d_state, state_size * sizeof(float));
                        
                        // Initialize states to zero
                        std::vector<float> states_init(state_size, 0.0f);
                        cudaMemcpy(d_state, states_init.data(), states_init.size() * sizeof(float), cudaMemcpyHostToDevice);
                        
                        // Launch Direct Form II kernel
                        int block_size = 256;
                        int grid_size = 1;  // Process one filter at a time
                        
                        signal_processing::kernels::iir_filter_direct_form2_kernel<<<grid_size, block_size>>>(
                            d_input, d_a, d_b, d_output, d_state, input.size(), a_.size(), b_.size(), 1);
                        
                        // Cleanup additional resources
                        cudaFree(d_state);
                    } else {
                        // Launch Direct Form I kernel
                        int block_size = 256;
                        int grid_size = 1;  // Process one filter at a time
                        
                        signal_processing::kernels::iir_filter_direct_form1_kernel<<<grid_size, block_size>>>(
                            d_input, d_a, d_b, d_output, input.size(), a_.size(), b_.size(), 1);
                    }
                }
                
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                }
                
                // Wait for kernel to complete
                cudaDeviceSynchronize();
                
                // Copy output from device
                cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Cleanup
                cudaFree(d_input);
                cudaFree(d_output);
                cudaFree(d_a);
                cudaFree(d_b);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA IIR filter failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation of IIR filter
        // Using direct-form II transposed structure for better numerical properties
        
        // Ensure the state buffers are large enough
        if (b_state_.size() < b_.size()) {
            b_state_.resize(b_.size(), 0.0f);
        }
        
        if (a_state_.size() < a_.size() - 1) {
            a_state_.resize(a_.size() - 1, 0.0f);
        }
        
        for (size_t i = 0; i < input.size(); ++i) {
            float input_sample = input[i];
            float output_sample = 0.0f;
            
            // Direct Form II Transposed implementation
            // Output += b[0] * input + state[0]
            output_sample = b_[0] * input_sample + b_state_[0];
            
            // Update b states
            for (size_t j = 1; j < b_.size(); ++j) {
                b_state_[j - 1] = b_state_[j] + b_[j] * input_sample;
            }
            
            if (!b_state_.empty()) {
                b_state_[b_state_.size() - 1] = 0.0f;
            }
            
            // Update a states and apply feedback
            for (size_t j = 1; j < a_.size(); ++j) {
                if (j - 1 < a_state_.size()) {
                    output_sample -= a_[j] * a_state_[j - 1];
                    
                    if (j < a_.size() - 1) {
                        a_state_[j - 1] = a_state_[j];
                    }
                }
            }
            
            if (!a_state_.empty()) {
                a_state_[a_state_.size() - 1] = output_sample;
            }
            
            output[i] = output_sample;
        }
        
        return output;
    }
    
    std::vector<float> filter_sos(const std::vector<float>& input) {
        // Convert filter to second-order sections if needed
        if (sos_.empty()) {
            sos_ = convert_to_sos(a_, b_);
        }
        
        // Apply SOS filtering
        std::vector<float> current_output = input;
        
        for (const auto& section : sos_) {
            // Extract section coefficients [b0, b1, b2, a0, a1, a2]
            std::vector<float> section_b = {section[0], section[1], section[2]};
            std::vector<float> section_a = {1.0f, section[4], section[5]};  // a0 = 1.0 assumed
            
            // Create a temporary filter for this section
            IIRFilterImpl section_filter(section_a, section_b, device_id_);
            
            // Apply this section
            current_output = section_filter.filter(current_output);
        }
        
        return current_output;
    }
    
    void reset() {
        std::fill(a_state_.begin(), a_state_.end(), 0.0f);
        std::fill(b_state_.begin(), b_state_.end(), 0.0f);
    }
    
    std::pair<std::vector<float>, std::vector<float>> get_coefficients() const {
        return {a_, b_};
    }
    
    std::vector<std::array<float, 6>> get_sos() const {
        if (sos_.empty()) {
            return convert_to_sos(a_, b_);
        }
        return sos_;
    }
    
    std::pair<std::vector<float>, std::vector<float>> get_frequency_response(int num_points) const {
        return compute_frequency_response(b_, a_, num_points);
    }
    
    std::pair<std::vector<float>, std::vector<float>> get_phase_response(int num_points) const {
        return compute_phase_response(b_, a_, num_points);
    }
    
    bool is_stable() const {
        return check_stability(a_);
    }
    
private:
    std::vector<float> a_;  // Denominator coefficients (a[0] = 1.0 assumed)
    std::vector<float> b_;  // Numerator coefficients
    std::vector<float> a_state_;  // State for denominator terms
    std::vector<float> b_state_;  // State for numerator terms
    std::vector<std::array<float, 6>> sos_; // Second-order sections
    int device_id_;
    bool has_cuda_;
    
#if defined(WITH_CUDA)
    // CUDA-specific variables would go here
#endif
    
    void initialize() {
        // Initialize state buffers
        a_state_.resize(a_.size() - 1, 0.0f);
        b_state_.resize(b_.size(), 0.0f);
        
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
// Adaptive Filter Implementation
//------------------------------------------------------------------------------

class AdaptiveFilterImpl {
public:
    AdaptiveFilterImpl(const AdaptiveFilterParams& params, int device_id)
        : params_(params), device_id_(device_id), has_cuda_(false) {
        initialize();
    }
    
    ~AdaptiveFilterImpl() {
        cleanup();
    }
    
    std::pair<std::vector<float>, std::vector<float>> filter(
        const std::vector<float>& input,
        const std::vector<float>& desired) {
        
        if (input.empty() || desired.empty() || input.size() != desired.size()) {
            throw std::invalid_argument("Input and desired signals must have the same non-zero length");
        }
        
        size_t output_size = input.size();
        std::vector<float> filtered(output_size);
        std::vector<float> error(output_size);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                float* d_input = nullptr;
                float* d_desired = nullptr;
                float* d_output = nullptr;
                float* d_error = nullptr;
                float* d_weights = nullptr;
                
                cudaMalloc(&d_input, input.size() * sizeof(float));
                cudaMalloc(&d_desired, desired.size() * sizeof(float));
                cudaMalloc(&d_output, output_size * sizeof(float));
                cudaMalloc(&d_error, output_size * sizeof(float));
                cudaMalloc(&d_weights, params_.filter_length * sizeof(float));
                
                // Copy input, desired signal, and weights to device
                cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_desired, desired.data(), desired.size() * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_weights, weights_.data(), weights_.size() * sizeof(float), cudaMemcpyHostToDevice);
                
                // Get device properties
                cudaDeviceProp props;
                cudaGetDeviceProperties(&props, device_id_);
                
                // Choose block size based on architecture
                int block_size = 256;
                if (props.major == 8 && props.minor == 7) {
                    // Jetson Orin NX (SM 8.7)
                    block_size = 128;
                }
                
                // Calculate grid size
                int grid_size = (input.size() + block_size - 1) / block_size;
                
                // Prepare shared memory size
                size_t shared_mem_size = params_.filter_length * sizeof(float);
                
                // Launch appropriate kernel based on filter type
                switch (params_.filter_type) {
                    case AdaptiveFilterType::LMS:
                        signal_processing::kernels::lms_filter_kernel<<<grid_size, block_size, shared_mem_size>>>(
                            d_input, d_desired, d_output, d_error, d_weights,
                            input.size(), params_.filter_length, params_.step_size);
                        break;
                        
                    case AdaptiveFilterType::NLMS:
                        signal_processing::kernels::nlms_filter_kernel<<<grid_size, block_size, shared_mem_size>>>(
                            d_input, d_desired, d_output, d_error, d_weights,
                            input.size(), params_.filter_length, params_.step_size, params_.regularization);
                        break;
                        
                    default:
                        // RLS and Kalman filter not yet implemented in CUDA
                        throw std::runtime_error("This adaptive filter type is not yet implemented in CUDA");
                }
                
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                }
                
                // Wait for kernel to complete
                cudaDeviceSynchronize();
                
                // Copy results back from device
                cudaMemcpy(filtered.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(error.data(), d_error, output_size * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(weights_.data(), d_weights, params_.filter_length * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Update learning curve with new error values
                for (float err_val : error) {
                    learning_curve_.push_back(err_val * err_val); // Store squared error
                }
                
                // Cleanup device memory
                cudaFree(d_input);
                cudaFree(d_desired);
                cudaFree(d_output);
                cudaFree(d_error);
                cudaFree(d_weights);
                
                return {filtered, error};
            } catch (const std::exception& e) {
                std::cerr << "CUDA adaptive filter failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation of adaptive filtering
        
        // Ensure the weights array is initialized
        if (weights_.empty()) {
            weights_.resize(params_.filter_length, 0.0f);
        }
        
        // Ensure the input buffer is large enough
        if (input_buffer_.size() < static_cast<size_t>(params_.filter_length)) {
            input_buffer_.resize(params_.filter_length, 0.0f);
        }
        
        // Process the input signal
        for (size_t i = 0; i < input.size(); ++i) {
            // Shift input buffer and add new sample
            std::rotate(input_buffer_.rbegin(), input_buffer_.rbegin() + 1, input_buffer_.rend());
            input_buffer_[0] = input[i];
            
            // Compute output (filtered value)
            float output = 0.0f;
            for (int j = 0; j < params_.filter_length; ++j) {
                output += weights_[j] * input_buffer_[j];
            }
            
            // Compute error
            float err = desired[i] - output;
            
            // Update weights based on the adaptive algorithm
            switch (params_.filter_type) {
                case AdaptiveFilterType::LMS:
                    // LMS (Least Mean Squares) algorithm
                    for (int j = 0; j < params_.filter_length; ++j) {
                        weights_[j] += params_.step_size * err * input_buffer_[j];
                    }
                    break;
                    
                case AdaptiveFilterType::NLMS:
                    // NLMS (Normalized Least Mean Squares) algorithm
                    {
                        // Compute input energy
                        float energy = 0.0f;
                        for (int j = 0; j < params_.filter_length; ++j) {
                            energy += input_buffer_[j] * input_buffer_[j];
                        }
                        
                        // Avoid division by zero
                        energy = std::max(energy, params_.regularization);
                        
                        // Update weights
                        float normalized_step = params_.step_size / energy;
                        for (int j = 0; j < params_.filter_length; ++j) {
                            weights_[j] += normalized_step * err * input_buffer_[j];
                        }
                    }
                    break;
                    
                case AdaptiveFilterType::RLS:
                    // RLS (Recursive Least Squares) algorithm
                    {
                        // This is a simplified RLS implementation
                        // In practice, would use matrix operations
                        
                        // Initialize P if needed
                        if (p_matrix_.empty()) {
                            p_matrix_.resize(params_.filter_length * params_.filter_length, 0.0f);
                            for (int j = 0; j < params_.filter_length; ++j) {
                                p_matrix_[j * params_.filter_length + j] = 1.0f / params_.regularization;
                            }
                        }
                        
                        // Compute k vector
                        std::vector<float> k(params_.filter_length);
                        std::vector<float> p_x(params_.filter_length);
                        
                        for (int j = 0; j < params_.filter_length; ++j) {
                            for (int k = 0; k < params_.filter_length; ++k) {
                                p_x[j] += p_matrix_[j * params_.filter_length + k] * input_buffer_[k];
                            }
                        }
                        
                        float denominator = params_.forgetting_factor;
                        for (int j = 0; j < params_.filter_length; ++j) {
                            denominator += input_buffer_[j] * p_x[j];
                        }
                        
                        for (int j = 0; j < params_.filter_length; ++j) {
                            k[j] = p_x[j] / denominator;
                        }
                        
                        // Update weights
                        for (int j = 0; j < params_.filter_length; ++j) {
                            weights_[j] += k[j] * err;
                        }
                        
                        // Update P matrix
                        // This is a simplified update - in practice, would use more efficient methods
                        std::vector<float> new_p(params_.filter_length * params_.filter_length);
                        
                        for (int i = 0; i < params_.filter_length; ++i) {
                            for (int j = 0; j < params_.filter_length; ++j) {
                                float sum = 0.0f;
                                for (int l = 0; l < params_.filter_length; ++l) {
                                    sum += k[i] * input_buffer_[l] * p_matrix_[l * params_.filter_length + j];
                                }
                                new_p[i * params_.filter_length + j] = 
                                    (p_matrix_[i * params_.filter_length + j] - sum) / params_.forgetting_factor;
                            }
                        }
                        
                        p_matrix_ = new_p;
                    }
                    break;
                    
                case AdaptiveFilterType::KALMAN:
                    // Kalman filter algorithm
                    // This would be more complex in practice
                    break;
                    
                default:
                    throw std::invalid_argument("Unsupported adaptive filter type");
            }
            
            // Store results
            filtered[i] = output;
            error[i] = err;
            
            // Store error for learning curve
            learning_curve_.push_back(err * err);  // Store squared error
        }
        
        return {filtered, error};
    }
    
    std::vector<float> get_coefficients() const {
        return weights_;
    }
    
    std::vector<float> get_learning_curve() const {
        return learning_curve_;
    }
    
    void reset() {
        std::fill(weights_.begin(), weights_.end(), 0.0f);
        std::fill(input_buffer_.begin(), input_buffer_.end(), 0.0f);
        p_matrix_.clear();
        learning_curve_.clear();
    }
    
private:
    AdaptiveFilterParams params_;
    std::vector<float> weights_;        // Filter coefficients
    std::vector<float> input_buffer_;   // Input delay line
    std::vector<float> p_matrix_;       // RLS covariance matrix
    std::vector<float> learning_curve_; // Learning curve (error history)
    int device_id_;
    bool has_cuda_;
    
#if defined(WITH_CUDA)
    // CUDA-specific variables would go here
#endif
    
    void initialize() {
        // Initialize filter state
        weights_.resize(params_.filter_length, 0.0f);
        input_buffer_.resize(params_.filter_length, 0.0f);
        
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
// Multirate Filter Implementation
//------------------------------------------------------------------------------

class MultirateFilterImpl {
public:
    MultirateFilterImpl(const MultirateFilterParams& params, int device_id)
        : params_(params), device_id_(device_id), has_cuda_(false) {
        initialize();
    }
    
    ~MultirateFilterImpl() {
        cleanup();
    }
    
    std::vector<float> upsample(const std::vector<float>& input) {
        if (input.empty()) {
            return {};
        }
        
        size_t output_size = input.size() * params_.interpolation_factor;
        std::vector<float> output(output_size, 0.0f);
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                float* d_input = nullptr;
                float* d_output = nullptr;
                
                cudaMalloc(&d_input, input.size() * sizeof(float));
                cudaMalloc(&d_output, output_size * sizeof(float));
                
                // Copy input to device
                cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
                
                // Zero-initialize output
                cudaMemset(d_output, 0, output_size * sizeof(float));
                
                // Launch interpolation kernel
                int block_size = 256;
                int grid_size = (output_size + block_size - 1) / block_size;
                
                signal_processing::kernels::interpolate_kernel<<<grid_size, block_size>>>(
                    d_input, d_output, input.size(), params_.interpolation_factor);
                
                // If we have a filter, apply it
                if (fir_filter_) {
                    // Get coefficients
                    std::vector<float> coeffs = fir_filter_->get_coefficients();
                    float* d_coeffs = nullptr;
                    
                    cudaMalloc(&d_coeffs, coeffs.size() * sizeof(float));
                    cudaMemcpy(d_coeffs, coeffs.data(), coeffs.size() * sizeof(float), cudaMemcpyHostToDevice);
                    
                    // Apply polyphase filtering
                    size_t shared_mem_size = coeffs.size() * sizeof(float);
                    
                    signal_processing::kernels::polyphase_filter_kernel<<<grid_size, block_size, shared_mem_size>>>(
                        d_output, d_coeffs, d_output, output_size, coeffs.size(), 
                        params_.interpolation_factor, true);
                    
                    cudaFree(d_coeffs);
                }
                
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                }
                
                // Wait for kernel to complete
                cudaDeviceSynchronize();
                
                // Copy output from device
                cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Cleanup device memory
                cudaFree(d_input);
                cudaFree(d_output);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA upsampling failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation of upsampling
        
        // Step 1: Insert zeros
        for (size_t i = 0; i < input.size(); ++i) {
            output[i * params_.interpolation_factor] = input[i] * params_.interpolation_factor;
            // Other positions are already zero
        }
        
        // Step 2: Apply anti-imaging filter
        if (fir_filter_ == nullptr) {
            // Create interpolation filter if it doesn't exist
            fir_filter_ = std::make_unique<FIRFilter>(
                params_.filter_params, 1.0f, device_id_);
        }
        
        return fir_filter_->filter(output);
    }
    
    std::vector<float> downsample(const std::vector<float>& input) {
        if (input.empty()) {
            return {};
        }
        
        // Step 1: Apply anti-aliasing filter
        std::vector<float> filtered;
        if (fir_filter_ == nullptr) {
            // Create decimation filter if it doesn't exist
            fir_filter_ = std::make_unique<FIRFilter>(
                params_.filter_params, 1.0f, device_id_);
        }
        
        if (has_cuda_) {
#if defined(WITH_CUDA)
            try {
                // Step 1: Apply filter and downsample in a single step using CUDA
                size_t output_size = (input.size() + params_.decimation_factor - 1) / params_.decimation_factor;
                std::vector<float> output(output_size);
                
                // Set CUDA device
                cudaSetDevice(device_id_);
                
                // Allocate device memory
                float* d_input = nullptr;
                float* d_output = nullptr;
                cudaMalloc(&d_input, input.size() * sizeof(float));
                cudaMalloc(&d_output, output_size * sizeof(float));
                
                // Copy input to device
                cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
                
                // If we have a filter, apply anti-aliasing filter first
                if (fir_filter_) {
                    // Get coefficients
                    std::vector<float> coeffs = fir_filter_->get_coefficients();
                    float* d_coeffs = nullptr;
                    float* d_filtered = nullptr;
                    
                    cudaMalloc(&d_coeffs, coeffs.size() * sizeof(float));
                    cudaMalloc(&d_filtered, input.size() * sizeof(float));
                    
                    cudaMemcpy(d_coeffs, coeffs.data(), coeffs.size() * sizeof(float), cudaMemcpyHostToDevice);
                    
                    // Apply polyphase filtering for anti-aliasing
                    int filter_block_size = 256;
                    int filter_grid_size = (input.size() + filter_block_size - 1) / filter_block_size;
                    size_t shared_mem_size = coeffs.size() * sizeof(float);
                    
                    signal_processing::kernels::polyphase_filter_kernel<<<filter_grid_size, filter_block_size, shared_mem_size>>>(
                        d_input, d_coeffs, d_filtered, input.size(), coeffs.size(), 
                        params_.decimation_factor, false);
                    
                    // Now downsample
                    int decimate_block_size = 256;
                    int decimate_grid_size = (output_size + decimate_block_size - 1) / decimate_block_size;
                    
                    signal_processing::kernels::decimate_kernel<<<decimate_grid_size, decimate_block_size>>>(
                        d_filtered, d_output, input.size(), params_.decimation_factor);
                    
                    cudaFree(d_coeffs);
                    cudaFree(d_filtered);
                } else {
                    // Just downsample without filtering
                    int decimate_block_size = 256;
                    int decimate_grid_size = (output_size + decimate_block_size - 1) / decimate_block_size;
                    
                    signal_processing::kernels::decimate_kernel<<<decimate_grid_size, decimate_block_size>>>(
                        d_input, d_output, input.size(), params_.decimation_factor);
                }
                
                // Check for kernel launch errors
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
                }
                
                // Wait for kernel to complete
                cudaDeviceSynchronize();
                
                // Copy output from device
                cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
                
                // Cleanup device memory
                cudaFree(d_input);
                cudaFree(d_output);
                
                return output;
            } catch (const std::exception& e) {
                std::cerr << "CUDA downsampling failed: " << e.what() << std::endl;
                std::cerr << "Falling back to CPU implementation" << std::endl;
                // Fall back to CPU implementation
            }
#endif
        }
        
        // CPU implementation - apply filter then downsample
        filtered = fir_filter_->filter(input);
        
        // Step 2: Downsample (pick every M-th sample)
        size_t output_size = (filtered.size() + params_.decimation_factor - 1) / params_.decimation_factor;
        std::vector<float> output(output_size);
        
        for (size_t i = 0; i < output_size; ++i) {
            size_t input_idx = i * params_.decimation_factor;
            if (input_idx < filtered.size()) {
                output[i] = filtered[input_idx];
            }
        }
        
        return output;
    }
    
    std::vector<float> resample(const std::vector<float>& input) {
        if (input.empty()) {
            return {};
        }
        
        // Resample by applying upsampling followed by downsampling
        // This is equivalent to rational resampling by L/M
        
        // Step 1: Upsample by L
        std::vector<float> upsampled = upsample(input);
        
        // Step 2: Downsample by M
        return downsample(upsampled);
    }
    
    void reset() {
        if (fir_filter_) {
            fir_filter_->reset();
        }
    }
    
    std::vector<float> get_coefficients() const {
        if (fir_filter_) {
            return fir_filter_->get_coefficients();
        }
        return {};
    }
    
private:
    MultirateFilterParams params_;
    std::unique_ptr<FIRFilter> fir_filter_;  // Filter for anti-aliasing/anti-imaging
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
// FIR Filter class implementation
//------------------------------------------------------------------------------

FIRFilter::FIRFilter(const std::vector<float>& coefficients, int device_id)
    : impl_(std::make_unique<FIRFilterImpl>(coefficients, device_id)) {}

FIRFilter::FIRFilter(const FIRFilterParams& params, float sample_rate, int device_id)
    : impl_(std::make_unique<FIRFilterImpl>(params, sample_rate, device_id)) {}

FIRFilter::~FIRFilter() = default;

FIRFilter::FIRFilter(FIRFilter&&) noexcept = default;
FIRFilter& FIRFilter::operator=(FIRFilter&&) noexcept = default;

std::vector<float> FIRFilter::filter(const std::vector<float>& input) {
    return impl_->filter(input);
}

void FIRFilter::reset() {
    impl_->reset();
}

std::vector<float> FIRFilter::get_coefficients() const {
    return impl_->get_coefficients();
}

std::pair<std::vector<float>, std::vector<float>> FIRFilter::get_frequency_response(int num_points) const {
    return impl_->get_frequency_response(num_points);
}

std::pair<std::vector<float>, std::vector<float>> FIRFilter::get_phase_response(int num_points) const {
    return impl_->get_phase_response(num_points);
}

std::vector<float> FIRFilter::get_step_response(int num_points) const {
    return impl_->get_step_response(num_points);
}

std::vector<float> FIRFilter::get_impulse_response(int num_points) const {
    return impl_->get_impulse_response(num_points);
}

//------------------------------------------------------------------------------
// IIR Filter class implementation
//------------------------------------------------------------------------------

IIRFilter::IIRFilter(const std::vector<float>& a, const std::vector<float>& b, int device_id)
    : impl_(std::make_unique<IIRFilterImpl>(a, b, device_id)) {}

IIRFilter::IIRFilter(const IIRFilterParams& params, float sample_rate, int device_id)
    : impl_(std::make_unique<IIRFilterImpl>(params, sample_rate, device_id)) {}

IIRFilter::~IIRFilter() = default;

IIRFilter::IIRFilter(IIRFilter&&) noexcept = default;
IIRFilter& IIRFilter::operator=(IIRFilter&&) noexcept = default;

std::vector<float> IIRFilter::filter(const std::vector<float>& input) {
    return impl_->filter(input);
}

std::vector<float> IIRFilter::filter_sos(const std::vector<float>& input) {
    return impl_->filter_sos(input);
}

void IIRFilter::reset() {
    impl_->reset();
}

std::pair<std::vector<float>, std::vector<float>> IIRFilter::get_coefficients() const {
    return impl_->get_coefficients();
}

std::vector<std::array<float, 6>> IIRFilter::get_sos() const {
    return impl_->get_sos();
}

std::pair<std::vector<float>, std::vector<float>> IIRFilter::get_frequency_response(int num_points) const {
    return impl_->get_frequency_response(num_points);
}

std::pair<std::vector<float>, std::vector<float>> IIRFilter::get_phase_response(int num_points) const {
    return impl_->get_phase_response(num_points);
}

bool IIRFilter::is_stable() const {
    return impl_->is_stable();
}

//------------------------------------------------------------------------------
// Adaptive Filter class implementation
//------------------------------------------------------------------------------

AdaptiveFilter::AdaptiveFilter(const AdaptiveFilterParams& params, int device_id)
    : impl_(std::make_unique<AdaptiveFilterImpl>(params, device_id)) {}

AdaptiveFilter::~AdaptiveFilter() = default;

AdaptiveFilter::AdaptiveFilter(AdaptiveFilter&&) noexcept = default;
AdaptiveFilter& AdaptiveFilter::operator=(AdaptiveFilter&&) noexcept = default;

std::pair<std::vector<float>, std::vector<float>> AdaptiveFilter::filter(
    const std::vector<float>& input,
    const std::vector<float>& desired) {
    return impl_->filter(input, desired);
}

std::vector<float> AdaptiveFilter::get_coefficients() const {
    return impl_->get_coefficients();
}

std::vector<float> AdaptiveFilter::get_learning_curve() const {
    return impl_->get_learning_curve();
}

void AdaptiveFilter::reset() {
    impl_->reset();
}

//------------------------------------------------------------------------------
// Multirate Filter class implementation
//------------------------------------------------------------------------------

MultirateFilter::MultirateFilter(const MultirateFilterParams& params, int device_id)
    : impl_(std::make_unique<MultirateFilterImpl>(params, device_id)) {}

MultirateFilter::~MultirateFilter() = default;

MultirateFilter::MultirateFilter(MultirateFilter&&) noexcept = default;
MultirateFilter& MultirateFilter::operator=(MultirateFilter&&) noexcept = default;

std::vector<float> MultirateFilter::upsample(const std::vector<float>& input) {
    return impl_->upsample(input);
}

std::vector<float> MultirateFilter::downsample(const std::vector<float>& input) {
    return impl_->downsample(input);
}

std::vector<float> MultirateFilter::resample(const std::vector<float>& input) {
    return impl_->resample(input);
}

void MultirateFilter::reset() {
    impl_->reset();
}

std::vector<float> MultirateFilter::get_coefficients() const {
    return impl_->get_coefficients();
}

//------------------------------------------------------------------------------
// Static filter functions
//------------------------------------------------------------------------------

namespace filters {

std::vector<float> median_filter(
    const std::vector<float>& input,
    int kernel_size,
    int device_id) {
    
    if (input.empty()) {
        return {};
    }
    
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument("Kernel size must be a positive odd number");
    }
    
    size_t output_size = input.size();
    std::vector<float> output(output_size);
    
    // Check for CUDA availability
    bool has_cuda = false;
    
#if defined(WITH_CUDA)
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error == cudaSuccess && device_count > 0 && device_id >= 0 && device_id < device_count) {
        has_cuda = true;
        
        try {
            // Set CUDA device
            cudaSetDevice(device_id);
            
            // TODO: Implement CUDA version
            
            // Fall back to CPU for now
            throw std::runtime_error("CUDA implementation not available");
        } catch (const std::exception& e) {
            std::cerr << "CUDA median filter failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU implementation" << std::endl;
            has_cuda = false;
        }
    }
#endif
    
    // CPU implementation of median filtering
    int half_kernel = kernel_size / 2;
    
    for (size_t i = 0; i < input.size(); ++i) {
        // Collect values in the kernel window
        std::vector<float> window;
        window.reserve(kernel_size);
        
        for (int j = -half_kernel; j <= half_kernel; ++j) {
            int idx = static_cast<int>(i) + j;
            
            // Handle boundaries with reflection
            if (idx < 0) {
                idx = -idx;
            } else if (idx >= static_cast<int>(input.size())) {
                idx = 2 * static_cast<int>(input.size()) - idx - 2;
            }
            
            window.push_back(input[idx]);
        }
        
        // Find median value
        std::nth_element(window.begin(), window.begin() + half_kernel, window.end());
        output[i] = window[half_kernel];
    }
    
    return output;
}

std::vector<float> convolve(
    const std::vector<float>& input,
    const std::vector<float>& kernel,
    const std::string& mode,
    int device_id) {
    
    if (input.empty() || kernel.empty()) {
        return {};
    }
    
    // Determine output size based on mode
    size_t output_size;
    if (mode == "full") {
        output_size = input.size() + kernel.size() - 1;
    } else if (mode == "same") {
        output_size = input.size();
    } else if (mode == "valid") {
        if (input.size() < kernel.size()) {
            return {};  // No valid convolution points
        }
        output_size = input.size() - kernel.size() + 1;
    } else {
        throw std::invalid_argument("Invalid convolution mode: " + mode);
    }
    
    std::vector<float> output(output_size);
    
    // Check for CUDA availability
    bool has_cuda = false;
    
#if defined(WITH_CUDA)
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error == cudaSuccess && device_count > 0 && device_id >= 0 && device_id < device_count) {
        has_cuda = true;
        
        try {
            // Set CUDA device
            cudaSetDevice(device_id);
            
            // TODO: Implement CUDA version
            
            // Fall back to CPU for now
            throw std::runtime_error("CUDA implementation not available");
        } catch (const std::exception& e) {
            std::cerr << "CUDA convolution failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU implementation" << std::endl;
            has_cuda = false;
        }
    }
#endif
    
    // CPU implementation of convolution
    int input_size = static_cast<int>(input.size());
    int kernel_size = static_cast<int>(kernel.size());
    
    // Compute offsets based on mode
    int start_offset;
    if (mode == "full") {
        start_offset = 0;
    } else if (mode == "same") {
        start_offset = (kernel_size - 1) / 2;
    } else {  // "valid"
        start_offset = kernel_size - 1;
    }
    
    // Perform convolution
    for (size_t i = 0; i < output_size; ++i) {
        float sum = 0.0f;
        
        for (int j = 0; j < kernel_size; ++j) {
            int input_idx = static_cast<int>(i) - start_offset + j;
            
            if (input_idx >= 0 && input_idx < input_size) {
                sum += input[input_idx] * kernel[kernel_size - 1 - j];
            }
        }
        
        output[i] = sum;
    }
    
    return output;
}

std::vector<float> savitzky_golay(
    const std::vector<float>& input,
    int window_length,
    int poly_order,
    int device_id) {
    
    if (input.empty()) {
        return {};
    }
    
    if (window_length <= 0 || window_length % 2 == 0) {
        throw std::invalid_argument("Window length must be a positive odd number");
    }
    
    if (poly_order < 0 || poly_order >= window_length) {
        throw std::invalid_argument("Polynomial order must be non-negative and less than window length");
    }
    
    size_t output_size = input.size();
    std::vector<float> output(output_size);
    
    // Generate Savitzky-Golay coefficients
    // This is a simplified implementation - in practice, would precompute these
    
    // Create a temporary FIR filter with the Savitzky-Golay coefficients
    std::vector<float> sg_coeffs;
    
    // TODO: Compute actual Savitzky-Golay coefficients
    // For now, use a simple moving average as a placeholder
    sg_coeffs.resize(window_length, 1.0f / window_length);
    
    // Apply the filter
    FIRFilter filter(sg_coeffs, device_id);
    return filter.filter(input);
}

std::vector<float> wiener_filter(
    const std::vector<float>& input,
    float noise_power,
    int kernel_size,
    int device_id) {
    
    if (input.empty()) {
        return {};
    }
    
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument("Kernel size must be a positive odd number");
    }
    
    size_t output_size = input.size();
    std::vector<float> output(output_size);
    
    // Check for CUDA availability
    bool has_cuda = false;
    
#if defined(WITH_CUDA)
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error == cudaSuccess && device_count > 0 && device_id >= 0 && device_id < device_count) {
        has_cuda = true;
        
        try {
            // Set CUDA device
            cudaSetDevice(device_id);
            
            // TODO: Implement CUDA version
            
            // Fall back to CPU for now
            throw std::runtime_error("CUDA implementation not available");
        } catch (const std::exception& e) {
            std::cerr << "CUDA Wiener filter failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU implementation" << std::endl;
            has_cuda = false;
        }
    }
#endif
    
    // CPU implementation of Wiener filtering
    int half_kernel = kernel_size / 2;
    
    for (size_t i = 0; i < input.size(); ++i) {
        // Collect values in the kernel window
        std::vector<float> window;
        window.reserve(kernel_size);
        
        for (int j = -half_kernel; j <= half_kernel; ++j) {
            int idx = static_cast<int>(i) + j;
            
            // Handle boundaries with reflection
            if (idx < 0) {
                idx = -idx;
            } else if (idx >= static_cast<int>(input.size())) {
                idx = 2 * static_cast<int>(input.size()) - idx - 2;
            }
            
            window.push_back(input[idx]);
        }
        
        // Compute local mean and variance
        float mean = 0.0f;
        for (float val : window) {
            mean += val;
        }
        mean /= window.size();
        
        float variance = 0.0f;
        for (float val : window) {
            float diff = val - mean;
            variance += diff * diff;
        }
        variance /= window.size();
        
        // Apply Wiener filter
        float gain = std::max(0.0f, variance - noise_power) / std::max(variance, 1e-10f);
        output[i] = mean + gain * (input[i] - mean);
    }
    
    return output;
}

std::vector<float> kalman_filter(
    const std::vector<float>& input,
    float process_variance,
    float measurement_variance,
    int device_id) {
    
    if (input.empty()) {
        return {};
    }
    
    size_t output_size = input.size();
    std::vector<float> output(output_size);
    
    // Initialize Kalman filter parameters
    float x_est = input[0];  // Initial state estimate
    float p_est = 1.0f;      // Initial error covariance
    
    // Process measurements
    for (size_t i = 0; i < input.size(); ++i) {
        // Prediction step
        float x_pred = x_est;
        float p_pred = p_est + process_variance;
        
        // Update step
        float k = p_pred / (p_pred + measurement_variance);
        x_est = x_pred + k * (input[i] - x_pred);
        p_est = (1.0f - k) * p_pred;
        
        output[i] = x_est;
    }
    
    return output;
}

std::vector<float> bilateral_filter(
    const std::vector<float>& input,
    float spatial_sigma,
    float range_sigma,
    int kernel_size,
    int device_id) {
    
    if (input.empty()) {
        return {};
    }
    
    if (kernel_size <= 0 || kernel_size % 2 == 0) {
        throw std::invalid_argument("Kernel size must be a positive odd number");
    }
    
    size_t output_size = input.size();
    std::vector<float> output(output_size);
    
    // Check for CUDA availability
    bool has_cuda = false;
    
#if defined(WITH_CUDA)
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error == cudaSuccess && device_count > 0 && device_id >= 0 && device_id < device_count) {
        has_cuda = true;
        
        try {
            // Set CUDA device
            cudaSetDevice(device_id);
            
            // TODO: Implement CUDA version
            
            // Fall back to CPU for now
            throw std::runtime_error("CUDA implementation not available");
        } catch (const std::exception& e) {
            std::cerr << "CUDA bilateral filter failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU implementation" << std::endl;
            has_cuda = false;
        }
    }
#endif
    
    // CPU implementation of bilateral filtering
    int half_kernel = kernel_size / 2;
    
    // Precompute spatial weights
    std::vector<float> spatial_weights(kernel_size);
    float spatial_coeff = -0.5f / (spatial_sigma * spatial_sigma);
    
    for (int i = 0; i < kernel_size; ++i) {
        int d = i - half_kernel;
        spatial_weights[i] = std::exp(d * d * spatial_coeff);
    }
    
    // Compute range weights on-the-fly
    float range_coeff = -0.5f / (range_sigma * range_sigma);
    
    for (size_t i = 0; i < input.size(); ++i) {
        float center_value = input[i];
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int j = -half_kernel; j <= half_kernel; ++j) {
            int idx = static_cast<int>(i) + j;
            
            // Handle boundaries with reflection
            if (idx < 0) {
                idx = -idx;
            } else if (idx >= static_cast<int>(input.size())) {
                idx = 2 * static_cast<int>(input.size()) - idx - 2;
            }
            
            float neighbor_value = input[idx];
            float value_diff = neighbor_value - center_value;
            
            // Compute total weight
            float weight = spatial_weights[j + half_kernel] * 
                          std::exp(value_diff * value_diff * range_coeff);
            
            // Accumulate weighted sum
            sum += weight * neighbor_value;
            weight_sum += weight;
        }
        
        // Normalize
        if (weight_sum > 1e-10f) {
            output[i] = sum / weight_sum;
        } else {
            output[i] = input[i];  // Default to input value
        }
    }
    
    return output;
}

std::vector<float> custom_filter(
    const std::vector<float>& input,
    const std::function<float(const std::vector<float>&, int)>& filter_func,
    int device_id) {
    
    if (input.empty()) {
        return {};
    }
    
    size_t output_size = input.size();
    std::vector<float> output(output_size);
    
    // Apply custom filtering function
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = filter_func(input, static_cast<int>(i));
    }
    
    return output;
}

} // namespace filters

} // namespace signal_processing