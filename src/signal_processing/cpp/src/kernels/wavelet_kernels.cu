// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <complex>
#include "signal_processing/wavelet_transform.h"

namespace signal_processing {

// Shared memory buffer size for tiled operations
constexpr int TILE_SIZE = 256;

// Utility function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel for discrete wavelet transform (convolution)
__global__ void dwt_convolution_kernel(const float* signal, const float* filter, 
                                     float* result, int signal_len, int filter_len) {
    extern __shared__ float s_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < signal_len + filter_len - 1) {
        float sum = 0.0f;
        
        for (int j = 0; j < filter_len; j++) {
            int signal_idx = idx - j;
            if (signal_idx >= 0 && signal_idx < signal_len) {
                sum += signal[signal_idx] * filter[j];
            }
        }
        
        result[idx] = sum;
    }
}

// Kernel for downsampling (using stride-2 access)
__global__ void downsample_kernel(const float* input, float* output, int input_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_idx = idx;
    int input_idx = idx * 2;
    
    if (input_idx < input_len) {
        output[output_idx] = input[input_idx];
    }
}

// Kernel for upsampling (inserting zeros)
__global__ void upsample_kernel(const float* input, float* output, int input_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_len) {
        output[idx * 2] = input[idx];
        output[idx * 2 + 1] = 0.0f;
    }
}

// Kernel for signal extension (handling boundaries)
__global__ void extend_signal_kernel(const float* signal, float* extended, 
                                   int signal_len, int extension_size, int mode) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int extended_len = signal_len + 2 * extension_size;
    
    if (idx < extended_len) {
        if (idx >= extension_size && idx < extension_size + signal_len) {
            // Copy the original signal to the middle
            extended[idx] = signal[idx - extension_size];
        } else if (idx < extension_size) {
            // Left boundary
            switch (mode) {
                case 0: // ZERO_PADDING (already initialized to zero)
                    extended[idx] = 0.0f;
                    break;
                case 1: // SYMMETRIC
                    extended[idx] = signal[extension_size - 1 - idx];
                    break;
                case 2: // PERIODIC
                    extended[idx] = signal[signal_len - extension_size + idx];
                    break;
                case 3: // REFLECT
                    extended[idx] = signal[extension_size - idx];
                    break;
            }
        } else {
            // Right boundary
            int offset = idx - (extension_size + signal_len);
            switch (mode) {
                case 0: // ZERO_PADDING
                    extended[idx] = 0.0f;
                    break;
                case 1: // SYMMETRIC
                    extended[idx] = signal[signal_len - 1 - offset];
                    break;
                case 2: // PERIODIC
                    extended[idx] = signal[offset];
                    break;
                case 3: // REFLECT
                    extended[idx] = signal[signal_len - 2 - offset];
                    break;
            }
        }
    }
}

// Kernel for combining approximation and detail coefficients
__global__ void dwt_combine_kernel(const float* approx, const float* detail, 
                                 float* result, int signal_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < signal_len) {
        result[idx] = approx[idx] + detail[idx];
    }
}

// Optimized kernel for continuous wavelet transform (Morlet)
__global__ void cwt_morlet_kernel(const float* signal, float2* coefficients,
                                const float* scales, int signal_len, int scale_idx, 
                                float omega0) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < signal_len) {
        float scale = scales[scale_idx];
        float2 sum = make_float2(0.0f, 0.0f);
        
        for (int k = 0; k < signal_len; k++) {
            float t = (k - pos) / scale;
            float t_squared = t * t;
            
            // Morlet wavelet: exp(-t²/2) * exp(iω₀t)
            float gauss = __expf(-0.5f * t_squared);
            float cos_term = __cosf(omega0 * t);
            float sin_term = __sinf(omega0 * t);
            
            // Complex multiplication with conjugate
            sum.x += signal[k] * gauss * cos_term;  // Real part
            sum.y -= signal[k] * gauss * sin_term;  // Imaginary part (negative for conjugate)
        }
        
        // Normalize by square root of scale
        float norm = __fsqrt_rn(scale);
        coefficients[pos + scale_idx * signal_len] = make_float2(sum.x / norm, sum.y / norm);
    }
}

// Optimized kernel for continuous wavelet transform (Mexican Hat)
__global__ void cwt_mexican_hat_kernel(const float* signal, float2* coefficients,
                                     const float* scales, int signal_len, int scale_idx) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pos < signal_len) {
        float scale = scales[scale_idx];
        float sum = 0.0f;
        
        for (int k = 0; k < signal_len; k++) {
            float t = (k - pos) / scale;
            float t_squared = t * t;
            
            // Mexican Hat formula: (1 - t²) * exp(-t²/2)
            float wavelet_val = (1.0f - t_squared) * __expf(-0.5f * t_squared);
            
            sum += signal[k] * wavelet_val;
        }
        
        // Normalize by square root of scale
        float norm = __fsqrt_rn(scale);
        coefficients[pos + scale_idx * signal_len] = make_float2(sum / norm, 0.0f);
    }
}

// Optimized CWT using FFT-based convolution
__global__ void cwt_fft_prepare_kernel(const float* wavelet, float2* wavelet_fft,
                                     float scale, int signal_len, int wavelet_len,
                                     int wavelet_family) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < signal_len) {
        float t = (idx - wavelet_len / 2) / scale;
        float wavelet_val = 0.0f;
        float imag_val = 0.0f;
        
        if (wavelet_family == 6) {  // MORLET
            float omega0 = 5.0f;
            float t_squared = t * t;
            float gauss = __expf(-0.5f * t_squared);
            wavelet_val = gauss * __cosf(omega0 * t);
            imag_val = gauss * __sinf(omega0 * t);
        } else if (wavelet_family == 7) {  // MEXICAN_HAT
            float t_squared = t * t;
            wavelet_val = (1.0f - t_squared) * __expf(-0.5f * t_squared);
        }
        
        // Store the wavelet for FFT
        if (idx < wavelet_len) {
            wavelet_fft[idx] = make_float2(wavelet_val, imag_val);
        } else {
            wavelet_fft[idx] = make_float2(0.0f, 0.0f);
        }
    }
}

// Helper function to launch convolution kernel
void gpu_convolution(const float* d_signal, const float* d_filter, float* d_result,
                  int signal_len, int filter_len) {
    int result_len = signal_len + filter_len - 1;
    int block_size = 256;
    int grid_size = (result_len + block_size - 1) / block_size;
    
    size_t shared_mem_size = (signal_len + filter_len - 1) * sizeof(float);
    
    dwt_convolution_kernel<<<grid_size, block_size, shared_mem_size>>>(
        d_signal, d_filter, d_result, signal_len, filter_len);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Helper function to launch downsampling kernel
void gpu_downsample(const float* d_input, float* d_output, int input_len) {
    int output_len = input_len / 2;
    int block_size = 256;
    int grid_size = (output_len + block_size - 1) / block_size;
    
    downsample_kernel<<<grid_size, block_size>>>(d_input, d_output, input_len);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Helper function to launch upsampling kernel
void gpu_upsample(const float* d_input, float* d_output, int input_len) {
    int block_size = 256;
    int grid_size = (input_len + block_size - 1) / block_size;
    
    upsample_kernel<<<grid_size, block_size>>>(d_input, d_output, input_len);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Helper function to extend signal on GPU
void gpu_extend_signal(const float* d_signal, float* d_extended, int signal_len,
                     int extension_size, BoundaryMode mode) {
    int extended_len = signal_len + 2 * extension_size;
    int block_size = 256;
    int grid_size = (extended_len + block_size - 1) / block_size;
    
    extend_signal_kernel<<<grid_size, block_size>>>(
        d_signal, d_extended, signal_len, extension_size, static_cast<int>(mode));
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Helper function to combine signals on GPU
void gpu_combine_signals(const float* d_approx, const float* d_detail, float* d_result,
                       int signal_len) {
    int block_size = 256;
    int grid_size = (signal_len + block_size - 1) / block_size;
    
    dwt_combine_kernel<<<grid_size, block_size>>>(d_approx, d_detail, d_result, signal_len);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// GPU-accelerated Discrete Wavelet Transform
WaveletTransformResult cuda_discrete_wavelet_transform(
        const std::vector<float>& signal,
        const std::vector<float>& decomp_low_pass,
        const std::vector<float>& decomp_high_pass,
        int levels,
        BoundaryMode mode) {
    
    int signal_len = signal.size();
    int filter_len = decomp_low_pass.size();
    
    // Initialize the result structure
    WaveletTransformResult result;
    result.approximation_coefficients.resize(levels + 1);
    result.detail_coefficients.resize(levels);
    
    // The initial approximation is the input signal
    result.approximation_coefficients[0] = signal;
    
    // Allocate device memory
    float *d_signal, *d_extended, *d_approx_conv, *d_detail_conv;
    float *d_next_approx, *d_detail, *d_low_pass, *d_high_pass;
    
    CUDA_CHECK(cudaMalloc(&d_signal, signal_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_extended, (signal_len + 2 * (filter_len - 1)) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_approx_conv, (signal_len + filter_len - 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_detail_conv, (signal_len + filter_len - 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_approx, signal_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_detail, signal_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_low_pass, filter_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_high_pass, filter_len * sizeof(float)));
    
    // Copy filters to device
    CUDA_CHECK(cudaMemcpy(d_low_pass, decomp_low_pass.data(), filter_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_high_pass, decomp_high_pass.data(), filter_len * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy the initial signal to device
    CUDA_CHECK(cudaMemcpy(d_signal, signal.data(), signal_len * sizeof(float), cudaMemcpyHostToDevice));
    
    int current_signal_len = signal_len;
    
    for (int level = 0; level < levels; level++) {
        // Extend signal
        gpu_extend_signal(d_signal, d_extended, current_signal_len, filter_len - 1, mode);
        
        // Apply filters
        gpu_convolution(d_extended, d_low_pass, d_approx_conv, current_signal_len + 2 * (filter_len - 1), filter_len);
        gpu_convolution(d_extended, d_high_pass, d_detail_conv, current_signal_len + 2 * (filter_len - 1), filter_len);
        
        // Downsample
        int next_signal_len = current_signal_len / 2;
        gpu_downsample(d_approx_conv + (filter_len - 1), d_next_approx, current_signal_len);
        gpu_downsample(d_detail_conv + (filter_len - 1), d_detail, current_signal_len);
        
        // Copy results back to host
        std::vector<float> next_approx(next_signal_len);
        std::vector<float> detail(next_signal_len);
        
        CUDA_CHECK(cudaMemcpy(next_approx.data(), d_next_approx, next_signal_len * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(detail.data(), d_detail, next_signal_len * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Store results
        result.approximation_coefficients[level + 1] = next_approx;
        result.detail_coefficients[level] = detail;
        
        // Update current signal for next level
        current_signal_len = next_signal_len;
        CUDA_CHECK(cudaMemcpy(d_signal, next_approx.data(), current_signal_len * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_extended));
    CUDA_CHECK(cudaFree(d_approx_conv));
    CUDA_CHECK(cudaFree(d_detail_conv));
    CUDA_CHECK(cudaFree(d_next_approx));
    CUDA_CHECK(cudaFree(d_detail));
    CUDA_CHECK(cudaFree(d_low_pass));
    CUDA_CHECK(cudaFree(d_high_pass));
    
    return result;
}

// GPU-accelerated Inverse Discrete Wavelet Transform
std::vector<float> cuda_inverse_discrete_wavelet_transform(
        const WaveletTransformResult& transform_result,
        const std::vector<float>& recon_low_pass,
        const std::vector<float>& recon_high_pass,
        BoundaryMode mode) {
    
    int levels = transform_result.detail_coefficients.size();
    int filter_len = recon_low_pass.size();
    
    // Start with the coarsest approximation
    std::vector<float> current_approx = transform_result.approximation_coefficients[levels];
    int current_len = current_approx.size();
    
    // Allocate device memory
    float *d_current, *d_detail, *d_upsampled_approx, *d_upsampled_detail;
    float *d_extended_approx, *d_extended_detail, *d_approx_conv, *d_detail_conv;
    float *d_result, *d_low_pass, *d_high_pass;
    
    int max_signal_len = transform_result.approximation_coefficients[0].size();
    
    CUDA_CHECK(cudaMalloc(&d_current, max_signal_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_detail, max_signal_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_upsampled_approx, max_signal_len * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_upsampled_detail, max_signal_len * 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_extended_approx, (max_signal_len * 2 + 2 * (filter_len - 1)) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_extended_detail, (max_signal_len * 2 + 2 * (filter_len - 1)) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_approx_conv, (max_signal_len * 2 + filter_len - 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_detail_conv, (max_signal_len * 2 + filter_len - 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_result, max_signal_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_low_pass, filter_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_high_pass, filter_len * sizeof(float)));
    
    // Copy filters to device
    CUDA_CHECK(cudaMemcpy(d_low_pass, recon_low_pass.data(), filter_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_high_pass, recon_high_pass.data(), filter_len * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy initial approximation to device
    CUDA_CHECK(cudaMemcpy(d_current, current_approx.data(), current_len * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int level = levels - 1; level >= 0; level--) {
        // Get detail coefficients for this level
        const std::vector<float>& detail = transform_result.detail_coefficients[level];
        int next_len = transform_result.approximation_coefficients[level].size();
        
        // Copy detail to device
        CUDA_CHECK(cudaMemcpy(d_detail, detail.data(), detail.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        // Upsample
        gpu_upsample(d_current, d_upsampled_approx, current_len);
        gpu_upsample(d_detail, d_upsampled_detail, detail.size());
        
        // Extend signals
        gpu_extend_signal(d_upsampled_approx, d_extended_approx, current_len * 2, filter_len - 1, mode);
        gpu_extend_signal(d_upsampled_detail, d_extended_detail, detail.size() * 2, filter_len - 1, mode);
        
        // Apply reconstruction filters
        gpu_convolution(d_extended_approx, d_low_pass, d_approx_conv, current_len * 2 + 2 * (filter_len - 1), filter_len);
        gpu_convolution(d_extended_detail, d_high_pass, d_detail_conv, detail.size() * 2 + 2 * (filter_len - 1), filter_len);
        
        // Combine results
        gpu_combine_signals(
            d_approx_conv + (filter_len - 1),
            d_detail_conv + (filter_len - 1),
            d_result,
            next_len
        );
        
        // Update current approximation
        current_len = next_len;
        CUDA_CHECK(cudaMemcpy(d_current, d_result, current_len * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    // Copy final result back to host
    std::vector<float> result(current_len);
    CUDA_CHECK(cudaMemcpy(result.data(), d_current, current_len * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_detail));
    CUDA_CHECK(cudaFree(d_upsampled_approx));
    CUDA_CHECK(cudaFree(d_upsampled_detail));
    CUDA_CHECK(cudaFree(d_extended_approx));
    CUDA_CHECK(cudaFree(d_extended_detail));
    CUDA_CHECK(cudaFree(d_approx_conv));
    CUDA_CHECK(cudaFree(d_detail_conv));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_low_pass));
    CUDA_CHECK(cudaFree(d_high_pass));
    
    return result;
}

// GPU-accelerated Continuous Wavelet Transform
std::vector<std::vector<std::complex<float>>> cuda_continuous_wavelet_transform(
        const std::vector<float>& signal,
        const std::vector<float>& scales,
        WaveletFamily family) {
    
    int signal_len = signal.size();
    int num_scales = scales.size();
    
    // Initialize the result
    std::vector<std::vector<std::complex<float>>> coefficients(num_scales);
    for (int i = 0; i < num_scales; i++) {
        coefficients[i].resize(signal_len);
    }
    
    // Allocate device memory
    float *d_signal, *d_scales;
    float2 *d_coefficients;
    
    CUDA_CHECK(cudaMalloc(&d_signal, signal_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, num_scales * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_coefficients, signal_len * num_scales * sizeof(float2)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_signal, signal.data(), signal_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, scales.data(), num_scales * sizeof(float), cudaMemcpyHostToDevice));
    
    // Block and grid dimensions
    int block_size = 256;
    int grid_size = (signal_len + block_size - 1) / block_size;
    
    // Process each scale
    for (int i = 0; i < num_scales; i++) {
        if (family == WaveletFamily::MORLET) {
            float omega0 = 5.0f; // Center frequency for Morlet
            cwt_morlet_kernel<<<grid_size, block_size>>>(
                d_signal, d_coefficients, d_scales, signal_len, i, omega0);
        } else if (family == WaveletFamily::MEXICAN_HAT) {
            cwt_mexican_hat_kernel<<<grid_size, block_size>>>(
                d_signal, d_coefficients, d_scales, signal_len, i);
        } else {
            // Unsupported wavelet family, handle error
            CUDA_CHECK(cudaFree(d_signal));
            CUDA_CHECK(cudaFree(d_scales));
            CUDA_CHECK(cudaFree(d_coefficients));
            throw std::invalid_argument("Unsupported wavelet family for CWT on GPU");
        }
        
        CUDA_CHECK(cudaPeekAtLastError());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    std::vector<float2> host_coefficients(signal_len * num_scales);
    CUDA_CHECK(cudaMemcpy(host_coefficients.data(), d_coefficients, 
                        signal_len * num_scales * sizeof(float2), cudaMemcpyDeviceToHost));
    
    // Reformat the results
    for (int i = 0; i < num_scales; i++) {
        for (int j = 0; j < signal_len; j++) {
            float2 c = host_coefficients[i * signal_len + j];
            coefficients[i][j] = std::complex<float>(c.x, c.y);
        }
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_coefficients));
    
    return coefficients;
}

// FFT-based CWT implementation for efficiency (advanced optimization)
std::vector<std::vector<std::complex<float>>> cuda_fft_continuous_wavelet_transform(
        const std::vector<float>& signal,
        const std::vector<float>& scales,
        WaveletFamily family) {
    
    int signal_len = signal.size();
    int num_scales = scales.size();
    
    // For FFT efficiency, pad to next power of 2
    int padded_len = 1;
    while (padded_len < signal_len * 2) {
        padded_len *= 2;
    }
    
    // Initialize the result
    std::vector<std::vector<std::complex<float>>> coefficients(num_scales);
    for (int i = 0; i < num_scales; i++) {
        coefficients[i].resize(signal_len);
    }
    
    // Allocate device memory
    float *d_signal, *d_scales;
    cufftComplex *d_signal_fft, *d_wavelet_fft, *d_result_fft;
    float2 *d_coefficients;
    
    CUDA_CHECK(cudaMalloc(&d_signal, padded_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales, num_scales * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_signal_fft, padded_len * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_wavelet_fft, padded_len * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_result_fft, padded_len * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_coefficients, signal_len * num_scales * sizeof(float2)));
    
    // Initialize padded signal
    CUDA_CHECK(cudaMemset(d_signal, 0, padded_len * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_signal, signal.data(), signal_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scales, scales.data(), num_scales * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create FFT plans
    cufftHandle plan_forward, plan_inverse;
    CUDA_CHECK(cufftPlan1d(&plan_forward, padded_len, CUFFT_R2C, 1));
    CUDA_CHECK(cufftPlan1d(&plan_inverse, padded_len, CUFFT_C2C, 1));
    
    // Execute forward FFT on signal
    CUDA_CHECK(cufftExecR2C(plan_forward, d_signal, d_signal_fft));
    
    // Block and grid dimensions
    int block_size = 256;
    int grid_size = (padded_len + block_size - 1) / block_size;
    
    // Process each scale
    for (int i = 0; i < num_scales; i++) {
        float scale = scales[i];
        int wavelet_len = std::min(padded_len, static_cast<int>(scale * 10)); // Adjust as needed
        
        // Generate wavelet in frequency domain
        cwt_fft_prepare_kernel<<<grid_size, block_size>>>(
            NULL, d_wavelet_fft, scale, padded_len, wavelet_len, static_cast<int>(family));
        CUDA_CHECK(cudaPeekAtLastError());
        
        // Execute forward FFT on wavelet
        CUDA_CHECK(cufftExecC2C(plan_forward, d_wavelet_fft, d_wavelet_fft, CUFFT_FORWARD));
        
        // Multiply in frequency domain (complex multiplication)
        // This would require a custom kernel for complex multiplication
        
        // Execute inverse FFT
        CUDA_CHECK(cufftExecC2C(plan_inverse, d_result_fft, d_result_fft, CUFFT_INVERSE));
        
        // Extract the relevant part of the result
        // This would require a custom kernel to copy and normalize
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy results back to host
    std::vector<float2> host_coefficients(signal_len * num_scales);
    CUDA_CHECK(cudaMemcpy(host_coefficients.data(), d_coefficients, 
                        signal_len * num_scales * sizeof(float2), cudaMemcpyDeviceToHost));
    
    // Reformat the results
    for (int i = 0; i < num_scales; i++) {
        for (int j = 0; j < signal_len; j++) {
            float2 c = host_coefficients[i * signal_len + j];
            coefficients[i][j] = std::complex<float>(c.x, c.y);
        }
    }
    
    // Clean up
    CUDA_CHECK(cufftDestroy(plan_forward));
    CUDA_CHECK(cufftDestroy(plan_inverse));
    CUDA_CHECK(cudaFree(d_signal));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_signal_fft));
    CUDA_CHECK(cudaFree(d_wavelet_fft));
    CUDA_CHECK(cudaFree(d_result_fft));
    CUDA_CHECK(cudaFree(d_coefficients));
    
    return coefficients;
}

// GPU-accelerated Wavelet Packet Transform
WaveletPacketResult cuda_wavelet_packet_transform(
        const std::vector<float>& signal,
        const std::vector<float>& decomp_low_pass,
        const std::vector<float>& decomp_high_pass,
        int levels,
        BoundaryMode mode) {
    
    int signal_len = signal.size();
    int filter_len = decomp_low_pass.size();
    
    // Initialize the result structure
    WaveletPacketResult result;
    result.coefficients.resize(levels + 1);
    
    // The first level has only one node - the original signal
    result.coefficients[0].resize(1);
    result.coefficients[0][0] = signal;
    
    // Allocate device memory for filters
    float *d_low_pass, *d_high_pass;
    CUDA_CHECK(cudaMalloc(&d_low_pass, filter_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_high_pass, filter_len * sizeof(float)));
    
    // Copy filters to device
    CUDA_CHECK(cudaMemcpy(d_low_pass, decomp_low_pass.data(), filter_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_high_pass, decomp_high_pass.data(), filter_len * sizeof(float), cudaMemcpyHostToDevice));
    
    // Perform wavelet packet decomposition
    for (int level = 0; level < levels; level++) {
        int num_nodes = 1 << level; // 2^level
        int next_num_nodes = 1 << (level + 1); // 2^(level+1)
        
        result.coefficients[level + 1].resize(next_num_nodes);
        
        // Process each node at the current level
        for (int node = 0; node < num_nodes; node++) {
            const std::vector<float>& current_signal = result.coefficients[level][node];
            int current_len = current_signal.size();
            
            // Allocate device memory for this node
            float *d_signal, *d_extended, *d_approx_conv, *d_detail_conv;
            float *d_approx, *d_detail;
            
            CUDA_CHECK(cudaMalloc(&d_signal, current_len * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_extended, (current_len + 2 * (filter_len - 1)) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_approx_conv, (current_len + filter_len - 1) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_detail_conv, (current_len + filter_len - 1) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_approx, (current_len / 2) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_detail, (current_len / 2) * sizeof(float)));
            
            // Copy current signal to device
            CUDA_CHECK(cudaMemcpy(d_signal, current_signal.data(), current_len * sizeof(float), cudaMemcpyHostToDevice));
            
            // Extend signal
            gpu_extend_signal(d_signal, d_extended, current_len, filter_len - 1, mode);
            
            // Apply filters
            gpu_convolution(d_extended, d_low_pass, d_approx_conv, current_len + 2 * (filter_len - 1), filter_len);
            gpu_convolution(d_extended, d_high_pass, d_detail_conv, current_len + 2 * (filter_len - 1), filter_len);
            
            // Downsample
            gpu_downsample(d_approx_conv + (filter_len - 1), d_approx, current_len);
            gpu_downsample(d_detail_conv + (filter_len - 1), d_detail, current_len);
            
            // Copy results back to host
            int next_len = current_len / 2;
            std::vector<float> approx(next_len);
            std::vector<float> detail(next_len);
            
            CUDA_CHECK(cudaMemcpy(approx.data(), d_approx, next_len * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(detail.data(), d_detail, next_len * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Store results
            result.coefficients[level + 1][2 * node] = approx;
            result.coefficients[level + 1][2 * node + 1] = detail;
            
            // Free device memory for this node
            CUDA_CHECK(cudaFree(d_signal));
            CUDA_CHECK(cudaFree(d_extended));
            CUDA_CHECK(cudaFree(d_approx_conv));
            CUDA_CHECK(cudaFree(d_detail_conv));
            CUDA_CHECK(cudaFree(d_approx));
            CUDA_CHECK(cudaFree(d_detail));
        }
    }
    
    // Free device memory for filters
    CUDA_CHECK(cudaFree(d_low_pass));
    CUDA_CHECK(cudaFree(d_high_pass));
    
    return result;
}

// GPU-accelerated Inverse Wavelet Packet Transform
std::vector<float> cuda_inverse_wavelet_packet_transform(
        const WaveletPacketResult& packet_result,
        const std::vector<float>& recon_low_pass,
        const std::vector<float>& recon_high_pass,
        BoundaryMode mode) {
    
    if (packet_result.coefficients.empty()) {
        throw std::invalid_argument("Packet result structure cannot be empty");
    }
    
    int levels = packet_result.coefficients.size() - 1;
    int filter_len = recon_low_pass.size();
    
    // Create a copy of the result
    WaveletPacketResult result = packet_result;
    
    // Allocate device memory for filters
    float *d_low_pass, *d_high_pass;
    CUDA_CHECK(cudaMalloc(&d_low_pass, filter_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_high_pass, filter_len * sizeof(float)));
    
    // Copy filters to device
    CUDA_CHECK(cudaMemcpy(d_low_pass, recon_low_pass.data(), filter_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_high_pass, recon_high_pass.data(), filter_len * sizeof(float), cudaMemcpyHostToDevice));
    
    // Perform reconstruction from bottom to top
    for (int level = levels; level > 0; level--) {
        int num_nodes = 1 << (level - 1); // 2^(level-1)
        
        // Process each node at the level we're reconstructing to
        for (int node = 0; node < num_nodes; node++) {
            // Get the left and right children
            const std::vector<float>& approx = result.coefficients[level][2 * node];
            const std::vector<float>& detail = result.coefficients[level][2 * node + 1];
            int approx_len = approx.size();
            
            // Allocate device memory for this node
            float *d_approx, *d_detail, *d_upsampled_approx, *d_upsampled_detail;
            float *d_extended_approx, *d_extended_detail, *d_approx_conv, *d_detail_conv;
            float *d_combined;
            
            CUDA_CHECK(cudaMalloc(&d_approx, approx_len * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_detail, approx_len * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_upsampled_approx, (approx_len * 2) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_upsampled_detail, (approx_len * 2) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_extended_approx, (approx_len * 2 + 2 * (filter_len - 1)) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_extended_detail, (approx_len * 2 + 2 * (filter_len - 1)) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_approx_conv, (approx_len * 2 + filter_len - 1) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_detail_conv, (approx_len * 2 + filter_len - 1) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_combined, (approx_len * 2) * sizeof(float)));
            
            // Copy data to device
            CUDA_CHECK(cudaMemcpy(d_approx, approx.data(), approx_len * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_detail, detail.data(), approx_len * sizeof(float), cudaMemcpyHostToDevice));
            
            // Upsample
            gpu_upsample(d_approx, d_upsampled_approx, approx_len);
            gpu_upsample(d_detail, d_upsampled_detail, approx_len);
            
            // Extend signals
            gpu_extend_signal(d_upsampled_approx, d_extended_approx, approx_len * 2, filter_len - 1, mode);
            gpu_extend_signal(d_upsampled_detail, d_extended_detail, approx_len * 2, filter_len - 1, mode);
            
            // Apply reconstruction filters
            gpu_convolution(d_extended_approx, d_low_pass, d_approx_conv, approx_len * 2 + 2 * (filter_len - 1), filter_len);
            gpu_convolution(d_extended_detail, d_high_pass, d_detail_conv, approx_len * 2 + 2 * (filter_len - 1), filter_len);
            
            // Combine results
            int combined_len = approx_len * 2;
            gpu_combine_signals(
                d_approx_conv + (filter_len - 1),
                d_detail_conv + (filter_len - 1),
                d_combined,
                combined_len
            );
            
            // Copy result back to host
            std::vector<float> combined(combined_len);
            CUDA_CHECK(cudaMemcpy(combined.data(), d_combined, combined_len * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Store result
            result.coefficients[level - 1][node] = combined;
            
            // Free device memory for this node
            CUDA_CHECK(cudaFree(d_approx));
            CUDA_CHECK(cudaFree(d_detail));
            CUDA_CHECK(cudaFree(d_upsampled_approx));
            CUDA_CHECK(cudaFree(d_upsampled_detail));
            CUDA_CHECK(cudaFree(d_extended_approx));
            CUDA_CHECK(cudaFree(d_extended_detail));
            CUDA_CHECK(cudaFree(d_approx_conv));
            CUDA_CHECK(cudaFree(d_detail_conv));
            CUDA_CHECK(cudaFree(d_combined));
        }
    }
    
    // Free device memory for filters
    CUDA_CHECK(cudaFree(d_low_pass));
    CUDA_CHECK(cudaFree(d_high_pass));
    
    // The final result is the single node at the top level
    return result.coefficients[0][0];
}

} // namespace signal_processing