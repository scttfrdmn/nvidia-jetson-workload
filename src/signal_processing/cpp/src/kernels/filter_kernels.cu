/**
 * @file filter_kernels.cu
 * @brief CUDA kernels for digital filtering operations
 * 
 * This file contains CUDA kernels and device functions for:
 * - FIR filtering
 * - IIR filtering
 * - Median and other non-linear filters
 * - Adaptive filtering
 * - Multirate filtering
 * 
 * Kernels are optimized for different GPU architectures:
 * - SM 8.7 for Jetson Orin NX
 * - SM 7.5 for AWS T4G
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

#include <cmath>
#include <complex>

namespace cg = cooperative_groups;

namespace signal_processing {
namespace kernels {

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

// Choose optimal filter implementation based on SM architecture
template<typename FilterFunc>
inline FilterFunc select_filter_implementation(
    cudaDeviceProp& props,
    FilterFunc sm_87_impl,
    FilterFunc sm_75_impl,
    FilterFunc default_impl) {
    
    if (props.major == 8 && props.minor == 7) {
        return sm_87_impl;
    } else if (props.major == 7 && props.minor == 5) {
        return sm_75_impl;
    } else {
        return default_impl;
    }
}

//------------------------------------------------------------------------------
// FIR Filter Kernels
//------------------------------------------------------------------------------

/**
 * @brief Naive FIR filter kernel
 * 
 * Basic implementation for any architecture
 */
__global__ void fir_filter_kernel(
    const float* __restrict__ input,
    const float* __restrict__ coeffs,
    float* __restrict__ output,
    int input_size,
    int filter_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_size) {
        float sum = 0.0f;
        
        for (int i = 0; i < filter_size; ++i) {
            int input_idx = idx - i;
            
            if (input_idx >= 0) {
                sum += coeffs[i] * input[input_idx];
            }
        }
        
        output[idx] = sum;
    }
}

/**
 * @brief Optimized FIR filter kernel for SM 8.7 architecture
 * 
 * Uses shared memory and thread coarsening for Jetson Orin NX
 */
__global__ void __launch_bounds__(128, 8)
fir_filter_sm87_kernel(
    const float* __restrict__ input,
    const float* __restrict__ coeffs,
    float* __restrict__ output,
    int input_size,
    int filter_size) {
    
    extern __shared__ float shared_mem[];
    
    // Load filter coefficients into shared memory
    if (threadIdx.x < filter_size) {
        shared_mem[threadIdx.x] = coeffs[threadIdx.x];
    }
    
    __syncthreads();
    
    // Each thread processes multiple output elements
    const int items_per_thread = 4;
    const int start_idx = (blockIdx.x * blockDim.x + threadIdx.x) * items_per_thread;
    
    for (int item = 0; item < items_per_thread; ++item) {
        int idx = start_idx + item;
        
        if (idx < input_size) {
            float sum = 0.0f;
            
            // Manual loop unrolling for better instruction-level parallelism
            int i = 0;
            for (; i + 3 < filter_size; i += 4) {
                int input_idx0 = idx - i;
                int input_idx1 = idx - (i + 1);
                int input_idx2 = idx - (i + 2);
                int input_idx3 = idx - (i + 3);
                
                float in0 = (input_idx0 >= 0) ? input[input_idx0] : 0.0f;
                float in1 = (input_idx1 >= 0) ? input[input_idx1] : 0.0f;
                float in2 = (input_idx2 >= 0) ? input[input_idx2] : 0.0f;
                float in3 = (input_idx3 >= 0) ? input[input_idx3] : 0.0f;
                
                sum += shared_mem[i] * in0;
                sum += shared_mem[i + 1] * in1;
                sum += shared_mem[i + 2] * in2;
                sum += shared_mem[i + 3] * in3;
            }
            
            // Handle remaining elements
            for (; i < filter_size; ++i) {
                int input_idx = idx - i;
                if (input_idx >= 0) {
                    sum += shared_mem[i] * input[input_idx];
                }
            }
            
            output[idx] = sum;
        }
    }
}

/**
 * @brief Optimized FIR filter kernel for SM 7.5 architecture
 * 
 * Uses shared memory and warp-level optimizations for AWS T4G
 */
__global__ void __launch_bounds__(256, 4)
fir_filter_sm75_kernel(
    const float* __restrict__ input,
    const float* __restrict__ coeffs,
    float* __restrict__ output,
    int input_size,
    int filter_size) {
    
    extern __shared__ float shared_mem[];
    
    // Load filter coefficients into shared memory
    if (threadIdx.x < filter_size) {
        shared_mem[threadIdx.x] = coeffs[threadIdx.x];
    }
    
    __syncthreads();
    
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_size) {
        float sum = 0.0f;
        
        // Use registers for frequently accessed input values
        float input_vals[16];  // Cache up to 16 input values
        const int cache_size = min(16, filter_size);
        
        for (int i = 0; i < cache_size; ++i) {
            int input_idx = idx - i;
            input_vals[i] = (input_idx >= 0) ? input[input_idx] : 0.0f;
        }
        
        // Compute using cached values
        for (int i = 0; i < cache_size; ++i) {
            sum += shared_mem[i] * input_vals[i];
        }
        
        // Handle remaining elements
        for (int i = cache_size; i < filter_size; ++i) {
            int input_idx = idx - i;
            if (input_idx >= 0) {
                sum += shared_mem[i] * input[input_idx];
            }
        }
        
        output[idx] = sum;
    }
}

//------------------------------------------------------------------------------
// IIR Filter Kernels
//------------------------------------------------------------------------------

/**
 * @brief IIR filter kernel (Direct Form I)
 * 
 * Due to sequential dependencies, IIR filters are less suited for
 * parallelization compared to FIR filters. This kernel processes
 * multiple independent filters in parallel.
 */
__global__ void iir_filter_direct_form1_kernel(
    const float* __restrict__ input,
    const float* __restrict__ a_coeffs,
    const float* __restrict__ b_coeffs,
    float* __restrict__ output,
    int input_size,
    int a_size,
    int b_size,
    int num_filters) {
    
    int filter_idx = blockIdx.x;
    int start_idx = threadIdx.x;
    
    if (filter_idx < num_filters) {
        // Offsets for this filter
        int input_offset = filter_idx * input_size;
        int output_offset = filter_idx * input_size;
        int a_offset = filter_idx * a_size;
        int b_offset = filter_idx * b_size;
        
        // Process each sample in sequence due to dependencies
        for (int i = start_idx; i < input_size; i += blockDim.x) {
            float sum = 0.0f;
            
            // Apply feedforward (input) terms
            for (int j = 0; j < b_size; ++j) {
                if (i - j >= 0) {
                    sum += b_coeffs[b_offset + j] * input[input_offset + i - j];
                }
            }
            
            // Apply feedback (output) terms
            for (int j = 1; j < a_size; ++j) {
                if (i - j >= 0) {
                    sum -= a_coeffs[a_offset + j] * output[output_offset + i - j];
                }
            }
            
            // Normalize by a[0]
            output[output_offset + i] = sum / a_coeffs[a_offset];
        }
    }
}

/**
 * @brief IIR filter kernel (Direct Form II)
 * 
 * Better numerical properties than Direct Form I
 */
__global__ void iir_filter_direct_form2_kernel(
    const float* __restrict__ input,
    const float* __restrict__ a_coeffs,
    const float* __restrict__ b_coeffs,
    float* __restrict__ output,
    float* __restrict__ state,
    int input_size,
    int a_size,
    int b_size,
    int num_filters) {
    
    int filter_idx = blockIdx.x;
    int start_idx = threadIdx.x;
    
    if (filter_idx < num_filters) {
        // Offsets for this filter
        int input_offset = filter_idx * input_size;
        int output_offset = filter_idx * input_size;
        int a_offset = filter_idx * a_size;
        int b_offset = filter_idx * b_size;
        int state_offset = filter_idx * (max(a_size, b_size) - 1);
        
        // Process each sample in sequence
        for (int i = start_idx; i < input_size; i += blockDim.x) {
            // Calculate new state
            float new_state = input[input_offset + i];
            
            for (int j = 1; j < a_size; ++j) {
                if (i - j >= 0) {
                    new_state -= a_coeffs[a_offset + j] * state[state_offset + j - 1];
                }
            }
            
            // Calculate output
            float sum = b_coeffs[b_offset] * new_state;
            
            for (int j = 1; j < b_size; ++j) {
                if (j - 1 < a_size - 1) {
                    sum += b_coeffs[b_offset + j] * state[state_offset + j - 1];
                }
            }
            
            // Update state
            for (int j = a_size - 2; j > 0; --j) {
                state[state_offset + j] = state[state_offset + j - 1];
            }
            
            state[state_offset] = new_state;
            
            // Store output
            output[output_offset + i] = sum;
        }
    }
}

/**
 * @brief Second-order sections (SOS) IIR filter kernel
 * 
 * Implements cascaded biquad filters for numerical stability
 */
__global__ void iir_filter_sos_kernel(
    const float* __restrict__ input,
    const float* __restrict__ sos_coeffs,  // [b0, b1, b2, a0, a1, a2]
    float* __restrict__ output,
    float* __restrict__ state,
    int input_size,
    int num_sections,
    int num_filters) {
    
    int filter_idx = blockIdx.x;
    int start_idx = threadIdx.x;
    
    if (filter_idx < num_filters) {
        // Offsets for this filter
        int input_offset = filter_idx * input_size;
        int output_offset = filter_idx * input_size;
        int sos_offset = filter_idx * num_sections * 6;  // 6 coeffs per section
        int state_offset = filter_idx * num_sections * 2;  // 2 states per section
        
        // Create temporary buffer for section I/O
        float temp_input[2048];  // Fixed size for now, should be dynamically allocated
        float temp_output[2048];
        
        // Copy input to temporary buffer
        for (int i = 0; i < input_size; ++i) {
            temp_input[i] = input[input_offset + i];
        }
        
        // Process each section
        for (int section = 0; section < num_sections; ++section) {
            int section_offset = sos_offset + section * 6;
            int section_state_offset = state_offset + section * 2;
            
            // Get coefficients for this section
            float b0 = sos_coeffs[section_offset];
            float b1 = sos_coeffs[section_offset + 1];
            float b2 = sos_coeffs[section_offset + 2];
            float a0 = sos_coeffs[section_offset + 3];  // Typically 1.0
            float a1 = sos_coeffs[section_offset + 4];
            float a2 = sos_coeffs[section_offset + 5];
            
            // Get states for this section
            float w1 = state[section_state_offset];
            float w2 = state[section_state_offset + 1];
            
            // Process each sample in sequence
            for (int i = start_idx; i < input_size; i += blockDim.x) {
                float x = temp_input[i];
                
                // Direct Form II Transposed
                float y = b0 * x + w1;
                w1 = b1 * x - a1 * y + w2;
                w2 = b2 * x - a2 * y;
                
                temp_output[i] = y;
            }
            
            // Update states
            state[section_state_offset] = w1;
            state[section_state_offset + 1] = w2;
            
            // Swap buffers for next section
            for (int i = 0; i < input_size; ++i) {
                temp_input[i] = temp_output[i];
            }
        }
        
        // Copy result to output
        for (int i = 0; i < input_size; ++i) {
            output[output_offset + i] = temp_input[i];
        }
    }
}

//------------------------------------------------------------------------------
// Median Filter Kernels
//------------------------------------------------------------------------------

/**
 * @brief Naive median filter kernel
 * 
 * Basic implementation for any architecture
 */
__global__ void median_filter_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size,
    int kernel_size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_size) {
        // Define window
        int half_kernel = kernel_size / 2;
        float window[64];  // Fixed size for now
        
        // Fill window with values from input
        for (int i = 0; i < kernel_size; ++i) {
            int input_idx = idx - half_kernel + i;
            
            // Handle boundaries with reflection
            if (input_idx < 0) {
                input_idx = -input_idx;
            } else if (input_idx >= input_size) {
                input_idx = 2 * input_size - input_idx - 2;
            }
            
            window[i] = input[input_idx];
        }
        
        // Sort window to find median
        // Simple bubble sort for small windows
        for (int i = 0; i < kernel_size - 1; ++i) {
            for (int j = 0; j < kernel_size - i - 1; ++j) {
                if (window[j] > window[j + 1]) {
                    float temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
            }
        }
        
        // Output median value
        output[idx] = window[half_kernel];
    }
}

/**
 * @brief Optimized median filter kernel for SM 8.7 architecture
 * 
 * Uses shared memory and thread coarsening for Jetson Orin NX
 */
__global__ void __launch_bounds__(128, 8)
median_filter_sm87_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size,
    int kernel_size) {
    
    extern __shared__ float shared_mem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_kernel = kernel_size / 2;
    
    if (idx < input_size) {
        // Each thread loads own window into shared memory
        float* window = shared_mem + threadIdx.x * kernel_size;
        
        // Fill window with values from input
        for (int i = 0; i < kernel_size; ++i) {
            int input_idx = idx - half_kernel + i;
            
            // Handle boundaries with reflection
            if (input_idx < 0) {
                input_idx = -input_idx;
            } else if (input_idx >= input_size) {
                input_idx = 2 * input_size - input_idx - 2;
            }
            
            window[i] = input[input_idx];
        }
        
        // Use bitonic sort for better performance
        for (int k = 2; k <= kernel_size; k *= 2) {
            for (int j = k / 2; j > 0; j /= 2) {
                for (int i = 0; i < kernel_size; ++i) {
                    int ixj = i ^ j;
                    if (ixj > i && ixj < kernel_size) {
                        if ((i & k) == 0) {
                            if (window[i] > window[ixj]) {
                                float temp = window[i];
                                window[i] = window[ixj];
                                window[ixj] = temp;
                            }
                        } else {
                            if (window[i] < window[ixj]) {
                                float temp = window[i];
                                window[i] = window[ixj];
                                window[ixj] = temp;
                            }
                        }
                    }
                }
            }
        }
        
        // Output median value
        output[idx] = window[half_kernel];
    }
}

/**
 * @brief Optimized median filter kernel for SM 7.5 architecture
 * 
 * Uses warp-level operations for AWS T4G
 */
__global__ void __launch_bounds__(256, 4)
median_filter_sm75_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size,
    int kernel_size) {
    
    extern __shared__ float shared_mem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_kernel = kernel_size / 2;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (idx < input_size) {
        // Use shared memory for window data
        float* window = shared_mem + warp_id * kernel_size;
        
        // Collaborate within warp to load window data
        for (int i = lane_id; i < kernel_size; i += 32) {
            int input_idx = idx - half_kernel + i;
            
            // Handle boundaries with reflection
            if (input_idx < 0) {
                input_idx = -input_idx;
            } else if (input_idx >= input_size) {
                input_idx = 2 * input_size - input_idx - 2;
            }
            
            window[i] = input[input_idx];
        }
        
        __syncwarp();
        
        // All threads in warp collaborate on sorting (first thread gets result)
        if (lane_id == 0) {
            // Simple insertion sort for small windows
            for (int i = 1; i < kernel_size; ++i) {
                float key = window[i];
                int j = i - 1;
                
                while (j >= 0 && window[j] > key) {
                    window[j + 1] = window[j];
                    j--;
                }
                
                window[j + 1] = key;
            }
            
            // Output median value
            output[idx] = window[half_kernel];
        }
    }
}

//------------------------------------------------------------------------------
// Adaptive Filter Kernels
//------------------------------------------------------------------------------

/**
 * @brief LMS (Least Mean Squares) adaptive filter kernel
 */
__global__ void lms_filter_kernel(
    const float* __restrict__ input,
    const float* __restrict__ desired,
    float* __restrict__ output,
    float* __restrict__ error,
    float* __restrict__ weights,
    int input_size,
    int filter_length,
    float step_size) {
    
    extern __shared__ float shared_mem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_size) {
        // Each thread processes one output sample
        float out = 0.0f;
        
        // Load weights to shared memory
        if (threadIdx.x < filter_length) {
            shared_mem[threadIdx.x] = weights[threadIdx.x];
        }
        
        __syncthreads();
        
        // Apply filter to compute output
        for (int i = 0; i < filter_length; ++i) {
            int input_idx = idx - i;
            if (input_idx >= 0) {
                out += shared_mem[i] * input[input_idx];
            }
        }
        
        // Compute error
        float err = desired[idx] - out;
        
        // Update weights (with atomic operations to handle race conditions)
        for (int i = 0; i < filter_length; ++i) {
            int input_idx = idx - i;
            if (input_idx >= 0) {
                float update = step_size * err * input[input_idx];
                atomicAdd(&weights[i], update);
            }
        }
        
        // Store output and error
        output[idx] = out;
        error[idx] = err;
    }
}

/**
 * @brief NLMS (Normalized Least Mean Squares) adaptive filter kernel
 */
__global__ void nlms_filter_kernel(
    const float* __restrict__ input,
    const float* __restrict__ desired,
    float* __restrict__ output,
    float* __restrict__ error,
    float* __restrict__ weights,
    int input_size,
    int filter_length,
    float step_size,
    float epsilon) {
    
    extern __shared__ float shared_mem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < input_size) {
        // Each thread processes one output sample
        float out = 0.0f;
        
        // Load weights to shared memory
        if (threadIdx.x < filter_length) {
            shared_mem[threadIdx.x] = weights[threadIdx.x];
        }
        
        __syncthreads();
        
        // Apply filter to compute output
        for (int i = 0; i < filter_length; ++i) {
            int input_idx = idx - i;
            if (input_idx >= 0) {
                out += shared_mem[i] * input[input_idx];
            }
        }
        
        // Compute error
        float err = desired[idx] - out;
        
        // Compute input energy
        float energy = 0.0f;
        for (int i = 0; i < filter_length; ++i) {
            int input_idx = idx - i;
            if (input_idx >= 0) {
                energy += input[input_idx] * input[input_idx];
            }
        }
        
        // Normalize step size
        float norm_step = energy > epsilon ? step_size / energy : step_size / epsilon;
        
        // Update weights (with atomic operations)
        for (int i = 0; i < filter_length; ++i) {
            int input_idx = idx - i;
            if (input_idx >= 0) {
                float update = norm_step * err * input[input_idx];
                atomicAdd(&weights[i], update);
            }
        }
        
        // Store output and error
        output[idx] = out;
        error[idx] = err;
    }
}

//------------------------------------------------------------------------------
// Multirate Filter Kernels
//------------------------------------------------------------------------------

/**
 * @brief Decimation kernel (downsample)
 */
__global__ void decimate_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size,
    int decimation_factor) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = (input_size + decimation_factor - 1) / decimation_factor;
    
    if (idx < output_size) {
        int input_idx = idx * decimation_factor;
        if (input_idx < input_size) {
            output[idx] = input[input_idx];
        }
    }
}

/**
 * @brief Interpolation kernel (upsample with zero insertion)
 */
__global__ void interpolate_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int input_size,
    int interpolation_factor) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size * interpolation_factor;
    
    if (idx < output_size) {
        if (idx % interpolation_factor == 0) {
            output[idx] = input[idx / interpolation_factor] * interpolation_factor;
        } else {
            output[idx] = 0.0f;
        }
    }
}

/**
 * @brief Polyphase FIR filter kernel
 * 
 * Efficient implementation for multirate filtering
 */
__global__ void polyphase_filter_kernel(
    const float* __restrict__ input,
    const float* __restrict__ coeffs,
    float* __restrict__ output,
    int input_size,
    int filter_size,
    int rate,
    bool is_interpolation) {
    
    extern __shared__ float shared_coeffs[];
    
    // Load coefficients into shared memory
    if (threadIdx.x < filter_size) {
        shared_coeffs[threadIdx.x] = coeffs[threadIdx.x];
    }
    
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (is_interpolation) {
        // Interpolation: Each output sample
        int output_size = input_size * rate;
        
        if (idx < output_size) {
            int phase = idx % rate;
            int input_idx = idx / rate;
            
            float sum = 0.0f;
            
            // Apply appropriate polyphase branch
            for (int i = 0; i < filter_size / rate; ++i) {
                int coeff_idx = i * rate + phase;
                int in_idx = input_idx - i;
                
                if (in_idx >= 0 && coeff_idx < filter_size) {
                    sum += shared_coeffs[coeff_idx] * input[in_idx];
                }
            }
            
            output[idx] = sum;
        }
    } else {
        // Decimation: Each output sample
        int output_size = (input_size + rate - 1) / rate;
        
        if (idx < output_size) {
            int input_idx = idx * rate;
            
            float sum = 0.0f;
            
            // Apply filter
            for (int i = 0; i < filter_size; ++i) {
                int in_idx = input_idx - i;
                
                if (in_idx >= 0 && in_idx < input_size) {
                    sum += shared_coeffs[i] * input[in_idx];
                }
            }
            
            output[idx] = sum;
        }
    }
}

} // namespace kernels
} // namespace signal_processing