/**
 * @file test_multirate_filter_cuda.cpp
 * @brief Test program for verifying MultirateFilter CUDA integration
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include "signal_processing/digital_filtering.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace signal_processing;

// Generate a chirp test signal (frequency sweep)
std::vector<float> generate_chirp_signal(int size, float start_freq, float end_freq, float sample_rate) {
    std::vector<float> signal(size);
    float rate = (end_freq - start_freq) / size;
    
    for (int i = 0; i < size; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float inst_freq = start_freq + i * rate;
        signal[i] = std::sin(2 * M_PI * t * inst_freq);
    }
    return signal;
}

// Print performance statistics
void print_performance(const std::string& name, size_t input_size, size_t output_size,
                       std::chrono::microseconds duration) {
    double throughput = static_cast<double>(input_size) / duration.count();
    std::cout << name << " Performance:" << std::endl;
    std::cout << "  Input Size: " << input_size << " samples" << std::endl;
    std::cout << "  Output Size: " << output_size << " samples" << std::endl;
    std::cout << "  Processing Time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput << " samples/μs" << std::endl;
    std::cout << "              " << throughput * 1e6 << " samples/s" << std::endl;
    std::cout << std::endl;
}

int main() {
    // Test parameters
    const int signal_size = 1000000;  // 1M samples
    const float sample_rate = 44100.0f;  // 44.1 kHz
    const float start_freq = 0.0f;    // Start at DC
    const float end_freq = 20000.0f;  // End at 20 kHz
    const int interpolation_factor = 4;
    const int decimation_factor = 4;
    
    // Generate test signal
    std::cout << "Generating test signal..." << std::endl;
    auto signal = generate_chirp_signal(signal_size, start_freq, end_freq, sample_rate);
    
    // Create filter parameters
    FIRFilterParams filter_params;
    filter_params.num_taps = 101;
    filter_params.filter_type = FilterType::LOWPASS;
    filter_params.cutoff_freqs = {0.4f * sample_rate / 2.0f};  // 0.4 * Nyquist
    filter_params.window_type = WindowType::HAMMING;
    
    // Create multirate filter parameters
    MultirateFilterParams params;
    params.interpolation_factor = interpolation_factor;
    params.decimation_factor = decimation_factor;
    params.filter_params = filter_params;
    
    // Test CPU implementation
    std::cout << "Testing CPU implementation..." << std::endl;
    
    // Upsample test
    MultirateFilter cpu_filter_up(params, -1);  // -1 means CPU
    auto cpu_start_up = std::chrono::high_resolution_clock::now();
    auto cpu_output_up = cpu_filter_up.upsample(signal);
    auto cpu_end_up = std::chrono::high_resolution_clock::now();
    auto cpu_duration_up = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end_up - cpu_start_up);
    
    print_performance("CPU Upsampling", signal.size(), cpu_output_up.size(), cpu_duration_up);
    
    // Downsample test
    MultirateFilter cpu_filter_down(params, -1);  // -1 means CPU
    auto cpu_start_down = std::chrono::high_resolution_clock::now();
    auto cpu_output_down = cpu_filter_down.downsample(signal);
    auto cpu_end_down = std::chrono::high_resolution_clock::now();
    auto cpu_duration_down = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end_down - cpu_start_down);
    
    print_performance("CPU Downsampling", signal.size(), cpu_output_down.size(), cpu_duration_down);
    
    // Test GPU implementation (if available)
    try {
        std::cout << "Testing GPU implementation..." << std::endl;
        
        // Upsample test
        MultirateFilter gpu_filter_up(params, 0);  // 0 means first GPU
        auto gpu_start_up = std::chrono::high_resolution_clock::now();
        auto gpu_output_up = gpu_filter_up.upsample(signal);
        auto gpu_end_up = std::chrono::high_resolution_clock::now();
        auto gpu_duration_up = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end_up - gpu_start_up);
        
        print_performance("GPU Upsampling", signal.size(), gpu_output_up.size(), gpu_duration_up);
        
        // Compare CPU and GPU results for upsampling
        double max_error_up = 0.0;
        if (cpu_output_up.size() == gpu_output_up.size()) {
            for (size_t i = 0; i < cpu_output_up.size(); ++i) {
                max_error_up = std::max(max_error_up, std::abs(cpu_output_up[i] - gpu_output_up[i]));
            }
            
            std::cout << "Upsampling Verification:" << std::endl;
            std::cout << "  Maximum Error: " << max_error_up << std::endl;
            std::cout << "  Implementation Matches: " << (max_error_up < 1e-5 ? "Yes" : "No") << std::endl;
            
            // Calculate speedup
            double speedup_up = static_cast<double>(cpu_duration_up.count()) / gpu_duration_up.count();
            std::cout << "  GPU Speedup: " << speedup_up << "x" << std::endl;
            std::cout << std::endl;
        } else {
            std::cout << "Upsampling Verification: Output sizes don't match!" << std::endl;
            std::cout << "  CPU output: " << cpu_output_up.size() << " samples" << std::endl;
            std::cout << "  GPU output: " << gpu_output_up.size() << " samples" << std::endl;
            std::cout << std::endl;
        }
        
        // Downsample test
        MultirateFilter gpu_filter_down(params, 0);  // 0 means first GPU
        auto gpu_start_down = std::chrono::high_resolution_clock::now();
        auto gpu_output_down = gpu_filter_down.downsample(signal);
        auto gpu_end_down = std::chrono::high_resolution_clock::now();
        auto gpu_duration_down = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end_down - gpu_start_down);
        
        print_performance("GPU Downsampling", signal.size(), gpu_output_down.size(), gpu_duration_down);
        
        // Compare CPU and GPU results for downsampling
        double max_error_down = 0.0;
        if (cpu_output_down.size() == gpu_output_down.size()) {
            for (size_t i = 0; i < cpu_output_down.size(); ++i) {
                max_error_down = std::max(max_error_down, std::abs(cpu_output_down[i] - gpu_output_down[i]));
            }
            
            std::cout << "Downsampling Verification:" << std::endl;
            std::cout << "  Maximum Error: " << max_error_down << std::endl;
            std::cout << "  Implementation Matches: " << (max_error_down < 1e-5 ? "Yes" : "No") << std::endl;
            
            // Calculate speedup
            double speedup_down = static_cast<double>(cpu_duration_down.count()) / gpu_duration_down.count();
            std::cout << "  GPU Speedup: " << speedup_down << "x" << std::endl;
            std::cout << std::endl;
        } else {
            std::cout << "Downsampling Verification: Output sizes don't match!" << std::endl;
            std::cout << "  CPU output: " << cpu_output_down.size() << " samples" << std::endl;
            std::cout << "  GPU output: " << gpu_output_down.size() << " samples" << std::endl;
            std::cout << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "GPU test failed: " << e.what() << std::endl;
        std::cerr << "GPU may not be available on this system." << std::endl;
    }
    
    return 0;
}