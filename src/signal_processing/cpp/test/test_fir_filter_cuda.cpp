/**
 * @file test_fir_filter_cuda.cpp
 * @brief Test program for verifying FIR filter CUDA integration
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

// Generate a simple test signal (sine wave with noise)
std::vector<float> generate_test_signal(int size, float frequency, float sample_rate) {
    std::vector<float> signal(size);
    for (int i = 0; i < size; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        signal[i] = std::sin(2 * M_PI * frequency * t);
        
        // Add some noise
        signal[i] += 0.1f * (std::rand() / static_cast<float>(RAND_MAX) - 0.5f);
    }
    return signal;
}

// Print performance statistics
void print_performance(const std::string& name, size_t signal_size, 
                      std::chrono::microseconds duration) {
    double throughput = static_cast<double>(signal_size) / duration.count();
    std::cout << name << " Performance:" << std::endl;
    std::cout << "  Signal Size: " << signal_size << " samples" << std::endl;
    std::cout << "  Processing Time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "  Throughput: " << throughput << " samples/μs" << std::endl;
    std::cout << "              " << throughput * 1e6 << " samples/s" << std::endl;
    std::cout << std::endl;
}

int main() {
    // Test parameters
    const int signal_size = 1000000;  // 1M samples
    const float sample_rate = 44100.0f;  // 44.1 kHz
    const float test_freq = 1000.0f;  // 1 kHz tone
    
    // Generate test signal
    std::cout << "Generating test signal..." << std::endl;
    auto signal = generate_test_signal(signal_size, test_freq, sample_rate);
    
    // Create a lowpass FIR filter
    FIRFilterParams params;
    params.num_taps = 101;  // 101 taps
    params.filter_type = FilterType::LOWPASS;
    params.cutoff_freqs = {2000.0f};  // 2 kHz cutoff
    params.window_type = WindowType::HAMMING;
    
    // Test CPU implementation
    std::cout << "Testing CPU implementation..." << std::endl;
    FIRFilter cpu_filter(params, sample_rate, -1);  // -1 means CPU
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto cpu_output = cpu_filter.filter(signal);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    print_performance("CPU", signal_size, cpu_duration);
    
    // Test GPU implementation (if available)
    try {
        std::cout << "Testing GPU implementation..." << std::endl;
        FIRFilter gpu_filter(params, sample_rate, 0);  // 0 means first GPU
        
        auto gpu_start = std::chrono::high_resolution_clock::now();
        auto gpu_output = gpu_filter.filter(signal);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
        
        print_performance("GPU", signal_size, gpu_duration);
        
        // Verify output
        double max_error = 0.0;
        for (size_t i = 0; i < signal_size; ++i) {
            max_error = std::max(max_error, std::abs(cpu_output[i] - gpu_output[i]));
        }
        
        std::cout << "Verification:" << std::endl;
        std::cout << "  Maximum Error: " << max_error << std::endl;
        std::cout << "  Implementation Matches: " << (max_error < 1e-5 ? "Yes" : "No") << std::endl;
        
        // Calculate speedup
        double speedup = static_cast<double>(cpu_duration.count()) / gpu_duration.count();
        std::cout << "  GPU Speedup: " << speedup << "x" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "GPU test failed: " << e.what() << std::endl;
        std::cerr << "GPU may not be available on this system." << std::endl;
    }
    
    return 0;
}