/**
 * @file test_adaptive_filter_cuda.cpp
 * @brief Test program for verifying AdaptiveFilter CUDA integration
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include "signal_processing/digital_filtering.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>

using namespace signal_processing;

// Generate test signals for adaptive filtering (reference and target with noise)
std::pair<std::vector<float>, std::vector<float>> generate_adaptive_test_signals(
    int size, float frequency, float sample_rate, float noise_level, float delay) {
    
    std::vector<float> reference(size);
    std::vector<float> desired(size);
    
    // Random number generator for noise
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, noise_level);
    
    // Generate the reference signal (sine wave)
    for (int i = 0; i < size; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        reference[i] = std::sin(2 * M_PI * frequency * t);
    }
    
    // Generate the desired signal (delayed and noisy version of reference)
    for (int i = 0; i < size; ++i) {
        int delay_idx = i - static_cast<int>(delay * sample_rate);
        
        if (delay_idx >= 0 && delay_idx < size) {
            desired[i] = 0.8f * reference[delay_idx];  // Attenuated
        } else {
            desired[i] = 0.0f;
        }
        
        // Add noise
        desired[i] += noise(gen);
    }
    
    return {reference, desired};
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

// Calculate MSE between two signals
double calculate_mse(const std::vector<float>& signal1, const std::vector<float>& signal2) {
    if (signal1.size() != signal2.size()) {
        throw std::invalid_argument("Signals must have the same length");
    }
    
    double sum_squared_error = 0.0;
    for (size_t i = 0; i < signal1.size(); ++i) {
        double error = signal1[i] - signal2[i];
        sum_squared_error += error * error;
    }
    
    return sum_squared_error / signal1.size();
}

int main() {
    // Test parameters
    const int signal_size = 50000;      // 50k samples
    const float sample_rate = 44100.0f; // 44.1 kHz
    const float test_freq = 1000.0f;    // 1 kHz tone
    const float noise_level = 0.1f;     // Noise standard deviation
    const float delay = 0.001f;         // 1 ms delay
    
    // Generate test signals
    std::cout << "Generating test signals..." << std::endl;
    auto signals = generate_adaptive_test_signals(signal_size, test_freq, sample_rate, noise_level, delay);
    auto& reference = signals.first;
    auto& desired = signals.second;
    
    // Create adaptive filter parameters
    AdaptiveFilterParams params;
    params.filter_length = 64;
    params.filter_type = AdaptiveFilterType::LMS;
    params.step_size = 0.01f;
    
    // Test CPU implementation
    std::cout << "Testing CPU implementation..." << std::endl;
    AdaptiveFilter cpu_filter(params, -1);  // -1 means CPU
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto cpu_results = cpu_filter.filter(reference, desired);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    auto& cpu_output = cpu_results.first;
    auto& cpu_error = cpu_results.second;
    
    print_performance("CPU", signal_size, cpu_duration);
    
    // Calculate error metrics
    double cpu_mse = calculate_mse(desired, cpu_output);
    std::cout << "CPU Mean Squared Error: " << cpu_mse << std::endl;
    std::cout << std::endl;
    
    // Test GPU implementation (if available)
    try {
        std::cout << "Testing GPU implementation..." << std::endl;
        AdaptiveFilter gpu_filter(params, 0);  // 0 means first GPU
        
        auto gpu_start = std::chrono::high_resolution_clock::now();
        auto gpu_results = gpu_filter.filter(reference, desired);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
        
        auto& gpu_output = gpu_results.first;
        auto& gpu_error = gpu_results.second;
        
        print_performance("GPU", signal_size, gpu_duration);
        
        // Calculate error metrics
        double gpu_mse = calculate_mse(desired, gpu_output);
        std::cout << "GPU Mean Squared Error: " << gpu_mse << std::endl;
        std::cout << std::endl;
        
        // Compare CPU and GPU results
        double cpu_gpu_diff = calculate_mse(cpu_output, gpu_output);
        std::cout << "Verification:" << std::endl;
        std::cout << "  CPU-GPU MSE Difference: " << cpu_gpu_diff << std::endl;
        std::cout << "  Implementation Matches: " << (cpu_gpu_diff < 1e-6 ? "Yes" : "No") << std::endl;
        
        // Calculate speedup
        double speedup = static_cast<double>(cpu_duration.count()) / gpu_duration.count();
        std::cout << "  GPU Speedup: " << speedup << "x" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "GPU test failed: " << e.what() << std::endl;
        std::cerr << "GPU may not be available on this system." << std::endl;
    }
    
    return 0;
}