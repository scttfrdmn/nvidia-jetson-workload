/**
 * @file test_time_frequency.cpp
 * @brief Test program for verifying time-frequency analysis functions
 * 
 * SPDX-License-Identifier: Apache-2.0
 * Copyright 2025 Scott Friedman and Project Contributors
 */

#include "signal_processing/time_frequency.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace signal_processing;

// Generate a chirp test signal (frequency sweep)
std::vector<float> generate_chirp_signal(int size, float start_freq, float end_freq, float sample_rate) {
    std::vector<float> signal(size);
    float duration = static_cast<float>(size) / sample_rate;
    float k = (end_freq - start_freq) / duration;
    
    for (int i = 0; i < size; ++i) {
        float t = static_cast<float>(i) / sample_rate;
        float phase = 2 * M_PI * (start_freq * t + 0.5f * k * t * t);
        signal[i] = std::sin(phase);
        
        // Add some noise
        signal[i] += 0.05f * (std::rand() / static_cast<float>(RAND_MAX) - 0.5f);
    }
    
    return signal;
}

// Print STFT performance statistics
void test_stft(const std::vector<float>& signal, float sample_rate, int device_id, const std::string& device_name) {
    // STFT parameters
    STFTParams params;
    params.window_size = 1024;
    params.hop_size = 256;
    params.window_type = WindowType::HANN;
    params.center = true;
    
    // Create STFT object
    STFT stft(params, device_id);
    
    // Measure transform time
    auto start = std::chrono::high_resolution_clock::now();
    auto stft_result = stft.transform(signal, sample_rate);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Measure inverse transform time
    auto start_inv = std::chrono::high_resolution_clock::now();
    auto reconstructed = stft.inverse_transform(stft_result);
    auto end_inv = std::chrono::high_resolution_clock::now();
    auto duration_inv = std::chrono::duration_cast<std::chrono::milliseconds>(end_inv - start_inv);
    
    // Report results
    std::cout << "STFT Performance (" << device_name << "):" << std::endl;
    std::cout << "  Forward STFT: " << duration.count() << " ms" << std::endl;
    std::cout << "  Inverse STFT: " << duration_inv.count() << " ms" << std::endl;
    std::cout << "  Number of frames: " << stft_result.spectrogram.size() << std::endl;
    std::cout << "  Number of frequency bins: " << stft_result.spectrogram[0].size() << std::endl;
    
    // Calculate reconstruction error
    float mse = 0.0f;
    size_t min_size = std::min(signal.size(), reconstructed.size());
    for (size_t i = 0; i < min_size; ++i) {
        float error = signal[i] - reconstructed[i];
        mse += error * error;
    }
    mse /= min_size;
    
    std::cout << "  Reconstruction MSE: " << mse << std::endl;
    std::cout << std::endl;
}

// Print CWT performance statistics
void test_cwt(const std::vector<float>& signal, float sample_rate, int device_id, const std::string& device_name) {
    // CWT parameters
    CWTParams params;
    params.wavelet_type = WaveletType::MORLET;
    params.num_scales = 32;
    params.min_scale = 1.0f;
    params.wavelet_param = 6.0f;
    
    // Create CWT object
    CWT cwt(params, device_id);
    
    // Measure transform time
    auto start = std::chrono::high_resolution_clock::now();
    auto cwt_result = cwt.transform(signal, sample_rate);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Report results
    std::cout << "CWT Performance (" << device_name << "):" << std::endl;
    std::cout << "  Transform Time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Number of scales: " << cwt_result.scalogram.size() << std::endl;
    std::cout << "  Number of time points: " << cwt_result.scalogram[0].size() << std::endl;
    
    // Frequency range
    std::cout << "  Frequency range: " << cwt_result.frequencies.front() << " - "
              << cwt_result.frequencies.back() << " Hz" << std::endl;
    std::cout << std::endl;
}

// Test high-level time-frequency functions
void test_high_level_functions(const std::vector<float>& signal, float sample_rate, int device_id, const std::string& device_name) {
    std::cout << "High-level Functions (" << device_name << "):" << std::endl;
    
    // Test spectrogram
    auto start_spec = std::chrono::high_resolution_clock::now();
    auto [spec, spec_axes] = time_frequency::spectrogram(signal, sample_rate, 1024, 256, WindowType::HANN, true, device_id);
    auto end_spec = std::chrono::high_resolution_clock::now();
    auto duration_spec = std::chrono::duration_cast<std::chrono::milliseconds>(end_spec - start_spec);
    
    std::cout << "  Spectrogram: " << duration_spec.count() << " ms "
              << "(" << spec.size() << " frames, " << spec[0].size() << " bins)" << std::endl;
    
    // Test scalogram
    auto start_scal = std::chrono::high_resolution_clock::now();
    auto [scal, scal_axes] = time_frequency::scalogram(signal, sample_rate, WaveletType::MORLET, 32, true, device_id);
    auto end_scal = std::chrono::high_resolution_clock::now();
    auto duration_scal = std::chrono::duration_cast<std::chrono::milliseconds>(end_scal - start_scal);
    
    std::cout << "  Scalogram: " << duration_scal.count() << " ms "
              << "(" << scal.size() << " scales, " << scal[0].size() << " time points)" << std::endl;
    
    // Test Hilbert transform
    auto start_hilb = std::chrono::high_resolution_clock::now();
    auto hilb = time_frequency::hilbert_transform(signal, device_id);
    auto end_hilb = std::chrono::high_resolution_clock::now();
    auto duration_hilb = std::chrono::duration_cast<std::chrono::milliseconds>(end_hilb - start_hilb);
    
    std::cout << "  Hilbert Transform: " << duration_hilb.count() << " ms" << std::endl;
    
    // Test instantaneous frequency
    auto start_inst = std::chrono::high_resolution_clock::now();
    auto inst_freq = time_frequency::instantaneous_frequency(signal, sample_rate, device_id);
    auto end_inst = std::chrono::high_resolution_clock::now();
    auto duration_inst = std::chrono::duration_cast<std::chrono::milliseconds>(end_inst - start_inst);
    
    std::cout << "  Instantaneous Frequency: " << duration_inst.count() << " ms" << std::endl;
    std::cout << std::endl;
}

int main() {
    // Test parameters
    const int signal_size = 44100 * 2;  // 2 seconds at 44.1 kHz
    const float sample_rate = 44100.0f;  // 44.1 kHz
    const float start_freq = 100.0f;    // Start at 100 Hz
    const float end_freq = 10000.0f;    // End at 10 kHz
    
    // Generate test signal
    std::cout << "Generating chirp signal..." << std::endl;
    auto signal = generate_chirp_signal(signal_size, start_freq, end_freq, sample_rate);
    std::cout << std::endl;
    
    // Test CPU implementation
    test_stft(signal, sample_rate, -1, "CPU");
    test_cwt(signal, sample_rate, -1, "CPU");
    test_high_level_functions(signal, sample_rate, -1, "CPU");
    
    // Test GPU implementation (if available)
    try {
        // Check for CUDA
#if defined(WITH_CUDA)
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        
        if (error == cudaSuccess && device_count > 0) {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, 0);
            
            std::string device_name = "GPU (" + std::string(props.name) + ", SM " + 
                                     std::to_string(props.major) + "." + std::to_string(props.minor) + ")";
            
            test_stft(signal, sample_rate, 0, device_name);
            test_cwt(signal, sample_rate, 0, device_name);
            test_high_level_functions(signal, sample_rate, 0, device_name);
        } else {
            std::cout << "No CUDA device available for testing" << std::endl;
        }
#else
        std::cout << "CUDA support not enabled in build" << std::endl;
#endif
    } catch (const std::exception& e) {
        std::cerr << "GPU test failed: " << e.what() << std::endl;
    }
    
    return 0;
}