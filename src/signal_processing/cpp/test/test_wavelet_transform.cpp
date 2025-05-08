// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <complex>

#include "signal_processing/wavelet_transform.h"

// Simple test framework
#define TEST_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion failed: " << #condition << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while (0)

// Function to generate a test signal
std::vector<float> generateTestSignal(int length, float frequency = 10.0f, float sampling_rate = 1000.0f) {
    std::vector<float> signal(length);
    float dt = 1.0f / sampling_rate;
    
    for (int i = 0; i < length; i++) {
        float t = i * dt;
        signal[i] = std::sin(2.0f * M_PI * frequency * t);
    }
    
    return signal;
}

// Function to generate a chirp signal (frequency changes with time)
std::vector<float> generateChirpSignal(int length, float f0 = 10.0f, float f1 = 100.0f, float sampling_rate = 1000.0f) {
    std::vector<float> signal(length);
    float dt = 1.0f / sampling_rate;
    float duration = length * dt;
    float rate = (f1 - f0) / duration;
    
    for (int i = 0; i < length; i++) {
        float t = i * dt;
        float instantaneous_freq = f0 + rate * t;
        signal[i] = std::sin(2.0f * M_PI * (f0 * t + 0.5f * rate * t * t));
    }
    
    return signal;
}

// Function to generate a step function with a rapid transition
std::vector<float> generateStepSignal(int length, int step_position) {
    std::vector<float> signal(length, 0.0f);
    for (int i = step_position; i < length; i++) {
        signal[i] = 1.0f;
    }
    return signal;
}

// Utility function to compute Mean Squared Error
float computeMSE(const std::vector<float>& original, const std::vector<float>& reconstructed) {
    if (original.size() != reconstructed.size()) {
        throw std::invalid_argument("Vectors must have the same size");
    }
    
    float mse = 0.0f;
    for (size_t i = 0; i < original.size(); i++) {
        float diff = original[i] - reconstructed[i];
        mse += diff * diff;
    }
    
    return mse / original.size();
}

// Utility function to print test results
void printResult(const std::string& test_name, bool result) {
    std::cout << test_name << ": " << (result ? "PASSED" : "FAILED") << std::endl;
}

// Test Haar wavelet transform with a step function
bool testHaarWaveletWithStep() {
    try {
        // Generate a step signal
        std::vector<float> signal = generateStepSignal(512, 256);
        
        // Create a Haar wavelet transform
        signal_processing::DiscreteWaveletTransform dwt(signal_processing::WaveletFamily::HAAR);
        
        // Perform forward transform
        signal_processing::WaveletTransformResult result = dwt.forward(signal, 3, signal_processing::BoundaryMode::SYMMETRIC);
        
        // The step should be captured in the detail coefficients
        // For a Haar wavelet, the step will create a large coefficient at the step position
        std::vector<float>& detail = result.detail_coefficients[0];
        
        // Find the coefficient with the largest absolute value
        auto max_it = std::max_element(detail.begin(), detail.end(),
            [](float a, float b) { return std::abs(a) < std::abs(b); });
        
        // For a step at position 256, the largest coefficient should be around position 128
        // (since downsampling by 2 halves the position)
        int max_pos = std::distance(detail.begin(), max_it);
        
        // Allow some flexibility due to boundary effects
        TEST_ASSERT(max_pos >= 120 && max_pos <= 136);
        
        // Perform inverse transform
        std::vector<float> reconstructed = dwt.inverse(result);
        
        // Compute MSE between original and reconstructed signals
        float mse = computeMSE(signal, reconstructed);
        
        // Haar should perfectly reconstruct a step function with no error
        TEST_ASSERT(mse < 1e-10);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testHaarWaveletWithStep: " << e.what() << std::endl;
        return false;
    }
}

// Test Daubechies wavelet transform with sinusoidal signal
bool testDaubechiesWaveletWithSine() {
    try {
        // Generate a sine wave
        std::vector<float> signal = generateTestSignal(512, 5.0f);
        
        // Create a Daubechies wavelet transform with 4 vanishing moments (db8)
        signal_processing::DiscreteWaveletTransform dwt(signal_processing::WaveletFamily::DAUBECHIES, 4);
        
        // Perform forward transform
        signal_processing::WaveletTransformResult result = dwt.forward(signal, 3, signal_processing::BoundaryMode::PERIODIC);
        
        // Most of the energy should be in specific frequency bands
        // For a 5Hz signal with 1000Hz sampling rate, the energy should be concentrated
        // in the appropriate frequency band
        
        // Perform inverse transform
        std::vector<float> reconstructed = dwt.inverse(result);
        
        // Compute MSE between original and reconstructed signals
        float mse = computeMSE(signal, reconstructed);
        
        // Daubechies should reconstruct sinusoidal signals well
        TEST_ASSERT(mse < 1e-4);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testDaubechiesWaveletWithSine: " << e.what() << std::endl;
        return false;
    }
}

// Test CWT with chirp signal (frequency changing with time)
bool testCwtWithChirp() {
    try {
        // Generate a chirp signal
        std::vector<float> signal = generateChirpSignal(512, 5.0f, 50.0f);
        
        // Create a CWT transform with Morlet wavelet
        signal_processing::ContinuousWaveletTransform cwt(signal_processing::WaveletFamily::MORLET);
        
        // Generate logarithmically spaced scales
        std::vector<float> scales = cwt.generateScales(32, 1.0f, 64.0f);
        
        // Perform forward transform
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<std::complex<float>>> coefficients = cwt.forward(signal, scales);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        
        std::cout << "CWT computation time: " << elapsed.count() << " seconds" << std::endl;
        
        // The chirp signal should create a ridge in the time-frequency plane
        // that follows the changing frequency
        
        // Verify dimensions of the result
        TEST_ASSERT(coefficients.size() == scales.size());
        TEST_ASSERT(coefficients[0].size() == signal.size());
        
        // Perform inverse transform (for completeness)
        std::vector<float> reconstructed = cwt.inverse(coefficients, scales);
        
        // CWT reconstruction is not perfect due to discretization, but should be reasonable
        float mse = computeMSE(signal, reconstructed);
        std::cout << "CWT reconstruction MSE: " << mse << std::endl;
        
        // Loose threshold due to the approximate nature of the inverse CWT
        TEST_ASSERT(mse < 0.1);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testCwtWithChirp: " << e.what() << std::endl;
        return false;
    }
}

// Test Wavelet Packet Transform
bool testWaveletPacket() {
    try {
        // Generate a test signal
        std::vector<float> signal = generateTestSignal(512, 20.0f);
        
        // Create a wavelet packet transform
        signal_processing::WaveletPacketTransform wpt(signal_processing::WaveletFamily::SYMLET, 4);
        
        // Perform forward transform
        signal_processing::WaveletPacketResult result = wpt.forward(signal, 3, signal_processing::BoundaryMode::SYMMETRIC);
        
        // Check the dimensions of the result
        TEST_ASSERT(result.coefficients.size() == 4); // levels + 1
        TEST_ASSERT(result.coefficients[0].size() == 1); // 2^0 nodes at level 0
        TEST_ASSERT(result.coefficients[1].size() == 2); // 2^1 nodes at level 1
        TEST_ASSERT(result.coefficients[2].size() == 4); // 2^2 nodes at level 2
        TEST_ASSERT(result.coefficients[3].size() == 8); // 2^3 nodes at level 3
        
        // Perform inverse transform
        std::vector<float> reconstructed = wpt.inverse(result);
        
        // Compute MSE between original and reconstructed signals
        float mse = computeMSE(signal, reconstructed);
        
        // WPT should reconstruct signals well
        TEST_ASSERT(mse < 1e-4);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testWaveletPacket: " << e.what() << std::endl;
        return false;
    }
}

// Test Maximal Overlap DWT with multiple levels
bool testMaximalOverlapDWT() {
    try {
        // Generate a test signal
        std::vector<float> signal = generateTestSignal(512, 10.0f);
        
        // Create a MODWT transform
        signal_processing::MaximalOverlapDWT modwt(signal_processing::WaveletFamily::DAUBECHIES, 2);
        
        // Perform forward transform
        signal_processing::WaveletTransformResult result = modwt.forward(signal, 4, signal_processing::BoundaryMode::PERIODIC);
        
        // Check dimensions of the result
        TEST_ASSERT(result.approximation_coefficients.size() == 5); // levels + 1
        TEST_ASSERT(result.detail_coefficients.size() == 4); // levels
        
        // MODWT does not downsample, so all coefficients should have the same size as the input
        for (const auto& approx : result.approximation_coefficients) {
            TEST_ASSERT(approx.size() == signal.size());
        }
        
        for (const auto& detail : result.detail_coefficients) {
            TEST_ASSERT(detail.size() == signal.size());
        }
        
        // Perform inverse transform
        std::vector<float> reconstructed = modwt.inverse(result);
        
        // Compute MSE between original and reconstructed signals
        float mse = computeMSE(signal, reconstructed);
        
        // MODWT should perfectly reconstruct signals
        TEST_ASSERT(mse < 1e-10);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testMaximalOverlapDWT: " << e.what() << std::endl;
        return false;
    }
}

// Performance test for the implemented wavelet transforms
bool testPerformance() {
    try {
        // Generate a larger test signal for performance testing
        int signal_length = 8192;
        std::vector<float> signal = generateTestSignal(signal_length, 20.0f);
        
        // Test DWT performance
        std::cout << "Performance test with signal length " << signal_length << std::endl;
        
        // DWT performance
        signal_processing::DiscreteWaveletTransform dwt(signal_processing::WaveletFamily::DAUBECHIES, 4);
        
        auto start_dwt = std::chrono::high_resolution_clock::now();
        signal_processing::WaveletTransformResult dwt_result = dwt.forward(signal, 6, signal_processing::BoundaryMode::SYMMETRIC);
        std::vector<float> dwt_reconstructed = dwt.inverse(dwt_result);
        auto end_dwt = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed_dwt = end_dwt - start_dwt;
        std::cout << "DWT (forward + inverse) time: " << elapsed_dwt.count() << " seconds" << std::endl;
        
        // WPT performance
        signal_processing::WaveletPacketTransform wpt(signal_processing::WaveletFamily::DAUBECHIES, 4);
        
        auto start_wpt = std::chrono::high_resolution_clock::now();
        signal_processing::WaveletPacketResult wpt_result = wpt.forward(signal, 4, signal_processing::BoundaryMode::SYMMETRIC);
        std::vector<float> wpt_reconstructed = wpt.inverse(wpt_result);
        auto end_wpt = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed_wpt = end_wpt - start_wpt;
        std::cout << "WPT (forward + inverse) time: " << elapsed_wpt.count() << " seconds" << std::endl;
        
        // MODWT is more computationally intensive as it doesn't downsample, so use fewer levels
        signal_processing::MaximalOverlapDWT modwt(signal_processing::WaveletFamily::DAUBECHIES, 4);
        
        auto start_modwt = std::chrono::high_resolution_clock::now();
        signal_processing::WaveletTransformResult modwt_result = modwt.forward(signal, 3, signal_processing::BoundaryMode::SYMMETRIC);
        std::vector<float> modwt_reconstructed = modwt.inverse(modwt_result);
        auto end_modwt = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed_modwt = end_modwt - start_modwt;
        std::cout << "MODWT (forward + inverse) time: " << elapsed_modwt.count() << " seconds" << std::endl;
        
        // The performance test passes if it completes without errors
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testPerformance: " << e.what() << std::endl;
        return false;
    }
}

#ifdef WITH_CUDA
// Test CUDA-accelerated DWT
bool testCudaDWT() {
    try {
        // Generate a test signal
        std::vector<float> signal = generateTestSignal(1024, 10.0f);
        
        // Create a DWT transform
        signal_processing::DiscreteWaveletTransform dwt(signal_processing::WaveletFamily::DAUBECHIES, 4);
        
        // Get filters for CUDA implementation
        const std::vector<float>& decomp_low_pass = dwt.getDecompositionLowPassFilter();
        const std::vector<float>& decomp_high_pass = dwt.getDecompositionHighPassFilter();
        const std::vector<float>& recon_low_pass = dwt.getReconstructionLowPassFilter();
        const std::vector<float>& recon_high_pass = dwt.getReconstructionHighPassFilter();
        
        // Perform CPU transform for comparison
        auto start_cpu = std::chrono::high_resolution_clock::now();
        signal_processing::WaveletTransformResult cpu_result = dwt.forward(signal, 3, signal_processing::BoundaryMode::SYMMETRIC);
        std::vector<float> cpu_reconstructed = dwt.inverse(cpu_result);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
        std::cout << "CPU DWT time: " << elapsed_cpu.count() << " seconds" << std::endl;
        
        // Perform CUDA transform
        auto start_cuda = std::chrono::high_resolution_clock::now();
        signal_processing::WaveletTransformResult cuda_result = signal_processing::cuda_discrete_wavelet_transform(
            signal, decomp_low_pass, decomp_high_pass, 3, signal_processing::BoundaryMode::SYMMETRIC);
        
        std::vector<float> cuda_reconstructed = signal_processing::cuda_inverse_discrete_wavelet_transform(
            cuda_result, recon_low_pass, recon_high_pass, signal_processing::BoundaryMode::SYMMETRIC);
        auto end_cuda = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed_cuda = end_cuda - start_cuda;
        std::cout << "CUDA DWT time: " << elapsed_cuda.count() << " seconds" << std::endl;
        std::cout << "Speedup: " << elapsed_cpu.count() / elapsed_cuda.count() << "x" << std::endl;
        
        // Compare CPU and CUDA results
        float approx_mse = 0.0f;
        for (size_t i = 0; i < cpu_result.approximation_coefficients.size(); i++) {
            approx_mse += computeMSE(
                cpu_result.approximation_coefficients[i],
                cuda_result.approximation_coefficients[i]
            );
        }
        approx_mse /= cpu_result.approximation_coefficients.size();
        
        float detail_mse = 0.0f;
        for (size_t i = 0; i < cpu_result.detail_coefficients.size(); i++) {
            detail_mse += computeMSE(
                cpu_result.detail_coefficients[i],
                cuda_result.detail_coefficients[i]
            );
        }
        detail_mse /= cpu_result.detail_coefficients.size();
        
        float reconstruction_mse = computeMSE(cpu_reconstructed, cuda_reconstructed);
        
        std::cout << "Approximation coefficients MSE: " << approx_mse << std::endl;
        std::cout << "Detail coefficients MSE: " << detail_mse << std::endl;
        std::cout << "Reconstruction MSE: " << reconstruction_mse << std::endl;
        
        // The results should be very close (not exactly the same due to floating-point differences)
        TEST_ASSERT(approx_mse < 1e-4);
        TEST_ASSERT(detail_mse < 1e-4);
        TEST_ASSERT(reconstruction_mse < 1e-4);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testCudaDWT: " << e.what() << std::endl;
        return false;
    }
}

// Test CUDA-accelerated CWT
bool testCudaCWT() {
    try {
        // Generate a chirp signal for time-frequency analysis
        std::vector<float> signal = generateChirpSignal(1024, 5.0f, 50.0f);
        
        // Create a CWT transform
        signal_processing::ContinuousWaveletTransform cwt(signal_processing::WaveletFamily::MORLET);
        
        // Generate scales
        std::vector<float> scales = cwt.generateScales(32, 1.0f, 64.0f);
        
        // Perform CPU transform for comparison
        auto start_cpu = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<std::complex<float>>> cpu_coefficients = cwt.forward(signal, scales);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
        std::cout << "CPU CWT time: " << elapsed_cpu.count() << " seconds" << std::endl;
        
        // Perform CUDA transform
        auto start_cuda = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<std::complex<float>>> cuda_coefficients = 
            signal_processing::cuda_continuous_wavelet_transform(signal, scales, signal_processing::WaveletFamily::MORLET);
        auto end_cuda = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed_cuda = end_cuda - start_cuda;
        std::cout << "CUDA CWT time: " << elapsed_cuda.count() << " seconds" << std::endl;
        std::cout << "Speedup: " << elapsed_cpu.count() / elapsed_cuda.count() << "x" << std::endl;
        
        // Compare CPU and CUDA results
        float mse_sum = 0.0f;
        int total_coeffs = 0;
        
        for (size_t i = 0; i < cpu_coefficients.size(); i++) {
            for (size_t j = 0; j < cpu_coefficients[i].size(); j++) {
                float real_diff = std::abs(cpu_coefficients[i][j].real() - cuda_coefficients[i][j].real());
                float imag_diff = std::abs(cpu_coefficients[i][j].imag() - cuda_coefficients[i][j].imag());
                mse_sum += real_diff * real_diff + imag_diff * imag_diff;
                total_coeffs++;
            }
        }
        
        float avg_mse = mse_sum / total_coeffs;
        std::cout << "Average MSE between CPU and CUDA CWT: " << avg_mse << std::endl;
        
        // The results should be close but not identical due to floating-point differences
        // and potential implementation differences in complex math operations
        TEST_ASSERT(avg_mse < 1e-3);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testCudaCWT: " << e.what() << std::endl;
        return false;
    }
}

// Test CUDA-accelerated Wavelet Packet Transform
bool testCudaWaveletPacket() {
    try {
        // Generate a test signal
        std::vector<float> signal = generateTestSignal(1024, 20.0f);
        
        // Create a wavelet packet transform
        signal_processing::WaveletPacketTransform wpt(signal_processing::WaveletFamily::DAUBECHIES, 4);
        
        // Get filters for CUDA implementation
        const std::vector<float>& decomp_low_pass = wpt.getDecompositionLowPassFilter();
        const std::vector<float>& decomp_high_pass = wpt.getDecompositionHighPassFilter();
        const std::vector<float>& recon_low_pass = wpt.getReconstructionLowPassFilter();
        const std::vector<float>& recon_high_pass = wpt.getReconstructionHighPassFilter();
        
        // Perform CPU transform for comparison
        auto start_cpu = std::chrono::high_resolution_clock::now();
        signal_processing::WaveletPacketResult cpu_result = wpt.forward(signal, 3, signal_processing::BoundaryMode::SYMMETRIC);
        std::vector<float> cpu_reconstructed = wpt.inverse(cpu_result);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed_cpu = end_cpu - start_cpu;
        std::cout << "CPU WPT time: " << elapsed_cpu.count() << " seconds" << std::endl;
        
        // Perform CUDA transform
        auto start_cuda = std::chrono::high_resolution_clock::now();
        signal_processing::WaveletPacketResult cuda_result = signal_processing::cuda_wavelet_packet_transform(
            signal, decomp_low_pass, decomp_high_pass, 3, signal_processing::BoundaryMode::SYMMETRIC);
        
        std::vector<float> cuda_reconstructed = signal_processing::cuda_inverse_wavelet_packet_transform(
            cuda_result, recon_low_pass, recon_high_pass, signal_processing::BoundaryMode::SYMMETRIC);
        auto end_cuda = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed_cuda = end_cuda - start_cuda;
        std::cout << "CUDA WPT time: " << elapsed_cuda.count() << " seconds" << std::endl;
        std::cout << "Speedup: " << elapsed_cpu.count() / elapsed_cuda.count() << "x" << std::endl;
        
        // Compare CPU and CUDA reconstructions
        float reconstruction_mse = computeMSE(cpu_reconstructed, cuda_reconstructed);
        std::cout << "Reconstruction MSE: " << reconstruction_mse << std::endl;
        
        // The results should be close
        TEST_ASSERT(reconstruction_mse < 1e-4);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Exception in testCudaWaveletPacket: " << e.what() << std::endl;
        return false;
    }
}
#endif // WITH_CUDA

// Main function to run all tests
int main() {
    std::cout << "Running Wavelet Transform Tests..." << std::endl;
    
    // CPU implementation tests
    bool result1 = testHaarWaveletWithStep();
    printResult("Haar Wavelet with Step Function", result1);
    
    bool result2 = testDaubechiesWaveletWithSine();
    printResult("Daubechies Wavelet with Sine Wave", result2);
    
    bool result3 = testCwtWithChirp();
    printResult("CWT with Chirp Signal", result3);
    
    bool result4 = testWaveletPacket();
    printResult("Wavelet Packet Transform", result4);
    
    bool result5 = testMaximalOverlapDWT();
    printResult("Maximal Overlap DWT", result5);
    
    bool result6 = testPerformance();
    printResult("Performance Tests", result6);
    
    // GPU-accelerated tests (if CUDA is available)
#ifdef WITH_CUDA
    std::cout << "\nRunning CUDA-accelerated tests..." << std::endl;
    
    bool result7 = testCudaDWT();
    printResult("CUDA Discrete Wavelet Transform", result7);
    
    bool result8 = testCudaCWT();
    printResult("CUDA Continuous Wavelet Transform", result8);
    
    bool result9 = testCudaWaveletPacket();
    printResult("CUDA Wavelet Packet Transform", result9);
    
    bool cuda_success = result7 && result8 && result9;
    std::cout << "CUDA tests: " << (cuda_success ? "PASSED" : "FAILED") << std::endl;
#else
    std::cout << "\nCUDA support not enabled, skipping GPU tests." << std::endl;
#endif
    
    bool cpu_success = result1 && result2 && result3 && result4 && result5 && result6;
    std::cout << "\nCPU tests: " << (cpu_success ? "ALL PASSED" : "SOME FAILED") << std::endl;
    
#ifdef WITH_CUDA
    return (cpu_success && cuda_success) ? 0 : 1;
#else
    return cpu_success ? 0 : 1;
#endif
}