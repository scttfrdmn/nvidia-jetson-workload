// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Scott Friedman and Project Contributors

#include "signal_processing/wavelet_transform.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <complex>
#include <vector>
#include <memory>

namespace signal_processing {

// ----- Base WaveletTransform Implementation -----

WaveletTransform::WaveletTransform(WaveletFamily family, int vanishing_moments) 
    : family_(family), vanishing_moments_(vanishing_moments) {
    generateFilters();
}

void WaveletTransform::generateFilters() {
    switch (family_) {
        case WaveletFamily::HAAR:
            generateHaarFilters();
            break;
        case WaveletFamily::DAUBECHIES:
            generateDaubechiesFilters();
            break;
        case WaveletFamily::SYMLET:
            generateSymletFilters();
            break;
        case WaveletFamily::COIFLET:
            generateCoifletFilters();
            break;
        case WaveletFamily::BIORTHOGONAL:
            generateBiorthogonalFilters();
            break;
        case WaveletFamily::MEYER:
            generateMeyerFilters();
            break;
        case WaveletFamily::MORLET:
            generateMorletFilters();
            break;
        case WaveletFamily::MEXICAN_HAT:
            generateMexicanHatFilters();
            break;
        default:
            throw std::invalid_argument("Unsupported wavelet family");
    }
}

void WaveletTransform::generateHaarFilters() {
    // Haar wavelet filters are the simplest
    decomposition_low_pass_ = {0.7071067811865475, 0.7071067811865475};
    decomposition_high_pass_ = {0.7071067811865475, -0.7071067811865475};
    
    // Reconstruction filters are the reverse of decomposition filters
    reconstruction_low_pass_ = decomposition_low_pass_;
    reconstruction_high_pass_ = {-decomposition_high_pass_[0], decomposition_high_pass_[1]};
}

void WaveletTransform::generateDaubechiesFilters() {
    if (vanishing_moments_ < 1) {
        throw std::invalid_argument("Vanishing moments must be at least 1 for Daubechies wavelets");
    }
    
    // For Daubechies wavelets, the filter length is 2*vanishing_moments_
    int N = vanishing_moments_ * 2;
    
    // Predefined coefficients for common Daubechies wavelets
    if (vanishing_moments_ == 1) {
        // This is actually Haar
        generateHaarFilters();
        return;
    } else if (vanishing_moments_ == 2) {
        // db4 (4 coefficients)
        decomposition_low_pass_ = {
            0.4829629131445341, 0.8365163037378079, 
            0.2241438680420134, -0.1294095225512604
        };
    } else if (vanishing_moments_ == 3) {
        // db6 (6 coefficients)
        decomposition_low_pass_ = {
            0.3326705529500825, 0.8068915093110924, 0.4598775021184914,
            -0.1350110200102546, -0.0854412738820267, 0.0352262918857095
        };
    } else if (vanishing_moments_ == 4) {
        // db8 (8 coefficients)
        decomposition_low_pass_ = {
            0.2303778133088964, 0.7148465705529154, 0.6308807679298587, 
            -0.0279837694168599, -0.1870348117190931, 0.0308413818355607, 
            0.0328830116668852, -0.0105974017850690
        };
    } else if (vanishing_moments_ == 5) {
        // db10 (10 coefficients)
        decomposition_low_pass_ = {
            0.1601023979741929, 0.6038292697971895, 0.7243085284377726, 
            0.1384281459013203, -0.2422948870663823, -0.0322448695846381, 
            0.0775714938400459, -0.0062414902127983, -0.0125807519990820, 
            0.0033357252854738
        };
    } else {
        // For higher orders, we should compute the coefficients
        // But for this implementation, we'll throw an error
        throw std::invalid_argument("Daubechies filters above order 5 not implemented");
    }
    
    // Generate high pass filter using quadrature mirror relationship
    decomposition_high_pass_.resize(N);
    for (int i = 0; i < N; i++) {
        decomposition_high_pass_[i] = std::pow(-1.0, i) * decomposition_low_pass_[N - 1 - i];
    }
    
    // Generate reconstruction filters
    reconstruction_low_pass_ = decomposition_high_pass_;
    reconstruction_high_pass_.resize(N);
    for (int i = 0; i < N; i++) {
        reconstruction_high_pass_[i] = std::pow(-1.0, i + 1) * decomposition_low_pass_[i];
    }
    
    // Reverse the order to get the correct phase
    std::reverse(reconstruction_low_pass_.begin(), reconstruction_low_pass_.end());
    std::reverse(reconstruction_high_pass_.begin(), reconstruction_high_pass_.end());
}

void WaveletTransform::generateSymletFilters() {
    if (vanishing_moments_ < 2) {
        throw std::invalid_argument("Vanishing moments must be at least 2 for Symlet wavelets");
    }
    
    // Symlets are nearly symmetrical variants of Daubechies wavelets
    // For this implementation, we'll define some common Symlet filters
    if (vanishing_moments_ == 2) {
        // sym4
        decomposition_low_pass_ = {
            -0.0757657147893407, -0.0296355276459541, 
            0.4976186676324578, 0.8037387518052163, 
            0.2978577956055422, -0.0992195435769354
        };
    } else if (vanishing_moments_ == 3) {
        // sym6
        decomposition_low_pass_ = {
            0.0154041093270274, 0.0034907120843304, -0.1179901111484105, 
            -0.0483117425859981, 0.4910559419276396, 0.7876411410287941, 
            0.3379294217282401, -0.0726375227866000, -0.0210602925126954, 
            0.0447249017707482
        };
    } else if (vanishing_moments_ == 4) {
        // sym8
        decomposition_low_pass_ = {
            0.0018899503327594, -0.0003029205147213, -0.0149522583367926, 
            0.0038087520138601, 0.0491371796734768, -0.0272190299168137, 
            -0.0519458381078751, 0.3644418948359564, 0.7771857517005235, 
            0.4813596512592012, -0.0612733590679088, -0.1432942383510542, 
            0.0076074873249176, 0.0316950878103452
        };
    } else {
        throw std::invalid_argument("Symlet filters for the specified vanishing moments not implemented");
    }
    
    // Generate high pass filter and reconstruction filters similar to Daubechies
    int N = decomposition_low_pass_.size();
    decomposition_high_pass_.resize(N);
    for (int i = 0; i < N; i++) {
        decomposition_high_pass_[i] = std::pow(-1.0, i) * decomposition_low_pass_[N - 1 - i];
    }
    
    reconstruction_low_pass_ = decomposition_high_pass_;
    std::reverse(reconstruction_low_pass_.begin(), reconstruction_low_pass_.end());
    
    reconstruction_high_pass_.resize(N);
    for (int i = 0; i < N; i++) {
        reconstruction_high_pass_[i] = std::pow(-1.0, i + 1) * decomposition_low_pass_[i];
    }
    std::reverse(reconstruction_high_pass_.begin(), reconstruction_high_pass_.end());
}

void WaveletTransform::generateCoifletFilters() {
    if (vanishing_moments_ < 1 || vanishing_moments_ > 5) {
        throw std::invalid_argument("Vanishing moments must be between 1 and 5 for Coiflet wavelets");
    }
    
    // For Coiflets, the filter length is 6*vanishing_moments_
    // Define coefficients for common Coiflet filters
    if (vanishing_moments_ == 1) {
        // coif6
        decomposition_low_pass_ = {
            -0.0156557285289848, -0.0727326213410511, 0.3848648565381134, 
            0.8525720416423900, 0.3378976709511590, -0.0727322757411889
        };
    } else if (vanishing_moments_ == 2) {
        // coif12
        decomposition_low_pass_ = {
            -0.0007205494453679, -0.0018232088707116, 0.0056114348194211, 
            0.0236801719464464, -0.0594344186467388, -0.0764885990786692, 
            0.4170051844236707, 0.8127236354493977, 0.3861100668229939, 
            -0.0673725547222826, -0.0414649367819558, 0.0163873364635998
        };
    } else if (vanishing_moments_ == 3) {
        // coif18
        decomposition_low_pass_ = {
            -0.0000345997728362, -0.0000709833031381, 0.0004662169601129, 
            0.0011175187708906, -0.0025745176887502, -0.0090079761366615, 
            0.0158805448636158, 0.0345550275730615, -0.0823019271068856, 
            -0.0717998216193117, 0.4284834763776168, 0.7937772226256169, 
            0.4051769024096150, -0.0611233900026726, -0.0657719112818552, 
            0.0234526961418362, 0.0077825964273254, -0.0037935128644910
        };
    } else {
        throw std::invalid_argument("Coiflet filters for the specified vanishing moments not implemented");
    }
    
    // Generate high pass filter and reconstruction filters
    int N = decomposition_low_pass_.size();
    decomposition_high_pass_.resize(N);
    for (int i = 0; i < N; i++) {
        decomposition_high_pass_[i] = std::pow(-1.0, i) * decomposition_low_pass_[N - 1 - i];
    }
    
    reconstruction_low_pass_ = decomposition_high_pass_;
    std::reverse(reconstruction_low_pass_.begin(), reconstruction_low_pass_.end());
    
    reconstruction_high_pass_.resize(N);
    for (int i = 0; i < N; i++) {
        reconstruction_high_pass_[i] = std::pow(-1.0, i + 1) * decomposition_low_pass_[i];
    }
    std::reverse(reconstruction_high_pass_.begin(), reconstruction_high_pass_.end());
}

void WaveletTransform::generateBiorthogonalFilters() {
    // We'll implement a common biorthogonal wavelet: Bior4.4
    // In practice, we'd have parameters to select different biorthogonal wavelets
    
    // Decomposition filters for Bior4.4
    decomposition_low_pass_ = {
        0.0, 0.03782845550726404, -0.023849465019556843, -0.11062440441843718, 
        0.37740285561283066, 0.8526986790088938, 0.37740285561283066, 
        -0.11062440441843718, -0.023849465019556843, 0.03782845550726404
    };
    
    decomposition_high_pass_ = {
        0.0, 0.0, 0.0, 0.17677669529663687, 
        -0.5303300858899107, 0.5303300858899107, 
        -0.17677669529663687, 0.0, 0.0, 0.0
    };
    
    // Reconstruction filters for Bior4.4
    reconstruction_low_pass_ = {
        0.0, 0.0, 0.0, -0.17677669529663687, 
        -0.5303300858899107, -0.5303300858899107, 
        -0.17677669529663687, 0.0, 0.0, 0.0
    };
    
    reconstruction_high_pass_ = {
        0.0, -0.03782845550726404, -0.023849465019556843, 0.11062440441843718, 
        0.37740285561283066, -0.8526986790088938, 0.37740285561283066, 
        0.11062440441843718, -0.023849465019556843, -0.03782845550726404
    };
}

void WaveletTransform::generateMeyerFilters() {
    // Meyer wavelets are defined in the frequency domain
    // For practical implementation, they are typically approximated by FIR filters
    // Here we'll use a common approximation with 62 coefficients
    
    // These coefficients approximate the Meyer wavelet
    decomposition_low_pass_ = {
        0.0012710883562970, 0.0022892635185499, 0.0020903374798204, 0.0006761197351911, 
        -0.0011704579340908, -0.0016932572858294, -0.0006831087594098, 0.0011167046387312, 
        0.0018091195070902, 0.0007601829253074, -0.0010264024796177, -0.0018963129912341, 
        -0.0008458946449452, 0.0009143958390320, 0.0019591023138738, 0.0009311774922303, 
        -0.0007823666916205, -0.0019945761959598, -0.0010147591729168, 0.0006315764435922, 
        0.0020002992522693, 0.0010942957350618, -0.0004632195636960, -0.0019755772087053, 
        -0.0011676498756524, 0.0002797017248833, 0.0019199452808967, 0.0012327739789566, 
        -0.0000837603570030, -0.0018332039355546, -0.0012879381483740, -0.0001214760298555, 
        0.0017161980030907, 0.0013317906162978, 0.0003326410905572, -0.0015701063384150, 
        -0.0013630339115952, -0.0005462275577305, 0.0013966553758087, 0.0013802881113284, 
        0.0007586778763298, -0.0011980553488053, -0.0013823273421587, -0.0009665982496464, 
        0.0009777452937725, 0.0013685881533489, 0.0011667906099799, -0.0007393194545523, 
        -0.0013386947219647, -0.0013560028421795, 0.0004866015612312, 0.0012925344654002, 
        0.0015306074132747, -0.0002236617747549, -0.0012303935952747, -0.0016866615335315, 
        -0.0000453462735421, 0.0011527913635811, 0.0018201119856549, 0.0003152517874437
    };
    
    // Generate high pass filter using quadrature mirror relationship
    int N = decomposition_low_pass_.size();
    decomposition_high_pass_.resize(N);
    for (int i = 0; i < N; i++) {
        decomposition_high_pass_[i] = std::pow(-1.0, i) * decomposition_low_pass_[N - 1 - i];
    }
    
    // For Meyer wavelets, the reconstruction filters are the same as decomposition filters
    reconstruction_low_pass_ = decomposition_low_pass_;
    reconstruction_high_pass_ = decomposition_high_pass_;
}

void WaveletTransform::generateMorletFilters() {
    // Morlet wavelet is primarily used for continuous wavelet transform
    // For CWT, we don't need the filter banks, but we'll define a sampled version
    // of the Morlet wavelet for reference
    
    // Parameters for Morlet wavelet
    double sigma = 1.0;
    double omega0 = 5.0; // Center frequency
    
    // Generate sampled Morlet wavelet
    int N = 64; // Number of sample points
    double dt = 0.125; // Time step
    
    decomposition_low_pass_.resize(N);
    for (int i = 0; i < N; i++) {
        double t = (i - N/2) * dt;
        double gauss = std::exp(-0.5 * t * t / (sigma * sigma));
        double cos_term = std::cos(omega0 * t);
        double normalization = 1.0 / (std::sqrt(2.0 * M_PI) * sigma);
        
        decomposition_low_pass_[i] = normalization * gauss * cos_term;
    }
    
    // For CWT, we don't use high pass or reconstruction filters
    // But we'll set them to something for completeness
    decomposition_high_pass_ = decomposition_low_pass_;
    reconstruction_low_pass_ = decomposition_low_pass_;
    reconstruction_high_pass_ = decomposition_low_pass_;
}

void WaveletTransform::generateMexicanHatFilters() {
    // Mexican Hat (Ricker) wavelet is the second derivative of a Gaussian
    // It's also primarily used for continuous wavelet transform
    
    // Parameters for Mexican Hat wavelet
    double sigma = 1.0;
    
    // Generate sampled Mexican Hat wavelet
    int N = 64; // Number of sample points
    double dt = 0.125; // Time step
    
    decomposition_low_pass_.resize(N);
    for (int i = 0; i < N; i++) {
        double t = (i - N/2) * dt;
        double t_squared = t * t;
        double sigma_squared = sigma * sigma;
        
        // Mexican Hat formula: (1 - t²/σ²) * exp(-t²/(2σ²))
        double gauss = std::exp(-0.5 * t_squared / sigma_squared);
        double factor = 1.0 - t_squared / sigma_squared;
        double normalization = 2.0 / (std::sqrt(3.0 * sigma) * std::pow(M_PI, 0.25));
        
        decomposition_low_pass_[i] = normalization * factor * gauss;
    }
    
    // For CWT, we don't use high pass or reconstruction filters
    // But we'll set them to something for completeness
    decomposition_high_pass_ = decomposition_low_pass_;
    reconstruction_low_pass_ = decomposition_low_pass_;
    reconstruction_high_pass_ = decomposition_low_pass_;
}

std::vector<float> WaveletTransform::convolve(const std::vector<float>& signal, 
                                             const std::vector<float>& filter) {
    int signal_len = signal.size();
    int filter_len = filter.size();
    int result_len = signal_len + filter_len - 1;
    
    std::vector<float> result(result_len, 0.0f);
    
    for (int i = 0; i < result_len; i++) {
        for (int j = 0; j < filter_len; j++) {
            int signal_idx = i - j;
            if (signal_idx >= 0 && signal_idx < signal_len) {
                result[i] += signal[signal_idx] * filter[j];
            }
        }
    }
    
    return result;
}

std::vector<float> WaveletTransform::downsample(const std::vector<float>& signal) {
    int original_size = signal.size();
    int new_size = original_size / 2;
    
    std::vector<float> downsampled(new_size);
    
    for (int i = 0; i < new_size; i++) {
        downsampled[i] = signal[i * 2];
    }
    
    return downsampled;
}

std::vector<float> WaveletTransform::upsample(const std::vector<float>& signal) {
    int original_size = signal.size();
    int new_size = original_size * 2;
    
    std::vector<float> upsampled(new_size, 0.0f);
    
    for (int i = 0; i < original_size; i++) {
        upsampled[i * 2] = signal[i];
    }
    
    return upsampled;
}

std::vector<float> WaveletTransform::extendSignal(const std::vector<float>& signal, 
                                                int filter_length, 
                                                BoundaryMode mode) {
    int extension_size = filter_length - 1;
    int signal_size = signal.size();
    std::vector<float> extended(signal_size + 2 * extension_size, 0.0f);
    
    // Copy the original signal to the middle
    for (int i = 0; i < signal_size; i++) {
        extended[i + extension_size] = signal[i];
    }
    
    // Handle the boundaries based on the specified mode
    switch (mode) {
        case BoundaryMode::ZERO_PADDING:
            // Already initialized to zeros
            break;
        
        case BoundaryMode::SYMMETRIC:
            // Left boundary (symmetric extension)
            for (int i = 0; i < extension_size; i++) {
                extended[extension_size - 1 - i] = signal[i];
            }
            // Right boundary (symmetric extension)
            for (int i = 0; i < extension_size; i++) {
                extended[signal_size + extension_size + i] = signal[signal_size - 1 - i];
            }
            break;
        
        case BoundaryMode::PERIODIC:
            // Left boundary (periodic extension)
            for (int i = 0; i < extension_size; i++) {
                extended[i] = signal[signal_size - extension_size + i];
            }
            // Right boundary (periodic extension)
            for (int i = 0; i < extension_size; i++) {
                extended[signal_size + extension_size + i] = signal[i];
            }
            break;
            
        case BoundaryMode::REFLECT:
            // Left boundary (reflection)
            for (int i = 0; i < extension_size; i++) {
                extended[i] = signal[extension_size - i];
            }
            // Right boundary (reflection)
            for (int i = 0; i < extension_size; i++) {
                extended[signal_size + extension_size + i] = signal[signal_size - 2 - i];
            }
            break;
    }
    
    return extended;
}

// ----- DiscreteWaveletTransform Implementation -----

DiscreteWaveletTransform::DiscreteWaveletTransform(WaveletFamily family, int vanishing_moments)
    : WaveletTransform(family, vanishing_moments) {
}

WaveletTransformResult DiscreteWaveletTransform::forward(const std::vector<float>& signal, 
                                                       int levels, 
                                                       BoundaryMode mode) {
    if (signal.empty()) {
        throw std::invalid_argument("Input signal cannot be empty");
    }
    
    if (levels <= 0) {
        throw std::invalid_argument("Number of decomposition levels must be positive");
    }
    
    // Initialize the result structure
    WaveletTransformResult result;
    result.approximation_coefficients.resize(levels + 1);
    result.detail_coefficients.resize(levels);
    
    // The initial approximation is the input signal
    result.approximation_coefficients[0] = signal;
    
    // Filter length
    int filter_length = decomposition_low_pass_.size();
    
    // Perform multi-level decomposition
    for (int level = 0; level < levels; level++) {
        const std::vector<float>& current_approx = result.approximation_coefficients[level];
        
        // Extend the signal to handle boundary effects
        std::vector<float> extended = extendSignal(current_approx, filter_length, mode);
        
        // Apply the low-pass filter and downsample
        std::vector<float> approx_convolved = convolve(extended, decomposition_low_pass_);
        std::vector<float> next_approx = downsample(approx_convolved);
        
        // Remove the boundary extension
        next_approx.erase(next_approx.begin(), next_approx.begin() + (filter_length - 1) / 2);
        next_approx.erase(next_approx.end() - (filter_length - 1) / 2, next_approx.end());
        
        // Apply the high-pass filter and downsample
        std::vector<float> detail_convolved = convolve(extended, decomposition_high_pass_);
        std::vector<float> detail = downsample(detail_convolved);
        
        // Remove the boundary extension
        detail.erase(detail.begin(), detail.begin() + (filter_length - 1) / 2);
        detail.erase(detail.end() - (filter_length - 1) / 2, detail.end());
        
        // Store the results
        result.approximation_coefficients[level + 1] = next_approx;
        result.detail_coefficients[level] = detail;
    }
    
    return result;
}

std::vector<float> DiscreteWaveletTransform::inverse(const WaveletTransformResult& transform_result, 
                                                  BoundaryMode mode) {
    int levels = transform_result.detail_coefficients.size();
    
    if (levels == 0 || transform_result.approximation_coefficients.size() != levels + 1) {
        throw std::invalid_argument("Invalid transform result structure");
    }
    
    // Start with the coarsest approximation
    std::vector<float> reconstruction = transform_result.approximation_coefficients[levels];
    
    // Filter length
    int filter_length = reconstruction_low_pass_.size();
    
    // Perform multi-level reconstruction
    for (int level = levels - 1; level >= 0; level--) {
        // Upsample the current reconstruction
        std::vector<float> upsampled_approx = upsample(reconstruction);
        
        // Upsample the detail coefficients at this level
        std::vector<float> upsampled_detail = upsample(transform_result.detail_coefficients[level]);
        
        // Extend signals to handle boundary effects
        std::vector<float> extended_approx = extendSignal(upsampled_approx, filter_length, mode);
        std::vector<float> extended_detail = extendSignal(upsampled_detail, filter_length, mode);
        
        // Apply reconstruction filters
        std::vector<float> approx_conv = convolve(extended_approx, reconstruction_low_pass_);
        std::vector<float> detail_conv = convolve(extended_detail, reconstruction_high_pass_);
        
        // Combine the approximation and detail components
        int result_size = transform_result.approximation_coefficients[level].size();
        reconstruction.resize(result_size);
        
        for (int i = 0; i < result_size; i++) {
            int idx = i + filter_length - 1;
            reconstruction[i] = approx_conv[idx] + detail_conv[idx];
        }
    }
    
    return reconstruction;
}

// ----- ContinuousWaveletTransform Implementation -----

ContinuousWaveletTransform::ContinuousWaveletTransform(WaveletFamily family, int vanishing_moments)
    : WaveletTransform(family, vanishing_moments) {
}

std::vector<std::vector<std::complex<float>>> ContinuousWaveletTransform::forward(
        const std::vector<float>& signal, 
        const std::vector<float>& scales) {
    if (signal.empty()) {
        throw std::invalid_argument("Input signal cannot be empty");
    }
    
    if (scales.empty()) {
        throw std::invalid_argument("Scales vector cannot be empty");
    }
    
    int signal_length = signal.size();
    int num_scales = scales.size();
    
    // Initialize the result - a 2D array of complex coefficients
    std::vector<std::vector<std::complex<float>>> coefficients(num_scales);
    
    // Handle different wavelet families
    if (family_ == WaveletFamily::MORLET) {
        // Parameters for Morlet wavelet
        double omega0 = 5.0; // Center frequency
        
        // Compute CWT for each scale
        for (int i = 0; i < num_scales; i++) {
            float scale = scales[i];
            coefficients[i].resize(signal_length);
            
            // For each position in the signal
            for (int b = 0; b < signal_length; b++) {
                std::complex<float> sum(0.0f, 0.0f);
                
                // Apply wavelet centered at position b with scale a
                for (int k = 0; k < signal_length; k++) {
                    double t = (k - b) / scale;
                    
                    // Morlet wavelet: exp(-t²/2) * exp(iω₀t)
                    double gauss = std::exp(-0.5 * t * t);
                    std::complex<float> wavelet(gauss * std::cos(omega0 * t), 
                                               gauss * std::sin(omega0 * t));
                    
                    // Multiply signal value by complex conjugate of wavelet
                    sum += signal[k] * std::conj(wavelet);
                }
                
                // Normalize by square root of scale
                coefficients[i][b] = sum / std::sqrt(scale);
            }
        }
    } else if (family_ == WaveletFamily::MEXICAN_HAT) {
        // Compute CWT for each scale
        for (int i = 0; i < num_scales; i++) {
            float scale = scales[i];
            coefficients[i].resize(signal_length);
            
            // For each position in the signal
            for (int b = 0; b < signal_length; b++) {
                std::complex<float> sum(0.0f, 0.0f);
                
                // Apply wavelet centered at position b with scale a
                for (int k = 0; k < signal_length; k++) {
                    double t = (k - b) / scale;
                    double t_squared = t * t;
                    
                    // Mexican Hat formula: (1 - t²) * exp(-t²/2)
                    double wavelet_val = (1.0 - t_squared) * std::exp(-0.5 * t_squared);
                    std::complex<float> wavelet(wavelet_val, 0.0f);
                    
                    // Multiply signal value by wavelet
                    sum += signal[k] * wavelet;
                }
                
                // Normalize by square root of scale
                coefficients[i][b] = sum / std::sqrt(scale);
            }
        }
    } else {
        throw std::invalid_argument("Unsupported wavelet family for CWT");
    }
    
    return coefficients;
}

std::vector<float> ContinuousWaveletTransform::inverse(
        const std::vector<std::vector<std::complex<float>>>& coefficients,
        const std::vector<float>& scales) {
    if (coefficients.empty() || scales.empty()) {
        throw std::invalid_argument("Coefficients or scales cannot be empty");
    }
    
    int num_scales = coefficients.size();
    if (num_scales != scales.size()) {
        throw std::invalid_argument("Number of coefficient rows must match number of scales");
    }
    
    int signal_length = coefficients[0].size();
    
    // Initialize the reconstructed signal
    std::vector<float> reconstruction(signal_length, 0.0f);
    
    // Handle different wavelet families
    if (family_ == WaveletFamily::MORLET) {
        // Parameters for Morlet wavelet
        double omega0 = 5.0; // Center frequency
        double Cpsi = 0.7764; // Admissibility constant for Morlet
        
        // Compute the reconstruction for each point
        for (int b = 0; b < signal_length; b++) {
            double sum = 0.0;
            
            // Sum over all scales
            for (int i = 0; i < num_scales; i++) {
                float scale = scales[i];
                
                // Add contribution from this coefficient
                sum += std::real(coefficients[i][b]) / (scale * scale);
                
                // For proper reconstruction, we should integrate over scales
                // In practice, we're using a discrete sum with appropriate spacing
                // If scales are logarithmically spaced, adjust accordingly
                if (i < num_scales - 1) {
                    double dscale = scales[i+1] - scale;
                    sum *= dscale;
                }
            }
            
            // Apply the admissibility factor
            reconstruction[b] = sum / Cpsi;
        }
    } else if (family_ == WaveletFamily::MEXICAN_HAT) {
        // Admissibility constant for Mexican Hat
        double Cpsi = 1.2533; // π^(-1/4)
        
        // Compute the reconstruction for each point
        for (int b = 0; b < signal_length; b++) {
            double sum = 0.0;
            
            // Sum over all scales
            for (int i = 0; i < num_scales; i++) {
                float scale = scales[i];
                
                // Add contribution from this coefficient
                sum += std::real(coefficients[i][b]) / (scale * scale);
                
                // Scale integration factor
                if (i < num_scales - 1) {
                    double dscale = scales[i+1] - scale;
                    sum *= dscale;
                }
            }
            
            // Apply the admissibility factor
            reconstruction[b] = sum / Cpsi;
        }
    } else {
        throw std::invalid_argument("Unsupported wavelet family for inverse CWT");
    }
    
    return reconstruction;
}

std::vector<float> ContinuousWaveletTransform::generateScales(int num_scales, float min_scale, float max_scale) {
    if (num_scales <= 0) {
        throw std::invalid_argument("Number of scales must be positive");
    }
    
    if (min_scale <= 0 || max_scale <= 0 || min_scale >= max_scale) {
        throw std::invalid_argument("Scale range must be positive and min_scale < max_scale");
    }
    
    std::vector<float> scales(num_scales);
    
    // Generate logarithmically spaced scales
    double log_min = std::log(min_scale);
    double log_max = std::log(max_scale);
    double step = (log_max - log_min) / (num_scales - 1);
    
    for (int i = 0; i < num_scales; i++) {
        scales[i] = std::exp(log_min + i * step);
    }
    
    return scales;
}

// ----- WaveletPacketTransform Implementation -----

WaveletPacketTransform::WaveletPacketTransform(WaveletFamily family, int vanishing_moments)
    : WaveletTransform(family, vanishing_moments) {
}

WaveletPacketResult WaveletPacketTransform::forward(const std::vector<float>& signal, 
                                                  int levels, 
                                                  BoundaryMode mode) {
    if (signal.empty()) {
        throw std::invalid_argument("Input signal cannot be empty");
    }
    
    if (levels <= 0) {
        throw std::invalid_argument("Number of decomposition levels must be positive");
    }
    
    // Initialize the result structure
    WaveletPacketResult result;
    result.coefficients.resize(levels + 1);
    
    // The first level has only one node - the original signal
    result.coefficients[0].resize(1);
    result.coefficients[0][0] = signal;
    
    // Filter length
    int filter_length = decomposition_low_pass_.size();
    
    // Perform wavelet packet decomposition
    for (int level = 0; level < levels; level++) {
        int num_nodes = 1 << level; // 2^level
        int next_num_nodes = 1 << (level + 1); // 2^(level+1)
        
        result.coefficients[level + 1].resize(next_num_nodes);
        
        // Process each node at the current level
        for (int node = 0; node < num_nodes; node++) {
            const std::vector<float>& current_signal = result.coefficients[level][node];
            
            // Extend the signal to handle boundary effects
            std::vector<float> extended = extendSignal(current_signal, filter_length, mode);
            
            // Apply the low-pass filter and downsample (approximation)
            std::vector<float> approx_convolved = convolve(extended, decomposition_low_pass_);
            std::vector<float> approximation = downsample(approx_convolved);
            
            // Remove the boundary extension
            approximation.erase(approximation.begin(), approximation.begin() + (filter_length - 1) / 2);
            approximation.erase(approximation.end() - (filter_length - 1) / 2, approximation.end());
            
            // Apply the high-pass filter and downsample (detail)
            std::vector<float> detail_convolved = convolve(extended, decomposition_high_pass_);
            std::vector<float> detail = downsample(detail_convolved);
            
            // Remove the boundary extension
            detail.erase(detail.begin(), detail.begin() + (filter_length - 1) / 2);
            detail.erase(detail.end() - (filter_length - 1) / 2, detail.end());
            
            // Store the results in the next level
            // Left child (approximation)
            result.coefficients[level + 1][2 * node] = approximation;
            // Right child (detail)
            result.coefficients[level + 1][2 * node + 1] = detail;
        }
    }
    
    return result;
}

std::vector<float> WaveletPacketTransform::inverse(const WaveletPacketResult& packet_result, 
                                                BoundaryMode mode) {
    if (packet_result.coefficients.empty()) {
        throw std::invalid_argument("Packet result structure cannot be empty");
    }
    
    int levels = packet_result.coefficients.size() - 1;
    
    if (levels <= 0) {
        throw std::invalid_argument("Invalid packet result structure");
    }
    
    // Create a copy of the result that we'll modify
    std::vector<std::vector<std::vector<float>>> reconstruction = packet_result.coefficients;
    
    // Filter length
    int filter_length = reconstruction_low_pass_.size();
    
    // Perform reconstruction from bottom to top
    for (int level = levels; level > 0; level--) {
        int num_nodes = 1 << (level - 1); // 2^(level-1)
        
        // Process each node at the level we're reconstructing to
        for (int node = 0; node < num_nodes; node++) {
            // Get the left and right children (approximation and detail)
            const std::vector<float>& approximation = reconstruction[level][2 * node];
            const std::vector<float>& detail = reconstruction[level][2 * node + 1];
            
            // Upsample
            std::vector<float> upsampled_approx = upsample(approximation);
            std::vector<float> upsampled_detail = upsample(detail);
            
            // Extend signals to handle boundary effects
            std::vector<float> extended_approx = extendSignal(upsampled_approx, filter_length, mode);
            std::vector<float> extended_detail = extendSignal(upsampled_detail, filter_length, mode);
            
            // Apply reconstruction filters
            std::vector<float> approx_conv = convolve(extended_approx, reconstruction_low_pass_);
            std::vector<float> detail_conv = convolve(extended_detail, reconstruction_high_pass_);
            
            // Combine the approximation and detail components
            int expected_size = reconstruction[level-1][node].size();
            std::vector<float> combined(expected_size);
            
            for (int i = 0; i < expected_size; i++) {
                int idx = i + filter_length - 1;
                combined[i] = approx_conv[idx] + detail_conv[idx];
            }
            
            // Store the result in the upper level
            reconstruction[level - 1][node] = combined;
        }
    }
    
    // The final result is the single node at the top level
    return reconstruction[0][0];
}

// ----- MaximalOverlapDWT Implementation -----

MaximalOverlapDWT::MaximalOverlapDWT(WaveletFamily family, int vanishing_moments)
    : WaveletTransform(family, vanishing_moments) {
}

WaveletTransformResult MaximalOverlapDWT::forward(const std::vector<float>& signal, 
                                               int levels, 
                                               BoundaryMode mode) {
    if (signal.empty()) {
        throw std::invalid_argument("Input signal cannot be empty");
    }
    
    if (levels <= 0) {
        throw std::invalid_argument("Number of decomposition levels must be positive");
    }
    
    // Initialize the result structure
    WaveletTransformResult result;
    result.approximation_coefficients.resize(levels + 1);
    result.detail_coefficients.resize(levels);
    
    int signal_length = signal.size();
    
    // The initial approximation is the input signal
    result.approximation_coefficients[0] = signal;
    
    // Filter length
    int filter_length = decomposition_low_pass_.size();
    
    // Perform multi-level MODWT decomposition
    for (int level = 0; level < levels; level++) {
        const std::vector<float>& current_approx = result.approximation_coefficients[level];
        
        // Extend the signal to handle boundary effects
        std::vector<float> extended = extendSignal(current_approx, filter_length, mode);
        
        // Scale the filters by 1/sqrt(2) for each level
        std::vector<float> scaled_low_pass = decomposition_low_pass_;
        std::vector<float> scaled_high_pass = decomposition_high_pass_;
        
        float scale_factor = 1.0f / std::sqrt(2.0f);
        for (int i = 0; i < filter_length; i++) {
            scaled_low_pass[i] *= scale_factor;
            scaled_high_pass[i] *= scale_factor;
        }
        
        // Apply the filters without downsampling
        std::vector<float> next_approx = convolve(extended, scaled_low_pass);
        std::vector<float> detail = convolve(extended, scaled_high_pass);
        
        // Trim to original length
        next_approx.resize(signal_length);
        detail.resize(signal_length);
        
        // Store the results
        result.approximation_coefficients[level + 1] = next_approx;
        result.detail_coefficients[level] = detail;
    }
    
    return result;
}

std::vector<float> MaximalOverlapDWT::inverse(const WaveletTransformResult& transform_result, 
                                           BoundaryMode mode) {
    int levels = transform_result.detail_coefficients.size();
    
    if (levels == 0 || transform_result.approximation_coefficients.size() != levels + 1) {
        throw std::invalid_argument("Invalid transform result structure");
    }
    
    int signal_length = transform_result.approximation_coefficients[0].size();
    
    // Start with the coarsest approximation
    std::vector<float> reconstruction = transform_result.approximation_coefficients[levels];
    
    // Filter length
    int filter_length = reconstruction_low_pass_.size();
    
    // Perform multi-level reconstruction
    for (int level = levels - 1; level >= 0; level--) {
        // Scale the filters by 1/sqrt(2) for each level
        std::vector<float> scaled_low_pass = reconstruction_low_pass_;
        std::vector<float> scaled_high_pass = reconstruction_high_pass_;
        
        float scale_factor = 1.0f / std::sqrt(2.0f);
        for (int i = 0; i < filter_length; i++) {
            scaled_low_pass[i] *= scale_factor;
            scaled_high_pass[i] *= scale_factor;
        }
        
        // Extend signals to handle boundary effects
        std::vector<float> extended_approx = extendSignal(reconstruction, filter_length, mode);
        std::vector<float> extended_detail = extendSignal(transform_result.detail_coefficients[level], filter_length, mode);
        
        // Apply reconstruction filters
        std::vector<float> approx_conv = convolve(extended_approx, scaled_low_pass);
        std::vector<float> detail_conv = convolve(extended_detail, scaled_high_pass);
        
        // Combine the approximation and detail components
        reconstruction.resize(signal_length);
        
        for (int i = 0; i < signal_length; i++) {
            reconstruction[i] = approx_conv[i + filter_length - 1] + detail_conv[i + filter_length - 1];
        }
    }
    
    return reconstruction;
}

} // namespace signal_processing