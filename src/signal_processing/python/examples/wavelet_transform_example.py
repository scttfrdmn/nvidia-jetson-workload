#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Wavelet Transform Example Script

This script demonstrates the wavelet transform capabilities
of the Signal Processing module.

It shows:
1. Discrete Wavelet Transform for multi-resolution analysis
2. Continuous Wavelet Transform for time-frequency analysis
3. Wavelet Packet Transform for decomposition flexibility
4. Maximal Overlap DWT for shift-invariant analysis
5. Wavelet-based denoising

The example is optimized for NVIDIA Jetson Orin NX and uses
CUDA acceleration when available.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Add parent directory to path to import signal_processing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from signal_processing.wavelet import (
    WaveletFamily, BoundaryMode,
    DiscreteWaveletTransform, ContinuousWaveletTransform,
    WaveletPacketTransform, MaximalOverlapDWT,
    denoise_signal, generate_test_signal, generate_chirp_signal
)


def plot_wavelet_decomposition(signal, coeffs, title):
    """Plot wavelet decomposition results."""
    levels = len(coeffs['detail'])
    plt.figure(figsize=(12, 8))
    
    # Plot original signal
    plt.subplot(levels+2, 1, 1)
    plt.plot(signal)
    plt.title(f'Original Signal - {title}')
    plt.grid(True)
    
    # Plot approximation at deepest level
    plt.subplot(levels+2, 1, 2)
    plt.plot(coeffs['approximation'][-1])
    plt.title(f'Approximation (Level {levels})')
    plt.grid(True)
    
    # Plot details from each level
    for i, detail in enumerate(coeffs['detail']):
        plt.subplot(levels+2, 1, i+3)
        plt.plot(detail)
        plt.title(f'Detail (Level {i+1})')
        plt.grid(True)
    
    plt.tight_layout()


def plot_cwt_scalogram(signal, coeffs, scales, title):
    """Plot CWT scalogram."""
    plt.figure(figsize=(12, 6))
    
    # Plot original signal
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title(f'Signal - {title}')
    plt.grid(True)
    
    # Plot scalogram
    plt.subplot(2, 1, 2)
    scalogram = np.abs(coeffs)**2
    plt.imshow(scalogram, aspect='auto', origin='lower', 
              extent=[0, len(signal), np.log10(scales[0]), np.log10(scales[-1])],
              cmap='jet', interpolation='bilinear', norm=LogNorm())
    plt.colorbar(label='Power')
    plt.ylabel('log10(Scale)')
    plt.title('Scalogram (CWT Power)')
    
    plt.tight_layout()


def example_dwt():
    """Demonstrate Discrete Wavelet Transform."""
    print("Discrete Wavelet Transform Example")
    
    # Generate a test signal with two frequency components
    t = np.linspace(0, 1, 1024)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    
    # Create DWT object with Daubechies 4 wavelet
    dwt = DiscreteWaveletTransform(WaveletFamily.DAUBECHIES, vanishing_moments=4)
    
    # Perform 5-level decomposition
    start_time = time.time()
    coeffs = dwt.forward(signal, levels=5, mode=BoundaryMode.SYMMETRIC)
    decomp_time = time.time() - start_time
    
    # Reconstruct signal
    start_time = time.time()
    reconstructed = dwt.inverse(coeffs)
    recon_time = time.time() - start_time
    
    # Compute error
    mse = np.mean((signal - reconstructed)**2)
    print(f"DWT Decomposition time: {decomp_time:.6f} seconds")
    print(f"DWT Reconstruction time: {recon_time:.6f} seconds")
    print(f"Reconstruction MSE: {mse:.10f}")
    
    # Plot results
    plot_wavelet_decomposition(signal, coeffs, "DWT Example")
    
    # Also plot the original and reconstructed signals
    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Original')
    plt.plot(reconstructed, 'r--', label='Reconstructed')
    plt.legend()
    plt.grid(True)
    plt.title('DWT - Original vs Reconstructed')


def example_cwt():
    """Demonstrate Continuous Wavelet Transform."""
    print("\nContinuous Wavelet Transform Example")
    
    # Generate a chirp signal (frequency changing with time)
    signal = generate_chirp_signal(length=1024, f0=5, f1=50, sample_rate=1000)
    
    # Create CWT object with Morlet wavelet
    cwt = ContinuousWaveletTransform(WaveletFamily.MORLET)
    
    # Generate logarithmically spaced scales
    scales = cwt.generate_scales(64, 1, 64)
    
    # Perform CWT
    start_time = time.time()
    coeffs = cwt.forward(signal, scales)
    transform_time = time.time() - start_time
    print(f"CWT computation time: {transform_time:.6f} seconds")
    
    # Plot time-frequency scalogram
    plot_cwt_scalogram(signal, coeffs, scales, "Chirp Signal")


def example_wpt():
    """Demonstrate Wavelet Packet Transform."""
    print("\nWavelet Packet Transform Example")
    
    # Generate a test signal with multiple frequency components
    t = np.linspace(0, 1, 1024)
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.3 * np.sin(2 * np.pi * 50 * t)
    
    # Create WPT object with Symlet wavelet
    wpt = WaveletPacketTransform(WaveletFamily.SYMLET, vanishing_moments=4)
    
    # Perform 3-level decomposition
    start_time = time.time()
    wpt_result = wpt.forward(signal, levels=3, mode=BoundaryMode.SYMMETRIC)
    transform_time = time.time() - start_time
    
    # Reconstruct signal
    start_time = time.time()
    reconstructed = wpt.inverse(wpt_result)
    recon_time = time.time() - start_time
    
    # Compute error
    mse = np.mean((signal - reconstructed)**2)
    print(f"WPT computation time: {transform_time:.6f} seconds")
    print(f"WPT reconstruction time: {recon_time:.6f} seconds")
    print(f"Reconstruction MSE: {mse:.10f}")
    
    # Plot original and reconstructed signals
    plt.figure(figsize=(10, 6))
    plt.plot(signal, label='Original')
    plt.plot(reconstructed, 'r--', label='Reconstructed')
    plt.legend()
    plt.grid(True)
    plt.title('WPT - Original vs Reconstructed')
    
    # Plot coefficient tree visualization
    coeffs = wpt_result['coefficients']
    plt.figure(figsize=(15, 10))
    
    # Plot all levels
    for level in range(wpt_result['levels'] + 1):
        plt.subplot(wpt_result['levels'] + 1, 1, level + 1)
        
        # Concatenate all coefficients at this level
        level_coeffs = []
        for node in range(2**level):
            node_key = (level, node)
            if node_key in coeffs:
                level_coeffs.append(coeffs[node_key])
        
        if level_coeffs:
            all_coeffs = np.concatenate(level_coeffs)
            plt.plot(all_coeffs)
            plt.title(f'Level {level} Coefficients - {2**level} nodes')
            plt.grid(True)
    
    plt.tight_layout()


def example_modwt():
    """Demonstrate Maximal Overlap Discrete Wavelet Transform."""
    print("\nMaximal Overlap DWT Example")
    
    # Generate a signal with an abrupt change
    t = np.linspace(0, 1, 1024)
    signal = np.sin(2 * np.pi * 5 * t)
    # Add step discontinuity
    signal[512:] += 0.5
    
    # Create MODWT object
    modwt = MaximalOverlapDWT(WaveletFamily.DAUBECHIES, vanishing_moments=4)
    
    # Perform 4-level decomposition
    start_time = time.time()
    coeffs = modwt.forward(signal, levels=4, mode=BoundaryMode.PERIODIC)
    transform_time = time.time() - start_time
    
    # Reconstruct signal
    start_time = time.time()
    reconstructed = modwt.inverse(coeffs)
    recon_time = time.time() - start_time
    
    # Compute error
    mse = np.mean((signal - reconstructed)**2)
    print(f"MODWT computation time: {transform_time:.6f} seconds")
    print(f"MODWT reconstruction time: {recon_time:.6f} seconds")
    print(f"Reconstruction MSE: {mse:.10f}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot original signal
    plt.subplot(6, 1, 1)
    plt.plot(signal)
    plt.title('Original Signal with Step')
    plt.grid(True)
    
    # Plot reconstructed signal
    plt.subplot(6, 1, 2)
    plt.plot(reconstructed)
    plt.title('Reconstructed Signal')
    plt.grid(True)
    
    # Plot scaling coefficients
    plt.subplot(6, 1, 3)
    plt.plot(coeffs['scaling'][-1])
    plt.title(f'Scaling Coefficients (Level 4)')
    plt.grid(True)
    
    # Plot wavelet coefficients from each level
    for i in range(3):
        plt.subplot(6, 1, i+4)
        plt.plot(coeffs['wavelet'][i])
        plt.title(f'Wavelet Coefficients (Level {i+1})')
        plt.grid(True)
    
    plt.tight_layout()


def example_denoising():
    """Demonstrate wavelet-based denoising."""
    print("\nWavelet Denoising Example")
    
    # Generate a clean signal
    t = np.linspace(0, 1, 1024)
    clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    
    # Add noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, 0.2, len(clean_signal))
    noisy_signal = clean_signal + noise
    
    # Apply wavelet denoising with different wavelets
    start_time = time.time()
    denoised_db4 = denoise_signal(noisy_signal, WaveletFamily.DAUBECHIES, 4, levels=4)
    denoised_sym4 = denoise_signal(noisy_signal, WaveletFamily.SYMLET, 4, levels=4)
    denoising_time = time.time() - start_time
    
    # Compute SNR
    def compute_snr(clean, noisy):
        noise_power = np.mean((clean - noisy)**2)
        signal_power = np.mean(clean**2)
        return 10 * np.log10(signal_power / noise_power)
    
    input_snr = compute_snr(clean_signal, noisy_signal)
    output_snr_db4 = compute_snr(clean_signal, denoised_db4)
    output_snr_sym4 = compute_snr(clean_signal, denoised_sym4)
    
    print(f"Denoising time: {denoising_time:.6f} seconds")
    print(f"Input SNR: {input_snr:.2f} dB")
    print(f"Output SNR (DB4): {output_snr_db4:.2f} dB")
    print(f"Output SNR (SYM4): {output_snr_sym4:.2f} dB")
    print(f"SNR Improvement (DB4): {output_snr_db4 - input_snr:.2f} dB")
    print(f"SNR Improvement (SYM4): {output_snr_sym4 - input_snr:.2f} dB")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot clean signal
    plt.subplot(4, 1, 1)
    plt.plot(clean_signal)
    plt.title('Clean Signal')
    plt.grid(True)
    
    # Plot noisy signal
    plt.subplot(4, 1, 2)
    plt.plot(noisy_signal)
    plt.title(f'Noisy Signal (SNR: {input_snr:.2f} dB)')
    plt.grid(True)
    
    # Plot denoised signal with DB4
    plt.subplot(4, 1, 3)
    plt.plot(denoised_db4)
    plt.title(f'Denoised with DB4 (SNR: {output_snr_db4:.2f} dB)')
    plt.grid(True)
    
    # Plot denoised signal with SYM4
    plt.subplot(4, 1, 4)
    plt.plot(denoised_sym4)
    plt.title(f'Denoised with SYM4 (SNR: {output_snr_sym4:.2f} dB)')
    plt.grid(True)
    
    plt.tight_layout()


def performance_test():
    """Test performance of different wavelet transforms."""
    print("\nPerformance Comparison")
    
    # Signal lengths to test
    lengths = [1024, 4096, 16384]
    
    # Results table
    print("\nExecution Time (seconds)")
    print("=" * 80)
    print(f"{'Signal Length':<15} {'DWT':<12} {'CWT':<12} {'WPT':<12} {'MODWT':<12}")
    print("-" * 80)
    
    for length in lengths:
        # Generate test signal
        signal = generate_test_signal(length)
        
        # Test DWT
        dwt = DiscreteWaveletTransform(WaveletFamily.DAUBECHIES, 4)
        start = time.time()
        dwt.forward(signal, levels=3)
        dwt_time = time.time() - start
        
        # Test CWT
        cwt = ContinuousWaveletTransform(WaveletFamily.MORLET)
        scales = cwt.generate_scales(32, 1, 32)
        start = time.time()
        cwt.forward(signal, scales)
        cwt_time = time.time() - start
        
        # Test WPT
        wpt = WaveletPacketTransform(WaveletFamily.DAUBECHIES, 4)
        start = time.time()
        wpt.forward(signal, levels=3)
        wpt_time = time.time() - start
        
        # Test MODWT
        modwt = MaximalOverlapDWT(WaveletFamily.DAUBECHIES, 4)
        start = time.time()
        modwt.forward(signal, levels=3)
        modwt_time = time.time() - start
        
        # Print results
        print(f"{length:<15} {dwt_time:<12.6f} {cwt_time:<12.6f} {wpt_time:<12.6f} {modwt_time:<12.6f}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Run examples
    print("Wavelet Transform Examples")
    print("=" * 50)
    
    # Check for GPU acceleration
    try:
        import _signal_processing
        has_cuda = hasattr(_signal_processing, 'has_cuda') and _signal_processing.has_cuda()
        print(f"CUDA acceleration: {'Available' if has_cuda else 'Not available'}")
    except ImportError:
        print("C++ bindings not available, using pure Python implementation")
    
    print("=" * 50)
    
    # Run examples
    example_dwt()
    example_cwt()
    example_wpt()
    example_modwt()
    example_denoising()
    performance_test()
    
    # Show all plots
    plt.show()