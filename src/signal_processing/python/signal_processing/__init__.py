"""
Signal Processing Module

This package provides GPU-accelerated signal processing functionality,
optimized for NVIDIA Jetson Orin NX and AWS Graviton g5g instances.

Key components:
- FFT and spectral analysis
- Digital filtering
- Time-frequency analysis
- Adaptive filtering
- Wavelet transforms (DWT, CWT, WPT, MODWT)

This module wraps the C++/CUDA implementation for performance,
while providing a Pythonic interface.
"""

from .spectral import (
    FFT,
    SpectralAnalyzer,
    compute_psd,
    compute_spectrogram,
    compute_coherence,
    detect_peaks
)

from .filters import (
    FIRFilter,
    IIRFilter,
    AdaptiveFilter,
    MultirateFilter,
    design_lowpass,
    design_highpass,
    design_bandpass,
    design_bandstop,
    median_filter,
    wiener_filter,
    kalman_filter
)

from .tf_analysis import (
    STFT,
    CWT,
    DWT,
    WignerVille,
    EMD,
    compute_scalogram,
    compute_mel_spectrogram,
    compute_mfcc
)

from .wavelet import (
    WaveletFamily,
    BoundaryMode,
    DiscreteWaveletTransform,
    ContinuousWaveletTransform,
    WaveletPacketTransform,
    MaximalOverlapDWT,
    denoise_signal,
    generate_test_signal,
    generate_chirp_signal
)

# Version information
__version__ = '0.1.0'