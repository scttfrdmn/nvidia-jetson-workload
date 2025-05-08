"""
Spectral Analysis Module

This module provides GPU-accelerated spectral analysis functionality,
including FFT, Power Spectral Density estimation, spectrograms, etc.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from enum import Enum

# Import C++ extension module
try:
    from ._signal_processing import (
        FFT as _FFT,
        SpectralAnalyzer as _SpectralAnalyzer,
        SpectralParams as _SpectralParams,
        WindowType as _WindowType,
        PSDResult as _PSDResult,
        CSDResult as _CSDResult,
        SpectrogramResult as _SpectrogramResult
    )
    _HAS_CUDA = True
except ImportError:
    # Fallback to numpy-based implementation if C++ module not available
    _HAS_CUDA = False


class WindowType(Enum):
    """Window types for spectral analysis"""
    RECTANGULAR = 0
    HANN = 1
    HAMMING = 2
    BLACKMAN = 3
    FLATTOP = 4
    KAISER = 5
    TUKEY = 6
    GAUSSIAN = 7


class FFT:
    """Fast Fourier Transform (FFT) with GPU acceleration

    This class provides FFT operations optimized for NVIDIA GPUs,
    with automatic fallback to CPU implementation when CUDA is not available.
    
    Parameters
    ----------
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    """
    
    def __init__(self, device_id: int = 0):
        """Initialize the FFT processor
        
        Parameters
        ----------
        device_id : int, optional
            CUDA device ID to use (-1 for CPU), by default 0
        """
        self.device_id = device_id
        if _HAS_CUDA:
            self._fft = _FFT(device_id)
        self.has_cuda = _HAS_CUDA and self.device_id >= 0

    def forward_1d_real(
        self, 
        input_signal: np.ndarray, 
        normalize: bool = False
    ) -> np.ndarray:
        """Compute 1D forward FFT of real input
        
        Parameters
        ----------
        input_signal : np.ndarray
            Real-valued input signal
        normalize : bool, optional
            Whether to normalize the result, by default False
        
        Returns
        -------
        np.ndarray
            Complex FFT result
        """
        # Ensure input is float32 numpy array
        input_signal = np.asarray(input_signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            result = self._fft.forward_1d_real(input_signal, normalize)
            return np.array(result, dtype=np.complex64)
        else:
            # Fallback to numpy implementation
            result = np.fft.rfft(input_signal)
            if normalize:
                result /= np.sqrt(len(input_signal))
            return result
    
    def forward_1d_complex(
        self, 
        input_signal: np.ndarray, 
        normalize: bool = False
    ) -> np.ndarray:
        """Compute 1D forward FFT of complex input
        
        Parameters
        ----------
        input_signal : np.ndarray
            Complex-valued input signal
        normalize : bool, optional
            Whether to normalize the result, by default False
        
        Returns
        -------
        np.ndarray
            Complex FFT result
        """
        # Ensure input is complex64 numpy array
        input_signal = np.asarray(input_signal, dtype=np.complex64)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            result = self._fft.forward_1d_complex(input_signal, normalize)
            return np.array(result, dtype=np.complex64)
        else:
            # Fallback to numpy implementation
            result = np.fft.fft(input_signal)
            if normalize:
                result /= np.sqrt(len(input_signal))
            return result
    
    def inverse_1d_real(
        self, 
        input_spectrum: np.ndarray, 
        normalize: bool = True
    ) -> np.ndarray:
        """Compute 1D inverse FFT to real output
        
        Parameters
        ----------
        input_spectrum : np.ndarray
            Complex-valued input frequency domain data
        normalize : bool, optional
            Whether to normalize the result, by default True
        
        Returns
        -------
        np.ndarray
            Real FFT result
        """
        # Ensure input is complex64 numpy array
        input_spectrum = np.asarray(input_spectrum, dtype=np.complex64)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            result = self._fft.inverse_1d_real(input_spectrum, normalize)
            return np.array(result, dtype=np.float32)
        else:
            # Fallback to numpy implementation
            result = np.fft.irfft(input_spectrum)
            if not normalize:
                result *= len(result)
            return result
    
    def inverse_1d_complex(
        self, 
        input_spectrum: np.ndarray, 
        normalize: bool = True
    ) -> np.ndarray:
        """Compute 1D inverse FFT to complex output
        
        Parameters
        ----------
        input_spectrum : np.ndarray
            Complex-valued input frequency domain data
        normalize : bool, optional
            Whether to normalize the result, by default True
        
        Returns
        -------
        np.ndarray
            Complex FFT result
        """
        # Ensure input is complex64 numpy array
        input_spectrum = np.asarray(input_spectrum, dtype=np.complex64)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            result = self._fft.inverse_1d_complex(input_spectrum, normalize)
            return np.array(result, dtype=np.complex64)
        else:
            # Fallback to numpy implementation
            result = np.fft.ifft(input_spectrum)
            if not normalize:
                result *= len(result)
            return result
    
    def forward_2d_real(
        self, 
        input_signal: np.ndarray, 
        normalize: bool = False
    ) -> np.ndarray:
        """Compute 2D forward FFT of real input
        
        Parameters
        ----------
        input_signal : np.ndarray
            Real-valued input signal (2D array)
        normalize : bool, optional
            Whether to normalize the result, by default False
        
        Returns
        -------
        np.ndarray
            Complex FFT result
        """
        # Ensure input is float32 numpy array
        input_signal = np.asarray(input_signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            rows, cols = input_signal.shape
            result = self._fft.forward_2d_real(input_signal.flatten(), rows, cols, normalize)
            result = np.array(result, dtype=np.complex64)
            return result.reshape(rows, cols // 2 + 1)
        else:
            # Fallback to numpy implementation
            result = np.fft.rfft2(input_signal)
            if normalize:
                result /= np.sqrt(input_signal.size)
            return result
    
    def forward_2d_complex(
        self, 
        input_signal: np.ndarray, 
        normalize: bool = False
    ) -> np.ndarray:
        """Compute 2D forward FFT of complex input
        
        Parameters
        ----------
        input_signal : np.ndarray
            Complex-valued input signal (2D array)
        normalize : bool, optional
            Whether to normalize the result, by default False
        
        Returns
        -------
        np.ndarray
            Complex FFT result
        """
        # Ensure input is complex64 numpy array
        input_signal = np.asarray(input_signal, dtype=np.complex64)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            rows, cols = input_signal.shape
            result = self._fft.forward_2d_complex(input_signal.flatten(), rows, cols, normalize)
            result = np.array(result, dtype=np.complex64)
            return result.reshape(rows, cols)
        else:
            # Fallback to numpy implementation
            result = np.fft.fft2(input_signal)
            if normalize:
                result /= np.sqrt(input_signal.size)
            return result


class SpectralAnalyzer:
    """Spectral analysis for signal processing
    
    This class provides functions for spectral analysis, including:
    - Power Spectral Density (PSD) estimation
    - Cross Spectral Density (CSD) estimation
    - Spectrogram computation
    - Coherence estimation
    - Periodogram computation
    
    Parameters
    ----------
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    """
    
    def __init__(self, device_id: int = 0):
        """Initialize the spectral analyzer
        
        Parameters
        ----------
        device_id : int, optional
            CUDA device ID to use (-1 for CPU), by default 0
        """
        self.device_id = device_id
        if _HAS_CUDA:
            self._analyzer = _SpectralAnalyzer(device_id)
        self.has_cuda = _HAS_CUDA and self.device_id >= 0
    
    def compute_psd(
        self, 
        signal: np.ndarray, 
        sample_rate: float = 1.0, 
        window_type: WindowType = WindowType.HANN,
        nfft: int = 0,
        overlap: int = 0,
        scaling: bool = True,
        return_onesided: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density using Welch's method
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sample_rate : float, optional
            Sample rate in Hz, by default 1.0
        window_type : WindowType, optional
            Window function type, by default WindowType.HANN
        nfft : int, optional
            FFT size (0 = auto), by default 0
        overlap : int, optional
            Overlap between segments (0 = 50%), by default 0
        scaling : bool, optional
            Apply scaling, by default True
        return_onesided : bool, optional
            Return one-sided spectrum, by default True
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Frequencies and PSD values
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Convert parameters to C++ types
            params = _SpectralParams()
            params.window_type = _WindowType(window_type.value)
            params.nfft = nfft
            params.overlap = overlap
            params.sample_rate = sample_rate
            params.scaling = scaling
            params.return_onesided = return_onesided
            
            # Compute PSD
            result = self._analyzer.compute_psd(signal, params)
            
            # Extract results
            frequencies = np.array(result.frequencies)
            psd = np.array(result.psd)
            
            return frequencies, psd
        else:
            # Fallback to numpy implementation
            from scipy import signal as sp_signal
            
            # Map window type to scipy name
            window_map = {
                WindowType.RECTANGULAR: 'boxcar',
                WindowType.HANN: 'hann',
                WindowType.HAMMING: 'hamming',
                WindowType.BLACKMAN: 'blackman',
                WindowType.FLATTOP: 'flattop',
                WindowType.KAISER: ('kaiser', 4.0),
                WindowType.TUKEY: ('tukey', 0.5),
                WindowType.GAUSSIAN: ('gaussian', 0.5)
            }
            window = window_map.get(window_type, 'hann')
            
            # Auto-set NFFT if not specified
            if nfft == 0:
                nfft = min(1024, len(signal))
            
            # Auto-set overlap if not specified
            if overlap == 0:
                overlap = nfft // 2
            
            # Compute PSD using scipy
            frequencies, psd = sp_signal.welch(
                signal,
                fs=sample_rate,
                window=window,
                nperseg=nfft,
                noverlap=overlap,
                scaling='density' if scaling else 'spectrum',
                return_onesided=return_onesided
            )
            
            return frequencies, psd
    
    def compute_spectrogram(
        self, 
        signal: np.ndarray, 
        sample_rate: float = 1.0, 
        window_type: WindowType = WindowType.HANN,
        nfft: int = 0,
        overlap: int = 0,
        scaling: bool = True,
        return_onesided: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram of a signal
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sample_rate : float, optional
            Sample rate in Hz, by default 1.0
        window_type : WindowType, optional
            Window function type, by default WindowType.HANN
        nfft : int, optional
            FFT size (0 = auto), by default 0
        overlap : int, optional
            Overlap between segments (0 = 50%), by default 0
        scaling : bool, optional
            Apply scaling, by default True
        return_onesided : bool, optional
            Return one-sided spectrum, by default True
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Times, frequencies and spectrogram values
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Convert parameters to C++ types
            params = _SpectralParams()
            params.window_type = _WindowType(window_type.value)
            params.nfft = nfft
            params.overlap = overlap
            params.sample_rate = sample_rate
            params.scaling = scaling
            params.return_onesided = return_onesided
            
            # Compute spectrogram
            result = self._analyzer.compute_spectrogram(signal, params)
            
            # Extract results
            times = np.array(result.times)
            frequencies = np.array(result.frequencies)
            spectrogram = np.array(result.spectrogram)
            
            return times, frequencies, spectrogram
        else:
            # Fallback to numpy implementation
            from scipy import signal as sp_signal
            
            # Map window type to scipy name
            window_map = {
                WindowType.RECTANGULAR: 'boxcar',
                WindowType.HANN: 'hann',
                WindowType.HAMMING: 'hamming',
                WindowType.BLACKMAN: 'blackman',
                WindowType.FLATTOP: 'flattop',
                WindowType.KAISER: ('kaiser', 4.0),
                WindowType.TUKEY: ('tukey', 0.5),
                WindowType.GAUSSIAN: ('gaussian', 0.5)
            }
            window = window_map.get(window_type, 'hann')
            
            # Auto-set NFFT if not specified
            if nfft == 0:
                nfft = min(1024, len(signal))
            
            # Auto-set overlap if not specified
            if overlap == 0:
                overlap = nfft // 2
            
            # Compute spectrogram using scipy
            frequencies, times, spectrogram = sp_signal.spectrogram(
                signal,
                fs=sample_rate,
                window=window,
                nperseg=nfft,
                noverlap=overlap,
                scaling='density' if scaling else 'spectrum',
                return_onesided=return_onesided
            )
            
            return times, frequencies, spectrogram
    
    def compute_coherence(
        self, 
        signal1: np.ndarray, 
        signal2: np.ndarray, 
        sample_rate: float = 1.0, 
        window_type: WindowType = WindowType.HANN,
        nfft: int = 0,
        overlap: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute coherence between two signals
        
        Parameters
        ----------
        signal1 : np.ndarray
            First input signal
        signal2 : np.ndarray
            Second input signal
        sample_rate : float, optional
            Sample rate in Hz, by default 1.0
        window_type : WindowType, optional
            Window function type, by default WindowType.HANN
        nfft : int, optional
            FFT size (0 = auto), by default 0
        overlap : int, optional
            Overlap between segments (0 = 50%), by default 0
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Frequencies and coherence values
        """
        # Ensure inputs are float32 numpy arrays
        signal1 = np.asarray(signal1, dtype=np.float32)
        signal2 = np.asarray(signal2, dtype=np.float32)
        
        if len(signal1) != len(signal2):
            raise ValueError("Input signals must have the same length")
        
        if self.has_cuda:
            # Convert parameters to C++ types
            params = _SpectralParams()
            params.window_type = _WindowType(window_type.value)
            params.nfft = nfft
            params.overlap = overlap
            params.sample_rate = sample_rate
            
            # Compute coherence
            result = self._analyzer.compute_coherence(signal1, signal2, params)
            
            # Extract results
            frequencies = np.array(result.frequencies)
            coherence = np.array(result.coherence)
            
            return frequencies, coherence
        else:
            # Fallback to numpy implementation
            from scipy import signal as sp_signal
            
            # Map window type to scipy name
            window_map = {
                WindowType.RECTANGULAR: 'boxcar',
                WindowType.HANN: 'hann',
                WindowType.HAMMING: 'hamming',
                WindowType.BLACKMAN: 'blackman',
                WindowType.FLATTOP: 'flattop',
                WindowType.KAISER: ('kaiser', 4.0),
                WindowType.TUKEY: ('tukey', 0.5),
                WindowType.GAUSSIAN: ('gaussian', 0.5)
            }
            window = window_map.get(window_type, 'hann')
            
            # Auto-set NFFT if not specified
            if nfft == 0:
                nfft = min(1024, len(signal1))
            
            # Auto-set overlap if not specified
            if overlap == 0:
                overlap = nfft // 2
            
            # Compute coherence using scipy
            frequencies, coherence = sp_signal.coherence(
                signal1,
                signal2,
                fs=sample_rate,
                window=window,
                nperseg=nfft,
                noverlap=overlap
            )
            
            return frequencies, coherence
    
    def detect_peaks(
        self, 
        spectrum: np.ndarray, 
        frequencies: np.ndarray, 
        threshold: float = 0.5, 
        min_distance: int = 1
    ) -> List[Tuple[float, float]]:
        """Detect peaks in a spectrum
        
        Parameters
        ----------
        spectrum : np.ndarray
            Input spectrum
        frequencies : np.ndarray
            Corresponding frequencies
        threshold : float, optional
            Peak detection threshold, by default 0.5
        min_distance : int, optional
            Minimum distance between peaks, by default 1
        
        Returns
        -------
        List[Tuple[float, float]]
            List of (frequency, magnitude) pairs for detected peaks
        """
        # Ensure inputs are numpy arrays
        spectrum = np.asarray(spectrum, dtype=np.float32)
        frequencies = np.asarray(frequencies, dtype=np.float32)
        
        if len(spectrum) != len(frequencies):
            raise ValueError("Spectrum and frequencies must have the same length")
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            peaks = self._analyzer.detect_peaks(spectrum, frequencies, threshold, min_distance)
            return [(freq, mag) for freq, mag in peaks]
        else:
            # Fallback to numpy implementation
            from scipy import signal as sp_signal
            
            # Normalize threshold to max value
            max_val = np.max(spectrum)
            norm_threshold = threshold * max_val
            
            # Find peaks using scipy
            peak_indices = sp_signal.find_peaks(
                spectrum, 
                height=norm_threshold, 
                distance=min_distance
            )[0]
            
            # Extract peak frequencies and magnitudes
            peaks = [(frequencies[i], spectrum[i]) for i in peak_indices]
            
            # Sort by frequency
            peaks.sort(key=lambda x: x[0])
            
            return peaks


# Convenience functions

def compute_psd(
    signal: np.ndarray, 
    sample_rate: float = 1.0, 
    window_type: WindowType = WindowType.HANN,
    nfft: int = 0,
    overlap: int = 0,
    scaling: bool = True,
    return_onesided: bool = True,
    device_id: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    sample_rate : float, optional
        Sample rate in Hz, by default 1.0
    window_type : WindowType, optional
        Window function type, by default WindowType.HANN
    nfft : int, optional
        FFT size (0 = auto), by default 0
    overlap : int, optional
        Overlap between segments (0 = 50%), by default 0
    scaling : bool, optional
        Apply scaling, by default True
    return_onesided : bool, optional
        Return one-sided spectrum, by default True
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequencies and PSD values
    """
    analyzer = SpectralAnalyzer(device_id)
    return analyzer.compute_psd(
        signal, 
        sample_rate, 
        window_type, 
        nfft, 
        overlap, 
        scaling, 
        return_onesided
    )


def compute_spectrogram(
    signal: np.ndarray, 
    sample_rate: float = 1.0, 
    window_type: WindowType = WindowType.HANN,
    nfft: int = 0,
    overlap: int = 0,
    scaling: bool = True,
    return_onesided: bool = True,
    device_id: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram of a signal
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    sample_rate : float, optional
        Sample rate in Hz, by default 1.0
    window_type : WindowType, optional
        Window function type, by default WindowType.HANN
    nfft : int, optional
        FFT size (0 = auto), by default 0
    overlap : int, optional
        Overlap between segments (0 = 50%), by default 0
    scaling : bool, optional
        Apply scaling, by default True
    return_onesided : bool, optional
        Return one-sided spectrum, by default True
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Times, frequencies and spectrogram values
    """
    analyzer = SpectralAnalyzer(device_id)
    return analyzer.compute_spectrogram(
        signal, 
        sample_rate, 
        window_type, 
        nfft, 
        overlap, 
        scaling, 
        return_onesided
    )


def compute_coherence(
    signal1: np.ndarray, 
    signal2: np.ndarray, 
    sample_rate: float = 1.0, 
    window_type: WindowType = WindowType.HANN,
    nfft: int = 0,
    overlap: int = 0,
    device_id: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute coherence between two signals
    
    Parameters
    ----------
    signal1 : np.ndarray
        First input signal
    signal2 : np.ndarray
        Second input signal
    sample_rate : float, optional
        Sample rate in Hz, by default 1.0
    window_type : WindowType, optional
        Window function type, by default WindowType.HANN
    nfft : int, optional
        FFT size (0 = auto), by default 0
    overlap : int, optional
        Overlap between segments (0 = 50%), by default 0
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Frequencies and coherence values
    """
    analyzer = SpectralAnalyzer(device_id)
    return analyzer.compute_coherence(
        signal1, 
        signal2, 
        sample_rate, 
        window_type, 
        nfft, 
        overlap
    )


def detect_peaks(
    spectrum: np.ndarray, 
    frequencies: np.ndarray, 
    threshold: float = 0.5, 
    min_distance: int = 1,
    device_id: int = 0
) -> List[Tuple[float, float]]:
    """Detect peaks in a spectrum
    
    Parameters
    ----------
    spectrum : np.ndarray
        Input spectrum
    frequencies : np.ndarray
        Corresponding frequencies
    threshold : float, optional
        Peak detection threshold, by default 0.5
    min_distance : int, optional
        Minimum distance between peaks, by default 1
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    
    Returns
    -------
    List[Tuple[float, float]]
        List of (frequency, magnitude) pairs for detected peaks
    """
    analyzer = SpectralAnalyzer(device_id)
    return analyzer.detect_peaks(
        spectrum, 
        frequencies, 
        threshold, 
        min_distance
    )