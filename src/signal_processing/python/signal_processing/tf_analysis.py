"""
Time-Frequency Analysis Module

This module provides GPU-accelerated time-frequency analysis functionality,
including STFT, CWT, DWT, Wigner-Ville distribution, and EMD.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from enum import Enum

# Import C++ extension module
try:
    from ._signal_processing import (
        STFT as _STFT,
        CWT as _CWT,
        DWT as _DWT,
        WignerVille as _WignerVille,
        EMD as _EMD,
        STFTParams as _STFTParams,
        CWTParams as _CWTParams,
        DWTParams as _DWTParams,
        WindowType as _WindowType,
        WaveletType as _WaveletType,
        STFTResult as _STFTResult,
        CWTResult as _CWTResult,
        DWTResult as _DWTResult,
        EMDResult as _EMDResult,
        IMF as _IMF
    )
    from ._signal_processing.time_frequency import (
        spectrogram as _spectrogram,
        scalogram as _scalogram,
        mel_spectrogram as _mel_spectrogram,
        mfcc as _mfcc,
        chroma as _chroma,
        hilbert_transform as _hilbert_transform,
        instantaneous_frequency as _instantaneous_frequency,
        reassigned_spectrogram as _reassigned_spectrogram
    )
    _HAS_CUDA = True
except ImportError:
    # Fallback to numpy-based implementation if C++ module not available
    _HAS_CUDA = False


class WindowType(Enum):
    """Window types for STFT"""
    RECTANGULAR = 0
    HANN = 1
    HAMMING = 2
    BLACKMAN = 3
    KAISER = 4


class WaveletType(Enum):
    """Wavelet types for CWT and DWT"""
    MORLET = 0
    MEXICAN_HAT = 1
    PAUL = 2
    DOG = 3
    HAAR = 4
    DB4 = 5


class STFT:
    """Short-Time Fourier Transform (STFT) with GPU acceleration
    
    This class provides STFT operations optimized for NVIDIA GPUs,
    with automatic fallback to CPU implementation when CUDA is not available.
    
    Parameters
    ----------
    window_size : int, optional
        Size of the window, by default 1024
    hop_size : int, optional
        Hop size between windows, by default 256
    window_type : WindowType, optional
        Window type, by default WindowType.HANN
    fft_size : int, optional
        FFT size (0 = window_size), by default 0
    center : bool, optional
        Center the windows, by default True
    pad_mode : bool, optional
        Pad the signal, by default True
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    """
    
    def __init__(
        self,
        window_size: int = 1024,
        hop_size: int = 256,
        window_type: WindowType = WindowType.HANN,
        fft_size: int = 0,
        center: bool = True,
        pad_mode: bool = True,
        device_id: int = 0
    ):
        """Initialize the STFT processor"""
        self.device_id = device_id
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_type = window_type
        self.fft_size = fft_size if fft_size > 0 else window_size
        self.center = center
        self.pad_mode = pad_mode
        
        if _HAS_CUDA:
            # Create parameters for C++ implementation
            params = _STFTParams()
            params.window_size = window_size
            params.hop_size = hop_size
            params.window_type = _WindowType(window_type.value)
            params.fft_size = self.fft_size
            params.center = center
            params.pad_mode = pad_mode
            
            # Create STFT processor
            self._stft = _STFT(params, device_id)
        
        self.has_cuda = _HAS_CUDA and self.device_id >= 0
    
    def transform(
        self, 
        signal: np.ndarray, 
        sample_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute STFT of a signal
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sample_rate : float
            Sample rate in Hz
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing:
            - Complex spectrogram (shape: time x frequency)
            - Time bins (in seconds)
            - Frequency bins (in Hz)
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            result = self._stft.transform(signal, sample_rate)
            
            # Convert to numpy arrays
            spectrogram = np.array(result.spectrogram)
            times = np.array(result.times)
            frequencies = np.array(result.frequencies)
            
            return spectrogram, times, frequencies
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            
            # Map window type to scipy name
            window_map = {
                WindowType.RECTANGULAR: 'boxcar',
                WindowType.HANN: 'hann',
                WindowType.HAMMING: 'hamming',
                WindowType.BLACKMAN: 'blackman',
                WindowType.KAISER: ('kaiser', 4.0)
            }
            window = window_map.get(self.window_type, 'hann')
            
            # Compute STFT
            frequencies, times, spectrogram = sp_signal.stft(
                signal,
                fs=sample_rate,
                window=window,
                nperseg=self.window_size,
                noverlap=self.window_size - self.hop_size,
                nfft=self.fft_size,
                return_onesided=True,
                boundary=self.pad_mode,
                padded=self.pad_mode
            )
            
            # Transpose to match our convention (time x frequency)
            spectrogram = spectrogram.T
            
            return spectrogram, times, frequencies
    
    def inverse_transform(
        self, 
        spectrogram: np.ndarray, 
        times: np.ndarray, 
        frequencies: np.ndarray,
        sample_rate: float
    ) -> np.ndarray:
        """Compute inverse STFT
        
        Parameters
        ----------
        spectrogram : np.ndarray
            Complex spectrogram (shape: time x frequency)
        times : np.ndarray
            Time bins (in seconds)
        frequencies : np.ndarray
            Frequency bins (in Hz)
        sample_rate : float
            Sample rate in Hz
        
        Returns
        -------
        np.ndarray
            Reconstructed signal
        """
        # Ensure input is complex64 numpy array
        spectrogram = np.asarray(spectrogram, dtype=np.complex64)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            stft_result = _STFTResult()
            stft_result.spectrogram = spectrogram.tolist()
            stft_result.times = times.tolist()
            stft_result.frequencies = frequencies.tolist()
            stft_result.sample_rate = sample_rate
            
            return np.array(self._stft.inverse_transform(stft_result))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            
            # Map window type to scipy name
            window_map = {
                WindowType.RECTANGULAR: 'boxcar',
                WindowType.HANN: 'hann',
                WindowType.HAMMING: 'hamming',
                WindowType.BLACKMAN: 'blackman',
                WindowType.KAISER: ('kaiser', 4.0)
            }
            window = window_map.get(self.window_type, 'hann')
            
            # Transpose to match scipy's convention (frequency x time)
            spectrogram = spectrogram.T
            
            # Compute inverse STFT
            _, reconstructed = sp_signal.istft(
                spectrogram,
                fs=sample_rate,
                window=window,
                nperseg=self.window_size,
                noverlap=self.window_size - self.hop_size,
                nfft=self.fft_size,
                input_onesided=True,
                boundary=self.pad_mode
            )
            
            return reconstructed
    
    def get_magnitude(
        self, 
        spectrogram: np.ndarray, 
        log_scale: bool = False
    ) -> np.ndarray:
        """Get magnitude spectrogram from STFT result
        
        Parameters
        ----------
        spectrogram : np.ndarray
            Complex spectrogram (shape: time x frequency)
        log_scale : bool, optional
            Apply log scaling to magnitudes, by default False
        
        Returns
        -------
        np.ndarray
            Magnitude spectrogram
        """
        # Ensure input is complex64 numpy array
        spectrogram = np.asarray(spectrogram, dtype=np.complex64)
        
        # Compute magnitude
        magnitude = np.abs(spectrogram)
        
        # Apply log scaling if requested
        if log_scale:
            magnitude = 20 * np.log10(np.maximum(magnitude, 1e-10))
        
        return magnitude
    
    def get_phase(self, spectrogram: np.ndarray) -> np.ndarray:
        """Get phase spectrogram from STFT result
        
        Parameters
        ----------
        spectrogram : np.ndarray
            Complex spectrogram (shape: time x frequency)
        
        Returns
        -------
        np.ndarray
            Phase spectrogram
        """
        # Ensure input is complex64 numpy array
        spectrogram = np.asarray(spectrogram, dtype=np.complex64)
        
        # Compute phase
        phase = np.angle(spectrogram)
        
        return phase
    
    def get_power(
        self, 
        spectrogram: np.ndarray, 
        log_scale: bool = False
    ) -> np.ndarray:
        """Get power spectrogram from STFT result
        
        Parameters
        ----------
        spectrogram : np.ndarray
            Complex spectrogram (shape: time x frequency)
        log_scale : bool, optional
            Apply log scaling to power values, by default False
        
        Returns
        -------
        np.ndarray
            Power spectrogram
        """
        # Ensure input is complex64 numpy array
        spectrogram = np.asarray(spectrogram, dtype=np.complex64)
        
        # Compute power
        power = np.abs(spectrogram) ** 2
        
        # Apply log scaling if requested
        if log_scale:
            power = 10 * np.log10(np.maximum(power, 1e-10))
        
        return power


class CWT:
    """Continuous Wavelet Transform (CWT) with GPU acceleration
    
    This class provides CWT operations optimized for NVIDIA GPUs,
    with automatic fallback to CPU implementation when CUDA is not available.
    
    Parameters
    ----------
    wavelet_type : WaveletType, optional
        Wavelet type, by default WaveletType.MORLET
    wavelet_param : float, optional
        Wavelet-specific parameter, by default 6.0
    num_scales : int, optional
        Number of scales, by default 32
    min_scale : float, optional
        Minimum scale, by default 1.0
    max_scale : float, optional
        Maximum scale (0 = auto), by default 0.0
    normalize_scales : bool, optional
        Normalize the scales, by default True
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    """
    
    def __init__(
        self,
        wavelet_type: WaveletType = WaveletType.MORLET,
        wavelet_param: float = 6.0,
        num_scales: int = 32,
        min_scale: float = 1.0,
        max_scale: float = 0.0,
        normalize_scales: bool = True,
        device_id: int = 0
    ):
        """Initialize the CWT processor"""
        self.device_id = device_id
        self.wavelet_type = wavelet_type
        self.wavelet_param = wavelet_param
        self.num_scales = num_scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.normalize_scales = normalize_scales
        
        if _HAS_CUDA:
            # Create parameters for C++ implementation
            params = _CWTParams()
            params.wavelet_type = _WaveletType(wavelet_type.value)
            params.wavelet_param = wavelet_param
            params.num_scales = num_scales
            params.min_scale = min_scale
            params.max_scale = max_scale
            params.normalize_scales = normalize_scales
            
            # Create CWT processor
            self._cwt = _CWT(params, device_id)
        
        self.has_cuda = _HAS_CUDA and self.device_id >= 0
    
    def transform(
        self, 
        signal: np.ndarray, 
        sample_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute CWT of a signal
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sample_rate : float
            Sample rate in Hz
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Tuple containing:
            - Complex scalogram (shape: scale x time)
            - Time bins (in seconds)
            - Scale bins
            - Frequency bins (in Hz)
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            result = self._cwt.transform(signal, sample_rate)
            
            # Convert to numpy arrays
            scalogram = np.array(result.scalogram)
            times = np.array(result.times)
            scales = np.array(result.scales)
            frequencies = np.array(result.frequencies)
            
            return scalogram, times, scales, frequencies
        else:
            # Fallback to pywavelets implementation
            try:
                import pywt
            except ImportError:
                raise ImportError("pywavelets is required for CWT when CUDA is not available")
            
            # Map wavelet type to pywt name
            wavelet_map = {
                WaveletType.MORLET: 'morl',
                WaveletType.MEXICAN_HAT: 'mexh',
                WaveletType.PAUL: 'paul',
                WaveletType.DOG: 'dog',
                WaveletType.HAAR: 'haar',
                WaveletType.DB4: 'db4'
            }
            wavelet = wavelet_map.get(self.wavelet_type, 'morl')
            
            # Compute scales
            if self.max_scale <= 0:
                max_scale = len(signal) / 2
            else:
                max_scale = self.max_scale
            
            scales = np.logspace(np.log10(self.min_scale), np.log10(max_scale), self.num_scales)
            
            # Compute CWT
            coefs, frequencies = pywt.cwt(signal, scales, wavelet, 1.0 / sample_rate)
            
            # Generate time bins
            times = np.arange(len(signal)) / sample_rate
            
            return coefs, times, scales, frequencies
    
    def get_magnitude(
        self, 
        scalogram: np.ndarray, 
        log_scale: bool = False
    ) -> np.ndarray:
        """Get magnitude scalogram from CWT result
        
        Parameters
        ----------
        scalogram : np.ndarray
            Complex scalogram (shape: scale x time)
        log_scale : bool, optional
            Apply log scaling to magnitudes, by default False
        
        Returns
        -------
        np.ndarray
            Magnitude scalogram
        """
        # Ensure input is complex64 numpy array
        scalogram = np.asarray(scalogram, dtype=np.complex64)
        
        # Compute magnitude
        magnitude = np.abs(scalogram)
        
        # Apply log scaling if requested
        if log_scale:
            magnitude = 20 * np.log10(np.maximum(magnitude, 1e-10))
        
        return magnitude
    
    def get_phase(self, scalogram: np.ndarray) -> np.ndarray:
        """Get phase scalogram from CWT result
        
        Parameters
        ----------
        scalogram : np.ndarray
            Complex scalogram (shape: scale x time)
        
        Returns
        -------
        np.ndarray
            Phase scalogram
        """
        # Ensure input is complex64 numpy array
        scalogram = np.asarray(scalogram, dtype=np.complex64)
        
        # Compute phase
        phase = np.angle(scalogram)
        
        return phase
    
    def get_power(
        self, 
        scalogram: np.ndarray, 
        log_scale: bool = False
    ) -> np.ndarray:
        """Get power scalogram from CWT result
        
        Parameters
        ----------
        scalogram : np.ndarray
            Complex scalogram (shape: scale x time)
        log_scale : bool, optional
            Apply log scaling to power values, by default False
        
        Returns
        -------
        np.ndarray
            Power scalogram
        """
        # Ensure input is complex64 numpy array
        scalogram = np.asarray(scalogram, dtype=np.complex64)
        
        # Compute power
        power = np.abs(scalogram) ** 2
        
        # Apply log scaling if requested
        if log_scale:
            power = 10 * np.log10(np.maximum(power, 1e-10))
        
        return power


class DWT:
    """Discrete Wavelet Transform (DWT) with GPU acceleration
    
    This class provides DWT operations optimized for NVIDIA GPUs,
    with automatic fallback to CPU implementation when CUDA is not available.
    
    Parameters
    ----------
    wavelet_type : WaveletType, optional
        Wavelet type, by default WaveletType.DB4
    levels : int, optional
        Number of decomposition levels (0 = auto), by default 0
    mode : str, optional
        Border extension mode, by default "reflect"
    use_swt : bool, optional
        Use stationary wavelet transform, by default False
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    """
    
    def __init__(
        self,
        wavelet_type: WaveletType = WaveletType.DB4,
        levels: int = 0,
        mode: str = "reflect",
        use_swt: bool = False,
        device_id: int = 0
    ):
        """Initialize the DWT processor"""
        self.device_id = device_id
        self.wavelet_type = wavelet_type
        self.levels = levels
        self.mode = mode
        self.use_swt = use_swt
        
        if _HAS_CUDA:
            # Create parameters for C++ implementation
            params = _DWTParams()
            params.wavelet_type = _WaveletType(wavelet_type.value)
            params.levels = levels
            params.mode = mode
            params.use_swt = use_swt
            
            # Create DWT processor
            self._dwt = _DWT(params, device_id)
        
        self.has_cuda = _HAS_CUDA and self.device_id >= 0
    
    def transform(self, signal: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Compute DWT of a signal
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        
        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            Tuple containing:
            - List of detail coefficients (from finest to coarsest)
            - Approximation coefficients
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            result = self._dwt.transform(signal)
            
            # Extract coefficients
            coeffs = np.array(result.coeffs)
            lengths = np.array(result.lengths)
            
            # Split into detail and approximation coefficients
            cA = coeffs[-1]
            cD = [coeffs[i] for i in range(len(coeffs) - 1)]
            
            return cD, cA
        else:
            # Fallback to pywavelets implementation
            try:
                import pywt
            except ImportError:
                raise ImportError("pywavelets is required for DWT when CUDA is not available")
            
            # Map wavelet type to pywt name
            wavelet_map = {
                WaveletType.MORLET: 'morl',
                WaveletType.MEXICAN_HAT: 'mexh',
                WaveletType.PAUL: 'paul',
                WaveletType.DOG: 'dog',
                WaveletType.HAAR: 'haar',
                WaveletType.DB4: 'db4'
            }
            wavelet = wavelet_map.get(self.wavelet_type, 'db4')
            
            # Auto-set levels if not specified
            if self.levels <= 0:
                levels = pywt.dwt_max_level(len(signal), wavelet)
            else:
                levels = self.levels
            
            # Compute DWT
            if self.use_swt:
                coeffs = pywt.swt(signal, wavelet, levels, self.mode)
                # Rearrange coefficients to match our convention
                cA = coeffs[0][0]
                cD = [c[1] for c in coeffs]
            else:
                coeffs = pywt.wavedec(signal, wavelet, mode=self.mode, level=levels)
                # Extract coefficients
                cA = coeffs[0]
                cD = coeffs[1:]
            
            return cD, cA
    
    def inverse_transform(
        self, 
        detail_coeffs: List[np.ndarray], 
        approx_coeffs: np.ndarray
    ) -> np.ndarray:
        """Compute inverse DWT
        
        Parameters
        ----------
        detail_coeffs : List[np.ndarray]
            Detail coefficients (from finest to coarsest)
        approx_coeffs : np.ndarray
            Approximation coefficients
        
        Returns
        -------
        np.ndarray
            Reconstructed signal
        """
        # Ensure inputs are float32 numpy arrays
        detail_coeffs = [np.asarray(c, dtype=np.float32) for c in detail_coeffs]
        approx_coeffs = np.asarray(approx_coeffs, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            dwt_result = _DWTResult()
            dwt_result.coeffs = detail_coeffs + [approx_coeffs]
            dwt_result.lengths = [len(c) for c in detail_coeffs] + [len(approx_coeffs)]
            dwt_result.levels = len(detail_coeffs)
            
            return np.array(self._dwt.inverse_transform(dwt_result))
        else:
            # Fallback to pywavelets implementation
            try:
                import pywt
            except ImportError:
                raise ImportError("pywavelets is required for DWT when CUDA is not available")
            
            # Map wavelet type to pywt name
            wavelet_map = {
                WaveletType.MORLET: 'morl',
                WaveletType.MEXICAN_HAT: 'mexh',
                WaveletType.PAUL: 'paul',
                WaveletType.DOG: 'dog',
                WaveletType.HAAR: 'haar',
                WaveletType.DB4: 'db4'
            }
            wavelet = wavelet_map.get(self.wavelet_type, 'db4')
            
            # Combine coefficients
            if self.use_swt:
                coeffs = [(approx_coeffs, d) for d in detail_coeffs]
                return pywt.iswt(coeffs, wavelet)
            else:
                coeffs = [approx_coeffs] + detail_coeffs
                return pywt.waverec(coeffs, wavelet, mode=self.mode)
    
    def denoise(
        self, 
        signal: np.ndarray, 
        threshold: float, 
        threshold_mode: str = "soft"
    ) -> np.ndarray:
        """Denoise a signal using wavelet thresholding
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        threshold : float
            Threshold value
        threshold_mode : str, optional
            Threshold mode ("soft" or "hard"), by default "soft"
        
        Returns
        -------
        np.ndarray
            Denoised signal
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._dwt.denoise(signal, threshold, threshold_mode))
        else:
            # Fallback to pywavelets implementation
            try:
                import pywt
            except ImportError:
                raise ImportError("pywavelets is required for DWT when CUDA is not available")
            
            # Map wavelet type to pywt name
            wavelet_map = {
                WaveletType.MORLET: 'morl',
                WaveletType.MEXICAN_HAT: 'mexh',
                WaveletType.PAUL: 'paul',
                WaveletType.DOG: 'dog',
                WaveletType.HAAR: 'haar',
                WaveletType.DB4: 'db4'
            }
            wavelet = wavelet_map.get(self.wavelet_type, 'db4')
            
            # Auto-set levels if not specified
            if self.levels <= 0:
                levels = pywt.dwt_max_level(len(signal), wavelet)
            else:
                levels = self.levels
            
            # Decompose signal
            coeffs = pywt.wavedec(signal, wavelet, mode=self.mode, level=levels)
            
            # Apply thresholding to detail coefficients
            for i in range(1, len(coeffs)):
                if threshold_mode == "soft":
                    coeffs[i] = pywt.threshold(coeffs[i], threshold, 'soft')
                else:
                    coeffs[i] = pywt.threshold(coeffs[i], threshold, 'hard')
            
            # Reconstruct signal
            return pywt.waverec(coeffs, wavelet, mode=self.mode)


class WignerVille:
    """Wigner-Ville Distribution with GPU acceleration
    
    This class provides Wigner-Ville Distribution operations optimized for NVIDIA GPUs,
    with automatic fallback to CPU implementation when CUDA is not available.
    
    Parameters
    ----------
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    """
    
    def __init__(self, device_id: int = 0):
        """Initialize the Wigner-Ville processor"""
        self.device_id = device_id
        
        if _HAS_CUDA:
            # Create Wigner-Ville processor
            self._wv = _WignerVille(device_id)
        
        self.has_cuda = _HAS_CUDA and self.device_id >= 0
    
    def transform(
        self, 
        signal: np.ndarray, 
        sample_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Wigner-Ville distribution of a signal
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sample_rate : float
            Sample rate in Hz
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing:
            - Distribution (shape: time x frequency)
            - Time bins (in seconds)
            - Frequency bins (in Hz)
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            result, axes = self._wv.transform(signal, sample_rate)
            
            # Extract axes
            times, frequencies = axes
            
            return np.array(result), np.array(times), np.array(frequencies)
        else:
            # Fallback to a simple numpy implementation
            # Generate analytic signal
            from scipy import signal as sp_signal
            analytic_signal = sp_signal.hilbert(signal)
            
            n = len(signal)
            wvd = np.zeros((n, n // 2 + 1), dtype=np.float32)
            
            # Compute Wigner-Ville distribution
            for t in range(n):
                for lag in range(-min(t, n - t - 1), min(t, n - t - 1) + 1):
                    # Auto-correlation
                    if 0 <= t + lag < n and 0 <= t - lag < n:
                        corr = analytic_signal[t + lag] * np.conj(analytic_signal[t - lag])
                        
                        # Add to all frequency bins
                        for f in range(n // 2 + 1):
                            angle = -2.0 * np.pi * f * lag / n
                            wvd[t, f] += corr.real * np.cos(angle) - corr.imag * np.sin(angle)
            
            # Generate time and frequency axes
            times = np.arange(n) / sample_rate
            frequencies = np.arange(n // 2 + 1) * sample_rate / n
            
            return wvd, times, frequencies
    
    def transform_pseudo(
        self, 
        signal: np.ndarray, 
        sample_rate: float, 
        window_size: int = 127
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute pseudo Wigner-Ville distribution (smoothed)
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sample_rate : float
            Sample rate in Hz
        window_size : int, optional
            Window size for time smoothing, by default 127
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing:
            - Distribution (shape: time x frequency)
            - Time bins (in seconds)
            - Frequency bins (in Hz)
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            result, axes = self._wv.transform_pseudo(signal, sample_rate, window_size)
            
            # Extract axes
            times, frequencies = axes
            
            return np.array(result), np.array(times), np.array(frequencies)
        else:
            # Fallback to using window-smoothed Wigner-Ville distribution
            from scipy import signal as sp_signal
            
            # Generate analytic signal
            analytic_signal = sp_signal.hilbert(signal)
            
            # Generate smoothing window
            window = sp_signal.windows.hann(window_size)
            window = window / np.sum(window)
            
            n = len(signal)
            pwvd = np.zeros((n, n // 2 + 1), dtype=np.float32)
            
            # Compute pseudo-Wigner-Ville distribution (smoothed)
            for t in range(n):
                max_lag = min(t, n - t - 1, window_size // 2)
                
                # Apply window to lag values
                for lag in range(-max_lag, max_lag + 1):
                    if 0 <= t + lag < n and 0 <= t - lag < n and abs(lag) < window_size // 2:
                        # Apply window
                        w = window[lag + window_size // 2]
                        
                        # Auto-correlation
                        corr = analytic_signal[t + lag] * np.conj(analytic_signal[t - lag]) * w
                        
                        # Add to all frequency bins
                        for f in range(n // 2 + 1):
                            angle = -2.0 * np.pi * f * lag / n
                            pwvd[t, f] += corr.real * np.cos(angle) - corr.imag * np.sin(angle)
            
            # Generate time and frequency axes
            times = np.arange(n) / sample_rate
            frequencies = np.arange(n // 2 + 1) * sample_rate / n
            
            return pwvd, times, frequencies


class EMD:
    """Empirical Mode Decomposition (EMD) with GPU acceleration
    
    This class provides Empirical Mode Decomposition operations optimized for NVIDIA GPUs,
    with automatic fallback to CPU implementation when CUDA is not available.
    
    Parameters
    ----------
    max_imfs : int, optional
        Maximum number of IMFs to extract (0 = automatic), by default 0
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    """
    
    def __init__(self, max_imfs: int = 0, device_id: int = 0):
        """Initialize the EMD processor"""
        self.device_id = device_id
        self.max_imfs = max_imfs
        
        if _HAS_CUDA:
            # Create EMD processor
            self._emd = _EMD(max_imfs, device_id)
        
        self.has_cuda = _HAS_CUDA and self.device_id >= 0
    
    def decompose(
        self, 
        signal: np.ndarray, 
        sample_rate: float
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Decompose a signal into IMFs
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sample_rate : float
            Sample rate in Hz
        
        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            Tuple containing:
            - List of IMFs
            - Residue signal
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            result = self._emd.decompose(signal, sample_rate)
            
            # Extract IMFs and residue
            imfs = [np.array(imf.signal) for imf in result.imfs]
            residue = np.array(result.residue)
            
            return imfs, residue
        else:
            # Fallback to PyEMD implementation
            try:
                from PyEMD import EMD as PyEMD
            except ImportError:
                raise ImportError("PyEMD is required for EMD when CUDA is not available")
            
            # Create EMD processor
            emd = PyEMD()
            
            # Set maximum number of IMFs
            if self.max_imfs > 0:
                emd.MAX_ITERATION = self.max_imfs
            
            # Decompose signal
            imfs = emd(signal)
            
            # Extract residue
            residue = signal - np.sum(imfs, axis=0)
            
            # Convert to list of IMFs
            imfs_list = [imf for imf in imfs]
            
            return imfs_list, residue
    
    def hilbert_huang_spectrum(
        self, 
        imfs: List[np.ndarray], 
        residue: np.ndarray, 
        sample_rate: float, 
        num_freqs: int = 256
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Hilbert-Huang spectrum
        
        Parameters
        ----------
        imfs : List[np.ndarray]
            IMFs from decompose method
        residue : np.ndarray
            Residue signal from decompose method
        sample_rate : float
            Sample rate in Hz
        num_freqs : int, optional
            Number of frequency bins, by default 256
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple containing:
            - Spectrum (shape: time x frequency)
            - Time bins (in seconds)
            - Frequency bins (in Hz)
        """
        # Ensure inputs are float32 numpy arrays
        imfs = [np.asarray(imf, dtype=np.float32) for imf in imfs]
        residue = np.asarray(residue, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            emd_result = _EMDResult()
            emd_result.imfs = [_IMF(imf, 0.0, []) for imf in imfs]
            emd_result.residue = residue.tolist()
            emd_result.num_imfs = len(imfs)
            
            spectrum, axes = self._emd.hilbert_huang_spectrum(emd_result, sample_rate, num_freqs)
            
            # Extract axes
            times, frequencies = axes
            
            return np.array(spectrum), np.array(times), np.array(frequencies)
        else:
            # Fallback to numpy implementation
            from scipy import signal as sp_signal
            
            # Get signal length and time axis
            n = len(imfs[0])
            times = np.arange(n) / sample_rate
            
            # Initialize spectrum
            spectrum = np.zeros((n, num_freqs), dtype=np.float32)
            
            # Frequency axis
            frequencies = np.linspace(0, sample_rate / 2, num_freqs)
            
            # Compute instantaneous frequency and amplitude for each IMF
            for imf in imfs:
                # Compute analytic signal
                analytic_signal = sp_signal.hilbert(imf)
                
                # Compute instantaneous amplitude
                amplitude = np.abs(analytic_signal)
                
                # Compute instantaneous phase
                phase = np.unwrap(np.angle(analytic_signal))
                
                # Compute instantaneous frequency
                inst_freq = np.diff(phase) / (2.0 * np.pi) * sample_rate
                inst_freq = np.concatenate([inst_freq[:1], inst_freq])
                
                # Add to spectrum
                for t in range(n):
                    freq = inst_freq[t]
                    amp = amplitude[t]
                    
                    if 0 <= freq <= sample_rate / 2:
                        # Find closest frequency bin
                        f_idx = int(freq / (sample_rate / 2) * (num_freqs - 1))
                        if 0 <= f_idx < num_freqs:
                            spectrum[t, f_idx] += amp
            
            return spectrum, times, frequencies
    
    def reconstruct(
        self, 
        imfs: List[np.ndarray], 
        residue: np.ndarray, 
        imf_indices: List[int] = []
    ) -> np.ndarray:
        """Reconstruct signal from IMFs
        
        Parameters
        ----------
        imfs : List[np.ndarray]
            IMFs from decompose method
        residue : np.ndarray
            Residue signal from decompose method
        imf_indices : List[int], optional
            Indices of IMFs to include (empty = all), by default []
        
        Returns
        -------
        np.ndarray
            Reconstructed signal
        """
        # Ensure inputs are float32 numpy arrays
        imfs = [np.asarray(imf, dtype=np.float32) for imf in imfs]
        residue = np.asarray(residue, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            emd_result = _EMDResult()
            emd_result.imfs = [_IMF(imf, 0.0, []) for imf in imfs]
            emd_result.residue = residue.tolist()
            emd_result.num_imfs = len(imfs)
            
            return np.array(self._emd.reconstruct(emd_result, imf_indices))
        else:
            # Fallback to numpy implementation
            if not imf_indices:
                # Include all IMFs by default
                imf_indices = list(range(len(imfs)))
            
            # Sum selected IMFs
            signal = np.zeros_like(residue)
            for idx in imf_indices:
                if 0 <= idx < len(imfs):
                    signal += imfs[idx]
            
            # Add residue
            signal += residue
            
            return signal


# Convenience functions

def compute_spectrogram(
    signal: np.ndarray, 
    sample_rate: float, 
    window_size: int = 1024, 
    hop_size: int = 256, 
    window_type: WindowType = WindowType.HANN, 
    log_scale: bool = True, 
    device_id: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram of a signal
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    sample_rate : float
        Sample rate in Hz
    window_size : int, optional
        Window size, by default 1024
    hop_size : int, optional
        Hop size, by default 256
    window_type : WindowType, optional
        Window type, by default WindowType.HANN
    log_scale : bool, optional
        Apply log scaling, by default True
    device_id : int, optional
        CUDA device ID (-1 for CPU), by default 0
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - Spectrogram (shape: time x frequency)
        - Time bins (in seconds)
        - Frequency bins (in Hz)
    """
    # Create STFT processor
    stft = STFT(window_size, hop_size, window_type, device_id=device_id)
    
    # Compute STFT
    spectrogram, times, frequencies = stft.transform(signal, sample_rate)
    
    # Compute magnitude spectrogram
    magnitude = stft.get_magnitude(spectrogram, log_scale)
    
    return magnitude, times, frequencies


def compute_scalogram(
    signal: np.ndarray, 
    sample_rate: float, 
    wavelet_type: WaveletType = WaveletType.MORLET, 
    num_scales: int = 32, 
    log_scale: bool = True, 
    device_id: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute scalogram of a signal
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    sample_rate : float
        Sample rate in Hz
    wavelet_type : WaveletType, optional
        Wavelet type, by default WaveletType.MORLET
    num_scales : int, optional
        Number of scales, by default 32
    log_scale : bool, optional
        Apply log scaling, by default True
    device_id : int, optional
        CUDA device ID (-1 for CPU), by default 0
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - Scalogram (shape: scale x time)
        - Time bins (in seconds)
        - Frequency bins (in Hz)
    """
    # Create CWT processor
    cwt = CWT(wavelet_type, num_scales=num_scales, device_id=device_id)
    
    # Compute CWT
    scalogram, times, scales, frequencies = cwt.transform(signal, sample_rate)
    
    # Compute magnitude scalogram
    magnitude = cwt.get_magnitude(scalogram, log_scale)
    
    return magnitude, times, frequencies


def compute_mel_spectrogram(
    signal: np.ndarray, 
    sample_rate: float, 
    n_fft: int = 2048, 
    hop_size: int = 512, 
    n_mels: int = 128, 
    fmin: float = 0.0, 
    fmax: float = 0.0, 
    log_scale: bool = True, 
    device_id: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Mel spectrogram of a signal
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    sample_rate : float
        Sample rate in Hz
    n_fft : int, optional
        FFT size, by default 2048
    hop_size : int, optional
        Hop size, by default 512
    n_mels : int, optional
        Number of Mel bands, by default 128
    fmin : float, optional
        Minimum frequency, by default 0.0
    fmax : float, optional
        Maximum frequency (0 = Nyquist), by default 0.0
    log_scale : bool, optional
        Apply log scaling, by default True
    device_id : int, optional
        CUDA device ID (-1 for CPU), by default 0
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - Mel spectrogram (shape: time x mel_bands)
        - Time bins (in seconds)
        - Mel frequency bins (in Hz)
    """
    # Ensure input is float32 numpy array
    signal = np.asarray(signal, dtype=np.float32)
    
    if _HAS_CUDA and device_id >= 0:
        # Use C++/CUDA implementation
        result, axes = _mel_spectrogram(
            signal, sample_rate, n_fft, hop_size, n_mels, 
            fmin, fmax, log_scale, device_id
        )
        
        # Extract axes
        times, frequencies = axes
        
        return np.array(result), np.array(times), np.array(frequencies)
    else:
        # Fallback to librosa implementation
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa is required for mel_spectrogram when CUDA is not available")
        
        # Compute Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal, 
            sr=sample_rate, 
            n_fft=n_fft, 
            hop_length=hop_size, 
            n_mels=n_mels, 
            fmin=fmin, 
            fmax=fmax if fmax > 0 else None
        )
        
        # Transpose to time x mel_bands
        mel_spec = mel_spec.T
        
        # Apply log scaling if requested
        if log_scale:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Generate time and frequency axes
        times = librosa.times_like(mel_spec.T, sr=sample_rate, hop_length=hop_size)
        frequencies = librosa.mel_frequencies(
            n_mels=n_mels, 
            fmin=fmin, 
            fmax=fmax if fmax > 0 else sample_rate / 2
        )
        
        return mel_spec, times, frequencies


def compute_mfcc(
    signal: np.ndarray, 
    sample_rate: float, 
    n_mfcc: int = 13, 
    n_fft: int = 2048, 
    hop_size: int = 512, 
    n_mels: int = 128, 
    device_id: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MFCC (Mel-frequency cepstral coefficients)
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    sample_rate : float
        Sample rate in Hz
    n_mfcc : int, optional
        Number of MFCCs, by default 13
    n_fft : int, optional
        FFT size, by default 2048
    hop_size : int, optional
        Hop size, by default 512
    n_mels : int, optional
        Number of Mel bands, by default 128
    device_id : int, optional
        CUDA device ID (-1 for CPU), by default 0
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - MFCCs (shape: time x n_mfcc)
        - Time bins (in seconds)
    """
    # Ensure input is float32 numpy array
    signal = np.asarray(signal, dtype=np.float32)
    
    if _HAS_CUDA and device_id >= 0:
        # Use C++/CUDA implementation
        result, times = _mfcc(
            signal, sample_rate, n_mfcc, n_fft, hop_size, n_mels, device_id
        )
        
        return np.array(result), np.array(times)
    else:
        # Fallback to librosa implementation
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa is required for mfcc when CUDA is not available")
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=signal, 
            sr=sample_rate, 
            n_mfcc=n_mfcc, 
            n_fft=n_fft, 
            hop_length=hop_size, 
            n_mels=n_mels
        )
        
        # Transpose to time x n_mfcc
        mfccs = mfccs.T
        
        # Generate time axis
        times = librosa.times_like(mfccs.T, sr=sample_rate, hop_length=hop_size)
        
        return mfccs, times