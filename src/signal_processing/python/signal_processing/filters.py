"""
Digital Filtering Module

This module provides GPU-accelerated digital filtering functionality,
including FIR filters, IIR filters, adaptive filters, and multirate filters.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from enum import Enum

# Import C++ extension module
try:
    from ._signal_processing import (
        FIRFilter as _FIRFilter,
        IIRFilter as _IIRFilter,
        AdaptiveFilter as _AdaptiveFilter,
        MultirateFilter as _MultirateFilter,
        FIRFilterParams as _FIRFilterParams,
        IIRFilterParams as _IIRFilterParams,
        AdaptiveFilterParams as _AdaptiveFilterParams,
        MultirateFilterParams as _MultirateFilterParams,
        WindowType as _WindowType,
        FilterType as _FilterType,
        FIRDesignMethod as _FIRDesignMethod,
        IIRDesignMethod as _IIRDesignMethod,
        AdaptiveFilterType as _AdaptiveFilterType
    )
    from ._signal_processing.filters import (
        median_filter as _median_filter,
        convolve as _convolve,
        savitzky_golay as _savitzky_golay,
        wiener_filter as _wiener_filter,
        kalman_filter as _kalman_filter,
        bilateral_filter as _bilateral_filter
    )
    _HAS_CUDA = True
except ImportError:
    # Fallback to numpy-based implementation if C++ module not available
    _HAS_CUDA = False


class FilterType(Enum):
    """Filter types for FIR and IIR filters"""
    LOWPASS = 0
    HIGHPASS = 1
    BANDPASS = 2
    BANDSTOP = 3


class WindowType(Enum):
    """Window types for FIR filter design"""
    RECTANGULAR = 0
    TRIANGULAR = 1
    HANN = 2
    HAMMING = 3
    BLACKMAN = 4
    KAISER = 5


class FIRDesignMethod(Enum):
    """FIR filter design methods"""
    WINDOW = 0
    LEAST_SQUARES = 1
    PARKS_MCCLELLAN = 2
    FREQUENCY_SAMPLING = 3


class IIRDesignMethod(Enum):
    """IIR filter design methods"""
    BUTTERWORTH = 0
    CHEBYSHEV1 = 1
    CHEBYSHEV2 = 2
    ELLIPTIC = 3
    BESSEL = 4


class AdaptiveFilterType(Enum):
    """Adaptive filter types"""
    LMS = 0
    NLMS = 1
    RLS = 2
    KALMAN = 3


class FIRFilter:
    """Finite Impulse Response (FIR) filter with GPU acceleration
    
    This class provides FIR filtering operations optimized for NVIDIA GPUs,
    with automatic fallback to CPU implementation when CUDA is not available.
    
    Parameters
    ----------
    coefficients : np.ndarray or None, optional
        Filter coefficients. If None, design parameters must be provided
    sample_rate : float, optional
        Sample rate in Hz (required if coefficients is None), by default None
    filter_type : FilterType, optional
        Filter type, by default FilterType.LOWPASS
    cutoff_freqs : list or float, optional
        Cutoff frequencies in Hz, by default None
    window_type : WindowType, optional
        Window type for window method, by default WindowType.HAMMING
    num_taps : int, optional
        Number of filter taps (required if coefficients is None), by default 0
    design_method : FIRDesignMethod, optional
        Design method, by default FIRDesignMethod.WINDOW
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    
    Raises
    ------
    ValueError
        If both coefficients and design parameters are None
    """
    
    def __init__(
        self,
        coefficients: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        filter_type: FilterType = FilterType.LOWPASS,
        cutoff_freqs: Optional[Union[List[float], float]] = None,
        window_type: WindowType = WindowType.HAMMING,
        num_taps: int = 0,
        design_method: FIRDesignMethod = FIRDesignMethod.WINDOW,
        device_id: int = 0
    ):
        """Initialize the FIR filter"""
        self.device_id = device_id
        
        if coefficients is not None:
            # Initialize with provided coefficients
            self.coefficients = np.asarray(coefficients, dtype=np.float32)
            if _HAS_CUDA:
                self._filter = _FIRFilter(self.coefficients, device_id)
        elif sample_rate is not None and cutoff_freqs is not None and num_taps > 0:
            # Initialize with design parameters
            if _HAS_CUDA:
                params = _FIRFilterParams()
                params.filter_type = _FilterType(filter_type.value)
                params.num_taps = num_taps
                params.design_method = _FIRDesignMethod(design_method.value)
                params.window_type = _WindowType(window_type.value)
                
                # Handle cutoff frequencies
                if isinstance(cutoff_freqs, (list, tuple)):
                    params.cutoff_freqs = cutoff_freqs
                else:
                    params.cutoff_freqs = [cutoff_freqs]
                
                # Create filter
                self._filter = _FIRFilter(params, sample_rate, device_id)
                self.coefficients = np.array(self._filter.get_coefficients())
            else:
                # Use scipy to design filter
                from scipy import signal as sp_signal
                
                # Convert to normalized frequency
                nyquist = sample_rate / 2.0
                if isinstance(cutoff_freqs, (list, tuple)):
                    norm_cutoffs = [freq / nyquist for freq in cutoff_freqs]
                else:
                    norm_cutoffs = cutoff_freqs / nyquist
                
                # Map window type to scipy name
                window_map = {
                    WindowType.RECTANGULAR: 'boxcar',
                    WindowType.TRIANGULAR: 'triang',
                    WindowType.HANN: 'hann',
                    WindowType.HAMMING: 'hamming',
                    WindowType.BLACKMAN: 'blackman',
                    WindowType.KAISER: ('kaiser', 4.0)
                }
                window = window_map.get(window_type, 'hamming')
                
                # Design filter using scipy
                if filter_type == FilterType.LOWPASS:
                    self.coefficients = sp_signal.firwin(
                        num_taps, norm_cutoffs, window=window, pass_zero='lowpass'
                    )
                elif filter_type == FilterType.HIGHPASS:
                    self.coefficients = sp_signal.firwin(
                        num_taps, norm_cutoffs, window=window, pass_zero='highpass'
                    )
                elif filter_type == FilterType.BANDPASS:
                    if len(norm_cutoffs) != 2:
                        raise ValueError("Bandpass filter requires two cutoff frequencies")
                    self.coefficients = sp_signal.firwin(
                        num_taps, norm_cutoffs, window=window, pass_zero='bandpass'
                    )
                elif filter_type == FilterType.BANDSTOP:
                    if len(norm_cutoffs) != 2:
                        raise ValueError("Bandstop filter requires two cutoff frequencies")
                    self.coefficients = sp_signal.firwin(
                        num_taps, norm_cutoffs, window=window, pass_zero='bandstop'
                    )
        else:
            raise ValueError("Either coefficients or design parameters must be provided")
        
        self.has_cuda = _HAS_CUDA and self.device_id >= 0
        
        # If no CUDA, initialize state for scipy lfilter
        if not self.has_cuda:
            self._zi = np.zeros(len(self.coefficients) - 1)
    
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply the filter to a signal
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        
        Returns
        -------
        np.ndarray
            Filtered signal
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.filter(signal))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            filtered_signal, self._zi = sp_signal.lfilter(
                self.coefficients, [1.0], signal, zi=self._zi
            )
            return filtered_signal
    
    def reset(self) -> None:
        """Reset the filter state"""
        if self.has_cuda:
            self._filter.reset()
        else:
            self._zi = np.zeros(len(self.coefficients) - 1)
    
    def get_frequency_response(self, num_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Get the filter frequency response
        
        Parameters
        ----------
        num_points : int, optional
            Number of frequency points, by default 512
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Frequencies and magnitude response
        """
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.get_frequency_response(num_points))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            w, h = sp_signal.freqz(self.coefficients, [1.0], worN=num_points)
            frequencies = w / np.pi
            magnitude = np.abs(h)
            return frequencies, magnitude
    
    def get_phase_response(self, num_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Get the filter phase response
        
        Parameters
        ----------
        num_points : int, optional
            Number of frequency points, by default 512
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Frequencies and phase response
        """
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.get_phase_response(num_points))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            w, h = sp_signal.freqz(self.coefficients, [1.0], worN=num_points)
            frequencies = w / np.pi
            phase = np.angle(h)
            return frequencies, phase
    
    def get_step_response(self, num_points: int = 100) -> np.ndarray:
        """Get the filter step response
        
        Parameters
        ----------
        num_points : int, optional
            Number of time points, by default 100
        
        Returns
        -------
        np.ndarray
            Step response
        """
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.get_step_response(num_points))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            
            # Create step input
            step_input = np.ones(num_points)
            
            # Apply filter
            step_response = sp_signal.lfilter(self.coefficients, [1.0], step_input)
            
            return step_response
    
    def get_impulse_response(self, num_points: int = 100) -> np.ndarray:
        """Get the filter impulse response
        
        Parameters
        ----------
        num_points : int, optional
            Number of time points, by default 100
        
        Returns
        -------
        np.ndarray
            Impulse response
        """
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.get_impulse_response(num_points))
        else:
            # For FIR filters, the impulse response is just the coefficients
            # padded with zeros if necessary
            if len(self.coefficients) >= num_points:
                return self.coefficients[:num_points]
            else:
                impulse_response = np.zeros(num_points)
                impulse_response[:len(self.coefficients)] = self.coefficients
                return impulse_response


class IIRFilter:
    """Infinite Impulse Response (IIR) filter with GPU acceleration
    
    This class provides IIR filtering operations optimized for NVIDIA GPUs,
    with automatic fallback to CPU implementation when CUDA is not available.
    
    Parameters
    ----------
    a : np.ndarray or None, optional
        Denominator coefficients (a[0] = 1.0 assumed), by default None
    b : np.ndarray or None, optional
        Numerator coefficients, by default None
    sample_rate : float, optional
        Sample rate in Hz (required if a and b are None), by default None
    filter_type : FilterType, optional
        Filter type, by default FilterType.LOWPASS
    cutoff_freqs : list or float, optional
        Cutoff frequencies in Hz, by default None
    order : int, optional
        Filter order (required if a and b are None), by default 0
    design_method : IIRDesignMethod, optional
        Design method, by default IIRDesignMethod.BUTTERWORTH
    ripple_db : float, optional
        Passband ripple in dB (for Chebyshev, Elliptic), by default 0.5
    stopband_atten_db : float, optional
        Stopband attenuation in dB (for Chebyshev II, Elliptic), by default 40.0
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    
    Raises
    ------
    ValueError
        If both coefficients and design parameters are None
    """
    
    def __init__(
        self,
        a: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        filter_type: FilterType = FilterType.LOWPASS,
        cutoff_freqs: Optional[Union[List[float], float]] = None,
        order: int = 0,
        design_method: IIRDesignMethod = IIRDesignMethod.BUTTERWORTH,
        ripple_db: float = 0.5,
        stopband_atten_db: float = 40.0,
        device_id: int = 0
    ):
        """Initialize the IIR filter"""
        self.device_id = device_id
        
        if a is not None and b is not None:
            # Initialize with provided coefficients
            self.a = np.asarray(a, dtype=np.float32)
            self.b = np.asarray(b, dtype=np.float32)
            if _HAS_CUDA:
                self._filter = _IIRFilter(self.a, self.b, device_id)
        elif sample_rate is not None and cutoff_freqs is not None and order > 0:
            # Initialize with design parameters
            if _HAS_CUDA:
                params = _IIRFilterParams()
                params.filter_type = _FilterType(filter_type.value)
                params.order = order
                params.design_method = _IIRDesignMethod(design_method.value)
                params.ripple_db = ripple_db
                params.stopband_atten_db = stopband_atten_db
                
                # Handle cutoff frequencies
                if isinstance(cutoff_freqs, (list, tuple)):
                    params.cutoff_freqs = cutoff_freqs
                else:
                    params.cutoff_freqs = [cutoff_freqs]
                
                # Create filter
                self._filter = _IIRFilter(params, sample_rate, device_id)
                self.a, self.b = self._filter.get_coefficients()
            else:
                # Use scipy to design filter
                from scipy import signal as sp_signal
                
                # Convert to normalized frequency
                nyquist = sample_rate / 2.0
                if isinstance(cutoff_freqs, (list, tuple)):
                    norm_cutoffs = [freq / nyquist for freq in cutoff_freqs]
                else:
                    norm_cutoffs = cutoff_freqs / nyquist
                
                # Design filter using scipy
                if design_method == IIRDesignMethod.BUTTERWORTH:
                    if filter_type == FilterType.LOWPASS:
                        self.b, self.a = sp_signal.butter(
                            order, norm_cutoffs, btype='lowpass'
                        )
                    elif filter_type == FilterType.HIGHPASS:
                        self.b, self.a = sp_signal.butter(
                            order, norm_cutoffs, btype='highpass'
                        )
                    elif filter_type == FilterType.BANDPASS:
                        self.b, self.a = sp_signal.butter(
                            order, norm_cutoffs, btype='bandpass'
                        )
                    elif filter_type == FilterType.BANDSTOP:
                        self.b, self.a = sp_signal.butter(
                            order, norm_cutoffs, btype='bandstop'
                        )
                elif design_method == IIRDesignMethod.CHEBYSHEV1:
                    if filter_type == FilterType.LOWPASS:
                        self.b, self.a = sp_signal.cheby1(
                            order, ripple_db, norm_cutoffs, btype='lowpass'
                        )
                    elif filter_type == FilterType.HIGHPASS:
                        self.b, self.a = sp_signal.cheby1(
                            order, ripple_db, norm_cutoffs, btype='highpass'
                        )
                    elif filter_type == FilterType.BANDPASS:
                        self.b, self.a = sp_signal.cheby1(
                            order, ripple_db, norm_cutoffs, btype='bandpass'
                        )
                    elif filter_type == FilterType.BANDSTOP:
                        self.b, self.a = sp_signal.cheby1(
                            order, ripple_db, norm_cutoffs, btype='bandstop'
                        )
                elif design_method == IIRDesignMethod.CHEBYSHEV2:
                    if filter_type == FilterType.LOWPASS:
                        self.b, self.a = sp_signal.cheby2(
                            order, stopband_atten_db, norm_cutoffs, btype='lowpass'
                        )
                    elif filter_type == FilterType.HIGHPASS:
                        self.b, self.a = sp_signal.cheby2(
                            order, stopband_atten_db, norm_cutoffs, btype='highpass'
                        )
                    elif filter_type == FilterType.BANDPASS:
                        self.b, self.a = sp_signal.cheby2(
                            order, stopband_atten_db, norm_cutoffs, btype='bandpass'
                        )
                    elif filter_type == FilterType.BANDSTOP:
                        self.b, self.a = sp_signal.cheby2(
                            order, stopband_atten_db, norm_cutoffs, btype='bandstop'
                        )
                elif design_method == IIRDesignMethod.ELLIPTIC:
                    if filter_type == FilterType.LOWPASS:
                        self.b, self.a = sp_signal.ellip(
                            order, ripple_db, stopband_atten_db, norm_cutoffs, btype='lowpass'
                        )
                    elif filter_type == FilterType.HIGHPASS:
                        self.b, self.a = sp_signal.ellip(
                            order, ripple_db, stopband_atten_db, norm_cutoffs, btype='highpass'
                        )
                    elif filter_type == FilterType.BANDPASS:
                        self.b, self.a = sp_signal.ellip(
                            order, ripple_db, stopband_atten_db, norm_cutoffs, btype='bandpass'
                        )
                    elif filter_type == FilterType.BANDSTOP:
                        self.b, self.a = sp_signal.ellip(
                            order, ripple_db, stopband_atten_db, norm_cutoffs, btype='bandstop'
                        )
                elif design_method == IIRDesignMethod.BESSEL:
                    if filter_type == FilterType.LOWPASS:
                        self.b, self.a = sp_signal.bessel(
                            order, norm_cutoffs, btype='lowpass'
                        )
                    elif filter_type == FilterType.HIGHPASS:
                        self.b, self.a = sp_signal.bessel(
                            order, norm_cutoffs, btype='highpass'
                        )
                    elif filter_type == FilterType.BANDPASS:
                        self.b, self.a = sp_signal.bessel(
                            order, norm_cutoffs, btype='bandpass'
                        )
                    elif filter_type == FilterType.BANDSTOP:
                        self.b, self.a = sp_signal.bessel(
                            order, norm_cutoffs, btype='bandstop'
                        )
        else:
            raise ValueError("Either coefficients or design parameters must be provided")
        
        self.has_cuda = _HAS_CUDA and self.device_id >= 0
        
        # If no CUDA, initialize state for scipy lfilter
        if not self.has_cuda:
            self._zi = np.zeros(max(len(self.a), len(self.b)) - 1)
    
    def filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply the filter to a signal
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        
        Returns
        -------
        np.ndarray
            Filtered signal
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.filter(signal))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            filtered_signal, self._zi = sp_signal.lfilter(
                self.b, self.a, signal, zi=self._zi
            )
            return filtered_signal
    
    def filter_sos(self, signal: np.ndarray) -> np.ndarray:
        """Apply the filter to a signal using second-order sections
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        
        Returns
        -------
        np.ndarray
            Filtered signal
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.filter_sos(signal))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            
            # Convert to SOS form if not already done
            if not hasattr(self, '_sos'):
                self._sos = sp_signal.tf2sos(self.b, self.a)
                self._sos_zi = sp_signal.sosfilt_zi(self._sos)
                
            # Apply filter
            filtered_signal, self._sos_zi = sp_signal.sosfilt(
                self._sos, signal, zi=self._sos_zi
            )
            
            return filtered_signal
    
    def reset(self) -> None:
        """Reset the filter state"""
        if self.has_cuda:
            self._filter.reset()
        else:
            self._zi = np.zeros(max(len(self.a), len(self.b)) - 1)
            if hasattr(self, '_sos_zi'):
                self._sos_zi = sp_signal.sosfilt_zi(self._sos)
    
    def get_frequency_response(self, num_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Get the filter frequency response
        
        Parameters
        ----------
        num_points : int, optional
            Number of frequency points, by default 512
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Frequencies and magnitude response
        """
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.get_frequency_response(num_points))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            w, h = sp_signal.freqz(self.b, self.a, worN=num_points)
            frequencies = w / np.pi
            magnitude = np.abs(h)
            return frequencies, magnitude
    
    def get_phase_response(self, num_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """Get the filter phase response
        
        Parameters
        ----------
        num_points : int, optional
            Number of frequency points, by default 512
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Frequencies and phase response
        """
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.get_phase_response(num_points))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            w, h = sp_signal.freqz(self.b, self.a, worN=num_points)
            frequencies = w / np.pi
            phase = np.angle(h)
            return frequencies, phase
    
    def is_stable(self) -> bool:
        """Check if the filter is stable
        
        Returns
        -------
        bool
            True if the filter is stable
        """
        if self.has_cuda:
            # Use C++/CUDA implementation
            return self._filter.is_stable()
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            return np.all(np.abs(np.roots(self.a)) < 1.0)


class AdaptiveFilter:
    """Adaptive filter with GPU acceleration
    
    This class provides adaptive filtering operations optimized for NVIDIA GPUs,
    with automatic fallback to CPU implementation when CUDA is not available.
    
    Parameters
    ----------
    filter_length : int
        Length of the adaptive filter
    filter_type : AdaptiveFilterType, optional
        Adaptive filter type, by default AdaptiveFilterType.LMS
    step_size : float, optional
        Step size (mu) parameter, by default 0.1
    forgetting_factor : float, optional
        Forgetting factor for RLS, by default 0.99
    regularization : float, optional
        Regularization parameter, by default 1e-6
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    """
    
    def __init__(
        self,
        filter_length: int,
        filter_type: AdaptiveFilterType = AdaptiveFilterType.LMS,
        step_size: float = 0.1,
        forgetting_factor: float = 0.99,
        regularization: float = 1e-6,
        device_id: int = 0
    ):
        """Initialize the adaptive filter"""
        self.device_id = device_id
        self.filter_length = filter_length
        self.filter_type = filter_type
        self.step_size = step_size
        self.forgetting_factor = forgetting_factor
        self.regularization = regularization
        
        if _HAS_CUDA:
            # Create parameters for C++ implementation
            params = _AdaptiveFilterParams()
            params.filter_length = filter_length
            params.filter_type = _AdaptiveFilterType(filter_type.value)
            params.step_size = step_size
            params.forgetting_factor = forgetting_factor
            params.regularization = regularization
            
            # Create filter
            self._filter = _AdaptiveFilter(params, device_id)
        
        self.has_cuda = _HAS_CUDA and self.device_id >= 0
        
        # If no CUDA, initialize weights for CPU implementation
        if not self.has_cuda:
            self.weights = np.zeros(filter_length, dtype=np.float32)
            self.input_buffer = np.zeros(filter_length, dtype=np.float32)
            
            # For RLS
            if filter_type == AdaptiveFilterType.RLS:
                self.P = np.eye(filter_length) / regularization
            
            # Learning curve
            self.learning_curve = []
    
    def filter(
        self, 
        input_signal: np.ndarray, 
        desired_signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the adaptive filter and update coefficients
        
        Parameters
        ----------
        input_signal : np.ndarray
            Input signal
        desired_signal : np.ndarray
            Desired signal
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Filtered signal and error signal
        """
        # Ensure inputs are float32 numpy arrays
        input_signal = np.asarray(input_signal, dtype=np.float32)
        desired_signal = np.asarray(desired_signal, dtype=np.float32)
        
        if len(input_signal) != len(desired_signal):
            raise ValueError("Input and desired signals must have the same length")
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            filtered, error = self._filter.filter(input_signal, desired_signal)
            return np.array(filtered), np.array(error)
        else:
            # Fallback to CPU implementation
            n_samples = len(input_signal)
            filtered = np.zeros(n_samples, dtype=np.float32)
            error = np.zeros(n_samples, dtype=np.float32)
            
            for i in range(n_samples):
                # Shift input buffer and add new sample
                self.input_buffer = np.roll(self.input_buffer, 1)
                self.input_buffer[0] = input_signal[i]
                
                # Compute output
                filtered[i] = np.dot(self.weights, self.input_buffer)
                
                # Compute error
                error[i] = desired_signal[i] - filtered[i]
                
                # Update weights based on filter type
                if self.filter_type == AdaptiveFilterType.LMS:
                    # LMS algorithm
                    self.weights += self.step_size * error[i] * self.input_buffer
                
                elif self.filter_type == AdaptiveFilterType.NLMS:
                    # NLMS algorithm
                    energy = np.dot(self.input_buffer, self.input_buffer)
                    if energy > self.regularization:
                        normalized_step = self.step_size / energy
                    else:
                        normalized_step = self.step_size / self.regularization
                    
                    self.weights += normalized_step * error[i] * self.input_buffer
                
                elif self.filter_type == AdaptiveFilterType.RLS:
                    # RLS algorithm
                    # Compute gain vector
                    k = np.dot(self.P, self.input_buffer)
                    denom = self.forgetting_factor + np.dot(self.input_buffer, k)
                    k = k / denom
                    
                    # Update weights
                    self.weights += error[i] * k
                    
                    # Update inverse correlation matrix
                    kxt = np.outer(k, self.input_buffer)
                    self.P = (self.P - np.dot(kxt, self.P)) / self.forgetting_factor
                
                # Store squared error for learning curve
                self.learning_curve.append(error[i] ** 2)
            
            return filtered, error
    
    def get_coefficients(self) -> np.ndarray:
        """Get the current filter coefficients
        
        Returns
        -------
        np.ndarray
            Current filter coefficients
        """
        if self.has_cuda:
            return np.array(self._filter.get_coefficients())
        else:
            return self.weights
    
    def get_learning_curve(self) -> np.ndarray:
        """Get the learning curve (error vs iteration)
        
        Returns
        -------
        np.ndarray
            Learning curve
        """
        if self.has_cuda:
            return np.array(self._filter.get_learning_curve())
        else:
            return np.array(self.learning_curve)
    
    def reset(self) -> None:
        """Reset the filter state and coefficients"""
        if self.has_cuda:
            self._filter.reset()
        else:
            self.weights = np.zeros(self.filter_length, dtype=np.float32)
            self.input_buffer = np.zeros(self.filter_length, dtype=np.float32)
            
            if self.filter_type == AdaptiveFilterType.RLS:
                self.P = np.eye(self.filter_length) / self.regularization
            
            self.learning_curve = []


class MultirateFilter:
    """Multirate filter for resampling operations
    
    This class provides multirate filtering operations optimized for NVIDIA GPUs,
    with automatic fallback to CPU implementation when CUDA is not available.
    
    Parameters
    ----------
    interpolation_factor : int
        Interpolation factor (upsampling)
    decimation_factor : int
        Decimation factor (downsampling)
    filter_params : dict, optional
        FIR filter parameters for anti-aliasing
    device_id : int, optional
        CUDA device ID to use (-1 for CPU), by default 0
    """
    
    def __init__(
        self,
        interpolation_factor: int,
        decimation_factor: int,
        filter_params: Optional[Dict] = None,
        device_id: int = 0
    ):
        """Initialize the multirate filter"""
        self.device_id = device_id
        self.interpolation_factor = interpolation_factor
        self.decimation_factor = decimation_factor
        
        # Default filter parameters if not provided
        if filter_params is None:
            filter_params = {
                'filter_type': FilterType.LOWPASS,
                'cutoff_freqs': 0.5 / max(interpolation_factor, decimation_factor),
                'window_type': WindowType.HAMMING,
                'num_taps': 31
            }
        
        self.filter_params = filter_params
        
        if _HAS_CUDA:
            # Create parameters for C++ implementation
            params = _MultirateFilterParams()
            params.interpolation_factor = interpolation_factor
            params.decimation_factor = decimation_factor
            
            # FIR filter parameters
            fir_params = _FIRFilterParams()
            fir_params.filter_type = _FilterType(filter_params.get('filter_type', FilterType.LOWPASS).value)
            fir_params.num_taps = filter_params.get('num_taps', 31)
            fir_params.window_type = _WindowType(filter_params.get('window_type', WindowType.HAMMING).value)
            
            # Handle cutoff frequencies
            cutoff_freqs = filter_params.get('cutoff_freqs', 0.5 / max(interpolation_factor, decimation_factor))
            if isinstance(cutoff_freqs, (list, tuple)):
                fir_params.cutoff_freqs = cutoff_freqs
            else:
                fir_params.cutoff_freqs = [cutoff_freqs]
            
            params.filter_params = fir_params
            
            # Create filter
            self._filter = _MultirateFilter(params, device_id)
        
        self.has_cuda = _HAS_CUDA and self.device_id >= 0
        
        # If no CUDA, create FIR filter for CPU implementation
        if not self.has_cuda:
            from scipy import signal as sp_signal
            
            # Design anti-aliasing filter
            cutoff = filter_params.get('cutoff_freqs', 0.5 / max(interpolation_factor, decimation_factor))
            if isinstance(cutoff, (list, tuple)):
                cutoff = cutoff[0]  # Use first cutoff frequency
            
            # Map window type to scipy name
            window_map = {
                WindowType.RECTANGULAR: 'boxcar',
                WindowType.TRIANGULAR: 'triang',
                WindowType.HANN: 'hann',
                WindowType.HAMMING: 'hamming',
                WindowType.BLACKMAN: 'blackman',
                WindowType.KAISER: ('kaiser', 4.0)
            }
            window = window_map.get(filter_params.get('window_type', WindowType.HAMMING), 'hamming')
            
            # Design lowpass filter
            self.fir_coeffs = sp_signal.firwin(
                filter_params.get('num_taps', 31),
                cutoff,
                window=window
            )
    
    def upsample(self, signal: np.ndarray) -> np.ndarray:
        """Apply upsampling (interpolation)
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        
        Returns
        -------
        np.ndarray
            Upsampled signal
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.upsample(signal))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            
            # Insert zeros
            up = np.zeros(len(signal) * self.interpolation_factor, dtype=np.float32)
            up[::self.interpolation_factor] = signal * self.interpolation_factor
            
            # Apply anti-aliasing filter
            upsampled = sp_signal.lfilter(self.fir_coeffs, [1.0], up)
            
            return upsampled
    
    def downsample(self, signal: np.ndarray) -> np.ndarray:
        """Apply downsampling (decimation)
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        
        Returns
        -------
        np.ndarray
            Downsampled signal
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.downsample(signal))
        else:
            # Fallback to scipy implementation
            from scipy import signal as sp_signal
            
            # Apply anti-aliasing filter
            filtered = sp_signal.lfilter(self.fir_coeffs, [1.0], signal)
            
            # Decimate
            downsampled = filtered[::self.decimation_factor]
            
            return downsampled
    
    def resample(self, signal: np.ndarray) -> np.ndarray:
        """Apply resampling (rational rate conversion)
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        
        Returns
        -------
        np.ndarray
            Resampled signal
        """
        # Ensure input is float32 numpy array
        signal = np.asarray(signal, dtype=np.float32)
        
        if self.has_cuda:
            # Use C++/CUDA implementation
            return np.array(self._filter.resample(signal))
        else:
            # Fallback to scipy implementation
            # First upsample, then downsample
            upsampled = self.upsample(signal)
            resampled = self.downsample(upsampled)
            
            return resampled
    
    def reset(self) -> None:
        """Reset the filter state"""
        if self.has_cuda:
            self._filter.reset()
    
    def get_coefficients(self) -> np.ndarray:
        """Get the filter coefficients
        
        Returns
        -------
        np.ndarray
            Filter coefficients
        """
        if self.has_cuda:
            return np.array(self._filter.get_coefficients())
        else:
            return self.fir_coeffs


# Convenience functions for filter design

def design_lowpass(
    cutoff_freq: float,
    sample_rate: float,
    num_taps: int = 31,
    window_type: WindowType = WindowType.HAMMING,
    design_method: FIRDesignMethod = FIRDesignMethod.WINDOW
) -> FIRFilter:
    """Design a lowpass FIR filter
    
    Parameters
    ----------
    cutoff_freq : float
        Cutoff frequency in Hz
    sample_rate : float
        Sample rate in Hz
    num_taps : int, optional
        Number of filter taps, by default 31
    window_type : WindowType, optional
        Window type, by default WindowType.HAMMING
    design_method : FIRDesignMethod, optional
        Design method, by default FIRDesignMethod.WINDOW
    
    Returns
    -------
    FIRFilter
        Designed lowpass filter
    """
    return FIRFilter(
        coefficients=None,
        sample_rate=sample_rate,
        filter_type=FilterType.LOWPASS,
        cutoff_freqs=cutoff_freq,
        window_type=window_type,
        num_taps=num_taps,
        design_method=design_method
    )


def design_highpass(
    cutoff_freq: float,
    sample_rate: float,
    num_taps: int = 31,
    window_type: WindowType = WindowType.HAMMING,
    design_method: FIRDesignMethod = FIRDesignMethod.WINDOW
) -> FIRFilter:
    """Design a highpass FIR filter
    
    Parameters
    ----------
    cutoff_freq : float
        Cutoff frequency in Hz
    sample_rate : float
        Sample rate in Hz
    num_taps : int, optional
        Number of filter taps, by default 31
    window_type : WindowType, optional
        Window type, by default WindowType.HAMMING
    design_method : FIRDesignMethod, optional
        Design method, by default FIRDesignMethod.WINDOW
    
    Returns
    -------
    FIRFilter
        Designed highpass filter
    """
    return FIRFilter(
        coefficients=None,
        sample_rate=sample_rate,
        filter_type=FilterType.HIGHPASS,
        cutoff_freqs=cutoff_freq,
        window_type=window_type,
        num_taps=num_taps,
        design_method=design_method
    )


def design_bandpass(
    cutoff_freqs: Tuple[float, float],
    sample_rate: float,
    num_taps: int = 31,
    window_type: WindowType = WindowType.HAMMING,
    design_method: FIRDesignMethod = FIRDesignMethod.WINDOW
) -> FIRFilter:
    """Design a bandpass FIR filter
    
    Parameters
    ----------
    cutoff_freqs : Tuple[float, float]
        Cutoff frequencies in Hz (low, high)
    sample_rate : float
        Sample rate in Hz
    num_taps : int, optional
        Number of filter taps, by default 31
    window_type : WindowType, optional
        Window type, by default WindowType.HAMMING
    design_method : FIRDesignMethod, optional
        Design method, by default FIRDesignMethod.WINDOW
    
    Returns
    -------
    FIRFilter
        Designed bandpass filter
    """
    return FIRFilter(
        coefficients=None,
        sample_rate=sample_rate,
        filter_type=FilterType.BANDPASS,
        cutoff_freqs=cutoff_freqs,
        window_type=window_type,
        num_taps=num_taps,
        design_method=design_method
    )


def design_bandstop(
    cutoff_freqs: Tuple[float, float],
    sample_rate: float,
    num_taps: int = 31,
    window_type: WindowType = WindowType.HAMMING,
    design_method: FIRDesignMethod = FIRDesignMethod.WINDOW
) -> FIRFilter:
    """Design a bandstop FIR filter
    
    Parameters
    ----------
    cutoff_freqs : Tuple[float, float]
        Cutoff frequencies in Hz (low, high)
    sample_rate : float
        Sample rate in Hz
    num_taps : int, optional
        Number of filter taps, by default 31
    window_type : WindowType, optional
        Window type, by default WindowType.HAMMING
    design_method : FIRDesignMethod, optional
        Design method, by default FIRDesignMethod.WINDOW
    
    Returns
    -------
    FIRFilter
        Designed bandstop filter
    """
    return FIRFilter(
        coefficients=None,
        sample_rate=sample_rate,
        filter_type=FilterType.BANDSTOP,
        cutoff_freqs=cutoff_freqs,
        window_type=window_type,
        num_taps=num_taps,
        design_method=design_method
    )


# Wrapper functions for static filter functions

def median_filter(
    signal: np.ndarray,
    kernel_size: int,
    device_id: int = 0
) -> np.ndarray:
    """Apply median filtering to a signal
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    kernel_size : int
        Size of the median filter kernel (must be odd)
    device_id : int, optional
        CUDA device ID (-1 for CPU), by default 0
    
    Returns
    -------
    np.ndarray
        Filtered signal
    """
    # Ensure input is float32 numpy array
    signal = np.asarray(signal, dtype=np.float32)
    
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    if _HAS_CUDA and device_id >= 0:
        # Use C++/CUDA implementation
        return np.array(_median_filter(signal, kernel_size, device_id))
    else:
        # Fallback to scipy implementation
        from scipy import signal as sp_signal
        return sp_signal.medfilt(signal, kernel_size)


def wiener_filter(
    signal: np.ndarray,
    noise_power: float,
    kernel_size: int = 5,
    device_id: int = 0
) -> np.ndarray:
    """Apply Wiener filter for noise reduction
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    noise_power : float
        Estimated noise power (variance)
    kernel_size : int, optional
        Size of the local variance estimation window, by default 5
    device_id : int, optional
        CUDA device ID (-1 for CPU), by default 0
    
    Returns
    -------
    np.ndarray
        Filtered signal
    """
    # Ensure input is float32 numpy array
    signal = np.asarray(signal, dtype=np.float32)
    
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    if _HAS_CUDA and device_id >= 0:
        # Use C++/CUDA implementation
        return np.array(_wiener_filter(signal, noise_power, kernel_size, device_id))
    else:
        # Fallback to scipy implementation
        from scipy import signal as sp_signal
        return sp_signal.wiener(signal, kernel_size, noise_power)


def kalman_filter(
    signal: np.ndarray,
    process_variance: float,
    measurement_variance: float,
    device_id: int = 0
) -> np.ndarray:
    """Apply Kalman filter to a signal
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    process_variance : float
        Process variance (Q)
    measurement_variance : float
        Measurement variance (R)
    device_id : int, optional
        CUDA device ID (-1 for CPU), by default 0
    
    Returns
    -------
    np.ndarray
        Filtered signal
    """
    # Ensure input is float32 numpy array
    signal = np.asarray(signal, dtype=np.float32)
    
    if _HAS_CUDA and device_id >= 0:
        # Use C++/CUDA implementation
        return np.array(_kalman_filter(signal, process_variance, measurement_variance, device_id))
    else:
        # Fallback to numpy implementation
        n = len(signal)
        filtered = np.zeros(n, dtype=np.float32)
        
        # Initial state estimate and covariance
        x_est = signal[0]
        p_est = 1.0
        
        # Process all measurements
        for i in range(n):
            # Prediction step
            x_pred = x_est
            p_pred = p_est + process_variance
            
            # Update step
            k = p_pred / (p_pred + measurement_variance)
            x_est = x_pred + k * (signal[i] - x_pred)
            p_est = (1 - k) * p_pred
            
            filtered[i] = x_est
        
        return filtered