# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Wavelet transform module for the Signal Processing package.

This module provides various wavelet transform functions and classes for 
signal analysis and processing, including Discrete Wavelet Transform (DWT),
Continuous Wavelet Transform (CWT), Wavelet Packet Transform, and 
Maximal Overlap Discrete Wavelet Transform (MODWT).

The implementation leverages GPU acceleration when available.
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import warnings
try:
    import _signal_processing as _sp
    HAS_CPP_BINDINGS = True
except ImportError:
    HAS_CPP_BINDINGS = False
    warnings.warn("C++ bindings not available, using pure Python implementation")


class WaveletFamily(Enum):
    """Enum defining available wavelet families."""
    HAAR = 0
    DAUBECHIES = 1
    SYMLET = 2
    COIFLET = 3
    BIORTHOGONAL = 4
    MEYER = 5
    MORLET = 6
    MEXICAN_HAT = 7


class BoundaryMode(Enum):
    """Enum defining boundary handling modes for wavelet transforms."""
    ZERO_PADDING = 0
    SYMMETRIC = 1
    PERIODIC = 2
    REFLECT = 3


class DiscreteWaveletTransform:
    """
    Discrete Wavelet Transform (DWT) implementation.
    
    The DWT decomposes a signal into approximation and detail coefficients
    using high-pass and low-pass filters, followed by downsampling. This
    implementation supports various wavelet families and boundary handling modes.
    
    GPU acceleration is used when available.
    """
    
    def __init__(self, wavelet_family=WaveletFamily.DAUBECHIES, vanishing_moments=4):
        """
        Initialize a new DWT object.
        
        Args:
            wavelet_family (WaveletFamily): The wavelet family to use
            vanishing_moments (int): Number of vanishing moments (for families that support it)
        """
        self.family = wavelet_family
        self.vanishing_moments = vanishing_moments
        self._use_cpp = HAS_CPP_BINDINGS
        
        # Initialize filters based on wavelet family
        self._init_filters()
    
    def _init_filters(self):
        """Initialize wavelet filters based on family and vanishing moments."""
        if self.family == WaveletFamily.HAAR:
            # Haar wavelet is the simplest
            self.decomp_low = np.array([0.7071067811865475, 0.7071067811865475])
            self.decomp_high = np.array([0.7071067811865475, -0.7071067811865475])
            self.recon_low = self.decomp_low
            self.recon_high = np.array([-self.decomp_high[0], self.decomp_high[1]])
        elif self.family == WaveletFamily.DAUBECHIES:
            # Use predefined coefficients for Daubechies wavelets
            if self.vanishing_moments == 1:
                # This is actually Haar
                self._init_filters(WaveletFamily.HAAR)
                return
            elif self.vanishing_moments == 2:
                # db4 (4 coefficients)
                self.decomp_low = np.array([
                    0.4829629131445341, 0.8365163037378079, 
                    0.2241438680420134, -0.1294095225512604
                ])
            elif self.vanishing_moments == 4:
                # db8 (8 coefficients)
                self.decomp_low = np.array([
                    0.2303778133088964, 0.7148465705529154, 0.6308807679298587, 
                    -0.0279837694168599, -0.1870348117190931, 0.0308413818355607, 
                    0.0328830116668852, -0.0105974017850690
                ])
            else:
                raise ValueError(f"Daubechies filter with {self.vanishing_moments} vanishing moments not implemented")
                
            # Generate high pass filter using quadrature mirror relationship
            self.decomp_high = np.zeros_like(self.decomp_low)
            N = len(self.decomp_low)
            for i in range(N):
                self.decomp_high[i] = (-1)**i * self.decomp_low[N-1-i]
            
            # Generate reconstruction filters
            self.recon_low = self.decomp_high[::-1]
            self.recon_high = np.array([(-1)**(i+1) * self.decomp_low[i] for i in range(N)])[::-1]
        
        elif self.family == WaveletFamily.SYMLET:
            # For simplicity, we use a predefined Symlet filter
            self.decomp_low = np.array([
                -0.0757657147893407, -0.0296355276459541, 
                0.4976186676324578, 0.8037387518052163, 
                0.2978577956055422, -0.0992195435769354
            ])
            
            # Generate high pass and reconstruction filters
            self.decomp_high = np.zeros_like(self.decomp_low)
            N = len(self.decomp_low)
            for i in range(N):
                self.decomp_high[i] = (-1)**i * self.decomp_low[N-1-i]
            
            self.recon_low = self.decomp_high[::-1]
            self.recon_high = np.array([(-1)**(i+1) * self.decomp_low[i] for i in range(N)])[::-1]
        
        else:
            # For other wavelet families, we'd add more filter definitions
            raise ValueError(f"Wavelet family {self.family} not implemented yet")
    
    def _extend_signal(self, signal, filter_length, mode):
        """
        Extend signal to handle boundary effects.
        
        Args:
            signal (numpy.ndarray): Input signal
            filter_length (int): Length of the filter
            mode (BoundaryMode): Method for handling boundaries
            
        Returns:
            numpy.ndarray: Extended signal
        """
        extension_size = filter_length - 1
        signal_size = len(signal)
        extended = np.zeros(signal_size + 2 * extension_size)
        
        # Copy original signal to the middle
        extended[extension_size:extension_size+signal_size] = signal
        
        # Handle boundaries
        if mode == BoundaryMode.ZERO_PADDING:
            # Already initialized to zeros
            pass
        elif mode == BoundaryMode.SYMMETRIC:
            # Left boundary
            for i in range(extension_size):
                extended[extension_size-1-i] = signal[i]
            # Right boundary
            for i in range(extension_size):
                extended[signal_size+extension_size+i] = signal[signal_size-1-i]
        elif mode == BoundaryMode.PERIODIC:
            # Left boundary
            extended[:extension_size] = signal[-(extension_size):]
            # Right boundary
            extended[signal_size+extension_size:] = signal[:extension_size]
        elif mode == BoundaryMode.REFLECT:
            # Left boundary
            for i in range(extension_size):
                extended[i] = signal[extension_size-i]
            # Right boundary
            for i in range(extension_size):
                extended[signal_size+extension_size+i] = signal[signal_size-2-i]
                
        return extended
    
    def _convolve_downsample(self, signal, filter_coef):
        """
        Apply convolution and downsample by 2.
        
        Args:
            signal (numpy.ndarray): Input signal
            filter_coef (numpy.ndarray): Filter coefficients
            
        Returns:
            numpy.ndarray: Filtered and downsampled signal
        """
        # Full convolution
        conv_result = np.convolve(signal, filter_coef, mode='full')
        
        # Downsample by 2 (take every other sample)
        downsampled = conv_result[::2]
        
        return downsampled
    
    def _upsample_convolve(self, signal, filter_coef):
        """
        Upsample signal by 2 and apply convolution.
        
        Args:
            signal (numpy.ndarray): Input signal
            filter_coef (numpy.ndarray): Filter coefficients
            
        Returns:
            numpy.ndarray: Upsampled and filtered signal
        """
        # Upsample by 2 (insert zeros)
        upsampled = np.zeros(2 * len(signal))
        upsampled[::2] = signal
        
        # Full convolution
        conv_result = np.convolve(upsampled, filter_coef, mode='full')
        
        return conv_result
    
    def forward(self, signal, levels=1, mode=BoundaryMode.SYMMETRIC):
        """
        Perform forward DWT.
        
        Args:
            signal (numpy.ndarray): Input signal
            levels (int): Number of decomposition levels
            mode (BoundaryMode): Method for handling boundaries
            
        Returns:
            dict: Dictionary containing approximation and detail coefficients
        """
        if len(signal) < 2**levels:
            raise ValueError(f"Signal length must be at least 2^{levels}")
        
        if self._use_cpp and False:  # TODO: Enable when C++ bindings are ready
            # Use C++ implementation
            pass
        else:
            # Python implementation
            signal = np.asarray(signal, dtype=np.float32)
            filter_length = len(self.decomp_low)
            
            # Initialize result
            approx_coeffs = [signal]
            detail_coeffs = []
            
            # Perform decomposition
            for level in range(levels):
                # Get current approximation
                current_approx = approx_coeffs[-1]
                
                # Extend signal
                extended = self._extend_signal(current_approx, filter_length, mode)
                
                # Apply low and high pass filters, then downsample
                next_approx = self._convolve_downsample(extended, self.decomp_low)
                detail = self._convolve_downsample(extended, self.decomp_high)
                
                # Remove boundary effects
                trim = (filter_length - 1) // 2
                next_approx = next_approx[trim:-trim]
                detail = detail[trim:-trim]
                
                # Store results
                approx_coeffs.append(next_approx)
                detail_coeffs.append(detail)
            
            return {
                'approximation': approx_coeffs,
                'detail': detail_coeffs
            }
    
    def inverse(self, coeffs, mode=BoundaryMode.SYMMETRIC):
        """
        Perform inverse DWT.
        
        Args:
            coeffs (dict): Dictionary with approximation and detail coefficients
            mode (BoundaryMode): Method for handling boundaries
            
        Returns:
            numpy.ndarray: Reconstructed signal
        """
        approx_coeffs = coeffs['approximation']
        detail_coeffs = coeffs['detail']
        
        if self._use_cpp and False:  # TODO: Enable when C++ bindings are ready
            # Use C++ implementation
            pass
        else:
            # Python implementation
            filter_length = len(self.recon_low)
            
            # Start with the coarsest approximation
            reconstruction = approx_coeffs[-1]
            
            # Perform reconstruction
            for level in range(len(detail_coeffs)-1, -1, -1):
                # Get detail coefficients for this level
                detail = detail_coeffs[level]
                
                # Get expected size from the level above
                expected_size = len(approx_coeffs[level])
                
                # Upsample and filter
                upsampled_approx = self._upsample_convolve(reconstruction, self.recon_low)
                upsampled_detail = self._upsample_convolve(detail, self.recon_high)
                
                # Combine approximation and detail
                combined = upsampled_approx + upsampled_detail
                
                # Trim to expected size
                start = filter_length - 1
                reconstruction = combined[start:start+expected_size]
            
            return reconstruction
    
    def analyze(self, signal, levels=3, mode=BoundaryMode.SYMMETRIC, plot=True):
        """
        Analyze signal using DWT and optionally plot the results.
        
        Args:
            signal (numpy.ndarray): Input signal
            levels (int): Number of decomposition levels
            mode (BoundaryMode): Method for handling boundaries
            plot (bool): Whether to plot the decomposition
            
        Returns:
            dict: Dictionary with analysis results
        """
        # Perform DWT
        coeffs = self.forward(signal, levels, mode)
        
        if plot:
            # Create figure for plotting
            plt.figure(figsize=(12, 8))
            
            # Plot original signal
            plt.subplot(levels+2, 1, 1)
            plt.plot(signal)
            plt.title('Original Signal')
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
            plt.show()
        
        return coeffs


class ContinuousWaveletTransform:
    """
    Continuous Wavelet Transform (CWT) implementation.
    
    The CWT decomposes a signal using wavelets scaled and translated continuously,
    providing better time-frequency localization than the DWT.
    
    GPU acceleration is used when available.
    """
    
    def __init__(self, wavelet_family=WaveletFamily.MORLET):
        """
        Initialize a new CWT object.
        
        Args:
            wavelet_family (WaveletFamily): The wavelet family to use
        """
        self.family = wavelet_family
        self._use_cpp = HAS_CPP_BINDINGS
        
        # Set wavelet parameters
        if self.family == WaveletFamily.MORLET:
            self.omega0 = 5.0  # Center frequency
        elif self.family == WaveletFamily.MEXICAN_HAT:
            pass  # No parameters needed
        else:
            raise ValueError(f"Wavelet family {self.family} not supported for CWT")
    
    def _morlet_wavelet(self, t, scale):
        """
        Compute Morlet wavelet at given time and scale.
        
        Args:
            t (float): Time point
            scale (float): Scale parameter
            
        Returns:
            complex: Complex wavelet value
        """
        scaled_t = t / scale
        gauss = np.exp(-0.5 * scaled_t**2)
        return gauss * np.exp(1j * self.omega0 * scaled_t) / np.sqrt(scale)
    
    def _mexican_hat_wavelet(self, t, scale):
        """
        Compute Mexican Hat wavelet at given time and scale.
        
        Args:
            t (float): Time point
            scale (float): Scale parameter
            
        Returns:
            float: Wavelet value
        """
        scaled_t = t / scale
        t_squared = scaled_t**2
        factor = (1.0 - t_squared)
        gauss = np.exp(-0.5 * t_squared)
        norm = 2.0 / (np.sqrt(3.0 * scale) * np.pi**0.25)
        return norm * factor * gauss
    
    def generate_scales(self, num_scales, min_scale=1.0, max_scale=32.0):
        """
        Generate logarithmically spaced scales for CWT.
        
        Args:
            num_scales (int): Number of scales to generate
            min_scale (float): Minimum scale
            max_scale (float): Maximum scale
            
        Returns:
            numpy.ndarray: Array of scales
        """
        return np.logspace(np.log10(min_scale), np.log10(max_scale), num_scales)
    
    def forward(self, signal, scales=None, num_scales=32):
        """
        Perform forward CWT.
        
        Args:
            signal (numpy.ndarray): Input signal
            scales (numpy.ndarray): Scales to use, or None to generate automatically
            num_scales (int): Number of scales if scales is None
            
        Returns:
            numpy.ndarray: 2D array of CWT coefficients (scales x time)
        """
        signal = np.asarray(signal, dtype=np.float32)
        N = len(signal)
        
        # Generate scales if not provided
        if scales is None:
            scales = self.generate_scales(num_scales)
        
        if self._use_cpp and False:  # TODO: Enable when C++ bindings are ready
            # Use C++ implementation
            pass
        else:
            # Python implementation
            # Initialize coefficient array
            coeffs = np.zeros((len(scales), N), dtype=complex)
            
            # Create time array
            t = np.arange(N)
            
            # For each scale, compute coefficients
            for i, scale in enumerate(scales):
                # Create time grid for this scale
                dt = np.outer(np.ones(N), t) - np.outer(t, np.ones(N))
                
                if self.family == WaveletFamily.MORLET:
                    # Compute wavelet for each time offset
                    scaled_t = dt / scale
                    gauss = np.exp(-0.5 * scaled_t**2)
                    wavelet = gauss * np.exp(1j * self.omega0 * scaled_t) / np.sqrt(scale)
                    
                    # Apply wavelet
                    for j in range(N):
                        coeffs[i, j] = np.sum(signal * np.conj(wavelet[j, :]))
                        
                elif self.family == WaveletFamily.MEXICAN_HAT:
                    # Compute wavelet for each time offset
                    scaled_t = dt / scale
                    t_squared = scaled_t**2
                    factor = (1.0 - t_squared)
                    gauss = np.exp(-0.5 * t_squared)
                    norm = 2.0 / (np.sqrt(3.0 * scale) * np.pi**0.25)
                    wavelet = norm * factor * gauss
                    
                    # Apply wavelet
                    for j in range(N):
                        coeffs[i, j] = np.sum(signal * wavelet[j, :])
            
            return coeffs
    
    def inverse(self, coeffs, scales):
        """
        Perform inverse CWT.
        
        Args:
            coeffs (numpy.ndarray): CWT coefficients (scales x time)
            scales (numpy.ndarray): Scales used in the forward transform
            
        Returns:
            numpy.ndarray: Reconstructed signal
        """
        # For simplicity, we only implement the inverse for Morlet
        if self.family != WaveletFamily.MORLET:
            raise NotImplementedError(f"Inverse CWT not implemented for {self.family}")
        
        # Get dimensions
        num_scales, signal_len = coeffs.shape
        
        # Admissibility constant for Morlet
        Cpsi = 0.776
        
        # Initialize reconstruction
        reconstruction = np.zeros(signal_len)
        
        # Integrate over scales
        for i in range(num_scales):
            scale = scales[i]
            
            # Add contribution from this scale
            reconstruction += np.real(coeffs[i, :]) / (scale**2)
            
            # Scale integration factor (if not the last scale)
            if i < num_scales - 1:
                dscale = scales[i+1] - scale
                reconstruction *= dscale
        
        # Apply admissibility factor
        reconstruction /= Cpsi
        
        return reconstruction
    
    def analyze(self, signal, scales=None, num_scales=32, plot=True):
        """
        Analyze signal using CWT and optionally plot the scalogram.
        
        Args:
            signal (numpy.ndarray): Input signal
            scales (numpy.ndarray): Scales to use, or None to generate automatically
            num_scales (int): Number of scales if scales is None
            plot (bool): Whether to plot the scalogram
            
        Returns:
            dict: Dictionary with analysis results
        """
        # Generate scales if not provided
        if scales is None:
            scales = self.generate_scales(num_scales)
        
        # Perform CWT
        coeffs = self.forward(signal, scales)
        
        # Compute scalogram (squared magnitude of coefficients)
        scalogram = np.abs(coeffs)**2
        
        if plot:
            # Create figure for plotting
            plt.figure(figsize=(12, 8))
            
            # Plot original signal
            plt.subplot(2, 1, 1)
            plt.plot(signal)
            plt.title('Original Signal')
            plt.grid(True)
            
            # Plot scalogram
            plt.subplot(2, 1, 2)
            plt.imshow(scalogram, aspect='auto', origin='lower', 
                      extent=[0, len(signal), np.log10(scales[0]), np.log10(scales[-1])],
                      cmap='jet', interpolation='bilinear')
            plt.colorbar(label='Power')
            plt.ylabel('log10(Scale)')
            plt.title('Scalogram')
            
            plt.tight_layout()
            plt.show()
        
        return {
            'coefficients': coeffs,
            'scalogram': scalogram,
            'scales': scales
        }


class WaveletPacketTransform:
    """
    Wavelet Packet Transform (WPT) implementation.
    
    The WPT is a generalization of the DWT where both approximation and
    detail coefficients are further decomposed. This provides a richer
    frequency decomposition at the cost of more computation.
    
    GPU acceleration is used when available.
    """
    
    def __init__(self, wavelet_family=WaveletFamily.DAUBECHIES, vanishing_moments=4):
        """
        Initialize a new WPT object.
        
        Args:
            wavelet_family (WaveletFamily): The wavelet family to use
            vanishing_moments (int): Number of vanishing moments (for families that support it)
        """
        # We use the DWT implementation for filtering operations
        self.dwt = DiscreteWaveletTransform(wavelet_family, vanishing_moments)
        self._use_cpp = HAS_CPP_BINDINGS
    
    def forward(self, signal, levels=1, mode=BoundaryMode.SYMMETRIC):
        """
        Perform forward WPT.
        
        Args:
            signal (numpy.ndarray): Input signal
            levels (int): Number of decomposition levels
            mode (BoundaryMode): Method for handling boundaries
            
        Returns:
            dict: Dictionary containing wavelet packet coefficients organized by level and node
        """
        if len(signal) < 2**levels:
            raise ValueError(f"Signal length must be at least 2^{levels}")
        
        if self._use_cpp and False:  # TODO: Enable when C++ bindings are ready
            # Use C++ implementation
            pass
        else:
            # Python implementation
            signal = np.asarray(signal, dtype=np.float32)
            
            # Initialize result
            coeffs = {}
            # Level 0 has only one node - the original signal
            coeffs[(0, 0)] = signal
            
            # Perform decomposition
            for level in range(levels):
                # For each node at this level
                for node in range(2**level):
                    # Get the signal at this node
                    parent_signal = coeffs[(level, node)]
                    
                    # Decompose using DWT
                    dwt_result = self.dwt.forward(parent_signal, 1, mode)
                    
                    # Store the approximation (left child) and detail (right child)
                    coeffs[(level+1, 2*node)] = dwt_result['approximation'][1]
                    coeffs[(level+1, 2*node+1)] = dwt_result['detail'][0]
            
            return {
                'coefficients': coeffs,
                'levels': levels
            }
    
    def inverse(self, wpt_result, mode=BoundaryMode.SYMMETRIC):
        """
        Perform inverse WPT.
        
        Args:
            wpt_result (dict): Dictionary with wavelet packet coefficients
            mode (BoundaryMode): Method for handling boundaries
            
        Returns:
            numpy.ndarray: Reconstructed signal
        """
        coeffs = wpt_result['coefficients']
        levels = wpt_result['levels']
        
        if self._use_cpp and False:  # TODO: Enable when C++ bindings are ready
            # Use C++ implementation
            pass
        else:
            # Python implementation
            # Create a copy of the coefficients that we'll modify
            result = coeffs.copy()
            
            # Perform reconstruction from bottom to top
            for level in range(levels, 0, -1):
                # For each node at the level we're reconstructing to
                for node in range(2**(level-1)):
                    # Get the approximation (left child) and detail (right child)
                    approx = result[(level, 2*node)]
                    detail = result[(level, 2*node+1)]
                    
                    # Reconstruct parent node using inverse DWT
                    dwt_coeffs = {
                        'approximation': [None, approx],  # The first element is not used in inverse
                        'detail': [detail]
                    }
                    result[(level-1, node)] = self.dwt.inverse(dwt_coeffs, mode)
            
            # The final result is the single node at the top level
            return result[(0, 0)]
    
    def analyze(self, signal, levels=3, mode=BoundaryMode.SYMMETRIC, plot=True):
        """
        Analyze signal using WPT and optionally plot the decomposition.
        
        Args:
            signal (numpy.ndarray): Input signal
            levels (int): Number of decomposition levels
            mode (BoundaryMode): Method for handling boundaries
            plot (bool): Whether to plot the decomposition
            
        Returns:
            dict: Dictionary with analysis results
        """
        # Perform WPT
        wpt_result = self.forward(signal, levels, mode)
        coeffs = wpt_result['coefficients']
        
        if plot:
            # Create figure for plotting
            plt.figure(figsize=(15, 10))
            
            # Plot original signal
            plt.subplot(levels+1, 1, 1)
            plt.plot(signal)
            plt.title('Original Signal')
            plt.grid(True)
            
            # Plot coefficients at each level
            for level in range(1, levels+1):
                plt.subplot(levels+1, 1, level+1)
                
                # Concatenate coefficients at this level
                level_coeffs = []
                for node in range(2**level):
                    if (level, node) in coeffs:
                        level_coeffs.append(coeffs[(level, node)])
                
                # Concatenate and plot
                if level_coeffs:
                    level_data = np.concatenate(level_coeffs)
                    plt.plot(level_data)
                    plt.title(f'Level {level} Coefficients')
                    plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return wpt_result


class MaximalOverlapDWT:
    """
    Maximal Overlap Discrete Wavelet Transform (MODWT) implementation.
    
    The MODWT is a non-decimated wavelet transform that does not downsample,
    making it translation-invariant. It is useful for time series analysis
    and feature extraction.
    
    GPU acceleration is used when available.
    """
    
    def __init__(self, wavelet_family=WaveletFamily.DAUBECHIES, vanishing_moments=4):
        """
        Initialize a new MODWT object.
        
        Args:
            wavelet_family (WaveletFamily): The wavelet family to use
            vanishing_moments (int): Number of vanishing moments (for families that support it)
        """
        # We'll reuse the DWT filters but scale them appropriately for MODWT
        self.dwt = DiscreteWaveletTransform(wavelet_family, vanishing_moments)
        self._use_cpp = HAS_CPP_BINDINGS
        
        # Scale filters by 1/sqrt(2) for MODWT
        self.decomp_low = self.dwt.decomp_low / np.sqrt(2)
        self.decomp_high = self.dwt.decomp_high / np.sqrt(2)
        self.recon_low = self.dwt.recon_low / np.sqrt(2)
        self.recon_high = self.dwt.recon_high / np.sqrt(2)
    
    def _extend_signal(self, signal, filter_length, mode):
        """Extend signal (same as in DWT)."""
        return self.dwt._extend_signal(signal, filter_length, mode)
    
    def _convolve(self, signal, filter_coef):
        """
        Apply convolution without downsampling (unlike DWT).
        
        Args:
            signal (numpy.ndarray): Input signal
            filter_coef (numpy.ndarray): Filter coefficients
            
        Returns:
            numpy.ndarray: Filtered signal
        """
        # Full convolution
        result = np.convolve(signal, filter_coef, mode='full')
        return result
    
    def forward(self, signal, levels=1, mode=BoundaryMode.SYMMETRIC):
        """
        Perform forward MODWT.
        
        Args:
            signal (numpy.ndarray): Input signal
            levels (int): Number of decomposition levels
            mode (BoundaryMode): Method for handling boundaries
            
        Returns:
            dict: Dictionary containing wavelet and scaling coefficients
        """
        if self._use_cpp and False:  # TODO: Enable when C++ bindings are ready
            # Use C++ implementation
            pass
        else:
            # Python implementation
            signal = np.asarray(signal, dtype=np.float32)
            N = len(signal)
            filter_length = len(self.decomp_low)
            
            # Initialize result
            wavelet_coeffs = []  # W_j (detail/wavelet coefficients)
            scaling_coeffs = [signal]  # V_j (approximation/scaling coefficients)
            
            # Perform decomposition
            for level in range(levels):
                # Get current scaling coefficients
                current_scaling = scaling_coeffs[-1]
                
                # Extend signal
                extended = self._extend_signal(current_scaling, filter_length, mode)
                
                # Apply filters without downsampling
                next_scaling = self._convolve(extended, self.decomp_low)
                wavelet = self._convolve(extended, self.decomp_high)
                
                # Trim to original length
                trim = filter_length - 1
                next_scaling = next_scaling[trim:trim+N]
                wavelet = wavelet[trim:trim+N]
                
                # Store results
                scaling_coeffs.append(next_scaling)
                wavelet_coeffs.append(wavelet)
            
            return {
                'wavelet': wavelet_coeffs,  # W_1, W_2, ..., W_J
                'scaling': scaling_coeffs    # V_0, V_1, ..., V_J
            }
    
    def inverse(self, coeffs, mode=BoundaryMode.SYMMETRIC):
        """
        Perform inverse MODWT.
        
        Args:
            coeffs (dict): Dictionary with wavelet and scaling coefficients
            mode (BoundaryMode): Method for handling boundaries
            
        Returns:
            numpy.ndarray: Reconstructed signal
        """
        wavelet_coeffs = coeffs['wavelet']
        scaling_coeffs = coeffs['scaling']
        
        if self._use_cpp and False:  # TODO: Enable when C++ bindings are ready
            # Use C++ implementation
            pass
        else:
            # Python implementation
            levels = len(wavelet_coeffs)
            N = len(scaling_coeffs[0])
            filter_length = len(self.recon_low)
            
            # Start with the coarsest scaling coefficients
            reconstruction = scaling_coeffs[-1].copy()
            
            # Perform reconstruction
            for level in range(levels-1, -1, -1):
                # Get wavelet coefficients for this level
                wavelet = wavelet_coeffs[level]
                
                # Extend signals
                extended_scaling = self._extend_signal(reconstruction, filter_length, mode)
                extended_wavelet = self._extend_signal(wavelet, filter_length, mode)
                
                # Apply reconstruction filters
                scaling_filtered = self._convolve(extended_scaling, self.recon_low)
                wavelet_filtered = self._convolve(extended_wavelet, self.recon_high)
                
                # Combine and trim
                trim = filter_length - 1
                reconstruction = scaling_filtered[trim:trim+N] + wavelet_filtered[trim:trim+N]
            
            return reconstruction
    
    def analyze(self, signal, levels=3, mode=BoundaryMode.SYMMETRIC, plot=True):
        """
        Analyze signal using MODWT and optionally plot the decomposition.
        
        Args:
            signal (numpy.ndarray): Input signal
            levels (int): Number of decomposition levels
            mode (BoundaryMode): Method for handling boundaries
            plot (bool): Whether to plot the decomposition
            
        Returns:
            dict: Dictionary with analysis results
        """
        # Perform MODWT
        coeffs = self.forward(signal, levels, mode)
        
        if plot:
            # Create figure for plotting
            plt.figure(figsize=(12, 8))
            
            # Plot original signal
            plt.subplot(levels+2, 1, 1)
            plt.plot(signal)
            plt.title('Original Signal')
            plt.grid(True)
            
            # Plot scaling coefficients at deepest level
            plt.subplot(levels+2, 1, 2)
            plt.plot(coeffs['scaling'][-1])
            plt.title(f'Scaling Coefficients (Level {levels})')
            plt.grid(True)
            
            # Plot wavelet coefficients from each level
            for i, wavelet in enumerate(coeffs['wavelet']):
                plt.subplot(levels+2, 1, i+3)
                plt.plot(wavelet)
                plt.title(f'Wavelet Coefficients (Level {i+1})')
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return coeffs


# Example usage functions

def generate_test_signal(length=1024, freq=5.0, sample_rate=100.0):
    """
    Generate a test sinusoidal signal.
    
    Args:
        length (int): Signal length
        freq (float): Frequency in Hz
        sample_rate (float): Sampling rate in Hz
        
    Returns:
        numpy.ndarray: Test signal
    """
    t = np.arange(length) / sample_rate
    return np.sin(2 * np.pi * freq * t)

def generate_chirp_signal(length=1024, f0=1.0, f1=20.0, sample_rate=100.0):
    """
    Generate a chirp signal (frequency changing with time).
    
    Args:
        length (int): Signal length
        f0 (float): Start frequency in Hz
        f1 (float): End frequency in Hz
        sample_rate (float): Sampling rate in Hz
        
    Returns:
        numpy.ndarray: Chirp signal
    """
    t = np.arange(length) / sample_rate
    duration = length / sample_rate
    k = (f1 - f0) / duration
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))

def denoise_signal(signal, wavelet_family=WaveletFamily.DAUBECHIES, 
                  vanishing_moments=4, levels=3, threshold_factor=1.5):
    """
    Simple wavelet-based signal denoising.
    
    Args:
        signal (numpy.ndarray): Input noisy signal
        wavelet_family (WaveletFamily): Wavelet family to use
        vanishing_moments (int): Number of vanishing moments
        levels (int): Number of decomposition levels
        threshold_factor (float): Threshold multiplier for noise estimation
        
    Returns:
        numpy.ndarray: Denoised signal
    """
    # Create DWT object
    dwt = DiscreteWaveletTransform(wavelet_family, vanishing_moments)
    
    # Perform forward transform
    coeffs = dwt.forward(signal, levels)
    
    # Threshold detail coefficients
    for i, detail in enumerate(coeffs['detail']):
        # Estimate noise level (MAD estimator)
        noise_level = np.median(np.abs(detail)) / 0.6745
        
        # Set threshold
        threshold = threshold_factor * noise_level
        
        # Apply soft thresholding
        detail_thresholded = np.sign(detail) * np.maximum(0, np.abs(detail) - threshold)
        
        # Update coefficients
        coeffs['detail'][i] = detail_thresholded
    
    # Perform inverse transform
    denoised = dwt.inverse(coeffs)
    
    return denoised

def show_wavelet_demo():
    """
    Demonstrate various wavelet transform capabilities.
    """
    # Generate signals
    t = np.linspace(0, 1, 1024)
    clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 15 * t)
    noise = np.random.normal(0, 0.1, len(t))
    noisy_signal = clean_signal + noise
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot original signals
    plt.subplot(3, 2, 1)
    plt.plot(t, clean_signal)
    plt.title('Clean Signal')
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    plt.plot(t, noisy_signal)
    plt.title('Noisy Signal')
    plt.grid(True)
    
    # Denoise using DWT
    denoised = denoise_signal(noisy_signal)
    plt.subplot(3, 2, 3)
    plt.plot(t, denoised)
    plt.title('Denoised Signal (DWT)')
    plt.grid(True)
    
    # Generate and analyze chirp signal with CWT
    chirp = generate_chirp_signal()
    t_chirp = np.linspace(0, 1, len(chirp))
    plt.subplot(3, 2, 4)
    plt.plot(t_chirp, chirp)
    plt.title('Chirp Signal')
    plt.grid(True)
    
    # Perform CWT
    cwt = ContinuousWaveletTransform()
    scales = cwt.generate_scales(64, 1, 64)
    coeffs = cwt.forward(chirp, scales)
    
    # Plot scalogram
    plt.subplot(3, 2, 5)
    plt.imshow(np.abs(coeffs)**2, aspect='auto', origin='lower', 
               extent=[0, 1, np.log10(scales[0]), np.log10(scales[-1])],
               cmap='jet')
    plt.colorbar()
    plt.ylabel('log10(Scale)')
    plt.title('CWT Scalogram')
    
    # Perform MODWT
    modwt = MaximalOverlapDWT()
    modwt_coeffs = modwt.forward(noisy_signal, 3)
    
    # Threshold and reconstruct
    for i in range(len(modwt_coeffs['wavelet'])):
        detail = modwt_coeffs['wavelet'][i]
        noise_level = np.median(np.abs(detail)) / 0.6745
        threshold = 2.0 * noise_level
        modwt_coeffs['wavelet'][i] = np.sign(detail) * np.maximum(0, np.abs(detail) - threshold)
    
    modwt_denoised = modwt.inverse(modwt_coeffs)
    
    plt.subplot(3, 2, 6)
    plt.plot(t, modwt_denoised)
    plt.title('Denoised Signal (MODWT)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_wavelet_demo()