# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Unit tests for the wavelet module.
"""

import unittest
import numpy as np
from signal_processing.wavelet import (
    WaveletFamily, BoundaryMode,
    DiscreteWaveletTransform, ContinuousWaveletTransform,
    WaveletPacketTransform, MaximalOverlapDWT,
    generate_test_signal, generate_chirp_signal
)


class TestWaveletTransforms(unittest.TestCase):
    """Test wavelet transform implementations."""
    
    def setUp(self):
        """Set up test signals."""
        # Create test signals
        self.signal_length = 512
        self.sine_signal = generate_test_signal(self.signal_length, freq=5.0)
        self.chirp_signal = generate_chirp_signal(self.signal_length, f0=5.0, f1=50.0)
        
        # Create step signal
        self.step_signal = np.zeros(self.signal_length)
        self.step_signal[self.signal_length // 2:] = 1.0

    def test_dwt_basic(self):
        """Test basic DWT functionality."""
        # Create DWT object
        dwt = DiscreteWaveletTransform(WaveletFamily.DAUBECHIES, 4)
        
        # Test with sine signal
        levels = 3
        result = dwt.forward(self.sine_signal, levels)
        
        # Check result structure
        self.assertIn('approximation', result)
        self.assertIn('detail', result)
        self.assertEqual(len(result['approximation']), levels + 1)
        self.assertEqual(len(result['detail']), levels)
        
        # Check approximation coefficient sizes (halved at each level)
        for i in range(levels + 1):
            expected_size = self.signal_length // (2**i)
            self.assertEqual(len(result['approximation'][i]), expected_size)

    def test_dwt_reconstruction(self):
        """Test DWT reconstruction accuracy."""
        # Create DWT object with different wavelets
        wavelets = [
            (WaveletFamily.HAAR, 1),
            (WaveletFamily.DAUBECHIES, 4),
            (WaveletFamily.SYMLET, 4)
        ]
        
        for family, vanishing_moments in wavelets:
            dwt = DiscreteWaveletTransform(family, vanishing_moments)
            
            # Forward transform
            result = dwt.forward(self.sine_signal, 3)
            
            # Inverse transform
            reconstructed = dwt.inverse(result)
            
            # Check reconstruction accuracy
            mse = np.mean((self.sine_signal - reconstructed)**2)
            self.assertLess(mse, 1e-10, f"Reconstruction failed for {family}")
    
    def test_dwt_boundary_modes(self):
        """Test different boundary handling modes."""
        dwt = DiscreteWaveletTransform(WaveletFamily.DAUBECHIES, 4)
        
        # Test all boundary modes
        for mode in [BoundaryMode.ZERO_PADDING, BoundaryMode.SYMMETRIC, 
                    BoundaryMode.PERIODIC, BoundaryMode.REFLECT]:
            # Forward transform
            result = dwt.forward(self.sine_signal, 3, mode)
            
            # Inverse transform
            reconstructed = dwt.inverse(result, mode)
            
            # Check reconstruction accuracy
            mse = np.mean((self.sine_signal - reconstructed)**2)
            self.assertLess(mse, 1e-10, f"Reconstruction failed for mode {mode}")

    def test_cwt_basic(self):
        """Test basic CWT functionality."""
        # Create CWT object
        cwt = ContinuousWaveletTransform(WaveletFamily.MORLET)
        
        # Generate scales
        num_scales = 32
        scales = cwt.generate_scales(num_scales, 1.0, 32.0)
        
        # Test with chirp signal
        coeffs = cwt.forward(self.chirp_signal, scales)
        
        # Check result dimensions
        self.assertEqual(coeffs.shape, (num_scales, self.signal_length))
        
        # Test Mexican Hat wavelet
        cwt_mh = ContinuousWaveletTransform(WaveletFamily.MEXICAN_HAT)
        coeffs_mh = cwt_mh.forward(self.chirp_signal, scales)
        
        # Check result dimensions
        self.assertEqual(coeffs_mh.shape, (num_scales, self.signal_length))
    
    def test_wpt_basic(self):
        """Test basic WPT functionality."""
        # Create WPT object
        wpt = WaveletPacketTransform(WaveletFamily.DAUBECHIES, 4)
        
        # Test with sine signal
        levels = 3
        result = wpt.forward(self.sine_signal, levels)
        
        # Check result structure
        self.assertIn('coefficients', result)
        self.assertIn('levels', result)
        self.assertEqual(result['levels'], levels)
        
        # Check coefficient structure
        coeffs = result['coefficients']
        
        # Level 0 should have 1 node
        self.assertIn((0, 0), coeffs)
        
        # Level 1 should have 2 nodes
        self.assertIn((1, 0), coeffs)
        self.assertIn((1, 1), coeffs)
        
        # Level 2 should have 4 nodes
        for node in range(4):
            self.assertIn((2, node), coeffs)
        
        # Level 3 should have 8 nodes
        for node in range(8):
            self.assertIn((3, node), coeffs)
    
    def test_wpt_reconstruction(self):
        """Test WPT reconstruction accuracy."""
        # Create WPT object
        wpt = WaveletPacketTransform(WaveletFamily.DAUBECHIES, 4)
        
        # Forward transform
        result = wpt.forward(self.sine_signal, 3)
        
        # Inverse transform
        reconstructed = wpt.inverse(result)
        
        # Check reconstruction accuracy
        mse = np.mean((self.sine_signal - reconstructed)**2)
        self.assertLess(mse, 1e-10)
    
    def test_modwt_basic(self):
        """Test basic MODWT functionality."""
        # Create MODWT object
        modwt = MaximalOverlapDWT(WaveletFamily.DAUBECHIES, 4)
        
        # Test with step signal
        levels = 3
        result = modwt.forward(self.step_signal, levels)
        
        # Check result structure
        self.assertIn('wavelet', result)
        self.assertIn('scaling', result)
        self.assertEqual(len(result['wavelet']), levels)
        self.assertEqual(len(result['scaling']), levels + 1)
        
        # MODWT doesn't downsample, so all coefficients should have the same length
        for coeffs in result['wavelet']:
            self.assertEqual(len(coeffs), self.signal_length)
        
        for coeffs in result['scaling']:
            self.assertEqual(len(coeffs), self.signal_length)
    
    def test_modwt_reconstruction(self):
        """Test MODWT reconstruction accuracy."""
        # Create MODWT object
        modwt = MaximalOverlapDWT(WaveletFamily.DAUBECHIES, 4)
        
        # Forward transform
        result = modwt.forward(self.step_signal, 3)
        
        # Inverse transform
        reconstructed = modwt.inverse(result)
        
        # Check reconstruction accuracy
        mse = np.mean((self.step_signal - reconstructed)**2)
        self.assertLess(mse, 1e-10)


if __name__ == '__main__':
    unittest.main()