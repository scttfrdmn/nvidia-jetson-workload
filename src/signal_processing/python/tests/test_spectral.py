import numpy as np
import pytest
from signal_processing import (
    FFT,
    SpectralAnalyzer,
    compute_psd,
    compute_spectrogram,
    compute_coherence,
    detect_peaks
)
from signal_processing.spectral import WindowType


def generate_test_signal(duration=1.0, sample_rate=44100):
    """Generate a test signal containing sine waves at 440 Hz and 880 Hz"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    return signal.astype(np.float32), sample_rate


class TestFFT:
    def test_forward_1d_real(self):
        signal, sample_rate = generate_test_signal()
        fft = FFT(device_id=-1)  # Use CPU for testing
        
        # Compute FFT
        result = fft.forward_1d_real(signal)
        
        # Verify shape
        assert result.shape == (len(signal) // 2 + 1,)
        
        # Verify peaks at expected frequencies
        freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
        magnitude = np.abs(result)
        
        # Find peaks
        peak_indices = np.argsort(magnitude)[-2:]
        peak_freqs = freqs[peak_indices]
        
        # Check that peaks are close to 440 Hz and 880 Hz
        assert any(abs(freq - 440) < 5 for freq in peak_freqs)
        assert any(abs(freq - 880) < 5 for freq in peak_freqs)
    
    def test_forward_inverse_round_trip(self):
        signal, _ = generate_test_signal()
        fft = FFT(device_id=-1)  # Use CPU for testing
        
        # Forward transform
        spectrum = fft.forward_1d_real(signal)
        
        # Inverse transform
        reconstructed = fft.inverse_1d_real(spectrum)
        
        # Check that the reconstructed signal is close to the original
        assert reconstructed.shape == signal.shape
        assert np.allclose(reconstructed, signal, rtol=1e-5, atol=1e-5)


class TestSpectralAnalyzer:
    def test_compute_psd(self):
        signal, sample_rate = generate_test_signal()
        analyzer = SpectralAnalyzer(device_id=-1)  # Use CPU for testing
        
        # Compute PSD
        frequencies, psd = analyzer.compute_psd(
            signal, 
            sample_rate=sample_rate,
            window_type=WindowType.HANN,
            nfft=1024,
            overlap=512
        )
        
        # Verify shape
        assert len(frequencies) == len(psd)
        assert len(frequencies) == 1024 // 2 + 1
        
        # Verify peaks at expected frequencies
        peak_indices = np.argsort(psd)[-2:]
        peak_freqs = frequencies[peak_indices]
        
        # Check that peaks are close to 440 Hz and 880 Hz
        assert any(abs(freq - 440) < 5 for freq in peak_freqs)
        assert any(abs(freq - 880) < 5 for freq in peak_freqs)
    
    def test_compute_spectrogram(self):
        signal, sample_rate = generate_test_signal()
        analyzer = SpectralAnalyzer(device_id=-1)  # Use CPU for testing
        
        # Compute spectrogram
        times, frequencies, spectrogram = analyzer.compute_spectrogram(
            signal, 
            sample_rate=sample_rate,
            window_type=WindowType.HANN,
            nfft=1024,
            overlap=768
        )
        
        # Verify shapes
        assert len(times) == spectrogram.shape[0]
        assert len(frequencies) == spectrogram.shape[1]
        
        # Verify that frequencies cover expected range
        assert frequencies[0] == 0
        assert frequencies[-1] <= sample_rate / 2
        
        # Verify that times cover expected range
        assert times[0] >= 0
        assert times[-1] <= 1.0  # Signal duration


class TestConvenienceFunctions:
    def test_compute_psd(self):
        signal, sample_rate = generate_test_signal()
        
        # Compute PSD using convenience function
        frequencies, psd = compute_psd(
            signal, 
            sample_rate=sample_rate,
            window_type=WindowType.HANN,
            nfft=1024,
            overlap=512,
            device_id=-1  # Use CPU for testing
        )
        
        # Verify shape
        assert len(frequencies) == len(psd)
        assert len(frequencies) == 1024 // 2 + 1
        
        # Verify peaks at expected frequencies
        peak_indices = np.argsort(psd)[-2:]
        peak_freqs = frequencies[peak_indices]
        
        # Check that peaks are close to 440 Hz and 880 Hz
        assert any(abs(freq - 440) < 5 for freq in peak_freqs)
        assert any(abs(freq - 880) < 5 for freq in peak_freqs)
    
    def test_detect_peaks(self):
        signal, sample_rate = generate_test_signal()
        
        # Compute PSD
        frequencies, psd = compute_psd(
            signal, 
            sample_rate=sample_rate,
            window_type=WindowType.HANN,
            nfft=1024,
            device_id=-1  # Use CPU for testing
        )
        
        # Detect peaks in the PSD
        peaks = detect_peaks(
            psd, 
            frequencies, 
            threshold=0.3,
            min_distance=10,
            device_id=-1  # Use CPU for testing
        )
        
        # Verify peaks
        assert len(peaks) >= 2
        peak_freqs = [freq for freq, _ in peaks]
        
        # Check that peaks are close to 440 Hz and 880 Hz
        assert any(abs(freq - 440) < 10 for freq in peak_freqs)
        assert any(abs(freq - 880) < 10 for freq in peak_freqs)