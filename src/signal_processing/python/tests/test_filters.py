import numpy as np
import pytest
from signal_processing import (
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
from signal_processing.filters import (
    FilterType,
    WindowType,
    FIRDesignMethod,
    IIRDesignMethod,
    AdaptiveFilterType
)


def generate_test_signal(duration=1.0, sample_rate=44100):
    """Generate a test signal containing sine waves at 440 Hz, 880 Hz, and 1760 Hz"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t) + 0.25 * np.sin(2 * np.pi * 1760 * t)
    
    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.1, size=len(signal))
    signal += noise
    
    return signal.astype(np.float32), sample_rate


def get_frequency_content(signal, sample_rate):
    """Compute the frequency content of a signal"""
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), 1/sample_rate)
    magnitude = np.abs(spectrum)
    return frequencies, magnitude


class TestFIRFilter:
    def test_lowpass_filter(self):
        signal, sample_rate = generate_test_signal()
        
        # Create a lowpass filter at 500 Hz
        lowpass = FIRFilter(
            coefficients=None,
            sample_rate=sample_rate,
            filter_type=FilterType.LOWPASS,
            cutoff_freqs=500,
            window_type=WindowType.HAMMING,
            num_taps=101,
            device_id=-1  # Use CPU for testing
        )
        
        # Apply filter
        filtered_signal = lowpass.filter(signal)
        
        # Check that filter preserved the signal length
        assert len(filtered_signal) == len(signal)
        
        # Check frequency content
        freq_orig, mag_orig = get_frequency_content(signal, sample_rate)
        freq_filt, mag_filt = get_frequency_content(filtered_signal, sample_rate)
        
        # Find indices for key frequencies
        idx_440 = np.argmin(np.abs(freq_filt - 440))
        idx_880 = np.argmin(np.abs(freq_filt - 880))
        idx_1760 = np.argmin(np.abs(freq_filt - 1760))
        
        # Check that 440 Hz is preserved
        assert mag_filt[idx_440] > 0.5 * mag_orig[idx_440]
        
        # Check that 880 Hz is attenuated
        assert mag_filt[idx_880] < 0.5 * mag_orig[idx_880]
        
        # Check that 1760 Hz is strongly attenuated
        assert mag_filt[idx_1760] < 0.1 * mag_orig[idx_1760]
    
    def test_bandpass_filter(self):
        signal, sample_rate = generate_test_signal()
        
        # Create a bandpass filter from 700 Hz to 1000 Hz
        bandpass = FIRFilter(
            coefficients=None,
            sample_rate=sample_rate,
            filter_type=FilterType.BANDPASS,
            cutoff_freqs=[700, 1000],
            window_type=WindowType.HAMMING,
            num_taps=101,
            device_id=-1  # Use CPU for testing
        )
        
        # Apply filter
        filtered_signal = bandpass.filter(signal)
        
        # Check frequency content
        freq_orig, mag_orig = get_frequency_content(signal, sample_rate)
        freq_filt, mag_filt = get_frequency_content(filtered_signal, sample_rate)
        
        # Find indices for key frequencies
        idx_440 = np.argmin(np.abs(freq_filt - 440))
        idx_880 = np.argmin(np.abs(freq_filt - 880))
        idx_1760 = np.argmin(np.abs(freq_filt - 1760))
        
        # Check that 440 Hz is attenuated
        assert mag_filt[idx_440] < 0.2 * mag_orig[idx_440]
        
        # Check that 880 Hz is relatively preserved
        assert mag_filt[idx_880] > 0.3 * mag_orig[idx_880]
        
        # Check that 1760 Hz is attenuated
        assert mag_filt[idx_1760] < 0.2 * mag_orig[idx_1760]


class TestIIRFilter:
    def test_butterworth_lowpass(self):
        signal, sample_rate = generate_test_signal()
        
        # Create a Butterworth lowpass filter at 500 Hz
        lowpass = IIRFilter(
            a=None,
            b=None,
            sample_rate=sample_rate,
            filter_type=FilterType.LOWPASS,
            cutoff_freqs=500,
            order=4,
            design_method=IIRDesignMethod.BUTTERWORTH,
            device_id=-1  # Use CPU for testing
        )
        
        # Apply filter
        filtered_signal = lowpass.filter(signal)
        
        # Check that filter preserved the signal length
        assert len(filtered_signal) == len(signal)
        
        # Check frequency content
        freq_orig, mag_orig = get_frequency_content(signal, sample_rate)
        freq_filt, mag_filt = get_frequency_content(filtered_signal, sample_rate)
        
        # Find indices for key frequencies
        idx_440 = np.argmin(np.abs(freq_filt - 440))
        idx_880 = np.argmin(np.abs(freq_filt - 880))
        idx_1760 = np.argmin(np.abs(freq_filt - 1760))
        
        # Check that 440 Hz is preserved
        assert mag_filt[idx_440] > 0.5 * mag_orig[idx_440]
        
        # Check that 880 Hz is attenuated
        assert mag_filt[idx_880] < 0.5 * mag_orig[idx_880]
        
        # Check that 1760 Hz is strongly attenuated
        assert mag_filt[idx_1760] < 0.1 * mag_orig[idx_1760]
    
    def test_filter_stability(self):
        # Create a Butterworth filter
        iir = IIRFilter(
            a=None,
            b=None,
            sample_rate=44100,
            filter_type=FilterType.LOWPASS,
            cutoff_freqs=1000,
            order=4,
            design_method=IIRDesignMethod.BUTTERWORTH,
            device_id=-1  # Use CPU for testing
        )
        
        # Check stability
        assert iir.is_stable()
        
        # Get filter coefficients
        a, b = iir.get_coefficients()
        
        # Verify coefficient shapes
        assert len(a) > 0
        assert len(b) > 0
        assert a[0] == 1.0  # First denominator coefficient should be 1


class TestAdaptiveFilter:
    def test_lms_filter(self):
        # Generate primary and reference signals
        np.random.seed(42)
        n = 10000
        noise = np.random.normal(0, 1, size=n).astype(np.float32)
        desired = np.sin(2 * np.pi * 0.05 * np.arange(n)).astype(np.float32)
        
        # Desired signal with noise
        primary = desired + 0.5 * noise
        
        # Create adaptive filter
        adaptive = AdaptiveFilter(
            filter_length=32,
            filter_type=AdaptiveFilterType.LMS,
            step_size=0.02,
            device_id=-1  # Use CPU for testing
        )
        
        # Apply filter
        filtered, error = adaptive.filter(noise, primary)
        
        # Check shapes
        assert len(filtered) == n
        assert len(error) == n
        
        # Get filter coefficients
        coeffs = adaptive.get_coefficients()
        
        # Verify coefficient shape
        assert len(coeffs) == 32
        
        # Check that error is reduced over time
        assert np.mean(np.abs(error[:n//4])) > np.mean(np.abs(error[3*n//4:]))


class TestMultirateFilter:
    def test_resample(self):
        signal, sample_rate = generate_test_signal()
        
        # Create a multirate filter with 2x upsampling and 3x downsampling (2/3 resampling)
        multirate = MultirateFilter(
            interpolation_factor=2,
            decimation_factor=3,
            device_id=-1  # Use CPU for testing
        )
        
        # Apply resampling
        resampled = multirate.resample(signal)
        
        # Check that the resampled signal has the expected length
        expected_length = int(len(signal) * 2 / 3)
        assert abs(len(resampled) - expected_length) <= 1  # Allow for minor rounding differences


class TestConvenienceFunctions:
    def test_design_filters(self):
        # Test convenience functions for filter design
        sample_rate = 44100
        
        # Design lowpass filter
        lowpass = design_lowpass(
            cutoff_freq=1000,
            sample_rate=sample_rate,
            num_taps=51,
            device_id=-1  # Use CPU for testing
        )
        
        # Design highpass filter
        highpass = design_highpass(
            cutoff_freq=5000,
            sample_rate=sample_rate,
            num_taps=51,
            device_id=-1  # Use CPU for testing
        )
        
        # Design bandpass filter
        bandpass = design_bandpass(
            cutoff_freqs=(1000, 5000),
            sample_rate=sample_rate,
            num_taps=51,
            device_id=-1  # Use CPU for testing
        )
        
        # Design bandstop filter
        bandstop = design_bandstop(
            cutoff_freqs=(1000, 5000),
            sample_rate=sample_rate,
            num_taps=51,
            device_id=-1  # Use CPU for testing
        )
        
        # Verify filter types
        assert isinstance(lowpass, FIRFilter)
        assert isinstance(highpass, FIRFilter)
        assert isinstance(bandpass, FIRFilter)
        assert isinstance(bandstop, FIRFilter)
    
    def test_median_filter(self):
        signal, _ = generate_test_signal()
        
        # Add some outliers
        signal_with_outliers = signal.copy()
        signal_with_outliers[1000] = 10
        signal_with_outliers[2000] = -10
        
        # Apply median filter
        filtered = median_filter(
            signal_with_outliers,
            kernel_size=5,
            device_id=-1  # Use CPU for testing
        )
        
        # Check that outliers are removed
        assert abs(filtered[1000]) < 2
        assert abs(filtered[2000]) < 2
    
    def test_kalman_filter(self):
        # Generate noisy signal
        np.random.seed(42)
        n = 1000
        true_signal = np.sin(2 * np.pi * 0.01 * np.arange(n)).astype(np.float32)
        noisy_signal = true_signal + np.random.normal(0, 0.5, size=n).astype(np.float32)
        
        # Apply Kalman filter
        filtered = kalman_filter(
            noisy_signal,
            process_variance=0.001,
            measurement_variance=0.1,
            device_id=-1  # Use CPU for testing
        )
        
        # Check shape
        assert len(filtered) == n
        
        # Check that filtered signal is closer to true signal than noisy signal
        error_noisy = np.mean((noisy_signal - true_signal) ** 2)
        error_filtered = np.mean((filtered - true_signal) ** 2)
        
        assert error_filtered < error_noisy