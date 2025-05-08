import numpy as np
import pytest
from signal_processing import (
    STFT,
    CWT,
    DWT,
    WignerVille,
    EMD,
    compute_spectrogram,
    compute_scalogram,
    compute_mel_spectrogram,
    compute_mfcc
)
from signal_processing.tf_analysis import (
    WindowType,
    WaveletType
)


def generate_chirp_signal(duration=1.0, sample_rate=44100):
    """Generate a chirp signal that sweeps from 100 Hz to 5000 Hz"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    f0 = 100  # Starting frequency (Hz)
    f1 = 5000  # Ending frequency (Hz)
    # Linear chirp formula
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t**2 / duration)
    signal = np.sin(phase)
    
    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, size=len(signal))
    signal += noise
    
    return signal.astype(np.float32), sample_rate


class TestSTFT:
    def test_stft_transform(self):
        signal, sample_rate = generate_chirp_signal()
        
        # Create STFT processor
        stft = STFT(
            window_size=1024,
            hop_size=256,
            window_type=WindowType.HANN,
            device_id=-1  # Use CPU for testing
        )
        
        # Compute STFT
        spectrogram, times, frequencies = stft.transform(signal, sample_rate)
        
        # Check shapes
        assert spectrogram.shape[0] == len(times)
        assert spectrogram.shape[1] == len(frequencies)
        
        # Check frequency range
        assert frequencies[0] == 0
        assert frequencies[-1] <= sample_rate / 2
        
        # Get magnitude spectrogram
        magnitude = stft.get_magnitude(spectrogram)
        
        # Check shape
        assert magnitude.shape == spectrogram.shape
        
        # Check that early times have energy at low frequencies
        early_time_idx = len(times) // 10
        early_spec = magnitude[early_time_idx, :]
        early_peak_idx = np.argmax(early_spec)
        assert frequencies[early_peak_idx] < 1000
        
        # Check that late times have energy at high frequencies
        late_time_idx = 9 * len(times) // 10
        late_spec = magnitude[late_time_idx, :]
        late_peak_idx = np.argmax(late_spec)
        assert frequencies[late_peak_idx] > 2000
    
    def test_stft_inverse_transform(self):
        signal, sample_rate = generate_chirp_signal()
        
        # Create STFT processor
        stft = STFT(
            window_size=1024,
            hop_size=256,
            window_type=WindowType.HANN,
            device_id=-1  # Use CPU for testing
        )
        
        # Compute STFT
        spectrogram, times, frequencies = stft.transform(signal, sample_rate)
        
        # Compute inverse STFT
        reconstructed = stft.inverse_transform(spectrogram, times, frequencies, sample_rate)
        
        # Check that the reconstructed signal has similar length
        length_diff = abs(len(reconstructed) - len(signal))
        assert length_diff <= stft.window_size
        
        # Check correlation between original and reconstructed
        min_len = min(len(signal), len(reconstructed))
        correlation = np.corrcoef(signal[:min_len], reconstructed[:min_len])[0, 1]
        assert correlation > 0.9  # High correlation expected


class TestCWT:
    def test_cwt_transform(self):
        signal, sample_rate = generate_chirp_signal()
        
        # Create CWT processor
        cwt = CWT(
            wavelet_type=WaveletType.MORLET,
            num_scales=32,
            device_id=-1  # Use CPU for testing
        )
        
        # Compute CWT
        scalogram, times, scales, frequencies = cwt.transform(signal, sample_rate)
        
        # Check shapes
        assert scalogram.shape[0] == len(scales)
        assert scalogram.shape[1] == len(times)
        assert len(frequencies) == len(scales)
        
        # Get magnitude scalogram
        magnitude = cwt.get_magnitude(scalogram)
        
        # Check shape
        assert magnitude.shape == scalogram.shape
        
        # Check frequency range
        assert min(frequencies) > 0
        assert max(frequencies) < sample_rate / 2
        
        # Check that chirp pattern is captured in the scalogram
        # Get frequency with maximum energy at different times
        early_time_idx = len(times) // 10
        late_time_idx = 9 * len(times) // 10
        
        early_peak_scale = np.argmax(magnitude[:, early_time_idx])
        late_peak_scale = np.argmax(magnitude[:, late_time_idx])
        
        early_freq = frequencies[early_peak_scale]
        late_freq = frequencies[late_peak_scale]
        
        # Verify chirp behavior (frequency increases over time)
        assert early_freq < late_freq


class TestDWT:
    def test_dwt_transform(self):
        signal, _ = generate_chirp_signal()
        
        # Create DWT processor
        dwt = DWT(
            wavelet_type=WaveletType.DB4,
            levels=4,
            device_id=-1  # Use CPU for testing
        )
        
        # Compute DWT
        detail_coeffs, approx_coeffs = dwt.transform(signal)
        
        # Check number of detail coefficient sets
        assert len(detail_coeffs) == 4  # Matches levels=4
        
        # Check that approximation coefficients exist
        assert len(approx_coeffs) > 0
        
        # Reconstruct signal
        reconstructed = dwt.inverse_transform(detail_coeffs, approx_coeffs)
        
        # Check that reconstructed signal has same length
        assert len(reconstructed) == len(signal)
        
        # Check correlation between original and reconstructed
        correlation = np.corrcoef(signal, reconstructed)[0, 1]
        assert correlation > 0.95  # High correlation expected for lossless transform
    
    def test_dwt_denoise(self):
        # Generate a clean and noisy signal
        t = np.linspace(0, 1, 44100, endpoint=False)
        clean_signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        np.random.seed(42)
        noise = np.random.normal(0, 0.2, size=len(t)).astype(np.float32)
        noisy_signal = clean_signal + noise
        
        # Create DWT processor
        dwt = DWT(
            wavelet_type=WaveletType.DB4,
            levels=4,
            device_id=-1  # Use CPU for testing
        )
        
        # Denoise signal
        denoised = dwt.denoise(noisy_signal, threshold=0.2, threshold_mode="soft")
        
        # Check length
        assert len(denoised) == len(noisy_signal)
        
        # Compare MSE with clean signal
        mse_noisy = np.mean((noisy_signal - clean_signal) ** 2)
        mse_denoised = np.mean((denoised - clean_signal) ** 2)
        
        # Verify denoising improved signal
        assert mse_denoised < mse_noisy


class TestWignerVille:
    def test_wigner_ville_transform(self):
        signal, sample_rate = generate_chirp_signal(duration=0.5)  # Shorter signal for faster test
        
        # Create Wigner-Ville processor
        wv = WignerVille(device_id=-1)  # Use CPU for testing
        
        # Compute Wigner-Ville distribution
        wvd, times, frequencies = wv.transform(signal, sample_rate)
        
        # Check shapes
        assert wvd.shape[0] == len(times)
        assert wvd.shape[1] == len(frequencies)
        
        # Check frequency range
        assert frequencies[0] == 0
        assert frequencies[-1] <= sample_rate / 2
        
        # Compute pseudo Wigner-Ville distribution
        pwvd, times_p, frequencies_p = wv.transform_pseudo(signal, sample_rate, window_size=127)
        
        # Check shapes
        assert pwvd.shape[0] == len(times_p)
        assert pwvd.shape[1] == len(frequencies_p)


class TestEMD:
    def test_emd_decompose(self):
        signal, sample_rate = generate_chirp_signal(duration=0.5)  # Shorter signal for faster test
        
        # Create EMD processor
        emd = EMD(max_imfs=4, device_id=-1)  # Use CPU for testing
        
        # Decompose signal
        imfs, residue = emd.decompose(signal, sample_rate)
        
        # Check IMF count
        assert len(imfs) <= 4  # Should not exceed max_imfs
        assert len(imfs) > 0  # Should have at least one IMF
        
        # Check that all IMFs have same length as input
        for imf in imfs:
            assert len(imf) == len(signal)
        
        # Check residue length
        assert len(residue) == len(signal)
        
        # Reconstruct signal
        reconstructed = emd.reconstruct(imfs, residue)
        
        # Check reconstructed length
        assert len(reconstructed) == len(signal)
        
        # Verify reconstruction accuracy
        assert np.allclose(reconstructed, signal, rtol=1e-5, atol=1e-5)


class TestConvenienceFunctions:
    def test_compute_spectrogram(self):
        signal, sample_rate = generate_chirp_signal()
        
        # Compute spectrogram
        spectrogram, times, frequencies = compute_spectrogram(
            signal, 
            sample_rate,
            window_size=1024,
            hop_size=256,
            window_type=WindowType.HANN,
            log_scale=True,
            device_id=-1  # Use CPU for testing
        )
        
        # Check shapes
        assert spectrogram.shape[0] == len(times)
        assert spectrogram.shape[1] == len(frequencies)
        
        # Check frequency range
        assert frequencies[0] == 0
        assert frequencies[-1] <= sample_rate / 2
    
    def test_compute_scalogram(self):
        signal, sample_rate = generate_chirp_signal()
        
        # Compute scalogram
        scalogram, times, frequencies = compute_scalogram(
            signal, 
            sample_rate,
            wavelet_type=WaveletType.MORLET,
            num_scales=32,
            log_scale=True,
            device_id=-1  # Use CPU for testing
        )
        
        # Check shapes
        assert scalogram.shape[0] == len(times)
        assert scalogram.shape[1] == len(frequencies)
        
        # Check frequency range
        assert min(frequencies) > 0
        assert max(frequencies) < sample_rate / 2
    
    def test_compute_mel_spectrogram(self):
        signal, sample_rate = generate_chirp_signal()
        
        try:
            # Compute mel spectrogram
            mel_spec, times, mel_freqs = compute_mel_spectrogram(
                signal, 
                sample_rate,
                n_fft=2048,
                hop_size=512,
                n_mels=128,
                log_scale=True,
                device_id=-1  # Use CPU for testing
            )
            
            # Check shapes
            assert mel_spec.shape[0] == len(times)
            assert mel_spec.shape[1] == len(mel_freqs)
            
            # Check frequency range
            assert mel_freqs[0] > 0
            assert mel_freqs[-1] < sample_rate / 2
        except ImportError:
            pytest.skip("librosa not installed for mel_spectrogram test")
    
    def test_compute_mfcc(self):
        signal, sample_rate = generate_chirp_signal()
        
        try:
            # Compute MFCCs
            mfccs, times = compute_mfcc(
                signal, 
                sample_rate,
                n_mfcc=13,
                n_fft=2048,
                hop_size=512,
                device_id=-1  # Use CPU for testing
            )
            
            # Check shapes
            assert mfccs.shape[0] == len(times)
            assert mfccs.shape[1] == 13  # n_mfcc=13
        except ImportError:
            pytest.skip("librosa not installed for mfcc test")