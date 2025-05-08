# Signal Processing

GPU-accelerated signal processing library optimized for NVIDIA Jetson Orin NX and AWS Graviton g5g instances.

## Features

- **FFT and Spectral Analysis**: Optimized Fast Fourier Transform operations, power spectral density estimation, spectrograms, and more.
- **Digital Filtering**: FIR, IIR, adaptive, and multirate filters, with both CPU and GPU implementations.
- **Time-Frequency Analysis**: STFT, continuous and discrete wavelet transforms, Wigner-Ville distribution, and empirical mode decomposition.
- **Cross-Platform**: Works on both x86 and ARM architectures, with CUDA acceleration for NVIDIA GPUs.
- **Python Interface**: Pythonic API with NumPy/SciPy fallbacks when CUDA is not available.

## Installation

### Prerequisites

- NVIDIA CUDA Toolkit (optional, for GPU acceleration)
- C++17 compatible compiler (GCC 7+, Clang 5+)
- CMake 3.14+
- Python 3.7+
- NumPy, SciPy

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-organization/nvidia-jetson-workload.git
cd nvidia-jetson-workload

# Set environment variables (optional)
export USE_CUDA=1  # Set to 0 to build without CUDA
export CUDA_HOME=/usr/local/cuda  # Path to CUDA installation

# Install the package
cd src/signal_processing/python
pip install -e .
```

## Usage

```python
import numpy as np
from signal_processing import FFT, compute_spectrogram, FIRFilter

# Generate a test signal
sample_rate = 44100  # Hz
duration = 1.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

# Compute FFT
fft = FFT(device_id=0)  # Use GPU 0, or -1 for CPU
spectrum = fft.forward_1d_real(signal)

# Compute spectrogram
spectrogram, times, frequencies = compute_spectrogram(
    signal, sample_rate, window_size=1024, hop_size=256
)

# Apply a lowpass filter
lowpass = FIRFilter(
    coefficients=None,
    sample_rate=sample_rate,
    filter_type=FilterType.LOWPASS,
    cutoff_freqs=1000,  # Hz
    num_taps=101
)
filtered_signal = lowpass.filter(signal)
```

## Modules

- `signal_processing.spectral`: FFT and spectral analysis operations
- `signal_processing.filters`: Digital filtering operations
- `signal_processing.tf_analysis`: Time-frequency analysis operations

## Performance

This library is optimized for the following hardware:

- **NVIDIA Jetson Orin NX**: Ampere architecture GPU with SM 8.7
- **AWS Graviton g5g**: T4G GPU with SM 7.5

Performance speedups over CPU implementations range from 5x to 50x depending on the operation and data size.

## License

This project is licensed under the MIT License - see the LICENSE file for details.