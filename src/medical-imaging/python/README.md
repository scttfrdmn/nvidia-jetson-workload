# Medical Imaging Workload

This package provides Python bindings for a GPU-accelerated medical imaging workload designed for Nvidia Jetson Orin NX and AWS Graviton g5g instances.

## Features

- GPU-accelerated medical image processing and analysis
- CT reconstruction (filtered backprojection, iterative methods)
- Image processing (filtering, denoising)
- Image segmentation (thresholding, watershed, level set, graph cut)
- Image registration
- Automatic GPU adaptability for different hardware architectures
- Hybrid CPU-GPU execution

## Requirements

- Python 3.7 or higher
- CUDA Toolkit 11.0 or higher
- CMake 3.18 or higher
- A CUDA-capable GPU (Nvidia Jetson Orin NX, T4, or other NVIDIA GPUs)
- NumPy
- Matplotlib (for visualization)
- scikit-image (for additional operations)

## Installation

```bash
# Clone the repository
git clone https://github.com/scttfrdmn/nvidia-jetson-workload.git
cd nvidia-jetson-workload/src/medical-imaging/python

# Install dependencies
pip install numpy matplotlib scikit-image

# Optional: Specify CUDA architecture (default: auto-detect)
export CUDA_ARCH="87"  # For Jetson Orin NX

# Install package
pip install .
```

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from medical_imaging import (
    load_image, save_image, 
    apply_filter, apply_segmentation, 
    reconstruct_ct, register_images,
    FilterMethod, SegmentationMethod, ReconstructionMethod
)

# Load an image
image = load_image("sample.png")

# Apply Gaussian filter
filtered = apply_filter(image, FilterMethod.Gaussian, sigma=1.5)

# Apply segmentation
segmented = apply_segmentation(image, SegmentationMethod.Thresholding, threshold=0.5)

# Display results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(image, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(filtered, cmap='gray')
axs[1].set_title('Filtered')
axs[2].imshow(segmented, cmap='gray')
axs[2].set_title('Segmented')
plt.show()

# Save result
save_image(segmented, "segmented.png")
```

## Advanced Usage

### CT Reconstruction

```python
# Load projection data
projections = load_image("projections.npy")

# Define projection angles (in radians)
angles = np.linspace(0, np.pi, projections.shape[0])

# Reconstruct using filtered backprojection
fbp_result = reconstruct_ct(projections, angles, ReconstructionMethod.FilteredBackProjection)

# Reconstruct using iterative method
iterative_result = reconstruct_ct(
    projections, angles, 
    ReconstructionMethod.IterativePrimalDual,
    num_iterations=10
)
```

### Image Registration

```python
# Load fixed and moving images
fixed = load_image("fixed.png")
moving = load_image("moving.png")

# Register moving to fixed
registered = register_images(fixed, moving, method="rigid")

# Display results
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(fixed, cmap='gray')
axs[0].set_title('Fixed')
axs[1].imshow(moving, cmap='gray')
axs[1].set_title('Moving')
axs[2].imshow(registered, cmap='gray')
axs[2].set_title('Registered')
plt.show()
```

### GPU Adaptability

```python
from medical_imaging import AdaptiveKernelManager, DeviceType

# Get device information
akm = AdaptiveKernelManager.get_instance()
device_caps = akm.get_device_capabilities()
print(device_caps.get_summary())

# Check device type
if device_caps.device_type == DeviceType.JetsonOrinNX:
    print("Running on Jetson Orin NX")
elif device_caps.device_type == DeviceType.T4:
    print("Running on T4 GPU")
```

## License

Copyright (c) 2025 Scott Friedman. All rights reserved.