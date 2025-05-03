# Medical Imaging Examples

This directory contains example scripts demonstrating the capabilities of the Medical Imaging workload.

## Examples

### CT Reconstruction

```bash
python ct_reconstruction_example.py
```

This example demonstrates CT image reconstruction from projections using filtered backprojection and iterative methods.

Options:
- `-s, --size`: Phantom size (default: 512)
- `-a, --num-angles`: Number of projection angles (default: 180)
- `-i, --input`: Input image path (instead of phantom)
- `-o, --output`: Output image path
- `-d, --device-id`: CUDA device ID (default: 0)
- `--iterative`: Perform iterative reconstruction
- `--iterations`: Number of iterations for iterative method (default: 10)
- `--no-display`: Don't display results (save only)

### Image Processing

```bash
python image_processing_example.py
```

This example demonstrates various image processing operations including filtering, denoising, and segmentation.

Options:
- `-i, --input`: Input image path
- `-o, --output`: Output image path
- `-d, --device-id`: CUDA device ID (default: 0)
- `--no-display`: Don't display results (save only)
- `--noise-type`: Type of noise (gaussian, salt_pepper, poisson)
- `--noise-amount`: Noise amount/intensity (default: 0.05)
- `--gaussian-sigma`: Gaussian filter sigma (default: 1.0)
- `--bilateral-spatial`: Bilateral filter spatial sigma (default: 3.0)
- `--bilateral-range`: Bilateral filter range sigma (default: 0.1)
- `--nlm`: Apply non-local means filter
- `--nlm-h`: Non-local means h parameter (default: 0.1)
- `--nlm-search`: Non-local means search radius (default: 7)
- `--nlm-patch`: Non-local means patch radius (default: 3)
- `--segment`: Apply segmentation
- `--segment-input`: Input for segmentation (original, noisy, gaussian, bilateral, nlm)
- `--threshold`: Threshold for segmentation (default: 0.5)
- `--watershed`: Apply watershed segmentation

### Image Registration

```bash
python image_registration_example.py
```

This example demonstrates various image registration methods for aligning a moving image to a fixed reference image.

Options:
- `--fixed`: Fixed image path
- `--moving`: Moving image path
- `-o, --output`: Output image path
- `-d, --device-id`: CUDA device ID (default: 0)
- `--no-display`: Don't display results (save only)
- `-s, --size`: Test image size (default: 512)
- `-r, --rotation`: Test image rotation in degrees (default: 10.0)
- `--scale`: Test image scale factor (default: 0.9)
- `-t, --translation`: Test image translation (x y) (default: 20 -15)
- `-n, --noise`: Test image noise level (default: 0.05)
- `-m, --method`: Registration method (rigid, affine, deformable) (default: rigid)
- `-i, --iterations`: Maximum number of iterations (default: 100)
- `-e, --epsilon`: Convergence criterion (default: 1e-5)
- `-c, --checker-size`: Checkerboard visualization tile size (default: 32)

### GPU Benchmark

```bash
python gpu_benchmark.py --all -o benchmark_results
```

This example benchmarks the performance of medical imaging workloads across different devices.

Options:
- `--ct`: Run CT reconstruction benchmark
- `--filtering`: Run image filtering benchmark
- `--nlm`: Run NLM filtering benchmark
- `--segmentation`: Run segmentation benchmark
- `--all`: Run all benchmarks
- `-d, --device-id`: CUDA device ID (default: 0)
- `-o, --output`: Output directory for results
- `-p, --plot`: Plot combined results
- `-s, --sizes`: Image sizes to benchmark (default: 256 512 1024 2048)
- `-r, --num-runs`: Number of runs per benchmark (default: 3)
- `-a, --num-angles`: Number of angles for CT (default: 180)
- `-k, --kernel-sizes`: Kernel sizes for filtering (default: 3 5 7 9)
- `--patch-sizes`: Patch sizes for NLM (default: 3 5 7)
- `--search-sizes`: Search window sizes for NLM (default: 7 11 15)

## Comparing Performance Across Devices

To compare performance across different devices (Jetson Orin NX, T4 GPU, etc.), run the benchmark on each device:

1. Run on first device:
   ```bash
   python gpu_benchmark.py --all -o benchmark_results
   ```

2. Run on second device:
   ```bash
   python gpu_benchmark.py --all -o benchmark_results
   ```

3. Generate comparison plots:
   ```bash
   python gpu_benchmark.py --plot -o benchmark_results
   ```

This will generate plots comparing performance across all devices for each benchmark.

## Copyright

Copyright (c) 2025 Scott Friedman. All rights reserved.