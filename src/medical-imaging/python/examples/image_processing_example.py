#!/usr/bin/env python3
"""
Image Processing Example

This example demonstrates various image processing operations including
filtering, denoising, and segmentation. It includes performance benchmarking
comparing different hardware architectures.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# Import the medical_imaging package
import medical_imaging as mi
from medical_imaging.visualization import (
    plot_images, plot_histogram, plot_comparison, 
    plot_overlay, plot_difference
)

def add_noise(image, noise_type='gaussian', amount=0.05):
    """Add noise to an image.
    
    Args:
        image: Input image
        noise_type: Type of noise ('gaussian', 'salt_pepper', 'poisson')
        amount: Noise amount/standard deviation
        
    Returns:
        Noisy image
    """
    noisy = image.copy()
    
    if noise_type == 'gaussian':
        # Gaussian noise
        noise = np.random.normal(0, amount, image.shape)
        noisy = image + noise
        noisy = np.clip(noisy, 0, 1)
    
    elif noise_type == 'salt_pepper':
        # Salt and pepper noise
        salt = np.random.random(image.shape) < amount/2
        pepper = np.random.random(image.shape) < amount/2
        noisy[salt] = 1.0
        noisy[pepper] = 0.0
    
    elif noise_type == 'poisson':
        # Poisson noise (requires positive image values)
        noisy = np.random.poisson(image * 1.0/amount) * amount
        noisy = np.clip(noisy, 0, 1)
    
    return noisy

def main(args):
    # Print device information
    akm = mi.AdaptiveKernelManager.get_instance()
    if akm.initialize():
        caps = akm.get_device_capabilities()
        print("Device capabilities:")
        print(caps.get_summary())
    
    # Load or create image
    if args.input:
        print(f"Loading input image: {args.input}")
        original = mi.load_image(args.input)
        
        # Ensure image is normalized
        original = (original - original.min()) / (original.max() - original.min())
    else:
        # Use a sample image
        print("Using sample image...")
        original = data.coins()
        
        # Normalize
        original = original.astype(np.float32) / 255.0
    
    # Add noise
    print(f"Adding {args.noise_type} noise with amount {args.noise_amount}...")
    noisy = add_noise(original, args.noise_type, args.noise_amount)
    
    # Configure processor
    print("Configuring image processor...")
    config = mi.ProcessingConfig()
    config.compute_backend = mi.ComputeBackend.CUDA
    config.device_id = args.device_id
    
    # Process with different filters
    filter_processor = mi.ProcessorFactory.get_instance().create_image_filter(config)
    
    # Results dictionary
    results = {}
    times = {}
    
    # Gaussian filter
    print("Applying Gaussian filter...")
    filter_processor.set_method(mi.FilterMethod.Gaussian)
    filter_processor.set_scalar_parameter("sigma", args.gaussian_sigma)
    
    start_time = time.time()
    gaussian_result = filter_processor.process(mi.from_numpy(noisy))
    gaussian_time = time.time() - start_time
    
    results['gaussian'] = mi.to_numpy(gaussian_result)
    times['gaussian'] = gaussian_time
    
    print(f"Gaussian filter time: {gaussian_time:.2f} seconds")
    print(f"Performance metrics:")
    filter_processor.get_performance_metrics().print()
    
    # Bilateral filter
    print("Applying Bilateral filter...")
    filter_processor.set_method(mi.FilterMethod.Bilateral)
    filter_processor.set_scalar_parameter("spatial_sigma", args.bilateral_spatial)
    filter_processor.set_scalar_parameter("range_sigma", args.bilateral_range)
    
    start_time = time.time()
    bilateral_result = filter_processor.process(mi.from_numpy(noisy))
    bilateral_time = time.time() - start_time
    
    results['bilateral'] = mi.to_numpy(bilateral_result)
    times['bilateral'] = bilateral_time
    
    print(f"Bilateral filter time: {bilateral_time:.2f} seconds")
    print(f"Performance metrics:")
    filter_processor.get_performance_metrics().print()
    
    # Non-local means filter
    if args.nlm:
        print("Applying Non-local means filter...")
        filter_processor.set_method(mi.FilterMethod.NonLocalMeans)
        filter_processor.set_scalar_parameter("h", args.nlm_h)
        filter_processor.set_scalar_parameter("search_radius", args.nlm_search)
        filter_processor.set_scalar_parameter("patch_radius", args.nlm_patch)
        
        start_time = time.time()
        nlm_result = filter_processor.process(mi.from_numpy(noisy))
        nlm_time = time.time() - start_time
        
        results['nlm'] = mi.to_numpy(nlm_result)
        times['nlm'] = nlm_time
        
        print(f"Non-local means filter time: {nlm_time:.2f} seconds")
        print(f"Performance metrics:")
        filter_processor.get_performance_metrics().print()
    
    # Segmentation
    if args.segment:
        print("Configuring image segmenter...")
        segmenter = mi.ProcessorFactory.get_instance().create_image_segmenter(config)
        
        # Thresholding
        print("Applying thresholding segmentation...")
        segmenter.set_method(mi.SegmentationMethod.Thresholding)
        segmenter.set_scalar_parameter("threshold", args.threshold)
        
        start_time = time.time()
        threshold_result = segmenter.process(mi.from_numpy(results.get(args.segment_input, original)))
        threshold_time = time.time() - start_time
        
        results['threshold'] = mi.to_numpy(threshold_result)
        times['threshold'] = threshold_time
        
        print(f"Thresholding time: {threshold_time:.2f} seconds")
        print(f"Performance metrics:")
        segmenter.get_performance_metrics().print()
        
        # Watershed segmentation
        if args.watershed:
            print("Applying watershed segmentation...")
            segmenter.set_method(mi.SegmentationMethod.Watershed)
            
            # Create markers for watershed
            markers = np.zeros_like(original)
            markers[original < 0.2] = 1  # Background
            markers[original > 0.7] = 2  # Foreground
            
            segmenter.set_scalar_parameter("use_markers", 1.0)  # Use markers
            
            start_time = time.time()
            watershed_result = segmenter.process(mi.from_numpy(results.get(args.segment_input, original)))
            watershed_time = time.time() - start_time
            
            results['watershed'] = mi.to_numpy(watershed_result)
            times['watershed'] = watershed_time
            
            print(f"Watershed time: {watershed_time:.2f} seconds")
            print(f"Performance metrics:")
            segmenter.get_performance_metrics().print()
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    # Original and noisy
    plt.subplot(231)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.colorbar()
    
    plt.subplot(232)
    plt.imshow(noisy, cmap='gray')
    plt.title(f'Noisy ({args.noise_type})')
    plt.colorbar()
    
    # Filtered results
    plt.subplot(233)
    plt.imshow(results['gaussian'], cmap='gray')
    plt.title(f'Gaussian (σ={args.gaussian_sigma})\n{times["gaussian"]:.2f}s')
    plt.colorbar()
    
    plt.subplot(234)
    plt.imshow(results['bilateral'], cmap='gray')
    plt.title(f'Bilateral (σ_s={args.bilateral_spatial}, σ_r={args.bilateral_range})\n{times["bilateral"]:.2f}s')
    plt.colorbar()
    
    if args.nlm:
        plt.subplot(235)
        plt.imshow(results['nlm'], cmap='gray')
        plt.title(f'Non-local Means\n{times["nlm"]:.2f}s')
        plt.colorbar()
    
    # Segmentation results
    if args.segment:
        plt_idx = 236 if args.nlm else 235
        plt.subplot(plt_idx)
        
        if args.watershed and 'watershed' in results:
            # Display watershed result
            plt.imshow(results['watershed'], cmap='viridis')
            plt.title(f'Watershed\n{times["watershed"]:.2f}s')
        else:
            # Display thresholding result
            plt.imshow(results['threshold'], cmap='gray')
            plt.title(f'Thresholding (t={args.threshold})\n{times["threshold"]:.2f}s')
        
        plt.colorbar()
    
    plt.tight_layout()
    
    # Save results
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plt.savefig(args.output)
        print(f"Results saved to {args.output}")
        
        # Save individual images
        base, ext = os.path.splitext(args.output)
        
        np.save(f"{base}_original.npy", original)
        np.save(f"{base}_noisy.npy", noisy)
        
        for name, result in results.items():
            np.save(f"{base}_{name}.npy", result)
    
    if not args.no_display:
        plt.show()
        
        # Show segmentation overlay if available
        if args.segment and 'threshold' in results:
            plt.figure(figsize=(10, 8))
            plt.subplot(121)
            plt.imshow(results.get(args.segment_input, original), cmap='gray')
            plt.title(f'Input for segmentation ({args.segment_input})')
            plt.colorbar()
            
            plt.subplot(122)
            plot_overlay(original, results['threshold'])
            plt.title('Segmentation Overlay')
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Processing Example")
    
    # Input/output options
    parser.add_argument("-i", "--input", type=str, help="Input image path")
    parser.add_argument("-o", "--output", type=str, help="Output image path")
    parser.add_argument("-d", "--device-id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--no-display", action="store_true", help="Don't display results (save only)")
    
    # Noise options
    parser.add_argument("--noise-type", type=str, default="gaussian", 
                        choices=["gaussian", "salt_pepper", "poisson"], help="Type of noise")
    parser.add_argument("--noise-amount", type=float, default=0.05, help="Noise amount/intensity")
    
    # Filter options
    parser.add_argument("--gaussian-sigma", type=float, default=1.0, help="Gaussian filter sigma")
    parser.add_argument("--bilateral-spatial", type=float, default=3.0, help="Bilateral filter spatial sigma")
    parser.add_argument("--bilateral-range", type=float, default=0.1, help="Bilateral filter range sigma")
    
    # Non-local means options
    parser.add_argument("--nlm", action="store_true", help="Apply non-local means filter")
    parser.add_argument("--nlm-h", type=float, default=0.1, help="Non-local means h parameter")
    parser.add_argument("--nlm-search", type=int, default=7, help="Non-local means search radius")
    parser.add_argument("--nlm-patch", type=int, default=3, help="Non-local means patch radius")
    
    # Segmentation options
    parser.add_argument("--segment", action="store_true", help="Apply segmentation")
    parser.add_argument("--segment-input", type=str, default="bilateral", 
                        choices=["original", "noisy", "gaussian", "bilateral", "nlm"],
                        help="Input for segmentation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for segmentation")
    parser.add_argument("--watershed", action="store_true", help="Apply watershed segmentation")
    
    args = parser.parse_args()
    main(args)