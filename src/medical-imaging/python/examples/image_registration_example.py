#!/usr/bin/env python3
"""
Image Registration Example

This example demonstrates various image registration methods for aligning
a moving image to a fixed reference image. It includes performance benchmarking
comparing different hardware architectures.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform

# Import the medical_imaging package
import medical_imaging as mi
from medical_imaging.visualization import (
    plot_images, plot_comparison, plot_difference
)

def create_test_images(size=512, rotation=10, scale=0.9, translation=(20, -15), noise_level=0.05):
    """Create a pair of test images for registration.
    
    Args:
        size: Image size
        rotation: Rotation angle in degrees
        scale: Scale factor
        translation: Translation (x, y) in pixels
        noise_level: Noise standard deviation
        
    Returns:
        fixed_image, moving_image
    """
    # Create a simple phantom with geometric shapes
    fixed = np.zeros((size, size), dtype=np.float32)
    
    # Add shapes
    center = size // 2
    
    # Circle
    y, x = np.ogrid[-center:size-center, -center:size-center]
    mask = x*x + y*y <= (size//4)**2
    fixed[mask] = 0.8
    
    # Rectangle
    fixed[center-size//8:center+size//8, center-size//6:center+size//6] = 1.0
    
    # Cross
    fixed[center-size//10:center+size//10, center-size//3:center+size//3] = 0.5
    fixed[center-size//3:center+size//3, center-size//10:center+size//10] = 0.5
    
    # Create moving image by applying transformation and adding noise
    # Convert to angles
    angle_rad = np.deg2rad(rotation)
    
    # Create transformation matrix
    tform = transform.AffineTransform(
        scale=(scale, scale),
        rotation=angle_rad,
        translation=translation
    )
    
    # Apply transformation
    moving = transform.warp(fixed, tform.inverse, mode='constant')
    
    # Add noise to both images
    fixed_noisy = fixed + np.random.normal(0, noise_level, fixed.shape)
    moving_noisy = moving + np.random.normal(0, noise_level, moving.shape)
    
    # Clip to valid range
    fixed_noisy = np.clip(fixed_noisy, 0, 1)
    moving_noisy = np.clip(moving_noisy, 0, 1)
    
    return fixed_noisy, moving_noisy

def main(args):
    # Print device information
    akm = mi.AdaptiveKernelManager.get_instance()
    if akm.initialize():
        caps = akm.get_device_capabilities()
        print("Device capabilities:")
        print(caps.get_summary())
    
    # Load or create images
    if args.fixed and args.moving:
        print(f"Loading fixed image: {args.fixed}")
        fixed = mi.load_image(args.fixed)
        
        print(f"Loading moving image: {args.moving}")
        moving = mi.load_image(args.moving)
        
        # Ensure images are normalized
        fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
        moving = (moving - moving.min()) / (moving.max() - moving.min())
    else:
        # Create test images
        print(f"Creating test images of size {args.size}x{args.size}...")
        print(f"  Rotation: {args.rotation} degrees")
        print(f"  Scale: {args.scale}")
        print(f"  Translation: {args.translation}")
        print(f"  Noise level: {args.noise}")
        
        fixed, moving = create_test_images(
            args.size, 
            args.rotation, 
            args.scale, 
            args.translation, 
            args.noise
        )
    
    # Configure processor
    print("Configuring image registration processor...")
    config = mi.ProcessingConfig()
    config.compute_backend = mi.ComputeBackend.CUDA
    config.device_id = args.device_id
    
    # Set registration method
    if args.method == "rigid":
        method_name = "Rigid Registration"
        config.string_params["registration_method"] = "rigid"
    elif args.method == "affine":
        method_name = "Affine Registration"
        config.string_params["registration_method"] = "affine"
    elif args.method == "deformable":
        method_name = "Deformable Registration"
        config.string_params["registration_method"] = "deformable_bspline"
    
    # Set other parameters
    config.scalar_params["max_iterations"] = args.iterations
    config.scalar_params["convergence_epsilon"] = args.epsilon
    
    # Create registration processor
    registration = mi.ProcessorFactory.get_instance().create_image_registration(config)
    registration.initialize()
    
    # Register images
    print(f"Performing {method_name}...")
    start_time = time.time()
    registered_result = registration.register_images(
        mi.from_numpy(fixed), 
        mi.from_numpy(moving)
    )
    registration_time = time.time() - start_time
    
    registered = mi.to_numpy(registered_result)
    
    print(f"Registration time: {registration_time:.2f} seconds")
    print(f"Performance metrics:")
    registration.get_performance_metrics().print()
    
    # Get transformation parameters
    transform_params = registration.get_transform_parameters()
    transform_matrix = registration.get_transform_matrix()
    
    print("Transformation parameters:")
    print(transform_params)
    
    if len(transform_matrix) > 0:
        print("Transformation matrix:")
        for row in transform_matrix:
            print("  ", row)
    
    # Calculate difference images
    diff_before = np.abs(fixed - moving)
    diff_after = np.abs(fixed - registered)
    
    # Calculate metrics
    mse_before = np.mean(np.square(fixed - moving))
    mse_after = np.mean(np.square(fixed - registered))
    
    print(f"Mean Squared Error before registration: {mse_before:.5f}")
    print(f"Mean Squared Error after registration: {mse_after:.5f}")
    print(f"Improvement: {100 * (mse_before - mse_after) / mse_before:.2f}%")
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    # Input images
    plt.subplot(231)
    plt.imshow(fixed, cmap='gray')
    plt.title('Fixed Image')
    plt.colorbar()
    
    plt.subplot(232)
    plt.imshow(moving, cmap='gray')
    plt.title('Moving Image')
    plt.colorbar()
    
    plt.subplot(233)
    plt.imshow(registered, cmap='gray')
    plt.title(f'{method_name}\n{registration_time:.2f}s')
    plt.colorbar()
    
    # Difference images and checkerboard
    plt.subplot(234)
    plt.imshow(diff_before, cmap='hot')
    plt.title(f'Difference Before\nMSE: {mse_before:.5f}')
    plt.colorbar()
    
    plt.subplot(235)
    plt.imshow(diff_after, cmap='hot')
    plt.title(f'Difference After\nMSE: {mse_after:.5f}')
    plt.colorbar()
    
    # Checkerboard visualization
    plt.subplot(236)
    checkerboard = np.zeros_like(fixed)
    check_size = args.checker_size
    for i in range(0, fixed.shape[0], check_size):
        for j in range(0, fixed.shape[1], check_size):
            if (i // check_size + j // check_size) % 2 == 0:
                checkerboard[i:i+check_size, j:j+check_size] = fixed[i:i+check_size, j:j+check_size]
            else:
                checkerboard[i:i+check_size, j:j+check_size] = registered[i:i+check_size, j:j+check_size]
    
    plt.imshow(checkerboard, cmap='gray')
    plt.title('Checkerboard Comparison')
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
        
        np.save(f"{base}_fixed.npy", fixed)
        np.save(f"{base}_moving.npy", moving)
        np.save(f"{base}_registered.npy", registered)
        np.save(f"{base}_diff_before.npy", diff_before)
        np.save(f"{base}_diff_after.npy", diff_after)
        
        # Save transform parameters
        np.save(f"{base}_transform_params.npy", np.array(transform_params))
        
        if len(transform_matrix) > 0:
            np.save(f"{base}_transform_matrix.npy", np.array(transform_matrix))
    
    if not args.no_display:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Registration Example")
    
    # Input/output options
    parser.add_argument("--fixed", type=str, help="Fixed image path")
    parser.add_argument("--moving", type=str, help="Moving image path")
    parser.add_argument("-o", "--output", type=str, help="Output image path")
    parser.add_argument("-d", "--device-id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--no-display", action="store_true", help="Don't display results (save only)")
    
    # Test image options
    parser.add_argument("-s", "--size", type=int, default=512, help="Test image size")
    parser.add_argument("-r", "--rotation", type=float, default=10.0, help="Test image rotation (degrees)")
    parser.add_argument("--scale", type=float, default=0.9, help="Test image scale factor")
    parser.add_argument("-t", "--translation", type=int, nargs=2, default=[20, -15], 
                        help="Test image translation (x y)")
    parser.add_argument("-n", "--noise", type=float, default=0.05, help="Test image noise level")
    
    # Registration options
    parser.add_argument("-m", "--method", type=str, default="rigid", 
                        choices=["rigid", "affine", "deformable"], 
                        help="Registration method")
    parser.add_argument("-i", "--iterations", type=int, default=100, 
                        help="Maximum number of iterations")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-5, 
                        help="Convergence criterion")
    parser.add_argument("-c", "--checker-size", type=int, default=32, 
                        help="Checkerboard visualization tile size")
    
    args = parser.parse_args()
    main(args)