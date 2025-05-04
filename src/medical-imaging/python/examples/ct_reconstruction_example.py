#!/usr/bin/env python3
"""
CT Reconstruction Example

This example demonstrates CT image reconstruction from projections using both
filtered backprojection and iterative methods. It includes performance benchmarking
to compare the reconstruction speed on different hardware architectures.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Import the medical_imaging package
import medical_imaging as mi
from medical_imaging.visualization import plot_images, plot_ct_projection

def create_phantom(size=512):
    """Create a Shepp-Logan phantom for testing."""
    # Create a circle
    x, y = np.mgrid[-1:1:size*1j, -1:1:size*1j]
    circle = x**2 + y**2 <= 0.8**2
    
    # Add some smaller circles and ellipses
    ellipse1 = ((x+0.3)**2 + (y+0.2)**2/0.3**2) <= 0.2**2
    ellipse2 = ((x-0.3)**2 + (y-0.3)**2/0.3**2) <= 0.15**2
    circle2 = (x-0.1)**2 + (y+0.1)**2 <= 0.05**2
    
    # Combine shapes
    phantom = circle.astype(float) * 0.5
    phantom[ellipse1] = 0.8
    phantom[ellipse2] = 0.3
    phantom[circle2] = 1.0
    
    return phantom

def simulate_projections(image, angles):
    """Simulate CT projections of an image."""
    n_angles = len(angles)
    n_detector = image.shape[0]
    
    # Initialize projections array
    projections = np.zeros((n_angles, n_detector))
    
    # Center of image
    center = n_detector // 2
    
    # Simulate projections for each angle
    for i, angle in enumerate(angles):
        # Rotate the image
        rotated = mi.to_numpy(
            mi.from_numpy(image).rotate_2d(angle, interpolation=1)
        )
        
        # Sum along columns (radon transform)
        projection = np.sum(rotated, axis=0)
        projections[i] = projection
    
    return projections

def main(args):
    # Print device information
    akm = mi.AdaptiveKernelManager.get_instance()
    if akm.initialize():
        caps = akm.get_device_capabilities()
        print("Device capabilities:")
        print(caps.get_summary())
    
    # Create or load phantom
    if args.input:
        print(f"Loading input image: {args.input}")
        phantom = mi.load_image(args.input)
    else:
        print(f"Creating Shepp-Logan phantom of size {args.size}x{args.size}")
        phantom = create_phantom(args.size)
    
    # Create projection angles
    angles = np.linspace(0, np.pi, args.num_angles, endpoint=False)
    
    # Simulate projections
    print(f"Simulating projections for {args.num_angles} angles...")
    start_time = time.time()
    projections = simulate_projections(phantom, angles)
    projection_time = time.time() - start_time
    print(f"Projection simulation time: {projection_time:.2f} seconds")
    
    # Configure reconstructor
    print("Configuring CT reconstructor...")
    config = mi.ProcessingConfig()
    config.compute_backend = mi.ComputeBackend.CUDA
    config.device_id = args.device_id
    
    # Create reconstructor
    reconstructor = mi.ProcessorFactory.get_instance().create_ct_reconstructor(config)
    reconstructor.set_projection_angles(angles)
    
    # Filtered Backprojection
    print("Performing filtered backprojection reconstruction...")
    reconstructor.set_method(mi.ReconstructionMethod.FilteredBackProjection)
    
    start_time = time.time()
    fbp_result = reconstructor.process(mi.from_numpy(projections))
    fbp_time = time.time() - start_time
    
    # Convert to numpy for display
    fbp_image = mi.to_numpy(fbp_result)
    
    print(f"Filtered backprojection time: {fbp_time:.2f} seconds")
    print(f"Performance metrics:")
    reconstructor.get_performance_metrics().print()
    
    # Iterative reconstruction
    if args.iterative:
        print("Performing iterative reconstruction...")
        reconstructor.set_method(mi.ReconstructionMethod.IterativePrimalDual)
        reconstructor.set_num_iterations(args.iterations)
        
        start_time = time.time()
        iterative_result = reconstructor.process(mi.from_numpy(projections))
        iterative_time = time.time() - start_time
        
        # Convert to numpy for display
        iterative_image = mi.to_numpy(iterative_result)
        
        print(f"Iterative reconstruction time: {iterative_time:.2f} seconds")
        print(f"Performance metrics:")
        reconstructor.get_performance_metrics().print()
    
    # Display results
    plt.figure(figsize=(12, 10))
    
    plt.subplot(221)
    plt.imshow(phantom, cmap='gray')
    plt.title('Original Phantom')
    plt.colorbar()
    
    plt.subplot(222)
    plt.imshow(projections, cmap='gray', aspect='auto')
    plt.title('Projections (Sinogram)')
    plt.xlabel('Detector Position')
    plt.ylabel('Projection Angle')
    plt.colorbar()
    
    plt.subplot(223)
    plt.imshow(fbp_image, cmap='gray')
    plt.title(f'Filtered Backprojection\n({fbp_time:.2f}s)')
    plt.colorbar()
    
    if args.iterative:
        plt.subplot(224)
        plt.imshow(iterative_image, cmap='gray')
        plt.title(f'Iterative Reconstruction\n({iterative_time:.2f}s, {args.iterations} iterations)')
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
        
        np.save(f"{base}_phantom.npy", phantom)
        np.save(f"{base}_projections.npy", projections)
        np.save(f"{base}_fbp.npy", fbp_image)
        
        if args.iterative:
            np.save(f"{base}_iterative.npy", iterative_image)
    
    if not args.no_display:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT Reconstruction Example")
    parser.add_argument("-s", "--size", type=int, default=512, help="Phantom size")
    parser.add_argument("-a", "--num-angles", type=int, default=180, help="Number of projection angles")
    parser.add_argument("-i", "--input", type=str, help="Input image path (instead of phantom)")
    parser.add_argument("-o", "--output", type=str, help="Output image path")
    parser.add_argument("-d", "--device-id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--iterative", action="store_true", help="Perform iterative reconstruction")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations for iterative method")
    parser.add_argument("--no-display", action="store_true", help="Don't display results (save only)")
    
    args = parser.parse_args()
    main(args)