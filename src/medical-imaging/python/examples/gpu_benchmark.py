#!/usr/bin/env python3
"""
GPU Performance Benchmark

This example benchmarks the performance of medical imaging workloads
across different devices (Jetson Orin NX, T4 GPU, CPU).
"""

import os
import time
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

# Import the medical_imaging package
import medical_imaging as mi

def benchmark_ct_reconstruction(sizes, device_id, num_angles=180, num_runs=3):
    """Benchmark CT reconstruction performance for different image sizes."""
    results = {}
    
    for size in sizes:
        print(f"Benchmarking CT Reconstruction for size {size}x{size}...")
        
        # Create phantom
        x, y = np.mgrid[-1:1:size*1j, -1:1:size*1j]
        phantom = (x**2 + y**2 <= 0.8**2).astype(np.float32)
        
        # Create projections
        angles = np.linspace(0, np.pi, num_angles, endpoint=False)
        projections = np.zeros((num_angles, size), dtype=np.float32)
        
        # Simple projection simulation
        center = size // 2
        for i, angle in enumerate(angles):
            # Rotate the phantom
            rot_img = mi.to_numpy(
                mi.from_numpy(phantom).rotate_2d(angle, interpolation=1)
            )
            
            # Sum along columns (radon transform)
            projections[i] = np.sum(rot_img, axis=0)
        
        # Configure CT reconstructor
        config = mi.ProcessingConfig()
        config.compute_backend = mi.ComputeBackend.CUDA
        config.device_id = device_id
        
        reconstructor = mi.ProcessorFactory.get_instance().create_ct_reconstructor(config)
        reconstructor.set_projection_angles(angles)
        
        # Benchmark filtered backprojection
        fbp_times = []
        for _ in range(num_runs):
            start_time = time.time()
            fbp_result = reconstructor.process(mi.from_numpy(projections))
            fbp_time = time.time() - start_time
            fbp_times.append(fbp_time)
        
        avg_fbp_time = sum(fbp_times) / len(fbp_times)
        
        # Store results
        results[size] = {
            'fbp_time': avg_fbp_time,
            'fbp_times': fbp_times,
            'compute_time': reconstructor.get_performance_metrics().compute_time_ms / 1000.0,
            'memory_time': reconstructor.get_performance_metrics().memory_transfer_time_ms / 1000.0
        }
        
        print(f"  Average FBP time: {avg_fbp_time:.3f} seconds")
    
    return results

def benchmark_image_filtering(sizes, device_id, kernel_sizes=[3, 5, 7, 9], num_runs=3):
    """Benchmark image filtering performance for different image sizes and kernel sizes."""
    results = {}
    
    for size in sizes:
        results[size] = {}
        
        print(f"Benchmarking Image Filtering for size {size}x{size}...")
        
        # Create test image
        image = np.random.rand(size, size).astype(np.float32)
        
        # Configure image filter
        config = mi.ProcessingConfig()
        config.compute_backend = mi.ComputeBackend.CUDA
        config.device_id = device_id
        
        filter_processor = mi.ProcessorFactory.get_instance().create_image_filter(config)
        
        # Benchmark different kernel sizes
        for kernel_size in kernel_sizes:
            print(f"  Kernel size: {kernel_size}x{kernel_size}")
            
            # Create kernel
            kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
            
            # Set convolution parameters
            filter_processor.set_method(mi.FilterMethod.Gaussian)
            filter_processor.set_scalar_parameter("sigma", kernel_size / 6.0)
            
            # Benchmark convolution
            conv_times = []
            for _ in range(num_runs):
                start_time = time.time()
                result = filter_processor.process(mi.from_numpy(image))
                conv_time = time.time() - start_time
                conv_times.append(conv_time)
            
            avg_conv_time = sum(conv_times) / len(conv_times)
            
            # Store results
            results[size][kernel_size] = {
                'conv_time': avg_conv_time,
                'conv_times': conv_times,
                'compute_time': filter_processor.get_performance_metrics().compute_time_ms / 1000.0,
                'memory_time': filter_processor.get_performance_metrics().memory_transfer_time_ms / 1000.0
            }
            
            print(f"    Average convolution time: {avg_conv_time:.3f} seconds")
    
    return results

def benchmark_nlm_filter(sizes, device_id, patch_sizes=[3, 5, 7], search_sizes=[7, 11, 15], num_runs=3):
    """Benchmark non-local means filtering performance."""
    results = {}
    
    for size in sizes:
        results[size] = {}
        
        print(f"Benchmarking NLM Filtering for size {size}x{size}...")
        
        # Create test image
        image = np.random.rand(size, size).astype(np.float32)
        
        # Configure image filter
        config = mi.ProcessingConfig()
        config.compute_backend = mi.ComputeBackend.CUDA
        config.device_id = device_id
        
        filter_processor = mi.ProcessorFactory.get_instance().create_image_filter(config)
        filter_processor.set_method(mi.FilterMethod.NonLocalMeans)
        
        # Benchmark different patch and search sizes
        for patch_radius in patch_sizes:
            results[size][f'patch_{patch_radius}'] = {}
            
            for search_radius in search_sizes:
                print(f"  Patch radius: {patch_radius}, Search radius: {search_radius}")
                
                # Set NLM parameters
                filter_processor.set_scalar_parameter("h", 0.1)
                filter_processor.set_scalar_parameter("patch_radius", float(patch_radius))
                filter_processor.set_scalar_parameter("search_radius", float(search_radius))
                
                # Benchmark NLM
                nlm_times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    result = filter_processor.process(mi.from_numpy(image))
                    nlm_time = time.time() - start_time
                    nlm_times.append(nlm_time)
                
                avg_nlm_time = sum(nlm_times) / len(nlm_times)
                
                # Store results
                results[size][f'patch_{patch_radius}'][f'search_{search_radius}'] = {
                    'nlm_time': avg_nlm_time,
                    'nlm_times': nlm_times,
                    'compute_time': filter_processor.get_performance_metrics().compute_time_ms / 1000.0,
                    'memory_time': filter_processor.get_performance_metrics().memory_transfer_time_ms / 1000.0
                }
                
                print(f"    Average NLM time: {avg_nlm_time:.3f} seconds")
    
    return results

def benchmark_segmentation(sizes, device_id, num_runs=3):
    """Benchmark segmentation performance for different image sizes."""
    results = {}
    
    for size in sizes:
        print(f"Benchmarking Segmentation for size {size}x{size}...")
        
        # Create test image
        image = np.random.rand(size, size).astype(np.float32)
        image = np.clip(image + np.sin(np.linspace(0, 10, size)) * np.sin(np.linspace(0, 10, size)).reshape(-1, 1), 0, 1)
        
        # Configure segmenter
        config = mi.ProcessingConfig()
        config.compute_backend = mi.ComputeBackend.CUDA
        config.device_id = device_id
        
        segmenter = mi.ProcessorFactory.get_instance().create_image_segmenter(config)
        
        # Benchmark thresholding
        segmenter.set_method(mi.SegmentationMethod.Thresholding)
        segmenter.set_scalar_parameter("threshold", 0.5)
        
        thresh_times = []
        for _ in range(num_runs):
            start_time = time.time()
            result = segmenter.process(mi.from_numpy(image))
            thresh_time = time.time() - start_time
            thresh_times.append(thresh_time)
        
        avg_thresh_time = sum(thresh_times) / len(thresh_times)
        
        # Benchmark watershed
        segmenter.set_method(mi.SegmentationMethod.Watershed)
        
        # Create markers for watershed
        markers = np.zeros_like(image)
        markers[image < 0.2] = 1  # Background
        markers[image > 0.8] = 2  # Foreground
        
        watershed_times = []
        for _ in range(num_runs):
            start_time = time.time()
            result = segmenter.process(mi.from_numpy(image))
            watershed_time = time.time() - start_time
            watershed_times.append(watershed_time)
        
        avg_watershed_time = sum(watershed_times) / len(watershed_times)
        
        # Store results
        results[size] = {
            'thresholding': {
                'time': avg_thresh_time,
                'times': thresh_times
            },
            'watershed': {
                'time': avg_watershed_time,
                'times': watershed_times
            }
        }
        
        print(f"  Average thresholding time: {avg_thresh_time:.3f} seconds")
        print(f"  Average watershed time: {avg_watershed_time:.3f} seconds")
    
    return results

def plot_benchmark_results(results, title, output_dir=None):
    """Plot benchmark results."""
    plt.figure(figsize=(12, 8))
    
    for device, device_results in results.items():
        # Extract data for plotting
        sizes = []
        times = []
        
        for size, size_results in device_results.items():
            sizes.append(int(size))
            times.append(size_results['fbp_time'] if 'fbp_time' in size_results else size_results['time'])
        
        # Sort by size
        sorted_data = sorted(zip(sizes, times))
        sizes, times = zip(*sorted_data)
        
        # Plot
        plt.plot(sizes, times, 'o-', label=device)
    
    plt.title(title)
    plt.xlabel('Image Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    if output_dir:
        filename = os.path.join(output_dir, title.replace(' ', '_') + '.png')
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
    
    plt.show()

def main(args):
    # Print device information
    akm = mi.AdaptiveKernelManager.get_instance()
    if akm.initialize():
        caps = akm.get_device_capabilities()
        print("Device capabilities:")
        print(caps.get_summary())
    
    # Get device type
    device_type = "Unknown"
    if caps.device_type == mi.DeviceType.JetsonOrinNX:
        device_type = "JetsonOrinNX"
    elif caps.device_type == mi.DeviceType.T4:
        device_type = "T4_GPU"
    elif caps.device_type == mi.DeviceType.HighEndGPU:
        device_type = "HighEndGPU"
    elif caps.device_type == mi.DeviceType.OtherGPU:
        device_type = "OtherGPU"
    elif caps.device_type == mi.DeviceType.CPU:
        device_type = "CPU"
    
    # Create output directory if needed
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    results = {}
    
    # CT Reconstruction benchmark
    if args.ct:
        print("\n=== CT Reconstruction Benchmark ===")
        ct_results = benchmark_ct_reconstruction(
            args.sizes, args.device_id, args.num_angles, args.num_runs
        )
        results['ct_reconstruction'] = ct_results
        
        # Save results
        if args.output:
            with open(os.path.join(args.output, f"ct_results_{device_type}.json"), 'w') as f:
                json.dump(ct_results, f, indent=2)
    
    # Image Filtering benchmark
    if args.filtering:
        print("\n=== Image Filtering Benchmark ===")
        filtering_results = benchmark_image_filtering(
            args.sizes, args.device_id, args.kernel_sizes, args.num_runs
        )
        results['image_filtering'] = filtering_results
        
        # Save results
        if args.output:
            with open(os.path.join(args.output, f"filtering_results_{device_type}.json"), 'w') as f:
                json.dump(filtering_results, f, indent=2)
    
    # NLM Filtering benchmark
    if args.nlm:
        print("\n=== Non-Local Means Filtering Benchmark ===")
        nlm_results = benchmark_nlm_filter(
            args.sizes[:2], args.device_id, args.patch_sizes, args.search_sizes, args.num_runs
        )
        results['nlm_filtering'] = nlm_results
        
        # Save results
        if args.output:
            with open(os.path.join(args.output, f"nlm_results_{device_type}.json"), 'w') as f:
                json.dump(nlm_results, f, indent=2)
    
    # Segmentation benchmark
    if args.segmentation:
        print("\n=== Segmentation Benchmark ===")
        segmentation_results = benchmark_segmentation(
            args.sizes, args.device_id, args.num_runs
        )
        results['segmentation'] = segmentation_results
        
        # Save results
        if args.output:
            with open(os.path.join(args.output, f"segmentation_results_{device_type}.json"), 'w') as f:
                json.dump(segmentation_results, f, indent=2)
    
    # Save combined results
    if args.output:
        with open(os.path.join(args.output, f"all_results_{device_type}.json"), 'w') as f:
            json.dump(results, f, indent=2)
    
    # Plot results if multiple devices are available
    if args.plot and os.path.exists(args.output):
        # Try to load results from other devices
        all_device_results = {'ct_reconstruction': {}, 'image_filtering': {}, 'nlm_filtering': {}, 'segmentation': {}}
        
        # Look for result files
        for filename in os.listdir(args.output):
            if filename.startswith("all_results_") and filename.endswith(".json"):
                device = filename[12:-5]  # Extract device name
                
                with open(os.path.join(args.output, filename), 'r') as f:
                    device_results = json.load(f)
                
                # Add to combined results
                for benchmark, bench_results in device_results.items():
                    if benchmark in all_device_results:
                        all_device_results[benchmark][device] = bench_results
        
        # Plot combined results if we have data from multiple devices
        if len(all_device_results['ct_reconstruction']) > 1:
            plot_benchmark_results(
                all_device_results['ct_reconstruction'], 
                "CT Reconstruction Performance", 
                args.output
            )
        
        if len(all_device_results['image_filtering']) > 1:
            # Plot for a specific kernel size
            kernel_size = args.kernel_sizes[0]
            kernel_results = {}
            
            for device, device_results in all_device_results['image_filtering'].items():
                kernel_results[device] = {}
                for size, size_results in device_results.items():
                    if kernel_size in size_results:
                        kernel_results[device][size] = size_results[kernel_size]
            
            plot_benchmark_results(
                kernel_results, 
                f"Image Filtering Performance ({kernel_size}x{kernel_size} kernel)", 
                args.output
            )
        
        if len(all_device_results['segmentation']) > 1:
            # Plot thresholding results
            thresh_results = {}
            
            for device, device_results in all_device_results['segmentation'].items():
                thresh_results[device] = {}
                for size, size_results in device_results.items():
                    thresh_results[device][size] = size_results['thresholding']
            
            plot_benchmark_results(
                thresh_results, 
                "Thresholding Segmentation Performance", 
                args.output
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Performance Benchmark")
    
    # Benchmark options
    parser.add_argument("--ct", action="store_true", help="Run CT reconstruction benchmark")
    parser.add_argument("--filtering", action="store_true", help="Run image filtering benchmark")
    parser.add_argument("--nlm", action="store_true", help="Run NLM filtering benchmark")
    parser.add_argument("--segmentation", action="store_true", help="Run segmentation benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    
    # Device options
    parser.add_argument("-d", "--device-id", type=int, default=0, help="CUDA device ID")
    
    # Output options
    parser.add_argument("-o", "--output", type=str, help="Output directory for results")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot combined results")
    
    # Benchmark parameters
    parser.add_argument("-s", "--sizes", type=int, nargs='+', default=[256, 512, 1024, 2048], 
                        help="Image sizes to benchmark")
    parser.add_argument("-r", "--num-runs", type=int, default=3, help="Number of runs per benchmark")
    parser.add_argument("-a", "--num-angles", type=int, default=180, help="Number of angles for CT")
    parser.add_argument("-k", "--kernel-sizes", type=int, nargs='+', default=[3, 5, 7, 9], 
                        help="Kernel sizes for filtering")
    parser.add_argument("--patch-sizes", type=int, nargs='+', default=[3, 5, 7], 
                        help="Patch sizes for NLM")
    parser.add_argument("--search-sizes", type=int, nargs='+', default=[7, 11, 15], 
                        help="Search window sizes for NLM")
    
    args = parser.parse_args()
    
    # If --all is specified, run all benchmarks
    if args.all:
        args.ct = True
        args.filtering = True
        args.nlm = True
        args.segmentation = True
    
    main(args)