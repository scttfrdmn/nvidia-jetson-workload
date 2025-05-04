#!/usr/bin/env python3
"""
Benchmark script for the Geospatial Analysis workload.
This script benchmarks the performance of the DEM processing and Point Cloud processing
operations on different hardware configurations.
"""

import os
import time
import argparse
import numpy as np
import json
import platform
import subprocess
from contextlib import contextmanager
import tempfile
import shutil
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Add parent directory to Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'python')))

# Import geospatial modules
from geospatial.dem import DEMProcessor
from geospatial.point_cloud import PointCloud

# Context manager for timing code execution
@contextmanager
def timer(operation):
    start = time.time()
    yield
    end = time.time()
    print(f"{operation}: {end - start:.4f} seconds")
    return end - start

def get_system_info():
    """Get system information for the benchmark report"""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'memory': ''
    }
    
    # Get total memory
    try:
        if platform.system() == 'Linux':
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        info['memory'] = line.split()[1]
        elif platform.system() == 'Darwin':  # macOS
            mem_info = subprocess.check_output(['sysctl', '-n', 'hw.memsize'])
            info['memory'] = f"{int(mem_info) // (1024 * 1024)} MB"
        elif platform.system() == 'Windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('ullAvailExtendedVirtual', c_ulonglong),
                ]
            
            memoryStatus = MEMORYSTATUSEX()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus))
            info['memory'] = f"{memoryStatus.ullTotalPhys // (1024 * 1024)} MB"
    except Exception as e:
        info['memory'] = f"Error getting memory info: {e}"
    
    # Get GPU information
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            info['cuda_version'] = torch.version.cuda
        else:
            info['gpu'] = 'No CUDA GPU available'
            info['cuda_version'] = 'N/A'
    except ImportError:
        # Try nvidia-smi directly
        try:
            nvidia_smi = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'],
                universal_newlines=True
            )
            if nvidia_smi.strip():
                parts = nvidia_smi.strip().split(',')
                info['gpu'] = parts[0].strip()
                info['cuda_version'] = parts[1].strip() if len(parts) > 1 else 'Unknown'
            else:
                info['gpu'] = 'No GPU found'
                info['cuda_version'] = 'N/A'
        except (subprocess.SubprocessError, FileNotFoundError):
            info['gpu'] = 'Unknown'
            info['cuda_version'] = 'Unknown'
    
    return info

def create_synthetic_dem(size=1024, temp_dir=None):
    """Create a synthetic DEM for benchmarking"""
    import numpy as np
    from osgeo import gdal, osr
    
    # Create a temporary directory if one wasn't provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        cleanup_temp_dir = True
    else:
        cleanup_temp_dir = False
    
    dem_path = os.path.join(temp_dir, f"benchmark_dem_{size}.tif")
    
    # Create a synthetic DEM with random terrain
    dem_data = np.zeros((size, size), dtype=np.float32)
    
    # Add some terrain features (hills, valleys)
    x, y = np.meshgrid(np.linspace(0, 10, size), np.linspace(0, 10, size))
    dem_data += 100 + 50 * np.sin(x) * np.cos(y * 0.8)
    
    # Add some random noise
    np.random.seed(42)  # For reproducibility
    dem_data += np.random.normal(0, 2, size=(size, size))
    
    # Add a central mountain range
    for i in range(size):
        for j in range(size):
            dist = min(abs(j - size//2), size//4)
            ridge_height = 200 * (1 - dist / (size//4))
            if ridge_height > 0:
                dem_data[i, j] += ridge_height
    
    # Create a new GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    dem_file = driver.Create(dem_path, size, size, 1, gdal.GDT_Float32)
    
    # Set geotransform and projection (10m resolution)
    dem_file.SetGeoTransform((0, 10, 0, 0, 0, 10))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    dem_file.SetProjection(srs.ExportToWkt())
    
    # Write data
    dem_file.GetRasterBand(1).WriteArray(dem_data)
    dem_file.GetRasterBand(1).SetNoDataValue(-9999)
    
    # Close file
    dem_file = None
    
    return dem_path, cleanup_temp_dir, temp_dir

def create_synthetic_point_cloud(num_points=1000000, temp_dir=None):
    """Create a synthetic point cloud for benchmarking"""
    import numpy as np
    import struct
    
    # Create a temporary directory if one wasn't provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
        cleanup_temp_dir = True
    else:
        cleanup_temp_dir = False
    
    pc_path = os.path.join(temp_dir, f"benchmark_pc_{num_points}.bin")
    
    # Generate random points
    np.random.seed(42)  # For reproducibility
    points = np.random.rand(num_points, 3)
    points[:, 0] *= 1000  # X range [0, 1000]
    points[:, 1] *= 1000  # Y range [0, 1000]
    points[:, 2] *= 100   # Z range [0, 100]
    
    # Add classification data (random classes 0-9)
    classifications = np.random.randint(0, 10, num_points, dtype=np.uint8)
    
    # Write to a simple binary format
    with open(pc_path, 'wb') as f:
        # Write header
        f.write(struct.pack('<I', num_points))
        
        # Write points with classification
        for i in range(num_points):
            f.write(struct.pack('<fffB', 
                                points[i, 0], points[i, 1], points[i, 2], 
                                classifications[i]))
    
    return pc_path, cleanup_temp_dir, temp_dir

def benchmark_dem_processing(dem_path, results_dir, device_id=0):
    """Benchmark DEM processing operations"""
    print(f"\nBenchmarking DEM processing on device {device_id}...")
    
    # Dictionary to store results
    results = {}
    
    # Initialize DEM processor
    dem_proc = DEMProcessor(dem_path, device_id)
    width, height = dem_proc.get_dimensions()
    print(f"DEM dimensions: {width}x{height}")
    
    # Benchmark viewshed computation
    observer_point = (width/2, height/2)  # Center point
    observer_height = 10.0
    radius = width / 4
    
    with timer("Viewshed computation") as elapsed:
        viewshed = dem_proc.compute_viewshed(observer_point, observer_height, radius)
    results['viewshed'] = elapsed
    
    # Benchmark slope computation
    with timer("Slope computation") as elapsed:
        slope = dem_proc.compute_slope()
    results['slope'] = elapsed
    
    # Benchmark aspect computation
    with timer("Aspect computation") as elapsed:
        aspect = dem_proc.compute_aspect()
    results['aspect'] = elapsed
    
    # Benchmark hillshade computation
    with timer("Hillshade computation") as elapsed:
        hillshade = dem_proc.compute_hillshade(315, 45)
    results['hillshade'] = elapsed
    
    # Benchmark statistics computation
    with timer("Statistics computation") as elapsed:
        stats = dem_proc.compute_statistics()
    results['statistics'] = elapsed
    
    # Benchmark flow direction computation
    with timer("Flow direction computation") as elapsed:
        flow_dir = dem_proc.compute_flow_direction()
    results['flow_direction'] = elapsed
    
    # Benchmark flow accumulation computation
    with timer("Flow accumulation computation") as elapsed:
        flow_acc = dem_proc.compute_flow_accumulation(flow_dir)
    results['flow_accumulation'] = elapsed
    
    # Benchmark cost path computation
    start_point = (width/4, height/4)
    end_point = (3*width/4, 3*height/4)
    
    with timer("Cost path computation") as elapsed:
        path = dem_proc.compute_cost_path(start_point, end_point)
    results['cost_path'] = elapsed
    
    # Benchmark resampling
    with timer("Resampling (50%)") as elapsed:
        resampled = dem_proc.resample(0.5)
    results['resample_50'] = elapsed
    
    return results

def benchmark_point_cloud_processing(pc_path, results_dir, device_id=0):
    """Benchmark point cloud processing operations"""
    print(f"\nBenchmarking point cloud processing on device {device_id}...")
    
    # Dictionary to store results
    results = {}
    
    # Initialize point cloud
    try:
        with timer("Point cloud loading") as elapsed:
            pc = PointCloud(pc_path, device_id)
        results['loading'] = elapsed
        
        # Get point count
        point_count = pc.get_point_count()
        print(f"Point cloud size: {point_count} points")
        
        # Benchmark classification
        with timer("Point classification") as elapsed:
            classified_pc = pc.classify_points()
        results['classification'] = elapsed
        
        # Benchmark filtering
        with timer("Height filtering") as elapsed:
            filtered_pc = pc.filter_by_height(min_height=20.0)
        results['filtering'] = elapsed
        
        # Benchmark DEM creation
        with timer("DEM creation") as elapsed:
            dem = pc.create_dem(resolution=1.0)
        results['dem_creation'] = elapsed
        
        # Benchmark DSM creation
        with timer("DSM creation") as elapsed:
            dsm = pc.create_dsm(resolution=1.0)
        results['dsm_creation'] = elapsed
        
        # Benchmark normal computation
        with timer("Normal computation") as elapsed:
            pc_with_normals = pc.compute_normals(radius=2.0)
        results['normal_computation'] = elapsed
        
        # Benchmark downsampling
        with timer("Downsampling") as elapsed:
            downsampled = pc.downsample(method="voxel", voxel_size=1.0)
        results['downsampling'] = elapsed
        
        # Benchmark segmentation
        with timer("Segmentation") as elapsed:
            segmented = pc.segment(distance_threshold=1.0, min_cluster_size=50)
        results['segmentation'] = elapsed
        
        # Benchmark feature extraction
        with timer("Building extraction") as elapsed:
            buildings = pc.extract_buildings()
        results['building_extraction'] = elapsed
        
    except Exception as e:
        print(f"Error during point cloud benchmarking: {e}")
        # If the binary format is not supported or other errors occur, return partial results
        print("Returning partial results")
    
    return results

def run_benchmarks(args):
    """Run all benchmarks"""
    # Create results directory if it doesn't exist
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # Get system information
    system_info = get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Create temporary directory for synthetic data
    temp_dir = tempfile.mkdtemp()
    
    # Create synthetic data
    dem_path, _, _ = create_synthetic_dem(args.dem_size, temp_dir)
    pc_path, _, _ = create_synthetic_point_cloud(args.pc_size, temp_dir)
    
    # Dictionary to store all results
    all_results = {
        'system_info': system_info,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'dem': {},
        'point_cloud': {}
    }
    
    # Run benchmarks on CPU
    if not args.gpu_only:
        print("\n=== Running CPU benchmarks ===")
        
        # Force CPU execution
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # DEM benchmarks
        all_results['dem']['cpu'] = benchmark_dem_processing(dem_path, args.results_dir, device_id=-1)
        
        # Point cloud benchmarks
        all_results['point_cloud']['cpu'] = benchmark_point_cloud_processing(pc_path, args.results_dir, device_id=-1)
    
    # Run benchmarks on GPU if available
    if not args.cpu_only:
        try:
            # Check if CUDA is available
            cuda_available = False
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                num_gpus = torch.cuda.device_count() if cuda_available else 0
            except ImportError:
                try:
                    from geospatial import _geospatial
                    cuda_available = _geospatial.is_cuda_available()
                    num_gpus = 1 if cuda_available else 0
                except (ImportError, AttributeError):
                    num_gpus = 0
            
            if cuda_available and num_gpus > 0:
                print(f"\n=== Running GPU benchmarks (found {num_gpus} GPUs) ===")
                
                # Set CUDA device environment variable
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                
                # Run benchmarks on each GPU
                for device_id in range(num_gpus):
                    print(f"\nRunning benchmarks on GPU {device_id}")
                    
                    # DEM benchmarks
                    all_results['dem'][f'gpu_{device_id}'] = benchmark_dem_processing(
                        dem_path, args.results_dir, device_id=device_id)
                    
                    # Point cloud benchmarks
                    all_results['point_cloud'][f'gpu_{device_id}'] = benchmark_point_cloud_processing(
                        pc_path, args.results_dir, device_id=device_id)
            else:
                print("No CUDA GPU available. Skipping GPU benchmarks.")
        except Exception as e:
            print(f"Error during GPU benchmarks: {e}")
    
    # Save results to JSON file
    result_file = os.path.join(args.results_dir, f"geospatial_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nBenchmark results saved to {result_file}")
    
    # Generate charts
    if args.charts:
        generate_charts(all_results, args.results_dir)
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    return all_results

def generate_charts(results, results_dir):
    """Generate charts from benchmark results"""
    print("\nGenerating benchmark charts...")
    
    # Create charts directory if it doesn't exist
    charts_dir = os.path.join(results_dir, 'charts')
    if not os.path.exists(charts_dir):
        os.makedirs(charts_dir)
    
    # Generate DEM processing charts
    if 'dem' in results:
        generate_operation_chart(results['dem'], 'DEM Processing', charts_dir)
    
    # Generate point cloud processing charts
    if 'point_cloud' in results:
        generate_operation_chart(results['point_cloud'], 'Point Cloud Processing', charts_dir)
    
    print(f"Charts saved to {charts_dir}")

def generate_operation_chart(operation_results, title, charts_dir):
    """Generate chart for a specific operation type (DEM or point cloud)"""
    # Get list of platforms (cpu, gpu_0, etc.)
    platforms = list(operation_results.keys())
    
    # Get list of operations
    operations = set()
    for platform in platforms:
        operations.update(operation_results[platform].keys())
    operations = sorted(operations)
    
    # Create data for the chart
    platform_labels = []
    operation_data = {op: [] for op in operations}
    
    for platform in platforms:
        if platform == 'cpu':
            platform_labels.append('CPU')
        else:
            platform_labels.append(f'GPU {platform.split("_")[1]}')
        
        # Get data for each operation
        for op in operations:
            if op in operation_results[platform]:
                operation_data[op].append(operation_results[platform][op])
            else:
                operation_data[op].append(0)
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Set up the bar chart
    x = np.arange(len(platform_labels))
    width = 0.8 / len(operations)
    
    # Plot each operation
    for i, op in enumerate(operations):
        plt.bar(x + i * width - 0.4 + width/2, operation_data[op], width, label=op.replace('_', ' ').title())
    
    # Add chart elements
    plt.xlabel('Platform')
    plt.ylabel('Time (seconds)')
    plt.title(f'{title} Benchmark')
    plt.xticks(x, platform_labels)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Use log scale for better visualization if there's a large range of values
    if any(v > 10 * min([v for v in [item for sublist in operation_data.values() for item in sublist] if v > 0]) 
          for v in [item for sublist in operation_data.values() for item in sublist]):
        plt.yscale('log')
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    
    # Save the chart
    chart_file = os.path.join(charts_dir, f"{title.lower().replace(' ', '_')}_benchmark.png")
    plt.savefig(chart_file)
    plt.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Benchmark Geospatial Analysis workload')
    
    parser.add_argument('--dem-size', type=int, default=1024,
                       help='Size of the synthetic DEM (default: 1024)')
    parser.add_argument('--pc-size', type=int, default=1000000,
                       help='Number of points in the synthetic point cloud (default: 1,000,000)')
    parser.add_argument('--results-dir', type=str, default='benchmark_results',
                       help='Directory to save benchmark results (default: benchmark_results)')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Run benchmarks only on CPU')
    parser.add_argument('--gpu-only', action='store_true',
                       help='Run benchmarks only on GPU')
    parser.add_argument('--charts', action='store_true',
                       help='Generate charts from benchmark results')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_benchmarks(args)