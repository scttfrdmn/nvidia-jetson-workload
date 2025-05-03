# Common Usage Examples

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This document provides examples of common usage scenarios for the GPU-accelerated scientific workloads package.

## Table of Contents

1. [N-body Simulation Examples](#n-body-simulation-examples)
2. [Molecular Dynamics Examples](#molecular-dynamics-examples)
3. [Weather Simulation Examples](#weather-simulation-examples)
4. [Medical Imaging Examples](#medical-imaging-examples)
5. [Benchmarking Examples](#benchmarking-examples)
6. [Deployment Examples](#deployment-examples)
7. [Integration Examples](#integration-examples)

## N-body Simulation Examples

### Example 1: Solar System Simulation

Create a solar system simulation with planets and simulate their movement over time:

```python
from nbody_sim.simulation import Simulation
from nbody_sim.integrator import IntegratorType
import matplotlib.pyplot as plt

# Create simulation
sim = Simulation()
sim.initialize(
    system_type="solar_system",
    integrator_type=IntegratorType.LEAPFROG,
    device_id=0  # Use first GPU
)

# Run simulation for 1 year with 1-day timesteps
for day in range(365):
    sim.step(86400)  # Step size: 1 day in seconds
    
    # Print progress every 30 days
    if day % 30 == 0:
        print(f"Simulated {day} days")
        
        # Calculate total energy
        kinetic = sim.calculate_total_kinetic_energy()
        potential = sim.calculate_total_potential_energy()
        total = kinetic + potential
        print(f"Energy: Kinetic={kinetic:.2e}, Potential={potential:.2e}, Total={total:.2e}")

# Visualize final state
from nbody_sim.visualization import plot_orbits
plt.figure(figsize=(10, 10))
plot_orbits(sim)
plt.savefig("solar_system.png")
plt.show()
```

### Example 2: Galaxy Collision

Simulate the collision of two galaxies:

```python
from nbody_sim.simulation import Simulation
from nbody_sim.galaxy import create_spiral_galaxy
import numpy as np

# Create simulation
sim = Simulation()

# Create two spiral galaxies with different positions and velocities
galaxy1 = create_spiral_galaxy(
    num_particles=5000, 
    radius=10000, 
    position=np.array([-15000, 0, 0]),
    velocity=np.array([20, 5, 0])
)

galaxy2 = create_spiral_galaxy(
    num_particles=5000,
    radius=8000,
    position=np.array([15000, 0, 0]),
    velocity=np.array([-20, -5, 0])
)

# Initialize simulation with both galaxies
sim.initialize_custom(
    positions=np.vstack([galaxy1["positions"], galaxy2["positions"]]),
    velocities=np.vstack([galaxy1["velocities"], galaxy2["velocities"]]),
    masses=np.concatenate([galaxy1["masses"], galaxy2["masses"]])
)

# Create output directory for frames
import os
os.makedirs("galaxy_collision", exist_ok=True)

# Run simulation and save frames
from nbody_sim.visualization import save_particle_plot
for step in range(200):
    sim.step(1.0)
    
    if step % 5 == 0:
        print(f"Step {step}/200")
        save_particle_plot(
            sim, 
            f"galaxy_collision/frame_{step:03d}.png",
            view_distance=50000,
            alpha=0.5,
            point_size=0.5
        )

# Create animation (requires ffmpeg)
os.system("ffmpeg -framerate 10 -i galaxy_collision/frame_%03d.png -c:v libx264 -pix_fmt yuv420p galaxy_collision.mp4")
```

## Molecular Dynamics Examples

### Example 1: Lennard-Jones Fluid

Simulate a Lennard-Jones fluid and calculate thermodynamic properties:

```python
import molecular_dynamics as md
import numpy as np
import matplotlib.pyplot as plt

# Create simulation
simulation = md.Simulation()

# Initialize a Lennard-Jones fluid
simulation.initialize(
    num_atoms=1000,
    box_size=20.0,
    density=0.8,
    temperature=1.0,
    force_field="lennard_jones"
)

# Equilibration
print("Equilibrating system...")
for step in range(1000):
    simulation.step(0.005)
    
    if step % 100 == 0:
        kinetic = simulation.calculate_kinetic_energy()
        potential = simulation.calculate_potential_energy()
        temp = simulation.calculate_temperature()
        print(f"Step {step}: T={temp:.2f}, E_k={kinetic:.2f}, E_p={potential:.2f}")

# Production run with data collection
temperatures = []
pressures = []
energies = []

print("Production run...")
for step in range(5000):
    simulation.step(0.005)
    
    if step % 10 == 0:
        # Collect data every 10 steps
        temperatures.append(simulation.calculate_temperature())
        pressures.append(simulation.calculate_pressure())
        energies.append(simulation.calculate_total_energy())

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(temperatures)
plt.xlabel("Sample")
plt.ylabel("Temperature")
plt.title("Temperature Fluctuations")

plt.subplot(1, 3, 2)
plt.plot(pressures)
plt.xlabel("Sample")
plt.ylabel("Pressure")
plt.title("Pressure Fluctuations")

plt.subplot(1, 3, 3)
plt.plot(energies)
plt.xlabel("Sample")
plt.ylabel("Total Energy")
plt.title("Energy Conservation")

plt.tight_layout()
plt.savefig("lj_fluid_properties.png")
plt.show()

# Calculate averages and standard deviations
print(f"Average temperature: {np.mean(temperatures):.4f} ± {np.std(temperatures):.4f}")
print(f"Average pressure: {np.mean(pressures):.4f} ± {np.std(pressures):.4f}")
print(f"Energy drift: {(energies[-1] - energies[0]) / energies[0] * 100:.6f}%")
```

### Example 2: Protein in Water

Simulate a protein in water with the CHARMM force field:

```python
import molecular_dynamics as md
from molecular_dynamics.utils import load_pdb, create_water_box

# Load protein from PDB file
protein = load_pdb("protein.pdb")

# Create water box around protein
system = create_water_box(
    protein=protein,
    padding=10.0,  # Å of water padding around protein
    water_model="tip3p"
)

# Create simulation
simulation = md.Simulation()

# Initialize with CHARMM force field
simulation.initialize_from_system(
    system=system,
    force_field="charmm27",
    cutoff=12.0,
    use_pme=True  # Particle Mesh Ewald for long-range electrostatics
)

# Add thermostat for constant temperature
simulation.add_thermostat(
    thermostat_type="nose_hoover",
    temperature=310.0,  # 310K (body temperature)
    coupling_constant=1.0
)

# Minimize energy to resolve bad contacts
print("Energy minimization...")
simulation.minimize(max_iterations=1000, tolerance=10.0)

# Heat system gradually
print("Heating system...")
for temp in range(0, 311, 20):
    simulation.set_thermostat_temperature(float(temp))
    for _ in range(50):
        simulation.step(0.001)
    print(f"Temperature: {temp}K")

# Equilibration
print("Equilibrating system...")
for step in range(5000):
    simulation.step(0.002)
    
    if step % 500 == 0:
        temp = simulation.calculate_temperature()
        energy = simulation.calculate_total_energy()
        print(f"Step {step}: T={temp:.2f}K, E={energy:.2f}")

# Production run with trajectory recording
print("Production run...")
simulation.enable_trajectory_recording("protein_traj.dcd", frame_interval=100)

for step in range(50000):
    simulation.step(0.002)
    
    if step % 1000 == 0:
        print(f"Production step {step}/50000")

# Save final state
simulation.save_state("protein_final.pdb")
```

## Weather Simulation Examples

### Example 1: Shallow Water Model

Simulate the shallow water equations on a periodic domain:

```python
import weather_sim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create simulation
sim = weather_sim.WeatherSimulation()

# Initialize with Shallow Water model
grid_size = 256
sim.initialize(
    model="shallow_water",
    grid_size=grid_size,
    dx=100.0,  # Grid spacing in meters
    dy=100.0,
    gravity=9.81,
    coriolis_parameter=1e-4
)

# Set initial conditions with a Gaussian bump in water height
x, y = np.meshgrid(
    np.linspace(0, 1, grid_size),
    np.linspace(0, 1, grid_size)
)
center_x, center_y = 0.5, 0.5
sigma = 0.05
height = 1.0 + 0.5 * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

# Set initial height field and zero initial velocities
sim.set_height_field(height)
sim.set_velocity_field_u(np.zeros((grid_size, grid_size)))
sim.set_velocity_field_v(np.zeros((grid_size, grid_size)))

# Create figure for animation
fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(x, y, height, levels=20, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
title = ax.set_title('Shallow Water Simulation: t = 0.0')
fig.colorbar(contour, ax=ax, label='Water Height')

# Update function for animation
def update(frame):
    # Step simulation
    for _ in range(10):  # Take 10 substeps per frame
        sim.step(0.1)  # dt = 0.1 seconds
    
    # Get current height field
    h = sim.get_height_field()
    
    # Update plot
    ax.clear()
    contour = ax.contourf(x, y, h, levels=20, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Shallow Water Simulation: t = {frame * 1.0:.1f} s')
    return contour,

# Create animation
anim = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
anim.save('shallow_water.gif', writer='pillow', fps=15)

# Show final state
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.contourf(x, y, sim.get_height_field(), levels=20, cmap='viridis')
plt.colorbar(label='Height')
plt.title('Water Height')

plt.subplot(1, 3, 2)
plt.contourf(x, y, sim.get_velocity_field_u(), levels=20, cmap='RdBu_r')
plt.colorbar(label='U')
plt.title('Zonal Velocity')

plt.subplot(1, 3, 3)
plt.contourf(x, y, sim.get_velocity_field_v(), levels=20, cmap='RdBu_r')
plt.colorbar(label='V')
plt.title('Meridional Velocity')

plt.tight_layout()
plt.savefig('shallow_water_final.png')
plt.show()
```

### Example 2: Weather Forecast Simulation

Run a primitive equations model to forecast weather over a region:

```python
import weather_sim
import numpy as np
import matplotlib.pyplot as plt
from weather_sim.visualization import create_weather_map

# Load real topography and initial conditions
topo = np.load('topography.npy')
initial_temperature = np.load('temperature.npy')
initial_pressure = np.load('pressure.npy')
initial_humidity = np.load('humidity.npy')

# Create simulation
sim = weather_sim.WeatherSimulation()

# Initialize with Primitive Equations model
sim.initialize(
    model="primitive",
    grid_size_x=topo.shape[1],
    grid_size_y=topo.shape[0],
    vertical_levels=20,
    dx=10000.0,  # 10 km grid spacing
    dy=10000.0,
    dt_max=60.0,  # Maximum time step: 60 seconds
    latitude_center=45.0,  # Center latitude (degrees)
    longitude_center=-100.0  # Center longitude (degrees)
)

# Set topography and initial conditions
sim.set_topography(topo)
sim.set_temperature_field(initial_temperature)
sim.set_pressure_field(initial_pressure)
sim.set_humidity_field(initial_humidity)

# Set up output manager to save results
output_dir = "forecast_output"
sim.setup_output_manager(
    output_dir=output_dir,
    output_interval=3600,  # Save output every hour
    output_variables=["temperature", "pressure", "humidity", "precipitation", "wind"]
)

# Run 48-hour forecast
total_hours = 48
seconds_per_hour = 3600
steps_per_report = 6

print(f"Running {total_hours} hour forecast...")
for hour in range(total_hours):
    for step in range(steps_per_report):
        # Run 10-minute chunks (600 seconds)
        sim.step_auto(600)
    
    # Report progress
    print(f"Forecast hour {hour+1}/{total_hours} completed")
    
    # Generate and save visualization for this hour
    if (hour + 1) % 6 == 0:  # Every 6 hours
        surface_temp = sim.get_temperature_field(level=0)
        surface_pressure = sim.get_pressure_field(level=0)
        precipitation = sim.get_precipitation_field()
        
        # Create weather map
        fig = create_weather_map(
            sim, 
            variables=["temperature", "pressure", "precipitation", "wind"],
            title=f"Weather Forecast: +{hour+1}h"
        )
        fig.savefig(f"{output_dir}/forecast_{hour+1:02d}h.png", dpi=150)
        plt.close(fig)

print(f"Forecast complete. Output saved to {output_dir}")

# Create forecast animation (requires ffmpeg)
import os
os.system(f"ffmpeg -framerate 4 -pattern_type glob -i '{output_dir}/forecast_*.png' -c:v libx264 -pix_fmt yuv420p forecast_animation.mp4")
```

## Medical Imaging Examples

### Example 1: CT Reconstruction

Reconstruct a CT image from sinogram data:

```python
import medical_imaging as mi
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Adaptive Kernel Manager for GPU detection
akm = mi.AdaptiveKernelManager.get_instance()
if akm.initialize():
    print("Initialized GPU acceleration")
    print(f"Device: {akm.get_device_capabilities().get_device_name()}")
    print(f"Compute capability: {akm.get_device_capabilities().get_compute_capability()}")
else:
    print("GPU not available, using CPU fallback")

# Create phantom image or load real data
use_real_data = False
if use_real_data:
    # Load real sinogram data
    sinogram = mi.load_image("patient_sinogram.raw")
    angles = np.load("projection_angles.npy")
else:
    # Create Shepp-Logan phantom for testing
    phantom_size = 512
    phantom = mi.create_phantom(phantom_size, phantom_type="shepp_logan")
    
    # Display phantom
    plt.figure(figsize=(7, 7))
    plt.imshow(mi.to_numpy(phantom), cmap='gray')
    plt.title("Original Phantom")
    plt.colorbar()
    plt.savefig("original_phantom.png")
    
    # Create projections (forward projection)
    num_angles = 180
    angles = np.linspace(0, np.pi, num_angles, dtype=np.float32)
    
    # Create CT projector
    projector = mi.CTProjector()
    projector.set_image_dimensions(phantom_size, phantom_size)
    projector.set_angles(angles)
    
    # Forward project to get sinogram
    print("Creating sinogram...")
    sinogram = projector.forward_project(phantom)
    
    # Display sinogram
    plt.figure(figsize=(12, 5))
    plt.imshow(mi.to_numpy(sinogram), cmap='gray', aspect='auto')
    plt.title("Sinogram")
    plt.xlabel("Detector Position")
    plt.ylabel("Angle")
    plt.colorbar()
    plt.savefig("sinogram.png")

# Create CT reconstructor
reconstructor = mi.CTReconstructor()
reconstructor.set_image_dimensions(sinogram.get_width(), sinogram.get_width())
reconstructor.set_angles(angles)

# Filtered Backprojection reconstruction
print("Performing filtered backprojection reconstruction...")
reconstructor.set_method(mi.ReconstructionMethod.FilteredBackProjection)
reconstructor.set_filter(mi.FilterType.RamLak)
fbp_result = reconstructor.process(sinogram)

# Iterative reconstruction (SIRT)
print("Performing iterative SIRT reconstruction...")
reconstructor.set_method(mi.ReconstructionMethod.SIRT)
reconstructor.set_iterations(50)
iterative_result = reconstructor.process(sinogram)

# Display results
plt.figure(figsize=(15, 5))

if use_real_data:
    plt.subplot(1, 2, 1)
    plt.imshow(mi.to_numpy(fbp_result), cmap='gray')
    plt.title("Filtered Backprojection")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(mi.to_numpy(iterative_result), cmap='gray')
    plt.title("SIRT (50 iterations)")
    plt.colorbar()
else:
    # Also show original for comparison
    plt.subplot(1, 3, 1)
    plt.imshow(mi.to_numpy(phantom), cmap='gray')
    plt.title("Original")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(mi.to_numpy(fbp_result), cmap='gray')
    plt.title("Filtered Backprojection")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(mi.to_numpy(iterative_result), cmap='gray')
    plt.title("SIRT (50 iterations)")
    plt.colorbar()

plt.tight_layout()
plt.savefig("reconstruction_comparison.png")
plt.show()

# Calculate error metrics (if using phantom)
if not use_real_data:
    # Convert to numpy arrays
    phantom_np = mi.to_numpy(phantom)
    fbp_np = mi.to_numpy(fbp_result)
    iterative_np = mi.to_numpy(iterative_result)
    
    # Normalize images to [0, 1] range
    phantom_np = (phantom_np - phantom_np.min()) / (phantom_np.max() - phantom_np.min())
    fbp_np = (fbp_np - fbp_np.min()) / (fbp_np.max() - fbp_np.min())
    iterative_np = (iterative_np - iterative_np.min()) / (iterative_np.max() - iterative_np.min())
    
    # Calculate mean squared error
    fbp_mse = ((phantom_np - fbp_np) ** 2).mean()
    iterative_mse = ((phantom_np - iterative_np) ** 2).mean()
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    fbp_psnr = 10 * np.log10(1.0 / fbp_mse)
    iterative_psnr = 10 * np.log10(1.0 / iterative_mse)
    
    print(f"FBP: MSE = {fbp_mse:.6f}, PSNR = {fbp_psnr:.2f} dB")
    print(f"SIRT: MSE = {iterative_mse:.6f}, PSNR = {iterative_psnr:.2f} dB")
```

### Example 2: Image Segmentation Pipeline

Segment specific structures in medical images:

```python
import medical_imaging as mi
import numpy as np
import matplotlib.pyplot as plt
from medical_imaging.visualization import plot_overlay, plot_contours

# Load medical image (e.g., MRI or CT slice)
image = mi.load_image("medical_scan.dcm")

# Convert to float and normalize
image_processor = mi.ImageProcessor()
image = image_processor.convert_to_float(image)
image = image_processor.normalize(image, 0, 1)

# Denoise the image
denoised = image_processor.apply_filter(image, mi.FilterType.Gaussian, kernel_size=3, sigma=1.0)

# Edge enhancement
edges = image_processor.detect_edges(denoised, mi.EdgeDetectorType.Sobel)

# Initialize segmenter
segmenter = mi.Segmenter()

# Method 1: Thresholding
segmenter.set_method(mi.SegmentationMethod.Thresholding)
segmenter.set_threshold(0.45)  # Threshold value determined empirically
thresholding_result = segmenter.process(denoised)

# Method 2: Watershed segmentation
segmenter.set_method(mi.SegmentationMethod.Watershed)
markers = np.zeros_like(mi.to_numpy(denoised), dtype=np.int32)
markers[100:150, 100:150] = 1  # Foreground marker (adjust based on image)
markers[0:10, 0:10] = 2  # Background marker
segmenter.set_markers(mi.from_numpy(markers))
watershed_result = segmenter.process(denoised)

# Method 3: Level Set segmentation
segmenter.set_method(mi.SegmentationMethod.LevelSet)
init_mask = np.zeros_like(mi.to_numpy(denoised), dtype=np.float32)
init_mask[200:300, 200:300] = 1.0  # Initial contour (adjust based on image)
segmenter.set_initial_contour(mi.from_numpy(init_mask))
segmenter.set_iterations(100)
level_set_result = segmenter.process(denoised)

# Display results
plt.figure(figsize=(20, 10))

# Original and preprocessed images
plt.subplot(2, 4, 1)
plt.imshow(mi.to_numpy(image), cmap='gray')
plt.title("Original Image")
plt.colorbar()

plt.subplot(2, 4, 2)
plt.imshow(mi.to_numpy(denoised), cmap='gray')
plt.title("Denoised Image")
plt.colorbar()

plt.subplot(2, 4, 3)
plt.imshow(mi.to_numpy(edges), cmap='gray')
plt.title("Edge Detection")
plt.colorbar()

# Segmentation results
plt.subplot(2, 4, 5)
plot_overlay(mi.to_numpy(image), mi.to_numpy(thresholding_result) > 0.5, alpha=0.3)
plt.title("Thresholding Segmentation")

plt.subplot(2, 4, 6)
plot_overlay(mi.to_numpy(image), mi.to_numpy(watershed_result) == 1, alpha=0.3)
plt.title("Watershed Segmentation")

plt.subplot(2, 4, 7)
plot_overlay(mi.to_numpy(image), mi.to_numpy(level_set_result) > 0.5, alpha=0.3)
plt.title("Level Set Segmentation")

# Contour comparison
plt.subplot(2, 4, 8)
plt.imshow(mi.to_numpy(image), cmap='gray')
plot_contours(mi.to_numpy(thresholding_result) > 0.5, color='red', linewidth=1, label='Thresholding')
plot_contours(mi.to_numpy(watershed_result) == 1, color='green', linewidth=1, label='Watershed')
plot_contours(mi.to_numpy(level_set_result) > 0.5, color='blue', linewidth=1, label='Level Set')
plt.title("Segmentation Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("segmentation_comparison.png", dpi=150)
plt.show()

# Volume calculation (assuming pixel spacing information is available)
pixel_spacing_x = 0.5  # mm
pixel_spacing_y = 0.5  # mm
pixel_area = pixel_spacing_x * pixel_spacing_y  # mm^2

threshold_volume = np.sum(mi.to_numpy(thresholding_result) > 0.5) * pixel_area
watershed_volume = np.sum(mi.to_numpy(watershed_result) == 1) * pixel_area
level_set_volume = np.sum(mi.to_numpy(level_set_result) > 0.5) * pixel_area

print(f"Thresholding segmentation area: {threshold_volume:.2f} mm²")
print(f"Watershed segmentation area: {watershed_volume:.2f} mm²")
print(f"Level Set segmentation area: {level_set_volume:.2f} mm²")
```

## Benchmarking Examples

### Example 1: Cross-Workload Performance Comparison

Compare performance of all workloads across different hardware:

```bash
#!/bin/bash

# Define output directory
OUTPUT_DIR="benchmark_results"
mkdir -p $OUTPUT_DIR

# Define device IDs to test (can be multiple GPUs)
DEVICES=(0)  # Use device 0

# Define workloads to benchmark
WORKLOADS=("--nbody" "--md" "--weather" "--medical")

# Run benchmarks for each device
for DEVICE in "${DEVICES[@]}"; do
    DEVICE_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $DEVICE)
    DEVICE_DIR="$OUTPUT_DIR/device_${DEVICE}_${DEVICE_NAME// /_}"
    mkdir -p "$DEVICE_DIR"
    
    echo "Running benchmarks on device $DEVICE: $DEVICE_NAME"
    
    # Run all workloads
    ./benchmark/scripts/run_benchmarks.sh --device $DEVICE --all --output "$DEVICE_DIR"
    
    # Run individual workloads with customized parameters
    ./benchmark/scripts/run_benchmarks.sh --device $DEVICE --nbody --nbody-particles 50000 --output "$DEVICE_DIR/nbody_large"
    ./benchmark/scripts/run_benchmarks.sh --device $DEVICE --weather --weather-grid 1024 --output "$DEVICE_DIR/weather_large"
done

# Generate comparative report
python -m benchmark.benchmark_suite --report --directory "$OUTPUT_DIR"

echo "Benchmarks complete. Results saved to $OUTPUT_DIR"
echo "Comparative report: $OUTPUT_DIR/benchmark_report.html"
```

### Example 2: Performance Scaling Analysis

Analyze performance scaling with problem size:

```python
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Create output directory
output_dir = "scaling_analysis"
os.makedirs(output_dir, exist_ok=True)

# Define problem sizes to test
nbody_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
weather_grids = [128, 256, 512, 768, 1024, 1536]
medical_sizes = [128, 256, 512, 768, 1024, 1536]

# Run nbody benchmarks with different problem sizes
nbody_times = []
for size in nbody_sizes:
    print(f"Running N-body benchmark with {size} particles...")
    result_dir = f"{output_dir}/nbody_{size}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Run benchmark
    subprocess.run([
        "./benchmark/scripts/run_benchmarks.sh",
        "--nbody",
        f"--nbody-particles={size}",
        f"--output={result_dir}"
    ])
    
    # Parse result
    result_files = [f for f in os.listdir(result_dir) if f.endswith(".json")]
    if result_files:
        with open(f"{result_dir}/{result_files[0]}", "r") as f:
            data = json.load(f)
            nbody_times.append(data["execution_time"])
    else:
        nbody_times.append(None)

# Run weather benchmarks with different grid sizes
weather_times = []
for grid in weather_grids:
    print(f"Running Weather benchmark with {grid}x{grid} grid...")
    result_dir = f"{output_dir}/weather_{grid}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Run benchmark
    subprocess.run([
        "./benchmark/scripts/run_benchmarks.sh",
        "--weather",
        f"--weather-grid={grid}",
        f"--output={result_dir}"
    ])
    
    # Parse result
    result_files = [f for f in os.listdir(result_dir) if f.endswith(".json")]
    if result_files:
        with open(f"{result_dir}/{result_files[0]}", "r") as f:
            data = json.load(f)
            weather_times.append(data["execution_time"])
    else:
        weather_times.append(None)

# Run medical benchmarks with different image sizes
medical_times = []
for size in medical_sizes:
    print(f"Running Medical benchmark with {size}x{size} image...")
    result_dir = f"{output_dir}/medical_{size}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Run benchmark
    subprocess.run([
        "./benchmark/scripts/run_benchmarks.sh",
        "--medical",
        f"--medical-size={size}",
        f"--output={result_dir}"
    ])
    
    # Parse result
    result_files = [f for f in os.listdir(result_dir) if f.endswith(".json")]
    if result_files:
        with open(f"{result_dir}/{result_files[0]}", "r") as f:
            data = json.load(f)
            medical_times.append(data["execution_time"])
    else:
        medical_times.append(None)

# Plot scaling results
plt.figure(figsize=(15, 5))

# N-body scaling (O(n²))
plt.subplot(1, 3, 1)
plt.plot(nbody_sizes, nbody_times, 'o-', linewidth=2, label='Measured')
# Plot theoretical O(n²) scaling
if nbody_times[0] is not None:
    theoretical = [nbody_times[0] * (size/nbody_sizes[0])**2 for size in nbody_sizes]
    plt.plot(nbody_sizes, theoretical, '--', label='O(n²)')
plt.xlabel('Number of Particles')
plt.ylabel('Execution Time (s)')
plt.title('N-body Simulation Scaling')
plt.legend()
plt.grid(True)

# Weather scaling (roughly O(n³))
plt.subplot(1, 3, 2)
plt.plot(weather_grids, weather_times, 'o-', linewidth=2, label='Measured')
# Plot theoretical scaling
if weather_times[0] is not None:
    theoretical = [weather_times[0] * (grid/weather_grids[0])**3 for grid in weather_grids]
    plt.plot(weather_grids, theoretical, '--', label='O(n³)')
plt.xlabel('Grid Size')
plt.ylabel('Execution Time (s)')
plt.title('Weather Simulation Scaling')
plt.legend()
plt.grid(True)

# Medical imaging scaling
plt.subplot(1, 3, 3)
plt.plot(medical_sizes, medical_times, 'o-', linewidth=2, label='Measured')
# Plot theoretical scaling (roughly O(n²log(n)))
if medical_times[0] is not None:
    theoretical = [medical_times[0] * (size/medical_sizes[0])**2 * np.log(size)/np.log(medical_sizes[0]) for size in medical_sizes]
    plt.plot(medical_sizes, theoretical, '--', label='O(n²log(n))')
plt.xlabel('Image Size')
plt.ylabel('Execution Time (s)')
plt.title('Medical Imaging Scaling')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{output_dir}/scaling_analysis.png", dpi=150)
plt.show()

# Print summary
print("Performance Scaling Analysis:")
print("----------------------------")
print("N-body Simulation:")
for i, size in enumerate(nbody_sizes):
    print(f"  {size} particles: {nbody_times[i]:.2f}s")

print("\nWeather Simulation:")
for i, grid in enumerate(weather_grids):
    print(f"  {grid}x{grid} grid: {weather_times[i]:.2f}s")

print("\nMedical Imaging:")
for i, size in enumerate(medical_sizes):
    print(f"  {size}x{size} image: {medical_times[i]:.2f}s")
```

## Deployment Examples

### Example 1: Deploying to Jetson Cluster

Deploy all workloads to a cluster of Jetson devices:

```bash
#!/bin/bash

# Define Jetson nodes
NODES=("jetson1" "jetson2" "jetson3" "jetson4")

# Check SSH connectivity
for NODE in "${NODES[@]}"; do
    echo "Checking SSH connectivity to $NODE..."
    if ! ssh -q -o BatchMode=yes -o ConnectTimeout=5 ubuntu@$NODE exit; then
        echo "Error: Cannot connect to $NODE without password."
        echo "Setting up SSH keys..."
        ./scripts/setup-ssh-keys.sh ubuntu@$NODE
    fi
done

# Build all workloads on this machine
echo "Building all workloads..."
./build.sh

# Deploy to all nodes
echo "Deploying to all nodes..."
./scripts/deploy-all.sh --workloads all --nodes "${NODES[*]}" --user ubuntu --build

# Verify deployment
for NODE in "${NODES[@]}"; do
    echo "Verifying deployment on $NODE..."
    ssh ubuntu@$NODE "cd /opt/nvidia-jetson-workload && python3 -c 'import nbody_sim; print(\"N-body module loaded successfully\")'"
    ssh ubuntu@$NODE "cd /opt/nvidia-jetson-workload && python3 -c 'import medical_imaging as mi; print(\"Medical imaging module loaded successfully\")'"
done

# Run a quick benchmark on all nodes
echo "Running benchmarks on all nodes..."
for NODE in "${NODES[@]}"; do
    echo "Running benchmark on $NODE..."
    ssh ubuntu@$NODE "cd /opt/nvidia-jetson-workload && ./benchmark/scripts/run_benchmarks.sh --all --output ~/benchmark_results"
    
    # Copy results back
    echo "Copying results from $NODE..."
    mkdir -p "benchmark_results/$NODE"
    scp -r ubuntu@$NODE:~/benchmark_results/* "benchmark_results/$NODE/"
done

echo "Deployment and verification complete."
echo "Benchmark results saved to benchmark_results/"
```

### Example 2: Deploying to AWS with CI/CD

Set up GitHub Actions to automatically deploy to AWS instances:

```yaml
name: Deploy to AWS

on:
  push:
    branches:
      - main
  workflow_dispatch:
    inputs:
      environment:
        description: 'Environment to deploy to'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production
      workloads:
        description: 'Workloads to deploy (comma-separated)'
        required: false
        default: 'all'
        type: string

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}
      
      - name: Get instance IDs
        id: get-instances
        run: |
          # Get instance IDs based on environment
          if [[ "${{ github.event.inputs.environment || 'staging' }}" == "staging" ]]; then
            INSTANCE_IDS=$(aws ec2 describe-instances \
              --filters "Name=tag:Environment,Values=staging" "Name=instance-state-name,Values=running" \
              --query "Reservations[].Instances[].InstanceId" --output text)
          else
            INSTANCE_IDS=$(aws ec2 describe-instances \
              --filters "Name=tag:Environment,Values=production" "Name=instance-state-name,Values=running" \
              --query "Reservations[].Instances[].InstanceId" --output text)
          fi
          
          # Replace spaces with commas for deploy-all.sh
          INSTANCE_IDS=$(echo $INSTANCE_IDS | tr ' ' ',')
          echo "instance_ids=$INSTANCE_IDS" >> $GITHUB_OUTPUT
      
      - name: Build workloads
        run: |
          # Install build dependencies
          sudo apt-get update
          sudo apt-get install -y build-essential cmake libboost-all-dev
          
          # Build workloads
          ./build.sh
      
      - name: Deploy to AWS instances
        run: |
          # Set up SSH key
          mkdir -p ~/.ssh
          echo "${{ secrets.SSH_PRIVATE_KEY }}" > ~/.ssh/id_rsa
          chmod 600 ~/.ssh/id_rsa
          
          # Deploy workloads
          ./scripts/deploy-all.sh \
            --aws \
            --instances ${{ steps.get-instances.outputs.instance_ids }} \
            --workloads ${{ github.event.inputs.workloads || 'all' }} \
            --key ~/.ssh/id_rsa \
            --user ec2-user
      
      - name: Run validation tests
        run: |
          # Get first instance IP for validation
          INSTANCE_ID=$(echo "${{ steps.get-instances.outputs.instance_ids }}" | cut -d',' -f1)
          INSTANCE_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
            --query "Reservations[].Instances[].PublicIpAddress" --output text)
          
          # Run validation tests
          ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no ec2-user@$INSTANCE_IP \
            "cd /opt/nvidia-jetson-workload && python3 -m pytest -xvs tests/validation"
      
      - name: Notify deployment success
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          fields: repo,message,commit,author,action,eventName,ref,workflow
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
        if: success()
```

## Integration Examples

### Example 1: Integration with Jupyter Notebook

Use the workloads in a Jupyter Notebook for interactive exploration:

```python
# Install custom jupyter widget for visualization (if not already installed)
!pip install ipywidgets matplotlib numpy

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output

# Import our workload libraries
import nbody_sim
import medical_imaging as mi

# Create interactive N-body simulation
def nbody_interactive_demo():
    # Create output for visualization
    output = widgets.Output()
    
    # Create simulation object
    sim = nbody_sim.Simulation()
    
    # Create widgets for parameters
    system_type = widgets.Dropdown(
        options=['random', 'solar_system', 'galaxy'],
        value='solar_system',
        description='System Type:'
    )
    
    num_particles = widgets.IntSlider(
        value=1000,
        min=100,
        max=10000,
        step=100,
        description='Particles:'
    )
    
    integrator = widgets.Dropdown(
        options=['euler', 'leapfrog', 'verlet', 'rk4'],
        value='leapfrog',
        description='Integrator:'
    )
    
    dt = widgets.FloatSlider(
        value=0.01,
        min=0.001,
        max=0.1,
        step=0.001,
        description='Time Step:'
    )
    
    steps_per_frame = widgets.IntSlider(
        value=10,
        min=1,
        max=100,
        step=1,
        description='Steps/Frame:'
    )
    
    # Button to start/stop simulation
    running = False
    button = widgets.Button(description="Start Simulation")
    
    def on_button_click(b):
        nonlocal running
        if not running:
            # Start simulation
            running = True
            button.description = "Stop Simulation"
            sim.initialize(
                num_particles=num_particles.value,
                system_type=system_type.value,
                integrator_type=getattr(nbody_sim.IntegratorType, integrator.value.upper())
            )
            run_simulation()
        else:
            # Stop simulation
            running = False
            button.description = "Start Simulation"
    
    button.on_click(on_button_click)
    
    def run_simulation():
        if not running:
            return
        
        # Step simulation
        for _ in range(steps_per_frame.value):
            sim.step(dt.value)
        
        # Get positions and plot
        with output:
            clear_output(wait=True)
            positions = sim.get_positions()
            
            # Create 3D plot for positions
            if system_type.value == 'solar_system':
                # 2D plot for solar system
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(positions[:, 0], positions[:, 1], s=sim.get_masses()*10, alpha=0.8)
                ax.set_aspect('equal')
                plt.title(f"N-body Simulation: {system_type.value}, {sim.get_particle_count()} particles")
            else:
                # 3D plot for other systems
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, alpha=0.5)
                plt.title(f"N-body Simulation: {system_type.value}, {sim.get_particle_count()} particles")
            
            # Add energy info
            kinetic = sim.calculate_total_kinetic_energy()
            potential = sim.calculate_total_potential_energy()
            total = kinetic + potential
            plt.figtext(0.5, 0.01, f"Energy: K={kinetic:.1e}, P={potential:.1e}, Total={total:.1e}", 
                        ha="center", fontsize=10)
            
            plt.tight_layout()
            plt.show()
        
        # Schedule next frame
        if running:
            import IPython
            IPython.display.display(IPython.core.display.Javascript('requestAnimationFrame(function() { IPython.notebook.kernel.execute("run_simulation()"); })'))
    
    # Display widgets and output
    controls = widgets.VBox([
        system_type, num_particles, integrator, dt, steps_per_frame, button
    ])
    display(widgets.HBox([controls, output]))

# Create interactive CT reconstruction demo
def ct_interactive_demo():
    # Create output for visualization
    output = widgets.Output()
    
    # Create phantom image
    phantom_size = 512
    phantom = mi.create_phantom(phantom_size, phantom_type="shepp_logan")
    
    # Convert to numpy for display
    phantom_np = mi.to_numpy(phantom)
    
    # Create widgets for parameters
    num_angles = widgets.IntSlider(
        value=180,
        min=10,
        max=360,
        step=10,
        description='Angles:'
    )
    
    filter_type = widgets.Dropdown(
        options=['RamLak', 'SheppLogan', 'Cosine', 'Hamming', 'Hann'],
        value='RamLak',
        description='Filter:'
    )
    
    reconstruction_method = widgets.Dropdown(
        options=['FilteredBackProjection', 'SIRT'],
        value='FilteredBackProjection',
        description='Method:'
    )
    
    iterations = widgets.IntSlider(
        value=50,
        min=10,
        max=200,
        step=10,
        description='Iterations:',
        disabled=True
    )
    
    # Enable/disable iterations based on method
    def on_method_change(change):
        iterations.disabled = change.new != 'SIRT'
    
    reconstruction_method.observe(on_method_change, names='value')
    
    # Button to run reconstruction
    button = widgets.Button(description="Run Reconstruction")
    
    def on_button_click(b):
        with output:
            clear_output(wait=True)
            
            # Create angles
            angles = np.linspace(0, np.pi, num_angles.value, dtype=np.float32)
            
            # Create projector
            projector = mi.CTProjector()
            projector.set_image_dimensions(phantom_size, phantom_size)
            projector.set_angles(angles)
            
            # Forward project to get sinogram
            print("Creating sinogram...")
            sinogram = projector.forward_project(phantom)
            
            # Create reconstructor
            reconstructor = mi.CTReconstructor()
            reconstructor.set_image_dimensions(phantom_size, phantom_size)
            reconstructor.set_angles(angles)
            
            # Set method and filter
            reconstructor.set_method(getattr(mi.ReconstructionMethod, reconstruction_method.value))
            reconstructor.set_filter(getattr(mi.FilterType, filter_type.value))
            
            if reconstruction_method.value == 'SIRT':
                reconstructor.set_iterations(iterations.value)
                print(f"Running SIRT reconstruction with {iterations.value} iterations...")
            else:
                print(f"Running filtered backprojection with {filter_type.value} filter...")
            
            # Perform reconstruction
            result = reconstructor.process(sinogram)
            result_np = mi.to_numpy(result)
            
            # Plot results
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original phantom
            axes[0].imshow(phantom_np, cmap='gray')
            axes[0].set_title("Original Phantom")
            
            # Sinogram
            axes[1].imshow(mi.to_numpy(sinogram), cmap='gray', aspect='auto')
            axes[1].set_title(f"Sinogram ({num_angles.value} angles)")
            
            # Reconstruction
            axes[2].imshow(result_np, cmap='gray')
            axes[2].set_title(f"Reconstruction ({reconstruction_method.value})")
            
            # Add error metrics
            mse = ((phantom_np - result_np) ** 2).mean()
            psnr = 10 * np.log10(1.0 / mse)
            plt.figtext(0.5, 0.01, f"Error Metrics: MSE={mse:.6f}, PSNR={psnr:.2f} dB", 
                        ha="center", fontsize=10)
            
            plt.tight_layout()
            plt.show()
    
    button.on_click(on_button_click)
    
    # Display widgets and output
    controls = widgets.VBox([
        num_angles, filter_type, reconstruction_method, iterations, button
    ])
    display(widgets.HBox([controls, output]))

# Run the interactive demos
print("N-body Simulation Interactive Demo")
nbody_interactive_demo()

print("\nCT Reconstruction Interactive Demo")
ct_interactive_demo()
```

### Example 2: Integration with Python Web API

Create a web API for remote execution of workloads:

```python
# app.py
from flask import Flask, request, jsonify
import tempfile
import os
import json
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Import our workloads
import nbody_sim
import medical_imaging as mi
import weather_sim
import molecular_dynamics as md

app = Flask(__name__)

@app.route('/api/nbody', methods=['POST'])
def run_nbody():
    data = request.json
    
    # Extract parameters
    num_particles = data.get('num_particles', 1000)
    system_type = data.get('system_type', 'random')
    integrator = data.get('integrator', 'leapfrog')
    num_steps = data.get('num_steps', 100)
    dt = data.get('dt', 0.01)
    
    # Create simulation
    sim = nbody_sim.Simulation()
    sim.initialize(
        num_particles=num_particles,
        system_type=system_type,
        integrator_type=getattr(nbody_sim.IntegratorType, integrator.upper())
    )
    
    # Run simulation
    for i in range(num_steps):
        sim.step(dt)
    
    # Get results
    positions = sim.get_positions().tolist()
    velocities = sim.get_velocities().tolist()
    kinetic_energy = sim.calculate_total_kinetic_energy()
    potential_energy = sim.calculate_total_potential_energy()
    
    # Create visualization
    plt.figure(figsize=(8, 8))
    positions_array = np.array(positions)
    plt.scatter(positions_array[:, 0], positions_array[:, 1], s=1, alpha=0.5)
    plt.title(f"N-body Simulation: {system_type}, {num_particles} particles")
    
    # Save plot to base64-encoded string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Return results
    return jsonify({
        'positions': positions,
        'velocities': velocities,
        'energy': {
            'kinetic': kinetic_energy,
            'potential': potential_energy,
            'total': kinetic_energy + potential_energy
        },
        'visualization': img_str
    })

@app.route('/api/medical/ct', methods=['POST'])
def run_ct_reconstruction():
    # Check if file was uploaded
    if 'sinogram' not in request.files:
        # Create phantom image
        phantom_size = int(request.form.get('size', 512))
        phantom = mi.create_phantom(phantom_size, phantom_type="shepp_logan")
        
        # Get angles
        num_angles = int(request.form.get('num_angles', 180))
        angles = np.linspace(0, np.pi, num_angles, dtype=np.float32)
        
        # Create projector and sinogram
        projector = mi.CTProjector()
        projector.set_image_dimensions(phantom_size, phantom_size)
        projector.set_angles(angles)
        sinogram = projector.forward_project(phantom)
    else:
        # Load uploaded sinogram
        sinogram_file = request.files['sinogram']
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            sinogram_file.save(tmp.name)
            sinogram = mi.load_image(tmp.name)
            os.unlink(tmp.name)
        
        # Get angles from request
        angles_json = request.form.get('angles', None)
        if angles_json:
            angles = np.array(json.loads(angles_json), dtype=np.float32)
        else:
            num_angles = sinogram.get_height()
            angles = np.linspace(0, np.pi, num_angles, dtype=np.float32)
    
    # Get reconstruction parameters
    method = request.form.get('method', 'FilteredBackProjection')
    filter_type = request.form.get('filter', 'RamLak')
    iterations = int(request.form.get('iterations', 50))
    
    # Create reconstructor
    reconstructor = mi.CTReconstructor()
    reconstructor.set_image_dimensions(sinogram.get_width(), sinogram.get_width())
    reconstructor.set_angles(angles)
    reconstructor.set_method(getattr(mi.ReconstructionMethod, method))
    reconstructor.set_filter(getattr(mi.FilterType, filter_type))
    
    if method == 'SIRT':
        reconstructor.set_iterations(iterations)
    
    # Perform reconstruction
    result = reconstructor.process(sinogram)
    
    # Convert to numpy and normalize for visualization
    result_np = mi.to_numpy(result)
    result_np = (result_np - result_np.min()) / (result_np.max() - result_np.min())
    
    # Convert to image
    img = Image.fromarray((result_np * 255).astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Return results
    return jsonify({
        'width': result.get_width(),
        'height': result.get_height(),
        'reconstruction': img_str,
        'parameters': {
            'method': method,
            'filter': filter_type,
            'angles': len(angles)
        }
    })

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    from benchmark.benchmark_suite import BenchmarkSuite
    
    data = request.json
    workload = data.get('workload', 'nbody_sim')
    params = data.get('parameters', {})
    
    # Create benchmark suite
    suite = BenchmarkSuite(device_id=0)
    
    # Run benchmark
    result = suite.run_benchmark(workload, **params)
    
    # Convert result to JSON-serializable dict
    result_dict = result.to_dict()
    
    # Return results
    return jsonify(result_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```