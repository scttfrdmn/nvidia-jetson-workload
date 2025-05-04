#!/bin/bash
# Script to generate synthetic test data for geospatial unit tests

TEST_DATA_DIR="$(dirname "$0")/../test_data"
mkdir -p "${TEST_DATA_DIR}"

echo "Generating synthetic test data in ${TEST_DATA_DIR}..."

# Generate synthetic DEM data (GeoTIFF)
if command -v gdal_create > /dev/null 2>&1; then
    echo "Generating synthetic DEM..."
    gdal_create -of GTiff -outsize 256 256 -bands 1 -burn 100 -a_srs EPSG:4326 \
                -a_ullr -180 90 -179 89 "${TEST_DATA_DIR}/test_dem.tif"
else
    echo "GDAL tools not found. Using Python to create a simple DEM file..."
    python3 - << EOF
import numpy as np
from osgeo import gdal, osr
import os

# Create a simple synthetic DEM
dem_size = 256
dem_data = np.zeros((dem_size, dem_size), dtype=np.float32)

# Add some terrain features (hills, valleys)
x, y = np.meshgrid(np.linspace(0, 1, dem_size), np.linspace(0, 1, dem_size))
dem_data += 100 + 50 * np.sin(x * 10) * np.cos(y * 8)

# Add a central peak
center_x, center_y = dem_size // 2, dem_size // 2
for i in range(dem_size):
    for j in range(dem_size):
        dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
        if dist < dem_size // 4:
            dem_data[i, j] += 100 * (1 - dist / (dem_size // 4))

# Create a new GeoTIFF
driver = gdal.GetDriverByName('GTiff')
dem_file = driver.Create(os.path.join("${TEST_DATA_DIR}", "test_dem.tif"), 
                         dem_size, dem_size, 1, gdal.GDT_Float32)

# Set geotransform and projection
dem_file.SetGeoTransform((-180, 1/256, 0, 90, 0, -1/256))
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
dem_file.SetProjection(srs.ExportToWkt())

# Write data
dem_file.GetRasterBand(1).WriteArray(dem_data)
dem_file.GetRasterBand(1).SetNoDataValue(-9999)

# Close file
dem_file = None
print("Created synthetic DEM: ${TEST_DATA_DIR}/test_dem.tif")
EOF
fi

# Generate synthetic point cloud data (LAS/LAZ)
if command -v pdal > /dev/null 2>&1; then
    echo "Generating synthetic point cloud..."
    pdal pipeline - << EOF
{
    "pipeline": [
        {
            "type": "readers.faux",
            "bounds": "([0, 100], [0, 100], [0, 10])",
            "count": 10000,
            "mode": "random"
        },
        {
            "type": "writers.las",
            "filename": "${TEST_DATA_DIR}/test_pointcloud.las",
            "compression": "none"
        }
    ]
}
EOF
else
    echo "PDAL not found. Using Python to create a simple point cloud file..."
    python3 - << EOF
import numpy as np
import struct
import os

# Create a simple synthetic point cloud in a binary format
# This is a very simplified format that's not LAS but can be read by our test code
num_points = 10000
points = np.random.rand(num_points, 3)
points[:, 0] *= 100  # X range [0, 100]
points[:, 1] *= 100  # Y range [0, 100]
points[:, 2] *= 10   # Z range [0, 10]

# Add classification data (random classes 0-9)
classifications = np.random.randint(0, 10, num_points, dtype=np.uint8)

# Write to a simple binary format
with open(os.path.join("${TEST_DATA_DIR}", "test_pointcloud.bin"), 'wb') as f:
    # Write header
    f.write(struct.pack('<I', num_points))
    
    # Write points with classification
    for i in range(num_points):
        f.write(struct.pack('<fffB', 
                            points[i, 0], points[i, 1], points[i, 2], 
                            classifications[i]))

print("Created synthetic point cloud: ${TEST_DATA_DIR}/test_pointcloud.bin")
EOF
fi

echo "Test data generation complete."