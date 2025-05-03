# AWS Open Data Resources for Workloads

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2024 nvidia-jetson-workload contributors -->

This document outlines the AWS Open Data resources selected for each workload type.

## Weather Simulation

### NOAA High-Resolution Rapid Refresh (HRRR) Model

**Description:** A real-time atmospheric model with 3-km resolution, updated hourly with cloud-resolving and convection-allowing capabilities. Radar data is assimilated every 15 minutes to provide detailed weather information.

**Usage in Our Project:**
- Source of real-world weather data for model initialization
- Validation data for our simulation results
- Historical weather pattern analysis

**Access Information:**
- AWS S3 Bucket: `noaa-hrrr-bdp-pds` (primary archive) and `hrrrzarr` (Zarr format)
- Example CLI Access: `aws s3 ls --no-sign-request s3://noaa-hrrr-bdp-pds/`
- No AWS account required

**Data Format:** GRIB2 files (standard meteorological format)

**Citation:** "NOAA High-Resolution Rapid Refresh (HRRR) Model was accessed from https://registry.opendata.aws/noaa-hrrr-pds."

## Medical Imaging Processing

### NYU FastMRI Dataset

**Description:** A comprehensive medical imaging resource containing deidentified raw k-space data and DICOM image files of over 1,500 knees and 6,970 brains.

**Usage in Our Project:**
- Training and testing data for segmentation algorithms
- Baseline MRI dataset for processing benchmarks
- Validation of reconstruction techniques

**Access Information:**
- AWS S3 Bucket: `arn:aws:s3:::fastmri-dataset`
- Requires authorization (contact: Florian Knoll - florian.knoll@nyuangone.org)
- Region: us-east-1

**Data Format:** DICOM and raw k-space data

**Citation:** "NYU fastMRI Initiative database was accessed from https://fastmri.med.nyu.edu/."

## N-body Simulation

For our N-body simulation, we will use synthetic data and established initial conditions, as there wasn't a directly suitable AWS Open Data dataset for gravitational N-body simulation.

### Alternative Resources:

1. **NASA Space Physics Data Facility**
   - Web: https://spdf.gsfc.nasa.gov/
   - Contains planetary and solar system data

2. **Initial Conditions from Standard Astrophysical Models**
   - Plummer model for globular clusters
   - Isothermal sphere model for dark matter halos
   - Disk galaxy models with bulge, disk, and halo components

3. **Synthetic Data Generation**
   - We'll implement procedural generation of initial particle distributions
   - Configurable parameters for mass distribution, velocity dispersion, etc.
   - Option to import custom particle data in standard formats