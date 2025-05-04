#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

import os
import sys
import platform
import subprocess
import setuptools
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# Enable verbose output
verbose = True

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep
        
        # Set build type
        cfg = "Debug" if self.debug else "Release"
        
        # CMake build directory
        build_dir = os.path.join(ext.sourcedir, "build")
        os.makedirs(build_dir, exist_ok=True)
        
        # CMake configure command
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        
        # Compiler-specific build args
        build_args = []
        
        # Platform-specific configuration
        if platform.system() == "Windows":
            # For Windows, use Visual Studio generator
            cmake_args += ["-G", "Visual Studio 16 2019"]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--config", cfg]
        else:
            # For Unix, use make by default
            build_args += ["--", "-j4"]
        
        # Run CMake configure
        subprocess.check_call(["cmake", "-S", ext.sourcedir, "-B", build_dir] + cmake_args)
        
        # Run CMake build
        subprocess.check_call(["cmake", "--build", build_dir] + build_args)

setup(
    name="geospatial",
    version="1.0.0",
    author="Scott Friedman",
    author_email="scttfrdmn@example.com",
    description="Geospatial Analysis Workload with GPU Acceleration",
    long_description=open(os.path.join(os.path.dirname(__file__), "../../README.md")).read(),
    long_description_content_type="text/markdown",
    url="https://github.com/scttfrdmn/nvidia-jetson-workload",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    ext_modules=[CMakeExtension("geospatial._geospatial", sourcedir="../..")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    install_requires=[
        "numpy>=1.19.0",
        "gdal>=3.0.0",
    ],
    extras_require={
        "full": [
            "pdal>=2.0.0",
            "matplotlib>=3.0.0",
            "rasterio>=1.0.0",
            "fiona>=1.8.0",
        ],
    },
)