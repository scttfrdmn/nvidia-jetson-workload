#!/usr/bin/env python3
"""
Setup script for the weather_sim package.

Author: Scott Friedman
Copyright 2025 Scott Friedman. All rights reserved.
"""

from setuptools import setup, find_packages

setup(
    name="weather_sim",
    version="0.1.0",
    description="Python bindings for the Weather Simulation workload",
    author="Scott Friedman",
    author_email="scott@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "pillow>=8.0.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    python_requires=">=3.8",
)