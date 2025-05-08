"""
Geospatial Financial Integration Module

This module provides integration between the Financial Modeling and Geospatial Analysis workloads,
enabling geospatial financial risk analysis and visualization.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

# Make sure the necessary modules are importable
try:
    import financial_modeling
except ImportError:
    raise ImportError(
        "Failed to import financial_modeling module. Please ensure the Financial Modeling library "
        "is properly built and installed."
    )

try:
    import geospatial
except ImportError:
    raise ImportError(
        "Failed to import geospatial module. Please ensure the Geospatial Analysis library "
        "is properly built and installed."
    )

# Import submodules
from .geo_risk import *
from .data_connectors import *
from .visualization import *

__version__ = "0.1.0"