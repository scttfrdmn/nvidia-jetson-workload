#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Configuration and fixtures for data transfer integration tests.
"""

import os
import sys
import pytest
import numpy as np
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Generator

# Add the project root to the path to access modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Use GPU tests only when CUDA is available
try:
    import cupy
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

@pytest.fixture
def numpy_arrays() -> Dict[str, np.ndarray]:
    """
    Create a dictionary of test arrays with different shapes and types.
    
    Returns:
        Dict containing test arrays with descriptive keys
    """
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    arrays = {
        "small_float32": rng.randn(100, 100).astype(np.float32),
        "medium_float32": rng.randn(1000, 1000).astype(np.float32),
        "large_float32": rng.randn(2000, 2000).astype(np.float32),
        "small_int32": rng.randint(-100, 100, (100, 100)).astype(np.int32),
        "medium_int32": rng.randint(-100, 100, (1000, 1000)).astype(np.int32),
        "vector_float64": rng.randn(10000).astype(np.float64),
        "3d_tensor": rng.randn(50, 50, 50).astype(np.float32),
    }
    
    return arrays

@pytest.fixture
def temp_shared_memory_dir() -> Generator[str, None, None]:
    """
    Create a temporary directory for shared memory files.
    
    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        old_env = os.environ.get('JETSON_WORKLOAD_SHARED_MEM_DIR')
        os.environ['JETSON_WORKLOAD_SHARED_MEM_DIR'] = temp_dir
        yield temp_dir
        if old_env is not None:
            os.environ['JETSON_WORKLOAD_SHARED_MEM_DIR'] = old_env
        else:
            del os.environ['JETSON_WORKLOAD_SHARED_MEM_DIR']

@pytest.fixture
def gpu_device_ids() -> List[int]:
    """
    Get available GPU device IDs for testing.
    
    Returns:
        List of available device IDs, empty if CUDA is not available
    """
    if not HAS_CUDA:
        return []
    
    try:
        return list(range(cupy.cuda.runtime.getDeviceCount()))
    except Exception:
        return []