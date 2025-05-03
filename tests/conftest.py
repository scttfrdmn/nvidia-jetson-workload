# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 nvidia-jetson-workload contributors

"""
Global pytest configuration.
"""

import os
import sys
import pytest

# Add the project root to the Python path so tests can import modules properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def test_data_dir():
    """Fixture to provide path to test data directory."""
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Fixture to provide a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def gpu_available():
    """Check if GPU is available for tests."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            import cupy
            return True
        except ImportError:
            return False


@pytest.fixture
def skip_if_no_gpu(gpu_available):
    """Skip test if GPU is not available."""
    if not gpu_available:
        pytest.skip("GPU not available")