import os
import sys
import pytest

# Add the parent directory to the Python path so that we can import the geospatial package
# when running tests directly from this directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define fixtures that can be used by multiple tests
@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        # If torch is not available, try to import the _geospatial module directly
        try:
            from geospatial import _geospatial
            return _geospatial.is_cuda_available()
        except (ImportError, AttributeError):
            return False