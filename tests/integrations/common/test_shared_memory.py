#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Integration tests for shared memory utilities.

These tests verify that the shared memory utilities properly handle
data transfer between different parts of the codebase.
"""

import os
import sys
import pytest
import numpy as np
from typing import Dict, Tuple, Any, List, Optional
import multiprocessing as mp
from multiprocessing import shared_memory
import tempfile

# Add the project root to the path to access modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.integrations.common.shared_memory import (
    DataTransferOptimizer,
    SharedMemoryManager,
    create_shared_array,
    attach_to_shared_array
)

# Skip GPU tests if CUDA is not available
try:
    import cupy
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# Test basic shared memory creation and attachment
def test_create_and_attach_shared_array(numpy_arrays, temp_shared_memory_dir):
    """Test creating and attaching to a shared array."""
    for name, arr in numpy_arrays.items():
        # Create shared array
        shared_info = create_shared_array(arr, f"test_{name}")
        
        # Verify the shared info contains required fields
        assert "shape" in shared_info
        assert "dtype" in shared_info
        assert "name" in shared_info
        
        # Attach to the shared array
        reconstructed = attach_to_shared_array(shared_info)
        
        # Verify the reconstructed array matches the original
        assert reconstructed.shape == arr.shape
        assert reconstructed.dtype == arr.dtype
        np.testing.assert_array_equal(reconstructed, arr)
        
        # Clean up
        shm = shared_memory.SharedMemory(name=shared_info["name"])
        shm.close()
        shm.unlink()

# Test the DataTransferOptimizer
def test_data_transfer_optimizer(numpy_arrays, temp_shared_memory_dir):
    """Test the DataTransferOptimizer for various array types."""
    optimizer = DataTransferOptimizer()
    
    for name, arr in numpy_arrays.items():
        # Optimize array transfer
        meta = optimizer.optimize_array_transfer(arr, f"test_dto_{name}")
        
        # Reconstruct array from metadata
        reconstructed = optimizer.reconstruct_array(meta)
        
        # Verify the reconstructed array matches the original
        assert reconstructed.shape == arr.shape
        assert reconstructed.dtype == arr.dtype
        np.testing.assert_array_equal(reconstructed, arr)
        
        # Clean up
        optimizer.release_array(meta)

# Test cross-process shared memory transfer
def _child_process_function(shared_info, queue):
    """Child process function for testing cross-process sharing."""
    try:
        # Reconstruct array in child process
        arr = attach_to_shared_array(shared_info)
        
        # Modify the array to test write access
        arr += 1
        
        # Send success to parent
        queue.put(True)
    except Exception as e:
        # Send the exception to parent
        queue.put(str(e))

def test_cross_process_sharing(numpy_arrays, temp_shared_memory_dir):
    """Test sharing arrays between processes."""
    
    # Select a subset of arrays to test for speed
    test_arrays = {
        k: numpy_arrays[k] 
        for k in ["small_float32", "small_int32", "vector_float64"]
    }
    
    for name, arr in test_arrays.items():
        # Create original copy
        original = arr.copy()
        
        # Create shared array
        shared_info = create_shared_array(arr, f"test_mproc_{name}")
        
        # Set up multiprocessing
        queue = mp.Queue()
        proc = mp.Process(
            target=_child_process_function,
            args=(shared_info, queue)
        )
        
        # Run child process
        proc.start()
        result = queue.get(timeout=5)
        proc.join(5)
        
        # Verify success
        assert result is True, f"Child process failed: {result}"
        
        # Verify the array was modified by the child
        modified_arr = attach_to_shared_array(shared_info)
        np.testing.assert_array_equal(modified_arr, original + 1)
        
        # Clean up
        shm = shared_memory.SharedMemory(name=shared_info["name"])
        shm.close()
        shm.unlink()

# Test SharedMemoryManager
def test_shared_memory_manager(numpy_arrays, temp_shared_memory_dir):
    """Test the SharedMemoryManager for handling multiple arrays."""
    manager = SharedMemoryManager()
    
    # Register multiple arrays
    registered_arrays = {}
    for name, arr in numpy_arrays.items():
        handle = manager.register_array(arr, f"mgr_{name}")
        registered_arrays[name] = handle
    
    # Verify all arrays are listed in manager
    assert len(manager.list_arrays()) == len(numpy_arrays)
    
    # Access arrays and verify contents
    for name, handle in registered_arrays.items():
        arr = manager.get_array(handle)
        np.testing.assert_array_equal(arr, numpy_arrays[name])
    
    # Test removal of specific array
    first_key = list(registered_arrays.keys())[0]
    manager.remove_array(registered_arrays[first_key])
    
    # Verify array was removed
    assert len(manager.list_arrays()) == len(numpy_arrays) - 1
    
    # Clean up all remaining arrays
    manager.clear()
    assert len(manager.list_arrays()) == 0

# GPU tests
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_gpu_array_transfer(numpy_arrays, gpu_device_ids, temp_shared_memory_dir):
    """Test transferring GPU arrays using the DataTransferOptimizer."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    optimizer = DataTransferOptimizer(device_id=device_id)
    
    # Test with a few arrays to keep the test fast
    test_arrays = {
        k: numpy_arrays[k] 
        for k in ["small_float32", "vector_float64"]
    }
    
    for name, arr in test_arrays.items():
        # Create GPU array
        gpu_arr = cupy.array(arr)
        
        # Optimize GPU array transfer
        meta = optimizer.optimize_array_transfer(gpu_arr, f"test_gpu_{name}")
        
        # Reconstruct array from metadata
        reconstructed = optimizer.reconstruct_array(meta)
        
        # Verify the reconstructed array matches the original
        assert isinstance(reconstructed, cupy.ndarray)
        assert reconstructed.shape == arr.shape
        assert reconstructed.dtype == arr.dtype
        cupy.testing.assert_array_equal(reconstructed, gpu_arr)
        
        # Test CPU-GPU transfer
        cpu_meta = optimizer.optimize_array_transfer(arr, f"test_cpu_gpu_{name}")
        cpu_reconstructed = optimizer.reconstruct_array(cpu_meta)
        
        # Transfer should return a GPU array
        assert isinstance(cpu_reconstructed, cupy.ndarray)
        cupy.testing.assert_array_equal(cpu_reconstructed, gpu_arr)
        
        # Clean up
        optimizer.release_array(meta)
        optimizer.release_array(cpu_meta)

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_shared_memory_manager_gpu(numpy_arrays, gpu_device_ids, temp_shared_memory_dir):
    """Test the SharedMemoryManager with GPU arrays."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    manager = SharedMemoryManager(device_id=device_id)
    
    # Register a mix of CPU and GPU arrays
    handles = {}
    
    # CPU arrays
    handles["cpu_small"] = manager.register_array(
        numpy_arrays["small_float32"], "cpu_small")
    handles["cpu_vector"] = manager.register_array(
        numpy_arrays["vector_float64"], "cpu_vector")
    
    # GPU arrays
    gpu_small = cupy.array(numpy_arrays["small_float32"])
    gpu_vector = cupy.array(numpy_arrays["vector_float64"])
    
    handles["gpu_small"] = manager.register_array(gpu_small, "gpu_small")
    handles["gpu_vector"] = manager.register_array(gpu_vector, "gpu_vector")
    
    # Verify all arrays are registered
    assert len(manager.list_arrays()) == 4
    
    # Access arrays and verify contents
    cpu_small = manager.get_array(handles["cpu_small"])
    cpu_vector = manager.get_array(handles["cpu_vector"])
    retrieved_gpu_small = manager.get_array(handles["gpu_small"])
    retrieved_gpu_vector = manager.get_array(handles["gpu_vector"])
    
    # CPU arrays should now be on GPU
    assert isinstance(cpu_small, cupy.ndarray)
    assert isinstance(cpu_vector, cupy.ndarray)
    
    # GPU arrays should remain on GPU
    assert isinstance(retrieved_gpu_small, cupy.ndarray)
    assert isinstance(retrieved_gpu_vector, cupy.ndarray)
    
    # Verify contents
    cupy.testing.assert_array_equal(
        cpu_small, cupy.array(numpy_arrays["small_float32"]))
    cupy.testing.assert_array_equal(
        cpu_vector, cupy.array(numpy_arrays["vector_float64"]))
    cupy.testing.assert_array_equal(retrieved_gpu_small, gpu_small)
    cupy.testing.assert_array_equal(retrieved_gpu_vector, gpu_vector)
    
    # Clean up
    manager.clear()
    assert len(manager.list_arrays()) == 0