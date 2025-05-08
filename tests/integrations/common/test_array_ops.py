#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Integration tests for array operations utilities.

These tests verify that array operations function correctly
across different device types and in integration scenarios.
"""

import os
import sys
import pytest
import numpy as np
from typing import Dict, Tuple, Any, List, Optional

# Add the project root to the path to access modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.integrations.common.array_ops import (
    transfer_array,
    elementwise_operation,
    matrix_multiply,
    reduction_operation,
    apply_function
)

# Skip GPU tests if CUDA is not available
try:
    import cupy
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# Test array transfer functions
def test_array_transfer_cpu(numpy_arrays):
    """Test transferring arrays within CPU memory."""
    for name, arr in numpy_arrays.items():
        # Transfer array (should create a copy)
        result = transfer_array(arr, device_id=-1)
        
        # Verify result is a new array with same data
        assert result is not arr  # Should be a different object
        assert isinstance(result, np.ndarray)
        assert result.shape == arr.shape
        assert result.dtype == arr.dtype
        np.testing.assert_array_equal(result, arr)

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_array_transfer_gpu(numpy_arrays, gpu_device_ids):
    """Test transferring arrays between CPU and GPU."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    
    # Test CPU to GPU transfer
    for name, arr in numpy_arrays.items():
        # Transfer to GPU
        gpu_arr = transfer_array(arr, device_id=device_id)
        
        # Verify result is a GPU array with same data
        assert isinstance(gpu_arr, cupy.ndarray)
        assert gpu_arr.shape == arr.shape
        assert gpu_arr.dtype == arr.dtype
        
        # Compare data (transfer back to CPU for comparison)
        cpu_copy = gpu_arr.get()
        np.testing.assert_array_equal(cpu_copy, arr)
        
        # Test GPU to CPU transfer
        cpu_arr = transfer_array(gpu_arr, device_id=-1)
        
        # Verify result is a CPU array with same data
        assert isinstance(cpu_arr, np.ndarray)
        np.testing.assert_array_equal(cpu_arr, arr)
        
        # Test GPU to same GPU transfer (should create a copy)
        gpu_copy = transfer_array(gpu_arr, device_id=device_id)
        assert gpu_copy is not gpu_arr  # Should be a different object
        cupy.testing.assert_array_equal(gpu_copy, gpu_arr)

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_array_transfer_multiple_gpus(numpy_arrays, gpu_device_ids):
    """Test transferring arrays between multiple GPUs."""
    if len(gpu_device_ids) < 2:
        pytest.skip("Need at least 2 GPU devices for this test")
        
    device1 = gpu_device_ids[0]
    device2 = gpu_device_ids[1]
    
    # Test transfers between GPUs
    test_array = numpy_arrays["small_float32"]
    
    # Transfer to first GPU
    gpu1_arr = transfer_array(test_array, device_id=device1)
    assert gpu1_arr.device.id == device1
    
    # Transfer from first GPU to second GPU
    gpu2_arr = transfer_array(gpu1_arr, device_id=device2)
    assert gpu2_arr.device.id == device2
    
    # Verify data is preserved
    gpu1_cpu = gpu1_arr.get()
    gpu2_cpu = gpu2_arr.get()
    np.testing.assert_array_equal(gpu1_cpu, gpu2_cpu)
    np.testing.assert_array_equal(test_array, gpu2_cpu)

# Test elementwise operations
def test_elementwise_operation_cpu(numpy_arrays):
    """Test elementwise operations on CPU arrays."""
    for name, arr in numpy_arrays.items():
        # Addition with scalar
        result_add = elementwise_operation(arr, 5, "add", device_id=-1)
        np.testing.assert_array_equal(result_add, arr + 5)
        
        # Multiplication with scalar
        result_mul = elementwise_operation(arr, 2, "multiply", device_id=-1)
        np.testing.assert_array_equal(result_mul, arr * 2)
        
        # Subtraction with scalar
        result_sub = elementwise_operation(arr, 3, "subtract", device_id=-1)
        np.testing.assert_array_equal(result_sub, arr - 3)
        
        # Division with scalar
        result_div = elementwise_operation(arr, 2, "divide", device_id=-1)
        np.testing.assert_array_almost_equal(result_div, arr / 2)
        
        # Power operation
        result_pow = elementwise_operation(arr, 2, "power", device_id=-1)
        np.testing.assert_array_almost_equal(result_pow, arr ** 2)
        
        if name == "small_float32":  # Test array-array operations with one case
            # Create a second array of the same shape
            arr2 = arr + 1
            
            # Addition with array
            result_add_arr = elementwise_operation(arr, arr2, "add", device_id=-1)
            np.testing.assert_array_equal(result_add_arr, arr + arr2)
            
            # Multiplication with array
            result_mul_arr = elementwise_operation(arr, arr2, "multiply", device_id=-1)
            np.testing.assert_array_equal(result_mul_arr, arr * arr2)

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_elementwise_operation_gpu(numpy_arrays, gpu_device_ids):
    """Test elementwise operations on GPU arrays."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    
    # Test with a subset of arrays for speed
    test_arrays = {
        k: numpy_arrays[k] 
        for k in ["small_float32", "vector_float64"]
    }
    
    for name, arr in test_arrays.items():
        # Create GPU array
        gpu_arr = transfer_array(arr, device_id=device_id)
        
        # Addition with scalar on GPU
        result_add = elementwise_operation(gpu_arr, 5, "add", device_id=device_id)
        assert isinstance(result_add, cupy.ndarray)
        np.testing.assert_array_equal(result_add.get(), arr + 5)
        
        # Test mixed device operands (CPU array, operation on GPU)
        result_mix = elementwise_operation(arr, 3, "multiply", device_id=device_id)
        assert isinstance(result_mix, cupy.ndarray)
        np.testing.assert_array_equal(result_mix.get(), arr * 3)
        
        # Test with another array
        if name == "small_float32":
            arr2 = arr + 1
            gpu_arr2 = transfer_array(arr2, device_id=device_id)
            
            # Addition with GPU array
            result_add_arr = elementwise_operation(
                gpu_arr, gpu_arr2, "add", device_id=device_id)
            np.testing.assert_array_equal(result_add_arr.get(), arr + arr2)
            
            # Test mixed array types (CPU + GPU)
            result_mix_arr = elementwise_operation(
                arr, gpu_arr2, "multiply", device_id=device_id)
            np.testing.assert_array_equal(result_mix_arr.get(), arr * arr2)

# Test matrix multiplication
def test_matrix_multiply_cpu(numpy_arrays):
    """Test matrix multiplication on CPU."""
    # Select matrix arrays
    matrices = {
        k: numpy_arrays[k] 
        for k in ["small_float32", "small_int32"]
    }
    
    for name, mat in matrices.items():
        # Need square matrices for this test
        assert mat.shape[0] == mat.shape[1]
        
        # Create a second matrix
        mat2 = np.eye(mat.shape[0], dtype=mat.dtype) * 2
        
        # Matrix multiplication
        result = matrix_multiply(mat, mat2, device_id=-1)
        
        # Verify result
        expected = np.matmul(mat, mat2)
        np.testing.assert_array_almost_equal(result, expected)

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_matrix_multiply_gpu(numpy_arrays, gpu_device_ids):
    """Test matrix multiplication on GPU."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    
    # Select matrix arrays
    matrices = {
        k: numpy_arrays[k] 
        for k in ["small_float32", "medium_float32"]
    }
    
    for name, mat in matrices.items():
        # Need square matrices for this test
        assert mat.shape[0] == mat.shape[1]
        
        # Create a second matrix
        mat2 = np.eye(mat.shape[0], dtype=mat.dtype) * 2
        
        # Transfer to GPU
        gpu_mat = transfer_array(mat, device_id=device_id)
        gpu_mat2 = transfer_array(mat2, device_id=device_id)
        
        # Test GPU matrix multiplication
        gpu_result = matrix_multiply(gpu_mat, gpu_mat2, device_id=device_id)
        assert isinstance(gpu_result, cupy.ndarray)
        
        # Verify result
        expected = np.matmul(mat, mat2)
        np.testing.assert_array_almost_equal(gpu_result.get(), expected)
        
        # Test mixed operands (CPU and GPU)
        mixed_result = matrix_multiply(mat, gpu_mat2, device_id=device_id)
        assert isinstance(mixed_result, cupy.ndarray)
        np.testing.assert_array_almost_equal(mixed_result.get(), expected)

# Test reduction operations
def test_reduction_operation_cpu(numpy_arrays):
    """Test reduction operations on CPU arrays."""
    for name, arr in numpy_arrays.items():
        # Sum reduction
        sum_result = reduction_operation(arr, "sum", device_id=-1)
        assert sum_result == np.sum(arr)
        
        # Mean reduction
        mean_result = reduction_operation(arr, "mean", device_id=-1)
        assert mean_result == pytest.approx(np.mean(arr))
        
        # Max reduction
        max_result = reduction_operation(arr, "max", device_id=-1)
        assert max_result == np.max(arr)
        
        # Min reduction
        min_result = reduction_operation(arr, "min", device_id=-1)
        assert min_result == np.min(arr)
        
        # Axis reduction
        if len(arr.shape) > 1:
            # Sum along axis 0
            sum_axis0 = reduction_operation(arr, "sum", axis=0, device_id=-1)
            np.testing.assert_array_equal(sum_axis0, np.sum(arr, axis=0))
            
            # Mean along axis 1
            mean_axis1 = reduction_operation(arr, "mean", axis=1, device_id=-1)
            np.testing.assert_array_almost_equal(mean_axis1, np.mean(arr, axis=1))

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_reduction_operation_gpu(numpy_arrays, gpu_device_ids):
    """Test reduction operations on GPU arrays."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    
    # Test with a subset of arrays for speed
    test_arrays = {
        k: numpy_arrays[k] 
        for k in ["small_float32", "3d_tensor"]
    }
    
    for name, arr in test_arrays.items():
        # Transfer to GPU
        gpu_arr = transfer_array(arr, device_id=device_id)
        
        # Sum reduction
        sum_result = reduction_operation(gpu_arr, "sum", device_id=device_id)
        assert sum_result == pytest.approx(np.sum(arr))
        
        # Mean reduction
        mean_result = reduction_operation(gpu_arr, "mean", device_id=device_id)
        assert mean_result == pytest.approx(np.mean(arr))
        
        # Test CPU array with GPU operation
        sum_result_mixed = reduction_operation(arr, "sum", device_id=device_id)
        assert sum_result_mixed == pytest.approx(np.sum(arr))
        
        # Axis reduction
        if len(arr.shape) > 1:
            # Sum along axis 0
            sum_axis0 = reduction_operation(gpu_arr, "sum", axis=0, device_id=device_id)
            np.testing.assert_array_almost_equal(sum_axis0.get(), np.sum(arr, axis=0))

# Test apply function
def test_apply_function_cpu(numpy_arrays):
    """Test applying custom functions to CPU arrays."""
    
    # Define test functions
    def square(x):
        return x * x
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Apply to arrays
    for name, arr in numpy_arrays.items():
        # Apply square function
        result_square = apply_function(arr, square, device_id=-1)
        np.testing.assert_array_almost_equal(result_square, arr * arr)
        
        # Apply sigmoid function
        result_sigmoid = apply_function(arr, sigmoid, device_id=-1)
        np.testing.assert_array_almost_equal(
            result_sigmoid, 1 / (1 + np.exp(-arr)))

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_apply_function_gpu(numpy_arrays, gpu_device_ids):
    """Test applying custom functions to GPU arrays."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    
    # Define test functions for both CPU and GPU
    def square_cpu(x):
        return x * x
    
    def square_gpu(x):
        return x * x
    
    # Apply to a subset of arrays
    test_arrays = {
        k: numpy_arrays[k] 
        for k in ["small_float32", "vector_float64"]
    }
    
    for name, arr in test_arrays.items():
        # Transfer to GPU
        gpu_arr = transfer_array(arr, device_id=device_id)
        
        # Apply function to GPU array
        result_gpu = apply_function(gpu_arr, square_gpu, device_id=device_id)
        assert isinstance(result_gpu, cupy.ndarray)
        np.testing.assert_array_almost_equal(result_gpu.get(), arr * arr)
        
        # Apply function to CPU array but execute on GPU
        result_mixed = apply_function(arr, square_cpu, device_id=device_id)
        assert isinstance(result_mixed, cupy.ndarray)
        np.testing.assert_array_almost_equal(result_mixed.get(), arr * arr)

# Test integration with cross-workload data transfer
def test_integration_with_shared_memory(numpy_arrays, temp_shared_memory_dir):
    """Test integration between array_ops and shared_memory."""
    from src.integrations.common.shared_memory import DataTransferOptimizer
    
    # Use a test array
    test_array = numpy_arrays["small_float32"]
    
    # Create optimizer for data transfer
    optimizer = DataTransferOptimizer()
    
    # Transfer array through shared memory
    meta = optimizer.optimize_array_transfer(test_array, "test_integration")
    
    # Reconstruct array
    reconstructed = optimizer.reconstruct_array(meta)
    
    # Apply operations to the shared array
    result = elementwise_operation(reconstructed, 5, "add", device_id=-1)
    
    # Verify result
    np.testing.assert_array_equal(result, test_array + 5)
    
    # Clean up
    optimizer.release_array(meta)

@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_integration_with_gpu_memory(numpy_arrays, gpu_device_ids):
    """Test integration between array_ops and gpu_memory_manager."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    
    from src.integrations.common.gpu_memory_manager import GpuMemoryManager
    
    # Use a test array
    test_array = numpy_arrays["small_float32"]
    array_size_bytes = test_array.nbytes
    
    # Create a memory manager
    manager = GpuMemoryManager()
    
    # Allocate memory through the manager
    block = manager.allocate(array_size_bytes, device_id)
    
    # Create a CuPy array using the allocated memory
    with cupy.cuda.Device(device_id):
        memptr = cupy.cuda.MemoryPointer(
            cupy.cuda.UnownedMemory(block.ptr, array_size_bytes, manager),
            0
        )
        
        shape = test_array.shape
        dtype = test_array.dtype
        gpu_array = cupy.ndarray(shape, dtype, memptr)
        
        # Copy data to the array
        gpu_array.set(test_array)
    
    # Apply operations to the GPU array
    result = matrix_multiply(
        gpu_array, 
        gpu_array,  # multiply by itself
        device_id=device_id
    )
    
    # Verify result
    expected = np.matmul(test_array, test_array)
    np.testing.assert_array_almost_equal(result.get(), expected)
    
    # Clean up
    manager.release(block)
    manager.clear()