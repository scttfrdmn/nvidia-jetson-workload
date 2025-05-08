#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Integration tests for GPU memory manager.

These tests verify that the GPU memory manager correctly handles
allocation, deallocation, and pooling across different workloads.
"""

import os
import sys
import pytest
import numpy as np
from typing import Dict, Tuple, Any, List, Optional
import gc

# Add the project root to the path to access modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.integrations.common.gpu_memory_manager import (
    GpuMemoryManager,
    MemoryBlock,
    MemoryPool
)

# Skip GPU tests if CUDA is not available
try:
    import cupy
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# Basic memory pool tests
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_memory_pool_basics(gpu_device_ids):
    """Test basic memory pool operations."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    pool = MemoryPool(device_id)
    
    # Test allocation from pool
    size = 1024 * 1024  # 1 MB
    block1 = pool.allocate(size)
    
    # Verify block properties
    assert block1.size >= size
    assert isinstance(block1.ptr, int) and block1.ptr > 0
    assert block1.in_use is True
    
    # Allocate a second block
    block2 = pool.allocate(size * 2)
    assert block2.size >= size * 2
    assert block2.ptr != block1.ptr
    assert block2.in_use is True
    
    # Release a block back to the pool
    pool.release(block1)
    assert block1.in_use is False
    
    # Allocate a block of the same size - should reuse the released block
    block3 = pool.allocate(size)
    assert block3.ptr == block1.ptr
    assert block3.in_use is True
    
    # Check pool stats
    stats = pool.get_stats()
    assert stats['allocated_blocks'] == 2
    assert stats['free_blocks'] == 0
    assert stats['total_bytes_allocated'] >= size * 3
    
    # Clean up
    pool.release(block2)
    pool.release(block3)
    pool.clear()

# Test memory block tracking
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_memory_block_tracking(gpu_device_ids):
    """Test tracking and lifecycle of memory blocks."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    
    # Create a memory block directly
    with cupy.cuda.Device(device_id):
        mem_ptr = cupy.cuda.alloc(1024 * 1024)
        ptr_value = mem_ptr.ptr
    
    # Create a MemoryBlock to track it
    block = MemoryBlock(ptr_value, 1024 * 1024, device_id)
    assert block.ptr == ptr_value
    assert block.size == 1024 * 1024
    assert block.device_id == device_id
    assert block.in_use is True
    
    # Test block release
    block.release()
    assert block.in_use is False
    
    # Clean up manually to avoid leak
    with cupy.cuda.Device(device_id):
        mem_ptr.free()

# Main GPU memory manager tests
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_gpu_memory_manager(gpu_device_ids):
    """Test the main GpuMemoryManager class."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    manager = GpuMemoryManager()
    
    # Allocate memory through the manager
    block1 = manager.allocate(1024 * 1024, device_id)  # 1 MB
    block2 = manager.allocate(2 * 1024 * 1024, device_id)  # 2 MB
    
    # Verify allocations
    assert block1.device_id == device_id
    assert block2.device_id == device_id
    assert block1.size >= 1024 * 1024
    assert block2.size >= 2 * 1024 * 1024
    
    # Get memory usage
    usage = manager.get_memory_usage(device_id)
    assert usage['allocated'] >= 3 * 1024 * 1024  # At least 3 MB
    
    # Release memory
    manager.release(block1)
    
    # Check that block was returned to pool
    usage_after_release = manager.get_memory_usage(device_id)
    assert usage_after_release['allocated'] == usage['allocated']
    assert usage_after_release['active'] < usage['allocated']
    
    # Re-allocate from pool
    block3 = manager.allocate(1024 * 1024, device_id)
    assert block3.ptr == block1.ptr  # Should reuse the same memory
    
    # Clean up
    manager.release(block2)
    manager.release(block3)
    manager.clear()

# Test creating arrays from blocks
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_array_creation(gpu_device_ids, numpy_arrays):
    """Test creating GPU arrays from allocated memory blocks."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    manager = GpuMemoryManager()
    
    test_array = numpy_arrays["medium_float32"]
    array_size_bytes = test_array.nbytes
    
    # Allocate memory for the array
    block = manager.allocate(array_size_bytes, device_id)
    
    # Create a CuPy array using the allocated memory
    with cupy.cuda.Device(device_id):
        # Get memory pointer
        memptr = cupy.cuda.MemoryPointer(
            cupy.cuda.UnownedMemory(block.ptr, array_size_bytes, manager),
            0
        )
        
        # Create array from pointer
        shape = test_array.shape
        dtype = test_array.dtype
        gpu_array = cupy.ndarray(shape, dtype, memptr)
        
        # Copy data to the array
        gpu_array.set(test_array)
        
        # Verify data
        cpu_copy = gpu_array.get()
        np.testing.assert_array_equal(cpu_copy, test_array)
    
    # Clean up
    manager.release(block)
    manager.clear()

# Test cross-workload memory sharing
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_cross_workload_memory_sharing(gpu_device_ids, numpy_arrays):
    """Test sharing memory between simulated workloads."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    manager = GpuMemoryManager()
    
    # Simulate first workload
    def workload1_alloc():
        array1 = numpy_arrays["small_float32"]
        array2 = numpy_arrays["vector_float64"]
        
        # Allocate memory for arrays
        block1 = manager.allocate(array1.nbytes, device_id)
        block2 = manager.allocate(array2.nbytes, device_id)
        
        # Create GPU arrays
        with cupy.cuda.Device(device_id):
            memptr1 = cupy.cuda.MemoryPointer(
                cupy.cuda.UnownedMemory(block1.ptr, array1.nbytes, manager),
                0
            )
            gpu_array1 = cupy.ndarray(array1.shape, array1.dtype, memptr1)
            gpu_array1.set(array1)
            
            memptr2 = cupy.cuda.MemoryPointer(
                cupy.cuda.UnownedMemory(block2.ptr, array2.nbytes, manager),
                0
            )
            gpu_array2 = cupy.ndarray(array2.shape, array2.dtype, memptr2)
            gpu_array2.set(array2)
        
        return block1, block2, gpu_array1, gpu_array2
    
    # Simulate second workload
    def workload2_use(block1, block2, shape1, dtype1, shape2, dtype2):
        # Create arrays from existing memory blocks
        with cupy.cuda.Device(device_id):
            memptr1 = cupy.cuda.MemoryPointer(
                cupy.cuda.UnownedMemory(block1.ptr, block1.size, manager),
                0
            )
            shared_array1 = cupy.ndarray(shape1, dtype1, memptr1)
            
            memptr2 = cupy.cuda.MemoryPointer(
                cupy.cuda.UnownedMemory(block2.ptr, block2.size, manager),
                0
            )
            shared_array2 = cupy.ndarray(shape2, dtype2, memptr2)
            
            # Modify the arrays
            shared_array1 += 1
            shared_array2 *= 2
        
        return shared_array1, shared_array2
    
    # Run the simulated workloads
    array1 = numpy_arrays["small_float32"]
    array2 = numpy_arrays["vector_float64"]
    
    # First workload allocates and initializes
    block1, block2, gpu_array1, gpu_array2 = workload1_alloc()
    
    # Second workload uses the allocated memory
    shared_array1, shared_array2 = workload2_use(
        block1, block2, 
        array1.shape, array1.dtype,
        array2.shape, array2.dtype
    )
    
    # Verify changes are visible in the original arrays
    # This confirms memory is shared
    with cupy.cuda.Device(device_id):
        # Get modified data back to CPU
        modified1 = gpu_array1.get()
        modified2 = gpu_array2.get()
        
        # Expected results after modifications
        expected1 = array1 + 1
        expected2 = array2 * 2
        
        # Verify
        np.testing.assert_array_equal(modified1, expected1)
        np.testing.assert_array_equal(modified2, expected2)
    
    # Clean up
    manager.release(block1)
    manager.release(block2)
    manager.clear()

# Test garbage collection
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_memory_garbage_collection(gpu_device_ids):
    """Test automatic garbage collection of unreferenced memory blocks."""
    if not gpu_device_ids:
        pytest.skip("No GPU devices available")
        
    device_id = gpu_device_ids[0]
    manager = GpuMemoryManager()
    
    # Get initial memory usage
    initial_usage = manager.get_memory_usage(device_id)
    
    # Allocate blocks but don't keep references to them
    for _ in range(10):
        manager.allocate(1024 * 1024, device_id)  # 1 MB each, references lost
    
    # Force garbage collection
    gc.collect()
    
    # Allow time for GC to work
    import time
    time.sleep(0.1)
    
    # Run the manager's internal cleanup
    manager._cleanup_unused_blocks()
    
    # Check memory usage - should have returned to pool
    current_usage = manager.get_memory_usage(device_id)
    assert current_usage['active'] == initial_usage['active']
    
    # Check that blocks were returned to pool, not freed
    assert manager.get_pool_stats(device_id)['free_blocks'] >= 10
    
    # Clean up
    manager.clear()

# Test multi-device support
@pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
def test_multi_device_support(gpu_device_ids):
    """Test memory management across multiple GPU devices."""
    if len(gpu_device_ids) < 2:
        pytest.skip("Need at least 2 GPU devices for this test")
        
    device1 = gpu_device_ids[0]
    device2 = gpu_device_ids[1]
    
    manager = GpuMemoryManager()
    
    # Allocate on both devices
    block1 = manager.allocate(1024 * 1024, device1)
    block2 = manager.allocate(1024 * 1024, device2)
    
    # Verify correct device assignment
    assert block1.device_id == device1
    assert block2.device_id == device2
    
    # Check memory usage per device
    usage1 = manager.get_memory_usage(device1)
    usage2 = manager.get_memory_usage(device2)
    
    assert usage1['allocated'] >= 1024 * 1024
    assert usage2['allocated'] >= 1024 * 1024
    
    # Release blocks
    manager.release(block1)
    manager.release(block2)
    
    # Check pool stats per device
    stats1 = manager.get_pool_stats(device1)
    stats2 = manager.get_pool_stats(device2)
    
    assert stats1['free_blocks'] >= 1
    assert stats2['free_blocks'] >= 1
    
    # Clean up
    manager.clear()