# Integration Tests for Data Transfer Utilities

This directory contains integration tests for the cross-workload data transfer utilities.

## Overview

The tests in this directory verify that the data transfer optimization utilities work correctly in integration scenarios, including:

1. Shared memory operations for efficient data transfer
2. GPU memory management for cross-workload operations
3. Array operations optimized for cross-device execution
4. Cross-workload integration patterns

## Running the Tests

To run all integration tests:

```bash
# From the project root
pytest tests/integrations/common/ -v
```

To run specific test files:

```bash
# Test shared memory utilities
pytest tests/integrations/common/test_shared_memory.py -v

# Test GPU memory manager
pytest tests/integrations/common/test_gpu_memory_manager.py -v

# Test array operations
pytest tests/integrations/common/test_array_ops.py -v
```

## Test Requirements

- Python 3.10+
- NumPy
- CuPy (for GPU tests)
- pytest

GPU tests will be skipped automatically if CUDA/CuPy is not available.

## Test Coverage

These tests cover:

1. **Data Transfer Between Workloads**
   - CPU-to-CPU transfer
   - CPU-to-GPU transfer
   - GPU-to-CPU transfer
   - GPU-to-GPU transfer
   - Cross-process sharing

2. **Memory Management**
   - Memory block allocation and tracking
   - Memory pooling for reuse
   - Garbage collection
   - Multi-device support

3. **Array Operations**
   - Element-wise operations
   - Matrix multiplication
   - Reduction operations
   - Custom function application

4. **Integration Scenarios**
   - Integration between shared memory and array operations
   - Integration between GPU memory manager and array operations
   - Cross-workload memory sharing

## Adding New Tests

When adding new integration tests:

1. Follow the existing pattern with clear test functions
2. Use pytest fixtures from conftest.py
3. Make tests skip gracefully when CUDA is not available
4. Add appropriate assertions to verify expected behavior
5. Clean up resources after tests complete