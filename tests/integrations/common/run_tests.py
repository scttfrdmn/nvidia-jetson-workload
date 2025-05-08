#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

"""
Test runner for data transfer utilities integration tests.

This script provides a convenient way to run the integration tests
for the data transfer utilities.
"""

import os
import sys
import argparse
import subprocess

def main():
    """Run integration tests for data transfer utilities."""
    parser = argparse.ArgumentParser(
        description="Run integration tests for data transfer utilities"
    )
    parser.add_argument(
        "--test-file", 
        choices=["shared_memory", "gpu_memory_manager", "array_ops", "all"],
        default="all",
        help="Specific test file to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Run only GPU tests"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run only CPU tests"
    )
    
    args = parser.parse_args()
    
    # Determine test path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up the pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Handle test selection
    if args.test_file == "all":
        cmd.append(script_dir)
    else:
        cmd.append(os.path.join(script_dir, f"test_{args.test_file}.py"))
    
    # Handle CPU/GPU selection
    if args.gpu_only:
        cmd.extend(["-k", "gpu"])
    elif args.cpu_only:
        cmd.extend(["-k", "not gpu"])
    
    # Run the tests
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main())