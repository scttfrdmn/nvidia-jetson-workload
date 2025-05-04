#!/usr/bin/env python3
"""
Test runner for the geospatial module.
This script runs all the tests in the tests directory.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run tests for the geospatial module')
    parser.add_argument('--verbose', '-v', action='store_true', help='Increase verbosity')
    parser.add_argument('--pattern', '-p', default='test_*.py', help='Pattern to match test files')
    parser.add_argument('--skip-gpu', action='store_true', help='Skip tests that require GPU')
    args = parser.parse_args()
    
    # Get the directory containing this script
    script_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Add parent directory to Python path so we can import the geospatial package
    sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))
    
    try:
        import pytest
        
        # Build the pytest arguments
        pytest_args = [script_dir]
        
        if args.verbose:
            pytest_args.append('-v')
        
        # Add pattern
        pytest_args.extend(['-k', args.pattern])
        
        # Skip GPU tests if requested
        if args.skip_gpu:
            pytest_args.extend(['-m', 'not gpu'])
        
        # Run the tests
        return pytest.main(pytest_args)
    
    except ImportError:
        print("pytest not found. Falling back to unittest discovery.")
        # Fallback to unittest discovery if pytest is not available
        import unittest
        
        # Build the test suite
        loader = unittest.TestLoader()
        suite = loader.discover(script_dir, pattern=args.pattern)
        
        # Configure the test runner
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        
        # Run the tests
        result = runner.run(suite)
        return not result.wasSuccessful()

if __name__ == '__main__':
    sys.exit(main())