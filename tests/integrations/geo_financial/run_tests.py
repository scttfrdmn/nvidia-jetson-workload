#!/usr/bin/env python3
"""
Test runner for geospatial financial integration tests.

This script runs all the integration tests for the geospatial financial integration.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import unittest

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(project_root)

def run_tests():
    """Run all geospatial financial integration tests."""
    # Discover tests in this directory
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run geospatial financial integration tests')
    parser.add_argument('--test', type=str, default=None, 
                      help='Specific test module to run (without .py extension)')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    args = parser.parse_args()
    
    # Set verbosity level
    verbosity = 2 if args.verbose else 1
    
    if args.test:
        # Run specific test module
        test_module = args.test
        if test_module.endswith('.py'):
            test_module = test_module[:-3]
        
        # Discover tests in the specified module
        suite = unittest.defaultTestLoader.loadTestsFromName(f"tests.integrations.geo_financial.{test_module}")
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run all tests
        sys.exit(run_tests())