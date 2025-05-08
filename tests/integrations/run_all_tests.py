#!/usr/bin/env python3
"""
Test runner for all integration tests.

This script runs all integration tests across different cross-workload integrations.

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors
"""

import os
import sys
import unittest
import importlib.util

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

def run_all_tests():
    """Run all integration tests."""
    # Discover all tests in the integrations directory
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1

def run_integration_tests(integration_name):
    """Run tests for a specific integration."""
    # Check if the integration exists
    integration_dir = os.path.join(script_dir, integration_name)
    if not os.path.isdir(integration_dir):
        print(f"Error: Integration '{integration_name}' not found")
        return 1
    
    # Check if the integration has a run_tests.py script
    run_tests_path = os.path.join(integration_dir, 'run_tests.py')
    if os.path.exists(run_tests_path):
        # Import and run the run_tests function from the script
        spec = importlib.util.spec_from_file_location(
            f"{integration_name}.run_tests", run_tests_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'run_tests'):
            return module.run_tests()
    
    # Fall back to discovering tests in the integration directory
    loader = unittest.TestLoader()
    suite = loader.discover(integration_dir, pattern='test_*.py')
    
    # Create a test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = runner.run(suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run integration tests')
    parser.add_argument('--integration', type=str, default=None, 
                      help='Name of specific integration to test')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    args = parser.parse_args()
    
    # Set verbosity level
    verbosity = 2 if args.verbose else 1
    
    if args.integration:
        # Run tests for specific integration
        sys.exit(run_integration_tests(args.integration))
    else:
        # Run all tests
        sys.exit(run_all_tests())