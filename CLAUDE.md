# Claude Code Information for NVIDIA Jetson Workload

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2024 nvidia-jetson-workload contributors -->

This file contains information for Claude Code to effectively assist with this project.

## Project Overview

NVIDIA Jetson Workload is a collection of scientific demo workloads for NVIDIA Jetson Orin NX systems. It includes implementations of weather simulation, medical image processing, and N-body gravitational simulation in both C++/CUDA and Python. The project includes a browser-based visualization dashboard and Slurm integration for job submission.

## Code Style Guidelines

- Follow PEP 8 for Python code
- Follow Google C++ Style Guide for C++ code
- Include SPDX license identifiers in all files
- Use clear, descriptive variable and function names
- Write comprehensive docstrings/comments for functions and classes
- Use consistent formatting in Protocol Buffer definitions

## Testing

When making changes, ensure to run the appropriate tests:

```bash
# Run pytest for Python tests
pytest

# Add more test commands as they become available
```

## Linting and Type Checking

Before committing changes, ensure to run the linting and type checking tools:

```bash
# For Python code
flake8 .
mypy .

# Add more lint/typecheck commands as they become available
```

## Documentation

Code should be well-documented with:
- Function/method docstrings
- Module docstrings
- Inline comments for complex logic

## Project Structure

The project is organized as follows:
- `/src` - Main source code
  - `/weather-sim` - Weather simulation workload
    - `/cpp` - C++/CUDA implementation
    - `/python` - Python implementation
  - `/medical-imaging` - Medical image processing workload
    - `/cpp` - C++/CUDA implementation
    - `/python` - Python implementation
  - `/nbody-sim` - N-body simulation workload
    - `/cpp` - C++/CUDA implementation
    - `/python` - Python implementation
  - `/proto` - Protocol Buffer definitions
  - `/visualization` - React-based dashboard
  - `/slurm` - Slurm integration scripts
- `/tests` - Test suite
- `/docs` - Documentation
- `/examples` - Example code and usage
- `/scripts` - Development and deployment scripts

## GitHub Workflow

Follow this process for commits and pushes:

1. Make small, focused commits with clear messages
2. Follow the conventional commit format: `type(scope): message`
   - Types: feat, fix, docs, style, refactor, perf, test, chore
   - Example: `feat(weather-sim): implement initial CUDA kernel`

3. Push changes regularly:
   ```bash
   # After making commits
   git push origin main
   ```

4. For significant features, use feature branches:
   ```bash
   # Create and switch to new branch
   git checkout -b feature/new-feature-name
   
   # When ready to merge
   git checkout main
   git merge feature/new-feature-name
   git push origin main
   ```

5. Before each commit, run appropriate linters and tests

## Cross-Compilation and Deployment

For Mac-to-Jetson development:

1. Use SSH/SCP to transfer files to Jetson nodes:
   ```bash
   # Example deployment script (run from project root)
   scripts/deploy.sh <jetson-node-ip>
   ```

2. Use containerization for consistent environments:
   ```bash
   # Build container on Jetson
   scripts/build-container.sh <workload-name>
   
   # Run workload in container
   scripts/run-container.sh <workload-name>
   ```

## License

This project is licensed under the Apache License 2.0. All contributions must adhere to this license.