# Claude Code Information for NVIDIA Jetson Workload

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2024 nvidia-jetson-workload contributors -->

This file contains information for Claude Code to effectively assist with this project.

## Project Overview

NVIDIA Jetson Workload is a toolkit for optimizing and managing workloads on NVIDIA Jetson devices. The project aims to simplify deployment, monitoring, and optimization of applications running on Jetson platforms.

## Code Style Guidelines

- Follow PEP 8 for Python code
- Include SPDX license identifiers in all files
- Use clear, descriptive variable and function names
- Write comprehensive docstrings for functions and classes

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
- `/tests` - Test suite
- `/docs` - Documentation
- `/examples` - Example code and usage

## License

This project is licensed under the Apache License 2.0. All contributions must adhere to this license.