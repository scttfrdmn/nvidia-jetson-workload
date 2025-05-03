# Claude Code Information for NVIDIA Jetson Workload

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2024 nvidia-jetson-workload contributors -->

This file contains information for Claude Code to effectively assist with this project.

## Project Overview

NVIDIA Jetson Workload is a collection of scientific demo workloads for NVIDIA Jetson Orin NX systems. It includes implementations of weather simulation, medical image processing, and N-body gravitational simulation in both C++/CUDA and Python. The project includes a browser-based visualization dashboard and Slurm integration for job submission.

## Development Principles

1. **Methodical Approach**: Build and test each component incrementally before moving to the next. Focus on having functional implementations rather than attempting to build everything at once.

2. **Test-Driven Development**: Write tests before or alongside implementation. Every feature should have associated tests before being considered complete.

3. **Continuous Validation**: Ensure code builds and passes tests after each meaningful change. Don't defer validation to later stages.

4. **Fix Issues Immediately**: Address bugs, lint errors, and compilation issues as soon as they're discovered, rather than accumulating technical debt.

5. **Tight Integration Loop**: Regularly integrate components to ensure they work together as expected. Avoid prolonged development in isolation.

6. **Focus on Functional Release**: Prioritize having a complete, working system with core functionality over adding advanced features. Start simple and iterate.

7. **Avoid Scope Creep**: Stay focused on the defined requirements. Suggestions for enhancements should be documented for future consideration but not immediately implemented if they distract from core goals.

## Code Style Guidelines

- Follow PEP 8 for Python code
- Follow Google C++ Style Guide for C++ code
- Include SPDX license identifiers in all files
- Use clear, descriptive variable and function names
- Write comprehensive docstrings/comments for functions and classes
- Use consistent formatting in Protocol Buffer definitions

## Testing

Every component must have associated tests. Write tests before or alongside implementation:

```bash
# Run pytest for Python tests
pytest

# Run C++ tests
cd build && ctest

# Run specific test suite
pytest tests/weather-sim/

# Run integration tests
pytest tests/integration/
```

## Validation Pipeline

Follow this validation sequence for all changes:

1. **Linting**: Check code style and formatting
2. **Type Checking**: Verify type annotations and interfaces
3. **Unit Testing**: Test isolated components
4. **Integration Testing**: Test component interactions
5. **Build Verification**: Ensure code compiles on all targets
6. **Performance Validation**: Verify meets timing requirements

```bash
# Python validation pipeline
flake8 .
mypy .
pytest

# C++ validation pipeline
clang-format -i src/**/*.cpp src/**/*.h
cmake --build build --target lint
cmake --build build
ctest -V
```

Run the full validation pipeline before each commit.

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

## Development Workflow

Follow this incremental development process for each component:

1. **Plan**: Define clear requirements and acceptance criteria
2. **Design**: Create minimal viable design with interfaces
3. **Implement**: Write code incrementally, following TDD practices
4. **Validate**: Run the full validation pipeline
5. **Integrate**: Combine with other components, test interactions
6. **Review**: Conduct self-review before committing
7. **Iterate**: Refine based on testing results

## GitHub Workflow

Each commit must represent a complete, validated change:

1. Run the full validation pipeline before each commit
2. Make small, focused commits with clear messages
3. Follow the conventional commit format: `type(scope): message`
   - Types: feat, fix, docs, style, refactor, perf, test, chore
   - Example: `feat(weather-sim): implement initial CUDA kernel`

4. For significant features, use feature branches:
   ```bash
   # Create and switch to new branch
   git checkout -b feature/new-feature-name
   
   # When ready to merge
   git checkout main
   git merge feature/new-feature-name
   git push origin main
   ```

5. Push changes after thorough validation:
   ```bash
   # After making commits and ensuring they pass validation
   git push origin main
   ```

## Cross-Compilation and Deployment

All target systems are running Ubuntu 22.04.x:
- Slurm head node VM on Mac (hostname: `linux-0`)
- Four Jetson Orin NX compute nodes (hostnames: `orin1`, `orin2`, `orin3`, `orin4`)

These systems can be accessed using their respective hostnames via SSH/SCP. SSH key distribution may be required for automated deployment and testing.

Follow this process for testing on target devices:

1. **Local Testing**: First validate all code on development machine
2. **Cross-Compile**: Build for ARM + CUDA target on Ubuntu 22.04
3. **Deploy**: Transfer to Jetson nodes
4. **Test on Device**: Validate functionality on actual hardware
5. **Benchmark**: Measure performance metrics
6. **Report**: Document results and any device-specific issues

```bash
# Deploy to Jetson (run from project root)
scripts/deploy.sh <jetson-node-ip>

# Build container on Jetson
scripts/build-container.sh <workload-name>

# Run workload and tests in container
scripts/run-container.sh <workload-name> --test

# Run performance benchmark
scripts/benchmark.sh <workload-name> --time-target 180
```

## Performance Requirements

Each workload must:
1. Complete within 2-5 minutes on Jetson Orin NX
2. Demonstrate effective GPU utilization (>80%)
3. Show measurable performance advantage over CPU-only version
4. Scale appropriately with available resources
5. Generate comparable results across implementations

## License

This project is licensed under the Apache License 2.0. All contributions must adhere to this license.