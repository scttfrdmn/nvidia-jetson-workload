# Molecular Dynamics Simulation

SPDX-License-Identifier: Apache-2.0
Copyright 2025 Scott Friedman and Project Contributors

## Overview
This module implements a molecular dynamics (MD) simulation for the NVIDIA Jetson workload demo. The simulation models atomic interactions in molecular systems and supports:

- Different molecular force fields (AMBER, CHARMM-like)
- Thermostats for temperature control
- Periodic boundary conditions
- GPU acceleration with CUDA
- Automatic scaling based on GPU capabilities

## Algorithms
The MD simulation implements several numerical integrators:
- Velocity Verlet
- Leapfrog
- Beeman's Algorithm

Force fields include:
- Lennard-Jones potential for van der Waals interactions
- Coulomb potential for electrostatic interactions
- Harmonic potentials for bond and angle terms

## Hardware Scaling
The implementation automatically scales based on the detected GPU hardware:
- NVIDIA Jetson Orin NX (Ampere architecture, SM 8.7)
- AWS Graviton g5g (Tesla T4, Turing architecture, SM 7.5)
- Fallback CPU implementation

## Usage
See the examples directory for complete demonstration of how to use this module.

```cpp
// Basic usage
auto system = molecular_dynamics::MolecularSystem::load_from_file("protein.pdb");
auto simulation = molecular_dynamics::Simulation(system);
simulation.run(10.0); // Run for 10 picoseconds
simulation.save_trajectory("trajectory.dcd");
```

## Python Bindings
Python bindings are available through pybind11. See the Python directory for examples.

## Performance
Performance metrics for different systems:
- Small proteins: ~10-50 ns/day on Jetson Orin NX
- Medium systems (up to 100K atoms): ~1-5 ns/day on AWS Graviton g5g instances
- Water boxes: scales with system size, ~20-100 ns/day depending on size

## References
1. Allen, M. P., & Tildesley, D. J. (2017). Computer simulation of liquids. Oxford University Press.
2. Frenkel, D., & Smit, B. (2002). Understanding molecular simulation: from algorithms to applications. Academic Press.