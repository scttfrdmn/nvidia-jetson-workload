# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added
- feat(benchmark): add comprehensive cost comparison between local computing and cloud options
- feat(benchmark): add support for DGX systems in cost modeling (A100, H100, DGX Station, SuperPOD)
- feat(benchmark): implement flexible Slurm cluster cost modeling with different node types
- feat(benchmark): add YAML configuration file support for customizing compute environments
- feat(benchmark): create example configuration files for various DGX and Slurm setups
- feat(benchmark): add visualization functions for cost comparison, scaling analysis, and enterprise deployment
- feat(benchmark): implement automated testing for cost modeling components
- docs: add detailed cost comparison user guide with enterprise deployment scenarios
- docs: add sample cost comparison reports with detailed examples and visualizations
- docs: create step-by-step walkthrough of cost calculations for different computing environments
- docs: add guide for interpreting cost metrics and making infrastructure decisions

### Improved
- refactor(benchmark): enhance benchmark_suite.py to support new cost modeling options
- refactor(benchmark): update run_benchmarks.sh with comprehensive cost modeling parameters
- refactor(benchmark): optimize visualization.py with specialized reports for different computing environments

## 1.0.0 (2025-05-03)

### Added
- Complete project implementation with all workloads, benchmarking, visualization, deployment, and release management
- docs: add architecture diagram and AWS dataset info
- docs: add development principles and validation process
- docs: add OS information for target environments
- feat: add cluster configuration and deployment tools
- feat: add containerization support and Python environment requirements
- feat(nbody-sim): implement Python N-body simulation

### Improved
- Enhance release script to handle systems without CUDA
- Update version files for 1.0.0 release

### Fixed
- Fix create_release.sh script to handle macOS sed limitations
- Fix Python handling in release script and clean up CHANGELOG



