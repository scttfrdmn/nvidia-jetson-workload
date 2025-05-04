# Release Process Guide

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2025 Scott Friedman and Project Contributors -->

This guide outlines the process for creating and publishing new releases of the GPU-accelerated scientific workloads package.

## Table of Contents

1. [Overview](#overview)
2. [Version Numbering](#version-numbering)
3. [Release Preparation](#release-preparation)
4. [Release Building](#release-building)
5. [Release Publishing](#release-publishing)
6. [Post-Release Activities](#post-release-activities)
7. [Hotfix Releases](#hotfix-releases)
8. [Release Automation](#release-automation)

## Overview

The release process ensures that each version of the software is properly versioned, tested, documented, and distributed to users. The process includes preparation, building, publishing, and post-release activities.

## Version Numbering

The project follows semantic versioning (SemVer):

```
MAJOR.MINOR.PATCH
```

Where:
- **MAJOR**: Incremented for incompatible API changes
- **MINOR**: Incremented for new features in a backward-compatible manner
- **PATCH**: Incremented for backward-compatible bug fixes

Examples:
- 1.0.0: Initial stable release
- 1.1.0: Added new feature
- 1.1.1: Bug fix

## Release Preparation

### 1. Update Documentation

- Update README.md with new features and changes
- Update API documentation if there are interface changes
- Update user guides with new functionality
- Check that all code examples work with the new version

### 2. Update Version Numbers

Update version numbers in:
- pyproject.toml
- package.json (for visualization dashboard)
- CMakeLists.txt files
- Documentation references

### 3. Update Changelog

Create or update CHANGELOG.md with:
- New features
- Improvements
- Bug fixes
- Breaking changes
- Deprecated features

Example format:

```markdown
## 1.2.0 (2025-05-15)

### Added
- Added weather simulation workload
- Added benchmarking support for weather simulation

### Improved
- Optimized N-body simulation performance by 30%
- Enhanced GPU adaptability pattern for T4 GPUs

### Fixed
- Fixed memory leak in medical imaging registration
- Fixed incorrect metric reporting in benchmark results

### Breaking Changes
- Changed parameter order in molecular dynamics API
```

### 4. Run Quality Checks

- Run all tests to ensure they pass
- Run code linters
- Run static analyzers
- Check for memory leaks
- Run benchmarks to ensure performance hasn't regressed

### 5. Create Release Branch

Create a release branch from the main branch:

```bash
git checkout -b release/v1.2.0
```

## Release Building

### 1. Build All Components

Build all components to ensure they compile correctly:

```bash
# Use the build script
./build.sh

# Or build individual workloads
cd src/nbody_sim/cpp && ./build_and_test.sh
cd src/molecular-dynamics/cpp && ./build.sh
cd src/weather-sim/cpp && cmake -B build -S . && cmake --build build
cd src/medical-imaging/cpp && cmake -B build -S . && cmake --build build
```

### 2. Run Integration Tests

Run integration tests to ensure all components work together:

```bash
cd tests
python -m pytest -xvs
```

### 3. Create Distribution Packages

#### Create Python Packages

```bash
# Build Python packages
python -m build

# Create wheel files
python -m pip wheel . -w dist/
```

#### Create Debian Packages (for Jetson)

```bash
# Create Debian packages
dpkg-buildpackage -b -rfakeroot -us -uc
```

#### Create Docker Images

```bash
# Build Docker images
docker build -t ghcr.io/username/nvidia-jetson-workload/benchmark:1.2.0 -f containers/benchmark.Dockerfile .
```

#### Create Singularity Containers

```bash
# Build Singularity image
singularity build benchmark-1.2.0.sif containers/benchmark.def
```

### 4. Create Release Package

Use the release script to create a complete release package:

```bash
./scripts/create_release.sh 1.2.0
```

This creates:
- Source code archive
- Binary packages for different platforms
- Documentation
- Examples
- Release notes

## Release Publishing

### 1. Create Git Tag

Create and push a Git tag for the release:

```bash
git tag -a v1.2.0 -m "Release v1.2.0"
git push origin v1.2.0
```

### 2. Create GitHub Release

Create a GitHub release:
1. Go to the GitHub repository
2. Click "Releases"
3. Click "Draft a new release"
4. Select the tag
5. Add release title and description
6. Upload release assets
7. Publish release

### 3. Publish Python Packages

Publish Python packages to PyPI:

```bash
python -m twine upload dist/*
```

### 4. Publish Docker Images

Push Docker images to GitHub Container Registry:

```bash
docker push ghcr.io/username/nvidia-jetson-workload/benchmark:1.2.0
docker push ghcr.io/username/nvidia-jetson-workload/benchmark:latest
```

### 5. Publish Documentation

Update documentation on the website or documentation hosting platform:

```bash
# Deploy documentation
./scripts/deploy_docs.sh
```

## Post-Release Activities

### 1. Announce Release

Announce the release through appropriate channels:
- Project website
- GitHub discussions
- Mailing lists
- Social media
- Blog posts

### 2. Merge Back to Main

Merge any changes from the release branch back to the main branch:

```bash
git checkout main
git merge --no-ff release/v1.2.0
git push origin main
```

### 3. Close Milestone

Close the milestone associated with the release on GitHub.

### 4. Update Roadmap

Update the project roadmap with the next planned features and milestones.

## Hotfix Releases

Hotfix releases address critical bugs in a released version.

### 1. Create Hotfix Branch

Create a hotfix branch from the release tag:

```bash
git checkout -b hotfix/v1.2.1 v1.2.0
```

### 2. Fix the Issue

Make the necessary changes to fix the issue.

### 3. Update Version Numbers

Update version numbers to the new patch version.

### 4. Update Changelog

Add the fix to the changelog.

### 5. Test the Fix

Thoroughly test the fix to ensure it resolves the issue without introducing new problems.

### 6. Release the Hotfix

Follow the same release process as for a regular release, but with a patch version increment.

## Release Automation

### 1. GitHub Actions Workflow

The `.github/workflows/release.yml` workflow can automate the release process:

```yaml
name: Release

on:
  workflow_dispatch:
    inputs:
      version:
        description: 'Version number (e.g., 1.2.0)'
        required: true
      type:
        description: 'Release type'
        required: true
        default: 'minor'
        type: choice
        options:
          - major
          - minor
          - patch
          - hotfix
```

This workflow automates:
- Version number updates
- Building all components
- Running tests
- Creating distribution packages
- Publishing to GitHub Releases
- Publishing to PyPI
- Publishing Docker images

### 2. Release Checklist

Use the release checklist to ensure all steps are completed:

```markdown
# Release Checklist for v1.2.0

## Preparation
- [ ] Update documentation
- [ ] Update version numbers
- [ ] Update changelog
- [ ] Run quality checks
- [ ] Create release branch

## Building
- [ ] Build all components
- [ ] Run integration tests
- [ ] Create distribution packages
- [ ] Create release package

## Publishing
- [ ] Create Git tag
- [ ] Create GitHub release
- [ ] Publish Python packages
- [ ] Publish Docker images
- [ ] Publish documentation

## Post-Release
- [ ] Announce release
- [ ] Merge back to main
- [ ] Close milestone
- [ ] Update roadmap
```

### 3. Release Script

Use the `create_release.sh` script to automate the release process:

```bash
./scripts/create_release.sh 1.2.0 --type minor --publish
```

#### Script Options

The script provides several command-line options:

```
Usage: ./scripts/create_release.sh VERSION [OPTIONS]
Create a release package for the project

Arguments:
  VERSION             Version number (e.g., 1.2.0)

Options:
  -h, --help          Show this help message
  -t, --type TYPE     Release type: major, minor, patch, hotfix (default: minor)
  -d, --dir DIR       Directory to store release files (default: release)
  -p, --publish       Publish the release (tag git, push to GitHub, etc.)
```

#### What the Script Does

The script performs these steps:

1. **Updates version numbers** in:
   - pyproject.toml
   - package.json for the visualization dashboard
   - CMakeLists.txt files throughout the project

2. **Updates changelog** by:
   - Creating a new entry in CHANGELOG.md
   - Categorizing commits since the last release as Added, Improved, Fixed, or Breaking Changes
   - Using commit messages to populate each category

3. **Builds all components** by running:
   - The main build.sh script
   - Individual build scripts for each workload

4. **Creates distribution packages**:
   - Creates a source distribution archive
   - Builds Python wheel packages for each workload
   - Copies binary libraries and executables to the release directory

5. **Packages documentation and support files**:
   - Copies all documentation to the release package
   - Includes README, LICENSE, CONTRIBUTING, etc.
   - Creates a detailed RELEASE_NOTES.md file

6. **When used with --publish**:
   - Commits version changes to git
   - Creates and pushes a git tag
   - Creates a GitHub release
   - Publishes Python packages to PyPI (using twine)
   - Builds and pushes Docker images to the configured registry
   - Builds and publishes Singularity containers if available

#### Environment Variables

The script uses these environment variables:

- **TWINE_USERNAME** and **TWINE_PASSWORD**: For PyPI publishing
- **DOCKER_REGISTRY**: The Docker registry to push images to (default: ghcr.io/scttfrdmn)
- **SYLABS_TOKEN**: Token for publishing Singularity containers to Sylabs.io library

#### Examples

Basic release creation:
```bash
./scripts/create_release.sh 1.2.0
```

Create and publish a minor release:
```bash
./scripts/create_release.sh 1.2.0 --type minor --publish
```

Create a patch release with custom output directory:
```bash
./scripts/create_release.sh 1.2.1 --type patch --dir custom_release
```

Create a hotfix release and publish it:
```bash
./scripts/create_release.sh 1.2.2 --type hotfix --publish
```

### 4. Configure Release Notifications

Set up automated notifications for new releases:
- Email notifications
- Slack/Discord messages
- RSS feeds
- Webhook integrations for CI/CD systems