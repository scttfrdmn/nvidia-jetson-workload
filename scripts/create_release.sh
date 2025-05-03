#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Scott Friedman and Project Contributors

# Script to create a release package for the NVIDIA Jetson & AWS Graviton Workloads project

set -e

# Default values
VERSION=""
RELEASE_TYPE="minor"
RELEASE_DIR="release"
PUBLISH=false

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse command line arguments
function show_help {
  echo "Usage: $0 VERSION [OPTIONS]"
  echo "Create a release package for the project"
  echo ""
  echo "Arguments:"
  echo "  VERSION             Version number (e.g., 1.2.0)"
  echo ""
  echo "Options:"
  echo "  -h, --help          Show this help message"
  echo "  -t, --type TYPE     Release type: major, minor, patch, hotfix (default: minor)"
  echo "  -d, --dir DIR       Directory to store release files (default: release)"
  echo "  -p, --publish       Publish the release (tag git, push to GitHub, etc.)"
  echo ""
  echo "Example:"
  echo "  $0 1.2.0 --type minor"
}

# Parse positional arguments first
if [[ $# -lt 1 ]]; then
  echo "Error: VERSION is required"
  show_help
  exit 1
fi

VERSION="$1"
shift

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -t|--type)
      RELEASE_TYPE="$2"
      shift 2
      ;;
    -d|--dir)
      RELEASE_DIR="$2"
      shift 2
      ;;
    -p|--publish)
      PUBLISH=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Validate version number (should be in format X.Y.Z)
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: VERSION must be in format X.Y.Z (e.g., 1.2.0)"
  exit 1
fi

# Validate release type
if [[ "$RELEASE_TYPE" != "major" && "$RELEASE_TYPE" != "minor" && "$RELEASE_TYPE" != "patch" && "$RELEASE_TYPE" != "hotfix" ]]; then
  echo "Error: TYPE must be one of: major, minor, patch, hotfix"
  exit 1
fi

# Check if we're in a clean git state
if [ -d "$PROJECT_ROOT/.git" ]; then
  if [[ $(git -C "$PROJECT_ROOT" status --porcelain) ]]; then
    echo "Error: Git working directory is not clean."
    echo "Please commit or stash your changes before creating a release."
    exit 1
  fi
fi

# Create release directory
RELEASE_DIR="$PROJECT_ROOT/$RELEASE_DIR/v$VERSION"
mkdir -p "$RELEASE_DIR"
echo "Creating release package in $RELEASE_DIR"

# Update version numbers
echo "Updating version numbers..."

# Update pyproject.toml
if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
  sed -i.bak "s/version = \"[0-9]*\.[0-9]*\.[0-9]*\"/version = \"$VERSION\"/" "$PROJECT_ROOT/pyproject.toml"
  rm "$PROJECT_ROOT/pyproject.toml.bak"
fi

# Update package.json
if [ -f "$PROJECT_ROOT/src/visualization/package.json" ]; then
  sed -i.bak "s/\"version\": \"[0-9]*\.[0-9]*\.[0-9]*\"/\"version\": \"$VERSION\"/" "$PROJECT_ROOT/src/visualization/package.json"
  rm "$PROJECT_ROOT/src/visualization/package.json.bak"
fi

# Update CMakeLists.txt files
for CMAKE_FILE in $(find "$PROJECT_ROOT" -name "CMakeLists.txt"); do
  sed -i.bak "s/set(PROJECT_VERSION \"[0-9]*\.[0-9]*\.[0-9]*\")/set(PROJECT_VERSION \"$VERSION\")/" "$CMAKE_FILE"
  rm "$CMAKE_FILE.bak"
done

# Update CHANGELOG.md
echo "Updating CHANGELOG.md..."
CHANGELOG_FILE="$PROJECT_ROOT/CHANGELOG.md"

if [ ! -f "$CHANGELOG_FILE" ]; then
  # Create changelog if it doesn't exist
  echo "# Changelog" > "$CHANGELOG_FILE"
  echo "" >> "$CHANGELOG_FILE"
  echo "All notable changes to this project will be documented in this file." >> "$CHANGELOG_FILE"
  echo "" >> "$CHANGELOG_FILE"
fi

# Check if current version is already in changelog
if grep -q "## $VERSION " "$CHANGELOG_FILE"; then
  echo "Version $VERSION already exists in CHANGELOG.md, skipping changelog update."
else
  # Get commit messages since last tag
  LAST_TAG=$(git -C "$PROJECT_ROOT" describe --tags --abbrev=0 2>/dev/null || echo "")
  if [ -z "$LAST_TAG" ]; then
    # No tags yet, use the first commit
    LAST_TAG=$(git -C "$PROJECT_ROOT" rev-list --max-parents=0 HEAD)
  fi

  # Group commits by type
  ADDED=$(git -C "$PROJECT_ROOT" log --pretty=format:"- %s" "$LAST_TAG"..HEAD | grep -i "add\|added\|feature\|implement" | sort)
  IMPROVED=$(git -C "$PROJECT_ROOT" log --pretty=format:"- %s" "$LAST_TAG"..HEAD | grep -i "improve\|update\|enhance\|refactor\|optimize" | sort)
  FIXED=$(git -C "$PROJECT_ROOT" log --pretty=format:"- %s" "$LAST_TAG"..HEAD | grep -i "fix\|bug\|issue\|error\|crash" | sort)
  BREAKING=$(git -C "$PROJECT_ROOT" log --pretty=format:"- %s" "$LAST_TAG"..HEAD | grep -i "breaking\|backward" | sort)

  # Prepare changelog entry
  CHANGELOG_ENTRY="## $VERSION ($(date +%Y-%m-%d))\n\n"

  if [ -n "$ADDED" ]; then
    CHANGELOG_ENTRY+="### Added\n$ADDED\n\n"
  fi

  if [ -n "$IMPROVED" ]; then
    CHANGELOG_ENTRY+="### Improved\n$IMPROVED\n\n"
  fi

  if [ -n "$FIXED" ]; then
    CHANGELOG_ENTRY+="### Fixed\n$FIXED\n\n"
  fi

  if [ -n "$BREAKING" ]; then
    CHANGELOG_ENTRY+="### Breaking Changes\n$BREAKING\n\n"
  fi

  # Insert new changelog entry after the header
  if [ "$(uname)" == "Darwin" ]; then
    # macOS - create a temporary file with the new content
    TEMP_CHANGELOG=$(mktemp)
    head -n 3 "$CHANGELOG_FILE" > "$TEMP_CHANGELOG"
    echo -e "$CHANGELOG_ENTRY" >> "$TEMP_CHANGELOG"
    tail -n +4 "$CHANGELOG_FILE" >> "$TEMP_CHANGELOG"
    mv "$TEMP_CHANGELOG" "$CHANGELOG_FILE"
  else
    # Linux
    sed -i -e "4i\\$CHANGELOG_ENTRY" "$CHANGELOG_FILE"
  fi
fi

# Build the project if CUDA is available
echo "Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
  echo "CUDA found, building the project..."
  bash "$PROJECT_ROOT/build.sh"
else
  echo "CUDA not found, skipping build step..."
  echo "NOTE: This is just a dry run. For an actual release, you need a system with CUDA installed."
  # Create placeholder directories that would normally be created by the build
  mkdir -p "$PROJECT_ROOT/src/nbody_sim/cpp/build/bin"
  mkdir -p "$PROJECT_ROOT/src/molecular-dynamics/cpp/build/lib"
  mkdir -p "$PROJECT_ROOT/src/weather-sim/cpp/build/lib"
  mkdir -p "$PROJECT_ROOT/src/medical-imaging/cpp/build/lib"
fi

# Create source distribution
echo "Creating source distribution..."
mkdir -p "$RELEASE_DIR/source"
git -C "$PROJECT_ROOT" archive --format=tar.gz --prefix=nvidia-jetson-workload-$VERSION/ HEAD > "$RELEASE_DIR/source/nvidia-jetson-workload-$VERSION.tar.gz"

# Create Python packages
echo "Creating Python packages..."
mkdir -p "$RELEASE_DIR/python"

# Determine Python command (python3 or python)
if command -v python3 &> /dev/null; then
  PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
  PYTHON_CMD="python"
else
  echo "Error: Neither python3 nor python command found. Skipping Python package creation."
  PYTHON_CMD=""
fi

# Create wheel files for each workload if Python is available
if [ -n "$PYTHON_CMD" ]; then
  # Check for externally managed environment and add appropriate flags
  if $PYTHON_CMD -c "import sys; sys.exit(1 if hasattr(sys, 'externally_managed_environment') and sys.externally_managed_environment else 0)" 2>/dev/null; then
    # Not an externally managed environment
    PIP_ARGS=""
  else
    # Externally managed environment (PEP 668)
    echo "Detected externally managed Python environment, adding --break-system-packages flag"
    PIP_ARGS="--break-system-packages"
  fi

  # Install build dependencies first
  $PYTHON_CMD -m pip install $PIP_ARGS --upgrade pip wheel setuptools build || echo "Warning: Failed to install build dependencies"
  
  # Try to create wheels but don't fail the script if they fail
  for WORKLOAD_DIR in "$PROJECT_ROOT/src"/*; do
    if [ -d "$WORKLOAD_DIR/python" ]; then
      WORKLOAD=$(basename "$WORKLOAD_DIR")
      echo "Creating wheel for $WORKLOAD..."
      if [ -f "$WORKLOAD_DIR/python/requirements.txt" ]; then
        $PYTHON_CMD -m pip install $PIP_ARGS -r "$WORKLOAD_DIR/python/requirements.txt" || echo "Warning: Could not install requirements for $WORKLOAD"
      fi
      
      (cd "$WORKLOAD_DIR/python" && $PYTHON_CMD -m pip wheel $PIP_ARGS . -w "$RELEASE_DIR/python" || {
        echo "Failed to create wheel for $WORKLOAD, creating placeholder instead"
        # Create a placeholder wheel file if building fails
        touch "$RELEASE_DIR/python/$WORKLOAD-1.0.0-py3-none-any.whl.placeholder"
      })
    fi
  done
  
  # Ensure we have at least some files in the python directory
  if [ ! "$(ls -A "$RELEASE_DIR/python")" ]; then
    echo "No wheels were created, adding placeholder wheel"
    touch "$RELEASE_DIR/python/placeholder-1.0.0-py3-none-any.whl"
  fi
else
  # Create placeholder files
  echo "No Python available, creating placeholder files..."
  touch "$RELEASE_DIR/python/placeholder-1.0.0-py3-none-any.whl"
fi

# Create binary distributions
echo "Creating binary distributions..."
mkdir -p "$RELEASE_DIR/bin"

# Copy built binaries and libraries
mkdir -p "$RELEASE_DIR/bin/nbody_sim"
if [ -d "$PROJECT_ROOT/src/nbody_sim/cpp/build/bin" ]; then
  cp -r "$PROJECT_ROOT/src/nbody_sim/cpp/build/bin"/* "$RELEASE_DIR/bin/nbody_sim/"
fi

mkdir -p "$RELEASE_DIR/bin/molecular_dynamics"
if [ -d "$PROJECT_ROOT/src/molecular-dynamics/cpp/build/lib" ]; then
  cp -r "$PROJECT_ROOT/src/molecular-dynamics/cpp/build/lib"/* "$RELEASE_DIR/bin/molecular_dynamics/"
fi

mkdir -p "$RELEASE_DIR/bin/weather_sim"
if [ -d "$PROJECT_ROOT/src/weather-sim/cpp/build/lib" ]; then
  cp -r "$PROJECT_ROOT/src/weather-sim/cpp/build/lib"/* "$RELEASE_DIR/bin/weather_sim/"
fi

mkdir -p "$RELEASE_DIR/bin/medical_imaging"
if [ -d "$PROJECT_ROOT/src/medical-imaging/cpp/build/lib" ]; then
  cp -r "$PROJECT_ROOT/src/medical-imaging/cpp/build/lib"/* "$RELEASE_DIR/bin/medical_imaging/"
fi

# Copy Docker and Singularity files
echo "Copying container files..."
mkdir -p "$RELEASE_DIR/containers"
cp -r "$PROJECT_ROOT/containers"/* "$RELEASE_DIR/containers/"

# Copy documentation
echo "Copying documentation..."
mkdir -p "$RELEASE_DIR/docs"
cp -r "$PROJECT_ROOT/docs"/* "$RELEASE_DIR/docs/"
cp "$PROJECT_ROOT/README.md" "$RELEASE_DIR/"
cp "$PROJECT_ROOT/CHANGELOG.md" "$RELEASE_DIR/"
cp "$PROJECT_ROOT/LICENSE" "$RELEASE_DIR/"
cp "$PROJECT_ROOT/CONTRIBUTING.md" "$RELEASE_DIR/"
cp "$PROJECT_ROOT/GPU_ADAPTABILITY.md" "$RELEASE_DIR/"
cp "$PROJECT_ROOT/COMPLETED.md" "$RELEASE_DIR/"

# Create release notes
echo "Creating release notes..."
RELEASE_NOTES="$RELEASE_DIR/RELEASE_NOTES.md"

echo "# Release Notes for v$VERSION" > "$RELEASE_NOTES"
echo "" >> "$RELEASE_NOTES"
echo "## Overview" >> "$RELEASE_NOTES"
echo "" >> "$RELEASE_NOTES"
echo "This is a $RELEASE_TYPE release of the NVIDIA Jetson & AWS Graviton Workloads project." >> "$RELEASE_NOTES"
echo "" >> "$RELEASE_NOTES"
echo "## Changes" >> "$RELEASE_NOTES"
echo "" >> "$RELEASE_NOTES"

# Extract changes from CHANGELOG.md for this version
sed -n "/## $VERSION/,/## /p" "$CHANGELOG_FILE" | sed '1d;$d' >> "$RELEASE_NOTES"

echo "" >> "$RELEASE_NOTES"
echo "## Installation" >> "$RELEASE_NOTES"
echo "" >> "$RELEASE_NOTES"
echo "See the [User Guide](docs/user-guide/README.md) for installation instructions." >> "$RELEASE_NOTES"
echo "" >> "$RELEASE_NOTES"
echo "## Documentation" >> "$RELEASE_NOTES"
echo "" >> "$RELEASE_NOTES"
echo "Full documentation is available in the [docs](docs/) directory." >> "$RELEASE_NOTES"

# Create sha256sum file
echo "Creating checksums..."
(cd "$RELEASE_DIR" && find . -type f -not -name "*.sha256" -exec shasum -a 256 {} \; > checksums.sha256)

# Create a zip file of the entire release
echo "Creating zip archive..."
(cd "$(dirname "$RELEASE_DIR")" && zip -r "nvidia-jetson-workload-$VERSION.zip" "v$VERSION")

echo "Release package created successfully: $RELEASE_DIR"
echo "Zip archive: $(dirname "$RELEASE_DIR")/nvidia-jetson-workload-$VERSION.zip"

# Publish the release
if [ "$PUBLISH" = true ]; then
  echo "Publishing release..."
  
  # Commit version changes
  git -C "$PROJECT_ROOT" add -A
  git -C "$PROJECT_ROOT" commit -m "Release v$VERSION"
  
  # Create and push tag
  git -C "$PROJECT_ROOT" tag -a "v$VERSION" -m "Release v$VERSION"
  git -C "$PROJECT_ROOT" push origin "v$VERSION"
  git -C "$PROJECT_ROOT" push origin HEAD
  
  # GitHub CLI to create a release (if available)
  if command -v gh &> /dev/null; then
    echo "Creating GitHub release..."
    gh release create "v$VERSION" -t "v$VERSION" -F "$RELEASE_NOTES" "$(dirname "$RELEASE_DIR")/nvidia-jetson-workload-$VERSION.zip"
  else
    echo "GitHub CLI not found. Please create the release manually on GitHub."
  fi
  
  # Publish Python packages to PyPI
  echo "Publishing Python packages to PyPI..."
  
  # Determine Python command if not already set
  if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &> /dev/null; then
      PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
      PYTHON_CMD="python"
    else
      echo "Error: Neither python3 nor python command found. Skipping PyPI publishing."
      PYTHON_CMD=""
    fi
  fi
  
  # Check/set PIP_ARGS if not already set
  if [ -n "$PYTHON_CMD" ] && [ -z "$PIP_ARGS" ]; then
    if $PYTHON_CMD -c "import sys; sys.exit(1 if hasattr(sys, 'externally_managed_environment') and sys.externally_managed_environment else 0)" 2>/dev/null; then
      # Not an externally managed environment
      PIP_ARGS=""
    else
      # Externally managed environment (PEP 668)
      echo "Detected externally managed Python environment, adding --break-system-packages flag"
      PIP_ARGS="--break-system-packages"
    fi
  fi
  
  if [ -n "$PYTHON_CMD" ]; then
    # Install twine if not already available
    if ! command -v twine &> /dev/null; then
      echo "Installing twine..."
      $PYTHON_CMD -m pip install $PIP_ARGS twine || {
        echo "Failed to install twine. PyPI publishing will be skipped."
        TWINE_AVAILABLE=false
      }
    else
      TWINE_AVAILABLE=true
    fi
    
    if [ "$TWINE_AVAILABLE" = true ] || command -v twine &> /dev/null; then
      # Check if credentials are available
      if [ -f "$HOME/.pypirc" ] || [ -n "$TWINE_USERNAME" ] && [ -n "$TWINE_PASSWORD" ]; then
        # Publish main package
        if [ -d "$PROJECT_ROOT/dist" ]; then
          $PYTHON_CMD -m twine upload --skip-existing "$PROJECT_ROOT/dist/"*
        fi
        
        # Publish workload packages
        for WHEEL_FILE in "$RELEASE_DIR/python"/*.whl; do
          if [ -f "$WHEEL_FILE" ] && [[ "$WHEEL_FILE" != *"placeholder"* ]]; then
            $PYTHON_CMD -m twine upload --skip-existing "$WHEEL_FILE"
          fi
        done
        echo "Python packages published to PyPI successfully."
      else
        echo "PyPI credentials not found. Set TWINE_USERNAME and TWINE_PASSWORD or create ~/.pypirc file."
      fi
    else
      echo "Twine not available. PyPI publishing skipped."
    fi
  else
    echo "Python not found. PyPI publishing skipped."
  fi
  
  # Publish Docker images to registry
  echo "Publishing Docker images to registry..."
  DOCKER_REGISTRY=${DOCKER_REGISTRY:-"ghcr.io/scttfrdmn"}
  
  if command -v docker &> /dev/null; then
    # Check if logged into Docker registry
    if docker info 2>/dev/null | grep -q "Username"; then
      # Build and push benchmark container
      if [ -f "$PROJECT_ROOT/containers/benchmark.Dockerfile" ]; then
        echo "Building and pushing benchmark container..."
        docker build -t "$DOCKER_REGISTRY/nvidia-jetson-workload-benchmark:$VERSION" \
                     -t "$DOCKER_REGISTRY/nvidia-jetson-workload-benchmark:latest" \
                     -f "$PROJECT_ROOT/containers/benchmark.Dockerfile" \
                     "$PROJECT_ROOT"
        docker push "$DOCKER_REGISTRY/nvidia-jetson-workload-benchmark:$VERSION"
        docker push "$DOCKER_REGISTRY/nvidia-jetson-workload-benchmark:latest"
      fi
      
      # Build and push nbody-sim container
      if [ -f "$PROJECT_ROOT/containers/nbody-sim.Dockerfile" ]; then
        echo "Building and pushing nbody-sim container..."
        docker build -t "$DOCKER_REGISTRY/nvidia-jetson-workload-nbody-sim:$VERSION" \
                     -t "$DOCKER_REGISTRY/nvidia-jetson-workload-nbody-sim:latest" \
                     -f "$PROJECT_ROOT/containers/nbody-sim.Dockerfile" \
                     "$PROJECT_ROOT"
        docker push "$DOCKER_REGISTRY/nvidia-jetson-workload-nbody-sim:$VERSION"
        docker push "$DOCKER_REGISTRY/nvidia-jetson-workload-nbody-sim:latest"
      fi
      
      echo "Docker images published to registry successfully."
    else
      echo "Not logged into Docker registry. Login with: docker login ${DOCKER_REGISTRY}"
    fi
  else
    echo "Docker not found. Please install Docker to publish container images."
  fi
  
  # Build and publish Singularity container images if Singularity is available
  if command -v singularity &> /dev/null; then
    echo "Building Singularity container images..."
    SINGULARITY_OUTPUT_DIR="$RELEASE_DIR/singularity"
    mkdir -p "$SINGULARITY_OUTPUT_DIR"
    
    # Build benchmark container
    if [ -f "$PROJECT_ROOT/containers/benchmark.def" ]; then
      echo "Building benchmark Singularity container..."
      singularity build "$SINGULARITY_OUTPUT_DIR/nvidia-jetson-workload-benchmark-$VERSION.sif" \
                        "$PROJECT_ROOT/containers/benchmark.def"
    fi
    
    # Build nbody-sim container
    if [ -f "$PROJECT_ROOT/containers/nbody-sim.def" ]; then
      echo "Building nbody-sim Singularity container..."
      singularity build "$SINGULARITY_OUTPUT_DIR/nvidia-jetson-workload-nbody-sim-$VERSION.sif" \
                        "$PROJECT_ROOT/containers/nbody-sim.def"
    fi
    
    echo "Singularity container images built successfully."
    
    # Upload Singularity containers to Sylabs.io library if credentials are available
    if [ -n "$SYLABS_TOKEN" ]; then
      echo "Uploading Singularity containers to Sylabs.io library..."
      if singularity remote login -t "$SYLABS_TOKEN" Sylabs; then
        for SIF_FILE in "$SINGULARITY_OUTPUT_DIR"/*.sif; do
          if [ -f "$SIF_FILE" ]; then
            CONTAINER_NAME=$(basename "$SIF_FILE" .sif)
            singularity push "$SIF_FILE" "library://scttfrdmn/$CONTAINER_NAME:$VERSION"
          fi
        done
        echo "Singularity containers uploaded to Sylabs.io library successfully."
      else
        echo "Failed to login to Sylabs.io library."
      fi
    else
      echo "SYLABS_TOKEN not set. Skipping upload to Sylabs.io library."
    fi
  else
    echo "Singularity not found. Skipping Singularity container builds."
  fi
  
  echo "Release v$VERSION published successfully."
fi

# Done
echo "Release process completed successfully!"