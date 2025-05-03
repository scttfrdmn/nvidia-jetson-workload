#!/bin/bash
# Script to update copyright information in all source files

find /Users/scttfrdmn/src/nvidia-jetson-workload -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.cu" -o -name "*.cuh" -o -name "*.py" -o -name "*.md" -o -name "*.cmake" -o -name "CMakeLists.txt" -o -name "*.sh" -o -name "*.def" -o -name "*.Dockerfile" -o -name "*.proto" -o -name "*.tsx" -o -name "*.ts" \) -print0 | 
while IFS= read -r -d $'\0' file; do
    # Skip this script to avoid issues
    if [[ "$file" == *"update_copyright.sh"* ]]; then
        continue
    fi
    
    # Replace old copyright lines
    sed -i '' 's|// SPDX-License-Identifier: Apache-2.0|// SPDX-License-Identifier: Apache-2.0|g' "$file"
    sed -i '' 's|# SPDX-License-Identifier: Apache-2.0|# SPDX-License-Identifier: Apache-2.0|g' "$file"
    sed -i '' 's|<!-- SPDX-License-Identifier: Apache-2.0 -->|<!-- SPDX-License-Identifier: Apache-2.0 -->|g' "$file"
    
    sed -i '' 's|// Copyright 2024 nvidia-jetson-workload contributors|// Copyright 2025 Scott Friedman and Project Contributors|g' "$file"
    sed -i '' 's|# Copyright 2024 nvidia-jetson-workload contributors|# Copyright 2025 Scott Friedman and Project Contributors|g' "$file"
    sed -i '' 's|<!-- Copyright 2024 nvidia-jetson-workload contributors -->|<!-- Copyright 2025 Scott Friedman and Project Contributors -->|g' "$file"
    
    echo "Updated copyright in $file"
done

echo "Copyright update completed"