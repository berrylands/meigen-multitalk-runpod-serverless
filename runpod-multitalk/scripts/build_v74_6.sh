#!/bin/bash

# Build script for MultiTalk V74.6 - Fixed src directory
set -e

echo "=============================================="
echo "Building MultiTalk V74.6 - Fixed src directory"
echo "=============================================="

# Navigate to the runpod-multitalk directory
cd "$(dirname "$0")/.."

# Get current timestamp for build tracking
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Build time: $BUILD_TIME"

# Verify required files exist
echo "Checking required files..."
required_files=(
    "Dockerfile.v74-6-fixed-src"
    "handler_v74_5_adaptive.py"
    "multitalk_v74_5_adaptive.py"
    "diagnostic_wan_model.py"
    "requirements_v74_3_compatible.txt"
    "setup_v74_6_fixed_src.sh"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Error: Required file $file not found"
        exit 1
    fi
    echo "✅ Found: $file"
done

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi

# Build the Docker image
IMAGE_TAG="berrylands/multitalk-v74-6:fixed-src"
echo "Building Docker image: $IMAGE_TAG"

docker build \
    --file Dockerfile.v74-6-fixed-src \
    --tag "$IMAGE_TAG" \
    --build-arg BUILD_TIME="$BUILD_TIME" \
    --progress=plain \
    .

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 BUILD SUCCESSFUL!"
    echo "Image: $IMAGE_TAG"
    echo "Features:"
    echo "  - Fixed src directory (no more 404 errors)"
    echo "  - Proper vram_management modules"
    echo "  - Fallback stubs for missing files"
    echo "  - High-quality lip-sync parameters (25fps)"
    echo "  - Comprehensive validation framework"
    echo ""
    echo "Ready to push to Docker Hub!"
    echo "Run: docker push $IMAGE_TAG"
else
    echo "❌ BUILD FAILED"
    exit 1
fi