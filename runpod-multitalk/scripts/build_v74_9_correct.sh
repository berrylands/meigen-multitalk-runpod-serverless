#!/bin/bash

# Build script for MultiTalk V74.9 - Correct interface implementation
set -e

echo "==============================================" 
echo "Building MultiTalk V74.9 - CORRECT Interface"
echo "Based on actual implementation analysis"
echo "=============================================="

# Navigate to the runpod-multitalk directory
cd "$(dirname "$0")/.."

# Get current timestamp
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Build time: $BUILD_TIME"

# Verify required files
echo "Checking required files..."
required_files=(
    "Dockerfile.v74-9-correct"
    "handler_v74_9_correct.py"
    "multitalk_v74_9_correct.py"
    "requirements_v74_3_compatible.txt"
    "setup_v74_8_fixed.sh"
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
IMAGE_TAG="berrylands/multitalk-v74-9:correct-interface"
echo "Building Docker image: $IMAGE_TAG"

docker build \
    --file Dockerfile.v74-9-correct \
    --tag "$IMAGE_TAG" \
    --build-arg BUILD_TIME="$BUILD_TIME" \
    --progress=plain \
    .

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 BUILD SUCCESSFUL!"
    echo "Image: $IMAGE_TAG"
    echo ""
    echo "🎯 KEY IMPROVEMENTS IN V74.9:"
    echo "  - Uses CORRECT command-line arguments based on official interface"
    echo "  - Properly handles --ckpt_dir, --wav2vec_dir, --frame_num"
    echo "  - Supports all official parameters like --use_teacache, --mode"
    echo "  - Based on analysis of zsxkib's working implementation"
    echo "  - No more guessing at arguments!"
    echo ""
    echo "📋 WHAT WE LEARNED:"
    echo "  - The official script uses different argument names than we assumed"
    echo "  - There's both a subprocess interface and Python API interface"
    echo "  - zsxkib uses the Python API, but we can use subprocess with correct args"
    echo "  - The model supports various optimization modes (turbo, teacache)"
    echo ""
    echo "Ready to push to Docker Hub!"
    echo "Run: docker push $IMAGE_TAG"
else
    echo "❌ BUILD FAILED"
    exit 1
fi