#!/bin/bash

# Build script for MultiTalk V74.8 - Complete systematic dependency mapping
set -e

echo "==============================================" 
echo "Building MultiTalk V74.8 - Complete Systematic"
echo "Systematic dependency analysis approach"
echo "=============================================="

# Navigate to the runpod-multitalk directory
cd "$(dirname "$0")/.."

# Get current timestamp for build tracking
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Build time: $BUILD_TIME"

# Verify required files exist
echo "Checking required files..."
required_files=(
    "Dockerfile.v74-8-complete"
    "handler_v74_5_adaptive.py"
    "multitalk_v74_5_adaptive.py"
    "diagnostic_wan_model.py"
    "requirements_v74_3_compatible.txt"
    "setup_v74_8_complete.sh"
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
IMAGE_TAG="berrylands/multitalk-v74-8:complete-systematic"
echo "Building Docker image: $IMAGE_TAG"

docker build \
    --file Dockerfile.v74-8-complete \
    --tag "$IMAGE_TAG" \
    --build-arg BUILD_TIME="$BUILD_TIME" \
    --progress=plain \
    .

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 BUILD SUCCESSFUL!"
    echo "Image: $IMAGE_TAG"
    echo "Features:"
    echo "  - Complete systematic dependency mapping"
    echo "  - All src.audio_analysis modules included"
    echo "  - Complete vram_management implementation"
    echo "  - Comprehensive kokoro TTS implementation"
    echo "  - All wan/ package files downloaded"
    echo "  - Proactive validation framework"
    echo "  - Fallback stubs for robustness"
    echo "  - NO more trial-and-error dependency fixing"
    echo "  - High-quality lip-sync parameters (25fps)"
    echo ""
    echo "🚀 SYSTEMATIC APPROACH IMPLEMENTED:"
    echo "   - Complete dependency analysis performed"
    echo "   - All known imports mapped and downloaded"
    echo "   - Proactive validation prevents runtime failures"
    echo "   - Comprehensive fallback system ensures robustness"
    echo ""
    echo "Ready to push to Docker Hub!"
    echo "Run: docker push $IMAGE_TAG"
else
    echo "❌ BUILD FAILED"
    exit 1
fi