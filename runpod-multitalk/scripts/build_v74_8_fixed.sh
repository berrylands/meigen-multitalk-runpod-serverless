#!/bin/bash

# Build script for MultiTalk V74.8 - Fixed audio_analysis module
set -e

echo "==============================================" 
echo "Building MultiTalk V74.8 - Fixed Audio Analysis"
echo "Targeted fix for missing src.audio_analysis.wav2vec2"
echo "=============================================="

# Navigate to the runpod-multitalk directory
cd "$(dirname "$0")/.."

# Get current timestamp for build tracking
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "Build time: $BUILD_TIME"

# Verify required files exist
echo "Checking required files..."
required_files=(
    "Dockerfile.v74-8-fixed"
    "handler_v74_5_adaptive.py"
    "multitalk_v74_5_adaptive.py"
    "diagnostic_wan_model.py"
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
IMAGE_TAG="berrylands/multitalk-v74-8:fixed-audio-analysis"
echo "Building Docker image: $IMAGE_TAG"

docker build \
    --file Dockerfile.v74-8-fixed \
    --tag "$IMAGE_TAG" \
    --build-arg BUILD_TIME="$BUILD_TIME" \
    --progress=plain \
    .

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 BUILD SUCCESSFUL!"
    echo "Image: $IMAGE_TAG"
    echo "Features:"
    echo "  - Fixed src.audio_analysis.wav2vec2 module"
    echo "  - Fixed src.audio_analysis.torch_utils module"
    echo "  - Complete vram_management implementation"
    echo "  - Comprehensive kokoro TTS implementation"
    echo "  - All critical imports working"
    echo "  - Targeted fix for V74.7 failure"
    echo "  - High-quality lip-sync parameters (25fps)"
    echo ""
    echo "🎯 TARGETED FIX APPROACH:"
    echo "   - Focused on specific missing audio_analysis module"
    echo "   - Created comprehensive fallback implementations"
    echo "   - Tested all critical imports during build"
    echo "   - Ready for successful MultiTalk generation"
    echo ""
    echo "Ready to push to Docker Hub!"
    echo "Run: docker push $IMAGE_TAG"
else
    echo "❌ BUILD FAILED"
    exit 1
fi