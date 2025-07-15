#!/bin/bash
# Build and deploy MultiTalk V74.4 - Complete Official Implementation

set -e

echo "=== MultiTalk V74.4 Complete Implementation Build Script ==="
echo "Fixes missing src directory and vram_management modules"

# Configuration
DOCKER_USERNAME="${1:-berrylands}"
VERSION="74.4"
TAG="${DOCKER_USERNAME}/multitalk-v${VERSION}:complete"

# Check Docker
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "✅ Docker is running"

# Navigate to runpod-multitalk directory
cd "$(dirname "$0")/.."
echo "📁 Working directory: $(pwd)"

# Build image
echo "🔨 Building V74.4 Complete Official Implementation..."
docker build -f Dockerfile.v74-4-complete -t ${TAG} .

echo "✅ Build complete: ${TAG}"

# Push to Docker Hub
read -p "📤 Push to Docker Hub? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker push ${TAG}
    echo "✅ Pushed to Docker Hub: ${TAG}"
    
    echo ""
    echo "🎯 Deployment Summary:"
    echo "  Docker Image: ${TAG}"
    echo ""
    echo "🔧 Key Fixes:"
    echo "    ✅ NumPy/SciPy compatibility fixed"
    echo "    ✅ Missing src directory downloaded"
    echo "    ✅ vram_management modules included"
    echo "    ✅ Complete official implementation"
    echo ""
    echo "📦 New Components:"
    echo "    • src/vram_management/__init__.py"
    echo "    • src/vram_management/layers.py"
    echo "    • src/utils.py"
    echo "    • src/audio_analysis/"
    echo ""
fi

echo ""
echo "🚀 V74.4 Complete Implementation ready for RunPod testing!"
echo ""
echo "💡 This version includes the complete official MultiTalk implementation"
echo "   with all required directories and modules."