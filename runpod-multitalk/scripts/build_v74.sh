#!/bin/bash
# Build and deploy MultiTalk V74 with actual gcc installation

set -e

echo "=== MultiTalk V74 Build Script ==="
echo "Installing gcc to fix runtime compilation errors"

# Configuration
DOCKER_USERNAME="${1:-berrylands}"
VERSION="74"
TAG="${DOCKER_USERNAME}/multitalk-v${VERSION}:gcc-install"

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
echo "🔨 Building V74 with gcc installation..."
docker build -f Dockerfile.v74-gcc-install -t ${TAG} .

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
    echo "  Key Fix: Actually installs gcc/g++ for runtime compilation"
    echo "  Expected: Should resolve 'gcc not found' error from V73"
fi

echo ""
echo "🚀 V74 ready for RunPod testing!"
echo "This version installs build-essential to provide gcc for Triton compilation"