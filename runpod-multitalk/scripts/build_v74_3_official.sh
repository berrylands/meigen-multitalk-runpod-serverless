#!/bin/bash
# Build and deploy MultiTalk V74.3 - Official Implementation with No Fallbacks

set -e

echo "=== MultiTalk V74.3 Official Implementation Build Script ==="
echo "Official MultiTalk implementation with no fallback logic"
echo "Fails fast if MultiTalk script or models are missing"

# Configuration
DOCKER_USERNAME="${1:-berrylands}"
VERSION="74.3"
TAG="${DOCKER_USERNAME}/multitalk-v${VERSION}:official"

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
echo "🔨 Building V74.3 Official Implementation..."
docker build -f Dockerfile.v74-3-official -t ${TAG} .

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
    echo "  Key Features:"
    echo "    - Official MultiTalk implementation downloaded from GitHub"
    echo "    - Complete wan/ directory structure with all subdirectories"
    echo "    - NO fallback logic - fails fast if implementation is incomplete"
    echo "    - Proper error messages for missing components"
    echo "    - Ready for production use with real MultiTalk models"
fi

echo ""
echo "🚀 V74.3 Official Implementation ready for RunPod testing!"
echo ""
echo "⚠️  IMPORTANT: This version will FAIL if the official MultiTalk"
echo "   implementation is not complete. This is intentional behavior"
echo "   to ensure we only use the real MultiTalk functionality."