#!/bin/bash
# Build and deploy MultiTalk V74.1 with syntax fix

set -e

echo "=== MultiTalk V74.1 Build Script ==="
echo "Fixed f-string syntax error and input format handling"

# Configuration
DOCKER_USERNAME="${1:-berrylands}"
VERSION="74.1"
TAG="${DOCKER_USERNAME}/multitalk-v${VERSION}:syntax-fix"

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
echo "🔨 Building V74.1 with syntax fixes..."
docker build -f Dockerfile.v74-1-syntax-fix -t ${TAG} .

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
    echo "  Key Fixes:"
    echo "    1. Fixed f-string backslash syntax error"
    echo "    2. Added support for your input format (audio_1, condition_image)"
    echo "    3. Gcc/g++ installed for runtime compilation"
fi

echo ""
echo "🚀 V74.1 ready for RunPod testing!"
echo "This version fixes the syntax error and supports your exact input format"