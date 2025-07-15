#!/bin/bash
# Build and deploy MultiTalk V74.2 with regression fixes

set -e

echo "=== MultiTalk V74.2 Build Script ==="
echo "Critical regression fixes - flexible model discovery"

# Configuration
DOCKER_USERNAME="${1:-berrylands}"
VERSION="74.2"
TAG="${DOCKER_USERNAME}/multitalk-v${VERSION}:regression-fix"

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
echo "🔨 Building V74.2 with critical regression fixes..."
docker build -f Dockerfile.v74-2-regression-fix -t ${TAG} .

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
    echo "  Critical Fixes:"
    echo "    1. No longer fails when models are missing"
    echo "    2. Flexible model discovery instead of hardcoded paths"
    echo "    3. Graceful fallbacks for video generation"
    echo "    4. Handler always starts successfully"
fi

echo ""
echo "🚀 V74.2 ready for RunPod testing!"
echo ""
echo "⚠️  IMPORTANT: This version fixes the critical regression where"
echo "   the handler would fail immediately if models weren't at exact paths."
echo "   Now it discovers available models and works with what's present."