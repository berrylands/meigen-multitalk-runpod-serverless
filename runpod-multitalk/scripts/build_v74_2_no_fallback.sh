#!/bin/bash
# Build and deploy MultiTalk V74.2 - No Fallback Logic

set -e

echo "=== MultiTalk V74.2 No Fallback Build Script ==="
echo "Removes all fallback logic - fails fast if MultiTalk implementation is incomplete"

# Configuration
DOCKER_USERNAME="${1:-berrylands}"
VERSION="74.2"
TAG="${DOCKER_USERNAME}/multitalk-v${VERSION}:no-fallback"

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
echo "🔨 Building V74.2 No Fallback..."
docker build -f Dockerfile.v74-2-no-fallback -t ${TAG} .

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
    echo "  Key Changes:"
    echo "    - Removed ALL fallback logic (no demo videos, no placeholders)"
    echo "    - Fails fast with clear error messages if MultiTalk script missing"
    echo "    - S3 upload works without ACL issues"
    echo "    - Only uses official MultiTalk implementation"
    echo "    - Flexible model discovery still works"
fi

echo ""
echo "🚀 V74.2 No Fallback ready for RunPod testing!"
echo ""
echo "⚠️  IMPORTANT: This version will FAIL if:"
echo "   - MultiTalk directory is missing"
echo "   - generate_multitalk.py script is missing"
echo "   - Official implementation is incomplete"
echo "   This is INTENTIONAL behavior to ensure real MultiTalk functionality."