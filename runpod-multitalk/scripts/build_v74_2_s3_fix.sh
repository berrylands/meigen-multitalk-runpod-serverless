#!/bin/bash
# Build and deploy MultiTalk V74.2 S3 fix

set -e

echo "=== MultiTalk V74.2 S3 Fix Build Script ==="
echo "Fixes S3 ACL upload error"

# Configuration
DOCKER_USERNAME="${1:-berrylands}"
VERSION="74.2"
TAG="${DOCKER_USERNAME}/multitalk-v${VERSION}:s3-fix"

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
echo "🔨 Building V74.2 S3 fix..."
docker build -f Dockerfile.v74-2-s3-fix -t ${TAG} .

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
    echo "  Fixed Issues:"
    echo "    - S3 upload no longer tries to set ACL"
    echo "    - Should work with buckets that don't support ACLs"
    echo "    - Maintains all other V74.2 improvements"
fi

echo ""
echo "🚀 V74.2 S3 fix ready for RunPod testing!"