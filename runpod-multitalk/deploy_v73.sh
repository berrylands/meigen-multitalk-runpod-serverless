#!/bin/bash
# Deploy MultiTalk V73 - Automated deployment script

set -e

echo "=== MultiTalk V73 Deployment Script ==="
echo "This script will build and deploy V73 with runtime dependency fixes"

# Configuration
DOCKER_USERNAME="${1:-berrylands}"
VERSION="73"

# Check if Docker is running
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"

# Ensure we're in the right directory
cd "$(dirname "$0")"

echo "📁 Current directory: $(pwd)"

# Option 1: Build minimal fix (faster, recommended)
echo ""
echo "🔧 Building V73 Minimal Fix..."
docker build -f Dockerfile.v73-minimal -t ${DOCKER_USERNAME}/multitalk-v73:minimal-fix .

echo "📤 Pushing minimal fix to Docker Hub..."
docker push ${DOCKER_USERNAME}/multitalk-v73:minimal-fix

echo ""
echo "✅ V73 Minimal Fix deployed successfully!"
echo "RunPod Image: ${DOCKER_USERNAME}/multitalk-v73:minimal-fix"

# Option 2: Build complete runtime deps (optional)
read -p "🤔 Do you want to also build the complete runtime deps version? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔧 Building V73 Complete Runtime Dependencies..."
    docker build -f Dockerfile.v73-runtime-deps -t ${DOCKER_USERNAME}/multitalk-v73:runtime-deps .
    
    echo "📤 Pushing complete version to Docker Hub..."
    docker push ${DOCKER_USERNAME}/multitalk-v73:runtime-deps
    
    echo "✅ V73 Complete version also deployed!"
    echo "RunPod Image: ${DOCKER_USERNAME}/multitalk-v73:runtime-deps"
fi

echo ""
echo "🎯 Deployment Summary:"
echo "  - Primary Image: ${DOCKER_USERNAME}/multitalk-v73:minimal-fix"
echo "  - Key Fixes: C compiler environment, xFormers compatibility"
echo "  - Expected: Resolves V72 runtime compilation errors"
echo ""
echo "🚀 Ready for RunPod testing!"
echo ""
echo "📋 Test with this RunPod configuration:"
echo "  Container Image: ${DOCKER_USERNAME}/multitalk-v73:minimal-fix"
echo "  Container Disk: 20 GB"  
echo "  Volume Disk: 100 GB"
echo "  GPU: A100 40GB or RTX 4090"