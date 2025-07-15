#!/bin/bash
# Build and deploy MultiTalk V74.3 - Comprehensive Validation Framework

set -e

echo "=== MultiTalk V74.3 Comprehensive Validation Build Script ==="
echo "Proactive validation of dependencies, resources, models, and inputs"
echo "Fixes NumPy/SciPy compatibility and prevents runtime failures"

# Configuration
DOCKER_USERNAME="${1:-berrylands}"
VERSION="74.3"
TAG="${DOCKER_USERNAME}/multitalk-v${VERSION}:comprehensive"

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
echo "🔨 Building V74.3 Comprehensive Validation Framework..."
docker build -f Dockerfile.v74-3-comprehensive -t ${TAG} .

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
    echo "🔧 Key Features:"
    echo "    ✅ Comprehensive Validation Framework"
    echo "    ✅ Dependency Compatibility (NumPy/SciPy/PyTorch)"
    echo "    ✅ Resource Validation (GPU/RAM/Disk)"
    echo "    ✅ Model Structure Validation"
    echo "    ✅ Input File Validation (Audio/Image)"
    echo "    ✅ S3 Configuration Validation"
    echo "    ✅ Runtime Monitoring & Timeouts"
    echo "    ✅ Detailed Error Diagnostics"
    echo ""
    echo "🚨 Proactive Issue Prevention:"
    echo "    • No more NumPy binary incompatibility errors"
    echo "    • No more resource exhaustion failures"
    echo "    • No more missing model file errors"
    echo "    • No more corrupted input file issues"
    echo "    • No more hung processes"
    echo "    • Clear diagnostic information for all failures"
fi

echo ""
echo "🚀 V74.3 Comprehensive Validation ready for RunPod testing!"
echo ""
echo "💡 This version validates EVERYTHING upfront to prevent runtime failures."
echo "   If something can go wrong, V74.3 will catch it early with clear error messages."