#!/bin/bash
set -e

# Build V141 Debug version
echo "🔧 Building MultiTalk V141 Debug..."

# Ensure we're in the correct directory
cd /Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk

# Build and push
echo "📦 Building Docker image..."
docker build -f Dockerfile.v141-debug -t berrylands/multitalk-runpod:v141-debug .

echo "📤 Pushing to Docker Hub..."
docker push berrylands/multitalk-runpod:v141-debug

echo "✅ V141 Debug build complete!"