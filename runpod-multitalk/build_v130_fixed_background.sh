#!/bin/bash

# Build V130 Fixed in background with PyTorch/torchvision compatibility fix
echo "Building MultiTalk V130 Fixed with pinned PyTorch 2.1.0 and torchvision 0.16.0..."
echo "This will run in the background and may take 10-15 minutes..."

# Set image tag
IMAGE_TAG="jasonedge/multitalk-runpod:v130"

# Build the Docker image in background
nohup docker build -f Dockerfile.v130-fixed -t $IMAGE_TAG . > build_v130_fixed.log 2>&1 &
BUILD_PID=$!

echo "🔨 Build started with PID: $BUILD_PID"
echo "📝 Build log: build_v130_fixed.log"
echo "🔍 Monitor progress with: tail -f build_v130_fixed.log"
echo "✅ Once build completes, run: docker push $IMAGE_TAG"
echo ""
echo "Expected completion time: 10-15 minutes"
echo "🔧 KEY FIXES:"
echo "1. xfuser installed with --no-deps to prevent PyTorch upgrades"
echo "2. PyTorch 2.1.0 and torchvision 0.16.0 force-reinstalled LAST"
echo "3. All other dependencies installed first"
echo "4. Version assertions to ensure compatibility"
echo ""
echo "After completion, update RunPod template and test V130"