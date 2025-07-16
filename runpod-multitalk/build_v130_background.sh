#!/bin/bash

# Build V130 in background with PyTorch/torchvision compatibility fix
echo "Building MultiTalk V130 with compatible PyTorch 2.1.0 and torchvision 0.16.0..."
echo "This will run in the background and may take 10-15 minutes..."

# Set image tag
IMAGE_TAG="jasonedge/multitalk-runpod:v130"

# Build the Docker image in background
nohup docker build -f Dockerfile.v130-pytorch-compat -t $IMAGE_TAG . > build_v130.log 2>&1 &
BUILD_PID=$!

echo "🔨 Build started with PID: $BUILD_PID"
echo "📝 Build log: build_v130.log"
echo "🔍 Monitor progress with: tail -f build_v130.log"
echo "✅ Once build completes, run: docker push $IMAGE_TAG"
echo ""
echo "Expected completion time: 10-15 minutes"
echo "After completion, update RunPod template and test V130"