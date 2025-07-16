#!/bin/bash

# Build V130 with PyTorch/torchvision compatibility fix
echo "Building MultiTalk V130 with compatible PyTorch 2.1.0 and torchvision 0.16.0..."

# Set image tag
IMAGE_TAG="jasonedge/multitalk-runpod:v130"

# Build the Docker image
docker build -f Dockerfile.v130-pytorch-compat -t $IMAGE_TAG .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful! Pushing to Docker Hub..."
    docker push $IMAGE_TAG
    
    if [ $? -eq 0 ]; then
        echo "✅ Push successful!"
        echo "🚀 V130 ready with compatible PyTorch/torchvision versions:"
        echo "   - PyTorch 2.1.0"
        echo "   - torchvision 0.16.0"
        echo "   - xformers 0.0.22"
        echo "   - xfuser, scikit-image, and all dependencies"
        echo "📝 Image: $IMAGE_TAG"
    else
        echo "❌ Push failed"
        exit 1
    fi
else
    echo "❌ Build failed"
    exit 1
fi