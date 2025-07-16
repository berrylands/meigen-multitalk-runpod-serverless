#!/bin/bash

# Build V130 Final with PyTorch/torchvision compatibility fix
echo "Building MultiTalk V130 Final with PyTorch 2.1.0 and torchvision 0.16.0..."

# Set image tag
IMAGE_TAG="jasonedge/multitalk-runpod:v130"

# Build the Docker image
docker build -f Dockerfile.v130-final -t $IMAGE_TAG .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful! Pushing to Docker Hub..."
    docker push $IMAGE_TAG
    
    if [ $? -eq 0 ]; then
        echo "✅ Push successful!"
        echo "🚀 V130 Final ready with PyTorch/torchvision compatibility:"
        echo "   - PyTorch 2.1.0+cu121 (compatible with cog-MultiTalk)"
        echo "   - torchvision 0.16.0 (compatible with cog-MultiTalk)"
        echo "   - NumPy 1.26.4 (pinned to avoid NumPy 2.x issues)"
        echo "   - xformers 0.0.22 (compatible with PyTorch 2.1.0)"
        echo "   - xfuser installed with --no-deps to prevent upgrades"
        echo "   - scikit-image, and all other dependencies"
        echo "📝 Image: $IMAGE_TAG"
    else
        echo "❌ Push failed"
        exit 1
    fi
else
    echo "❌ Build failed"
    exit 1
fi