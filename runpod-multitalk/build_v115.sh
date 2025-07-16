#!/bin/bash

# Build V115 - Proper MeiGen-MultiTalk Implementation
set -e

echo "======================================"
echo "Building MultiTalk V115"
echo "======================================"

# Build configuration
IMAGE_NAME="multitalk-v115"
DOCKERFILE="Dockerfile.v115"
TAG="proper-meigen-multitalk"

# Check if Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo "❌ Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# Check if required files exist
REQUIRED_FILES=(
    "multitalk_v115_implementation.py"
    "handler_v115.py"
    "s3_handler.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file not found: $file"
        exit 1
    fi
done

echo "✅ All required files found"

# Build the image
echo "🏗️  Building Docker image: $IMAGE_NAME:$TAG"
docker build -f "$DOCKERFILE" -t "$IMAGE_NAME:$TAG" .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "📦 Image: $IMAGE_NAME:$TAG"
    
    # Show image size
    echo "📊 Image size:"
    docker images "$IMAGE_NAME:$TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    
    # Tag for pushing
    echo "🏷️  Tagging for push..."
    docker tag "$IMAGE_NAME:$TAG" "berrylands/$IMAGE_NAME:$TAG"
    
    echo ""
    echo "🚀 Ready to push with:"
    echo "   docker push berrylands/$IMAGE_NAME:$TAG"
    echo ""
    echo "🎯 RunPod Configuration:"
    echo "   Container Image: berrylands/$IMAGE_NAME:$TAG"
    echo "   Container Disk: 20 GB"
    echo "   Volume Disk: 100 GB"
    echo "   Volume Mount: /runpod-volume"
    echo "   GPU: A100 40GB or RTX 4090"
    echo ""
    echo "✅ V115 build complete - ready for deployment!"
    
else
    echo "❌ Build failed!"
    exit 1
fi