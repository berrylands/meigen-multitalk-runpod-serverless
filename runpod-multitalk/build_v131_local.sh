#!/bin/bash

# Load Docker Hub credentials from .env
source ../.env

echo "🚀 Building V131-fixed locally and pushing to Docker Hub..."
echo "Using Docker Hub account: $DOCKERHUB_USERNAME"

# Login to Docker Hub
echo "$DOCKERHUB_PASSWORD" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

if [ $? -ne 0 ]; then
    echo "❌ Docker login failed"
    exit 1
fi

echo "✅ Docker login successful"

# Build the image
echo ""
echo "📦 Building V131-fixed..."
docker build -f Dockerfile.v131-fixed -t berrylands/multitalk-runpod:v131 -t berrylands/multitalk-runpod:v131-fixed .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Push both tags
    echo ""
    echo "📤 Pushing to Docker Hub..."
    docker push berrylands/multitalk-runpod:v131
    docker push berrylands/multitalk-runpod:v131-fixed
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Push successful!"
        echo "🎉 V131 is now available at:"
        echo "   - berrylands/multitalk-runpod:v131"
        echo "   - berrylands/multitalk-runpod:v131-fixed"
        echo ""
        echo "📋 RunPod template is already set to use v131"
        echo "🧪 Ready to test with: python ../test_v131_fixed.py"
    else
        echo "❌ Push failed"
        exit 1
    fi
else
    echo "❌ Build failed"
    exit 1
fi

# Logout from Docker Hub
docker logout