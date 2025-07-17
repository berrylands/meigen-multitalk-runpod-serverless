#!/bin/bash

# Manual build script for V131
echo "🚀 Building MultiTalk V131 manually..."

cd /Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk

# Build the image
echo "Building V131..."
docker build -f Dockerfile.v131 -t berrylands/multitalk-runpod:v131 .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Push to Docker Hub
    echo "Pushing to Docker Hub..."
    docker push berrylands/multitalk-runpod:v131
    
    if [ $? -eq 0 ]; then
        echo "✅ Push successful!"
        echo "🎉 V131 is now available at: berrylands/multitalk-runpod:v131"
    else
        echo "❌ Push failed - you may need to login to Docker Hub"
        echo "Run: docker login"
    fi
else
    echo "❌ Build failed"
    exit 1
fi