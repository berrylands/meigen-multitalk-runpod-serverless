#!/bin/bash
# Build script for MultiTalk V58 with device fixes

echo "Building MultiTalk V58 - Device Fix"
echo "=================================="

# Build the Docker image
echo "Starting Docker build..."
docker build -f Dockerfile.v58-device-fix -t berrylands/multitalk-v58:device-fix .

if [ $? -eq 0 ]; then
    echo "Build successful!"
    
    # Push to DockerHub
    echo "Pushing to DockerHub..."
    docker push berrylands/multitalk-v58:device-fix
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pushed berrylands/multitalk-v58:device-fix to DockerHub"
    else
        echo "❌ Failed to push to DockerHub"
    fi
else
    echo "❌ Build failed"
fi