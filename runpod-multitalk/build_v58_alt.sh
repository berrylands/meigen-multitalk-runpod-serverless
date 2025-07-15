#!/bin/bash
# Alternative build script for MultiTalk V58

echo "Alternative Build for MultiTalk V58"
echo "==================================="

# Check Docker status
echo "Checking Docker status..."
docker info --format "Docker version: {{.ServerVersion}}" 2>/dev/null || echo "Docker not responding"

# Try to restart Docker daemon
echo "Attempting to restart Docker daemon..."
killall Docker 2>/dev/null || true
sleep 2
open -a Docker 2>/dev/null || echo "Could not restart Docker Desktop"

# Wait for Docker to be ready
echo "Waiting for Docker to be ready..."
for i in {1..30}; do
    if docker info >/dev/null 2>&1; then
        echo "Docker is ready!"
        break
    fi
    echo -n "."
    sleep 1
done

# Now try the build
echo ""
echo "Attempting build..."
docker build --no-cache -f Dockerfile.v58-device-fix -t berrylands/multitalk-v58:device-fix . || {
    echo "Build failed. You may need to:"
    echo "1. Restart Docker Desktop manually"
    echo "2. Check available disk space: df -h"
    echo "3. Clean Docker cache: docker system prune -a"
    exit 1
}

# If build succeeded, push
echo "Build successful! Pushing to DockerHub..."
docker push berrylands/multitalk-v58:device-fix || {
    echo "Push failed. Make sure you're logged in: docker login"
    exit 1
}

echo "✅ Successfully built and pushed berrylands/multitalk-v58:device-fix"