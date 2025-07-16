#!/bin/bash

# Build script for V113 - Complete MeiGen-MultiTalk Implementation

set -e

echo "=========================================="
echo "Building MultiTalk V113"
echo "Complete MeiGen-MultiTalk Implementation"
echo "=========================================="

# Change to the runpod-multitalk directory
cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.v113 -t multitalk-v113:latest .

# Tag for DockerHub
DOCKERHUB_USERNAME="multitalk"
docker tag multitalk-v113:latest $DOCKERHUB_USERNAME/multitalk-runpod:v113

echo "=========================================="
echo "Build complete!"
echo "To push: docker push $DOCKERHUB_USERNAME/multitalk-runpod:v113"
echo "=========================================="