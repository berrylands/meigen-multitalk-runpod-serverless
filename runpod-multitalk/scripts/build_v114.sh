#!/bin/bash

# Build script for V114 - Complete Offline Operation

set -e

echo "=========================================="
echo "Building MultiTalk V114"
echo "Complete Offline Operation with Network Storage"
echo "=========================================="

# Change to the runpod-multitalk directory
cd "$(dirname "$0")/.."

# Build the Docker image
echo "Building Docker image..."
docker build -f Dockerfile.v114 -t multitalk-v114:latest .

# Tag for DockerHub
DOCKERHUB_USERNAME="multitalk"
docker tag multitalk-v114:latest $DOCKERHUB_USERNAME/multitalk-runpod:v114

echo "=========================================="
echo "Build complete!"
echo "To push: docker push $DOCKERHUB_USERNAME/multitalk-runpod:v114"
echo "=========================================="