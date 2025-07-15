#!/bin/bash

# Build script for RunPod pod with more disk space
# This script should be run on the RunPod pod to build and push the Docker image

set -e

echo "=== RunPod MultiTalk Builder ==="
echo "Starting build process..."

# Update system and install Docker
echo "Installing Docker..."
apt-get update
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git

# Add Docker's official GPG key
mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start Docker service
systemctl start docker
systemctl enable docker

# Clone repository
echo "Cloning repository..."
cd /workspace
git clone https://github.com/berrylands/meigen-multitalk-runpod-serverless.git
cd meigen-multitalk-runpod-serverless

# Determine latest version
echo "Determining latest version..."
LATEST_VERSION=$(ls runpod-multitalk/Dockerfile.v* | grep -E 'Dockerfile\.v[0-9]+$' | sort -V | tail -1 | sed 's/.*Dockerfile\.v//')
echo "Latest version: $LATEST_VERSION"

# Build Docker image
echo "Building Docker image v$LATEST_VERSION..."
cd runpod-multitalk
docker build -f Dockerfile.v$LATEST_VERSION -t berrylands/multitalk-runpod:v$LATEST_VERSION .

# Login to DockerHub
echo "Logging into DockerHub..."
echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

# Push to DockerHub
echo "Pushing to DockerHub..."
docker push berrylands/multitalk-runpod:v$LATEST_VERSION

# Also tag as latest
docker tag berrylands/multitalk-runpod:v$LATEST_VERSION berrylands/multitalk-runpod:latest
docker push berrylands/multitalk-runpod:latest

echo "Build and push completed successfully!"
echo "Image: berrylands/multitalk-runpod:v$LATEST_VERSION"

# Clean up
docker system prune -f

echo "Done!"