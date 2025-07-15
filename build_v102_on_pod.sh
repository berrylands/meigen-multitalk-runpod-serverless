#!/bin/bash

# Build script for V102 with proper xfuser installation
# Run this on RunPod pod with sufficient disk space

set -e

echo "=== Building MultiTalk V102 with xfuser ==="
echo "This build requires significant disk space for xfuser dependencies"

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get update
    apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

    systemctl start docker
    systemctl enable docker
fi

# Clone repository
echo "Cloning repository..."
cd /workspace
rm -rf meigen-multitalk-runpod-serverless
git clone https://github.com/berrylands/meigen-multitalk-runpod-serverless.git
cd meigen-multitalk-runpod-serverless/runpod-multitalk

# Build V102
echo "Building V102..."
docker build -f Dockerfile.v102 -t berrylands/multitalk-runpod:v102 .

# Login to DockerHub
echo "Logging into DockerHub..."
echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

# Push to DockerHub
echo "Pushing to DockerHub..."
docker push berrylands/multitalk-runpod:v102

echo "Build completed successfully!"
echo "Image: berrylands/multitalk-runpod:v102"

# Cleanup
docker system prune -f

echo "Done!"