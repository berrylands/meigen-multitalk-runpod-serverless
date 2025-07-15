#!/bin/bash
# Single command to build V102 on RunPod pod
# This can be executed via: curl -sSL https://raw.githubusercontent.com/berrylands/meigen-multitalk-runpod-serverless/master/build_v102_single_command.sh | bash

set -e

echo "=== Building MultiTalk V102 with real xfuser ==="
echo "Starting automated build process..."

# Install prerequisites
apt-get update -y
apt-get install -y git ca-certificates curl gnupg lsb-release

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
    systemctl start docker
    systemctl enable docker
fi

# Clone repository
cd /workspace
rm -rf meigen-multitalk-runpod-serverless
git clone https://github.com/berrylands/meigen-multitalk-runpod-serverless.git
cd meigen-multitalk-runpod-serverless/runpod-multitalk

# Check disk space
echo "Disk space before build:"
df -h

# Build V102
echo "Building V102 with real xfuser..."
docker build -f Dockerfile.v102 -t berrylands/multitalk-runpod:v102 .

# Login and push to DockerHub
if [ -n "$DOCKERHUB_TOKEN" ]; then
    echo "Logging into DockerHub..."
    echo "$DOCKERHUB_TOKEN" | docker login -u berrylands --password-stdin
    
    echo "Pushing to DockerHub..."
    docker push berrylands/multitalk-runpod:v102
    
    echo "Build and push completed successfully!"
else
    echo "DOCKERHUB_TOKEN not set, skipping push"
    echo "Image built locally as: berrylands/multitalk-runpod:v102"
fi

# Clean up
docker system prune -f

echo "Done!"