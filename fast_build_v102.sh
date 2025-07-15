#!/bin/bash
# Fast build script for V102 with real xfuser
# Optimized for speed and reliability

set -e

echo "=== FAST BUILD: MultiTalk V102 with REAL xfuser ==="
echo "Available space: 500GB volume + 200GB container = 700GB total"
date

# Quick Docker install
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get update -qq
    apt-get install -y -qq docker.io
    systemctl start docker
fi

# Clone and build immediately
cd /workspace
rm -rf meigen-multitalk-runpod-serverless
git clone -q https://github.com/berrylands/meigen-multitalk-runpod-serverless.git
cd meigen-multitalk-runpod-serverless/runpod-multitalk

echo "Building V102 with REAL xfuser..."
docker build -f Dockerfile.v102 -t berrylands/multitalk-runpod:v102-real-xfuser .

# Push if credentials available
if [ -n "$DOCKERHUB_TOKEN" ]; then
    echo "$DOCKERHUB_TOKEN" | docker login -u berrylands --password-stdin
    docker push berrylands/multitalk-runpod:v102-real-xfuser
    echo "SUCCESS: V102 with REAL xfuser pushed to DockerHub!"
else
    echo "SUCCESS: V102 with REAL xfuser built locally!"
fi

date
echo "Build completed!"