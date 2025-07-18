#!/bin/bash
# V154 Build Script for RunPod
set -e

echo "========================================="
echo "MultiTalk V154 Build"
echo "========================================="

# Install buildah
apt-get update -qq && apt-get install -y buildah wget curl

# Setup directory
cd /workspace
mkdir -p multitalk-build
cd multitalk-build

# Download files
BASE_URL="https://raw.githubusercontent.com/berrylands/meigen-multitalk-runpod-serverless/master/runpod-multitalk"
wget -q -O handler.py "$BASE_URL/handler_v150_graceful.py"
wget -q -O multitalk_reference_wrapper.py "$BASE_URL/multitalk_reference_wrapper_v150.py"
wget -q -O s3_handler.py "$BASE_URL/s3_handler.py"
wget -q -O cog_multitalk_reference.tar.gz "https://github.com/berrylands/meigen-multitalk-runpod-serverless/raw/master/runpod-multitalk/cog_multitalk_reference.tar.gz"

# Download Dockerfile
wget -q -O Dockerfile "https://raw.githubusercontent.com/berrylands/meigen-multitalk-runpod-serverless/master/runpod-multitalk/Dockerfile.v154-full"

echo "Files downloaded:"
ls -la

# Build
echo "Building image..."
buildah bud --format docker -t multitalk-v154:latest .

# Tag and push
if [ $? -eq 0 ]; then
    buildah tag multitalk-v154:latest docker.io/berrylands/multitalk-runpod:v154-full
    echo "j2AZQp*rp" | buildah login -u berrylands --password-stdin docker.io
    buildah push docker.io/berrylands/multitalk-runpod:v154-full
    echo "✅ Build and push complete!"
else
    echo "❌ Build failed!"
fi