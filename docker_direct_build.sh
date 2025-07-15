#!/bin/bash
# Direct Docker build without systemd

echo "=== Starting Docker daemon manually ==="
dockerd &
sleep 5

echo "=== Cloning repository ==="
cd /workspace
rm -rf meigen-multitalk-runpod-serverless
git clone https://github.com/berrylands/meigen-multitalk-runpod-serverless.git
cd meigen-multitalk-runpod-serverless/runpod-multitalk

echo "=== Building V102 with REAL xfuser ==="
docker build -f Dockerfile.v102 -t berrylands/multitalk-runpod:v102-real-xfuser .

echo "=== Logging into DockerHub ==="
echo "$DOCKERHUB_TOKEN" | docker login -u berrylands --password-stdin

echo "=== Pushing to DockerHub ==="
docker push berrylands/multitalk-runpod:v102-real-xfuser

echo "=== Build completed! ==="