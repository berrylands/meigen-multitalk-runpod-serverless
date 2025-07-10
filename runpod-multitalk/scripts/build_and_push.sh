#!/bin/bash

# Build and push Docker image to DockerHub
# Usage: ./build_and_push.sh <dockerhub-username>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <dockerhub-username>"
    exit 1
fi

DOCKER_USERNAME=$1
IMAGE_NAME="multitalk-runpod"
TAG="latest"

echo "Building Docker image..."
docker build -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} .

echo "Logging in to DockerHub..."
docker login

echo "Pushing image to DockerHub..."
docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}

echo "Image pushed successfully: ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo ""
echo "To deploy on RunPod, use this image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"