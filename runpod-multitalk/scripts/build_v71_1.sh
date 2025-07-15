#!/bin/bash
# Build script for MultiTalk V71.1 - Fixed Input Handling

set -e

# Configuration
VERSION="71.1"
DOCKER_USERNAME="${1:-berrylands}"
IMAGE_NAME="multitalk-v${VERSION}"
TAG="fixed-input"

echo "=== Building MultiTalk V${VERSION} Docker Image ==="
echo "Docker Hub Username: ${DOCKER_USERNAME}"
echo "Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Check required files exist
required_files=(
    "Dockerfile.v71.1-fixed"
    "handler_v71_1.py"
    "multitalk_v71_1_official_wrapper.py"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Required file missing: $file"
        exit 1
    fi
done

# Build the Docker image
echo "Building Docker image..."
docker build \
    -f Dockerfile.v71.1-fixed \
    -t ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

# Tag as latest
docker tag ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} ${DOCKER_USERNAME}/${IMAGE_NAME}:latest

echo "✓ Build complete!"
echo ""
echo "To push to Docker Hub:"
echo "  docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo "  docker push ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
echo ""
echo "To test locally:"
echo "  docker run --gpus all -e MODEL_PATH=/path/to/models ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo ""
echo "RunPod deployment image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"