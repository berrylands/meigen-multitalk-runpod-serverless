#!/bin/bash
# Build script for MultiTalk V73 - Runtime Dependencies Fixed

set -e

# Configuration
VERSION="73"
DOCKER_USERNAME="${1:-berrylands}"
IMAGE_NAME="multitalk-v${VERSION}"
TAG="runtime-deps"

echo "=== Building MultiTalk V${VERSION} Docker Image ==="
echo "Docker Hub Username: ${DOCKER_USERNAME}"
echo "Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Check required files exist
required_files=(
    "Dockerfile.v73-runtime-deps"
    "handler_v73.py"
    "multitalk_v73_official_wrapper.py"
    "setup_official_multitalk_v72.sh"
    "requirements_multitalk_v73.txt"
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
    -f Dockerfile.v73-runtime-deps \
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