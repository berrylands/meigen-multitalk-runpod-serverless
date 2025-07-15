#!/bin/bash
# Build script for MultiTalk V71 - Pre-installed Dependencies

set -e

# Configuration
VERSION="71"
DOCKER_USERNAME="${1:-berrylands}"
IMAGE_NAME="multitalk-v${VERSION}"
TAG="preinstalled-deps"

echo "=== Building MultiTalk V${VERSION} Docker Image ==="
echo "Docker Hub Username: ${DOCKER_USERNAME}"
echo "Image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Check required files exist
required_files=(
    "Dockerfile.v71-preinstalled"
    "handler_v71.py"
    "multitalk_v71_official_wrapper.py"
    "setup_official_multitalk_v71.sh"
    "requirements_multitalk_official.txt"
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
    -f Dockerfile.v71-preinstalled \
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