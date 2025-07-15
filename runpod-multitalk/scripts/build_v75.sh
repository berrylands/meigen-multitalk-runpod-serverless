#!/bin/bash
# Build script for MultiTalk V75.0 - JSON Input Format

# Exit on error
set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "Building MultiTalk V75.0 - JSON Input Format"
echo "==========================================${NC}"

# Check if docker hub username is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Docker Hub username required${NC}"
    echo "Usage: $0 <dockerhub-username>"
    exit 1
fi

DOCKER_USERNAME=$1
IMAGE_NAME="${DOCKER_USERNAME}/multitalk-runpod"
VERSION="v75.0"
TAG="${IMAGE_NAME}:${VERSION}"
LATEST_TAG="${IMAGE_NAME}:latest"

echo -e "${YELLOW}Docker Hub Username: ${DOCKER_USERNAME}${NC}"
echo -e "${YELLOW}Image: ${TAG}${NC}"

# Change to runpod-multitalk directory
cd "$(dirname "$0")/.."
pwd

# Check required files exist
echo -e "\n${YELLOW}Checking required files...${NC}"
required_files=(
    "Dockerfile.v75-json-input"
    "multitalk_v75_0_json_input.py"
    "handler_v75.py"
    "setup_v75_0_json_input.sh"
    "requirements.txt"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓ Found: $file${NC}"
    else
        echo -e "${RED}✗ Missing: $file${NC}"
        exit 1
    fi
done

# Build the image
echo -e "\n${YELLOW}Building Docker image...${NC}"
docker build \
    -f Dockerfile.v75-json-input \
    -t ${TAG} \
    -t ${LATEST_TAG} \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --progress=plain \
    .

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Build successful!${NC}"
    
    # Show image info
    echo -e "\n${YELLOW}Image info:${NC}"
    docker images | grep "${IMAGE_NAME}" | head -2
    
    # Push to Docker Hub
    echo -e "\n${YELLOW}Pushing to Docker Hub...${NC}"
    docker push ${TAG}
    docker push ${LATEST_TAG}
    
    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}✅ Push successful!${NC}"
        echo -e "\n${GREEN}Image available at:${NC}"
        echo -e "  - ${TAG}"
        echo -e "  - ${LATEST_TAG}"
        
        echo -e "\n${YELLOW}To deploy on RunPod, use:${NC}"
        echo -e "  ${TAG}"
    else
        echo -e "\n${RED}❌ Push failed!${NC}"
        exit 1
    fi
else
    echo -e "\n${RED}❌ Build failed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}✨ MultiTalk V75.0 build complete!${NC}"