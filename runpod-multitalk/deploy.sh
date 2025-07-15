#!/bin/bash
# Deploy script for RunPod with guaranteed fresh image

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}RunPod MultiTalk Deployment Script${NC}"
echo "======================================"

# Generate unique tag with timestamp
TAG="v2.1.0-$(date +%Y%m%d-%H%M%S)"
IMAGE="berrylands/multitalk-complete:$TAG"
BUILD_ID=$(date +%s)

echo -e "\n${YELLOW}Build Information:${NC}"
echo "  Image: $IMAGE"
echo "  Build ID: $BUILD_ID"
echo "  Timestamp: $(date)"

# Save build info
echo "BUILD_INFO=\"Built at $(date) with tag $TAG\"" > build_info.sh

# Update handler to include build info
echo -e "\n${YELLOW}Updating handler with build info...${NC}"
sed -i.bak "s/\"version\": \"2.1.0\"/\"version\": \"2.1.0\", \"build_id\": \"$BUILD_ID\", \"image_tag\": \"$TAG\"/" complete_multitalk_handler.py

# Build and push
echo -e "\n${YELLOW}Building Docker image...${NC}"
docker buildx build \
  --platform linux/amd64 \
  -t $IMAGE \
  -f Dockerfile.complete \
  --push \
  --no-cache \
  --build-arg BUILD_ID=$BUILD_ID \
  --build-arg BUILD_TIME="$(date)" \
  .

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Build successful!${NC}"
    
    # Restore original handler
    mv complete_multitalk_handler.py.bak complete_multitalk_handler.py
    
    # Get image digest
    echo -e "\n${YELLOW}Getting image digest...${NC}"
    DIGEST=$(docker inspect $IMAGE --format='{{index .RepoDigests 0}}' 2>/dev/null || echo "Unable to get digest")
    
    # Save deployment info
    cat > ../last_deployment.txt << EOF
Deployment Information
======================
Date: $(date)
Image: $IMAGE
Digest: $DIGEST
Build ID: $BUILD_ID

To update RunPod:
1. Go to: https://www.runpod.io/console/serverless
2. Click on your endpoint
3. Update Docker image to: $IMAGE
4. Or use digest: $DIGEST

To verify deployment:
python test_s3_integration.py
EOF
    
    echo -e "\n${GREEN}Deployment information saved to: last_deployment.txt${NC}"
    echo -e "\n${YELLOW}Next Steps:${NC}"
    echo "1. Update your RunPod endpoint to use: ${GREEN}$IMAGE${NC}"
    echo "2. Or for absolute certainty, use digest: ${GREEN}$DIGEST${NC}"
    echo "3. Wait for endpoint to update (check worker status)"
    echo "4. Test with: python test_s3_integration.py"
    
else
    echo -e "\n${RED}❌ Build failed!${NC}"
    # Restore original handler
    mv complete_multitalk_handler.py.bak complete_multitalk_handler.py
    exit 1
fi