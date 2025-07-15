#!/bin/bash
# Build in background to avoid timeout

cd /Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk

TAG="v2.1.0-s3-$(date +%Y%m%d-%H%M%S)"
IMAGE="berrylands/multitalk-complete:$TAG"

echo "Starting background build of: $IMAGE"
echo "This will continue even if this terminal closes."
echo ""

# Start build in background with nohup
nohup docker buildx build \
  --platform linux/amd64 \
  -t "$IMAGE" \
  -f Dockerfile.complete \
  --push \
  . > build.log 2>&1 &

BUILD_PID=$!

echo "Build started with PID: $BUILD_PID"
echo "Log file: $(pwd)/build.log"
echo ""
echo "Monitor progress with:"
echo "  tail -f build.log"
echo ""
echo "Check if still running:"
echo "  ps -p $BUILD_PID"
echo ""
echo "Once complete, update RunPod to use:"
echo "  $IMAGE"
echo ""
echo "Build info saved to: build_info.txt"

# Save build info
cat > build_info.txt << EOF
Build Started: $(date)
Image: $IMAGE
PID: $BUILD_PID
Log: $(pwd)/build.log

To check status:
  ps -p $BUILD_PID
  tail -f build.log

To update RunPod after completion:
  Image: $IMAGE
EOF