#!/bin/bash
# Build complete image in background

cd /Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk

TAG="berrylands/multitalk-pytorch:latest"

echo "Starting background build of complete MultiTalk image..."
echo "This will take 10-15 minutes due to PyTorch installation."
echo ""

# Start build in background
nohup docker build \
  --platform linux/amd64 \
  -t "$TAG" \
  -f Dockerfile.add-torch \
  --build-arg BUILD_TIME="$(date)" \
  --build-arg BUILD_ID="$(date +%s)" \
  . > pytorch_build.log 2>&1 &

BUILD_PID=$!

echo "Build started with PID: $BUILD_PID"
echo "Log file: $(pwd)/pytorch_build.log"
echo ""
echo "Monitor progress with:"
echo "  tail -f pytorch_build.log"
echo ""
echo "Check if complete:"
echo "  ps -p $BUILD_PID || echo 'Build complete!'"
echo ""
echo "Once complete, push with:"
echo "  docker push $TAG"
echo ""
echo "Then update RunPod to use: $TAG"

# Save build info
cat > pytorch_build_info.txt << EOF
Build Started: $(date)
Image: $TAG
PID: $BUILD_PID
Log: $(pwd)/pytorch_build.log

Commands:
  Monitor: tail -f pytorch_build.log
  Check: ps -p $BUILD_PID
  Push: docker push $TAG
EOF