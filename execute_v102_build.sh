#!/bin/bash

# Execute V102 build on RunPod pod

POD_IP="94.101.98.96"
POD_PORT="34493"
BUILD_SCRIPT="build_v102_on_pod.sh"

echo "=== Executing V102 Build on RunPod Pod ==="
echo "Pod: $POD_IP:$POD_PORT"
echo "Available disk space: 200GB volume + 100GB container disk"

# Upload build script to pod
echo "Uploading build script..."
scp -P $POD_PORT -o StrictHostKeyChecking=no $BUILD_SCRIPT root@$POD_IP:/workspace/

# Execute build script
echo "Executing build..."
ssh -p $POD_PORT -o StrictHostKeyChecking=no root@$POD_IP "cd /workspace && chmod +x $BUILD_SCRIPT && ./$BUILD_SCRIPT"

echo "Build execution completed!"