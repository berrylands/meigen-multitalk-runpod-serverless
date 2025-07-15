#!/bin/bash

# Execute build on RunPod pod
# This runs the build script on the remote pod

POD_IP="47.47.180.200"
POD_PORT="11102"
BUILD_SCRIPT="build_on_runpod.sh"

echo "=== Executing Build on RunPod Pod ==="
echo "Pod: $POD_IP:$POD_PORT"
echo "Script: $BUILD_SCRIPT"

# Upload build script to pod
echo "Uploading build script..."
scp -P $POD_PORT -o StrictHostKeyChecking=no $BUILD_SCRIPT root@$POD_IP:/workspace/

# Execute build script on pod
echo "Executing build script..."
ssh -p $POD_PORT -o StrictHostKeyChecking=no root@$POD_IP "cd /workspace && chmod +x $BUILD_SCRIPT && ./$BUILD_SCRIPT"

echo "Build execution completed!"