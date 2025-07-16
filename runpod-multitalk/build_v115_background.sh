#!/bin/bash

# Build V115 in background
set -e

echo "Starting V115 build in background..."

# Build with reduced verbosity
docker build -f Dockerfile.v115 -t multitalk-v115:proper-meigen-multitalk . > build_v115.log 2>&1 &

BUILD_PID=$!

echo "Build started with PID: $BUILD_PID"
echo "Monitoring build progress..."

# Monitor the build
while kill -0 $BUILD_PID 2>/dev/null; do
    if [ -f build_v115.log ]; then
        tail -n 1 build_v115.log 2>/dev/null | grep -E "(Step|RUN|COPY|FROM)" || true
    fi
    sleep 10
done

# Check if build succeeded
wait $BUILD_PID
BUILD_EXIT_CODE=$?

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "✅ Build completed successfully!"
    docker images multitalk-v115:proper-meigen-multitalk
    
    # Tag for pushing
    docker tag multitalk-v115:proper-meigen-multitalk berrylands/multitalk-v115:proper-meigen-multitalk
    
    echo "🚀 Ready to push: docker push berrylands/multitalk-v115:proper-meigen-multitalk"
else
    echo "❌ Build failed with exit code: $BUILD_EXIT_CODE"
    echo "Check build_v115.log for details"
    exit 1
fi