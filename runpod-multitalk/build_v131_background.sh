#!/bin/bash

# Load Docker Hub credentials from .env
source ../.env

echo "🚀 Building V131-fixed in background..."
echo "Using Docker Hub account: $DOCKERHUB_USERNAME"

# Login to Docker Hub
echo "$DOCKERHUB_PASSWORD" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin

if [ $? -ne 0 ]; then
    echo "❌ Docker login failed"
    exit 1
fi

echo "✅ Docker login successful"

# Build the image in background with log file
echo ""
echo "📦 Building V131-fixed (this may take 10-15 minutes)..."
echo "📝 Build log: v131_build.log"
echo "🔍 Monitor with: tail -f v131_build.log"

# Build with output to log file
(
    docker build -f Dockerfile.v131-fixed -t berrylands/multitalk-runpod:v131 -t berrylands/multitalk-runpod:v131-fixed . 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Build successful!" | tee -a v131_build.log
        
        # Push both tags
        echo "📤 Pushing to Docker Hub..." | tee -a v131_build.log
        docker push berrylands/multitalk-runpod:v131 2>&1
        docker push berrylands/multitalk-runpod:v131-fixed 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ Push successful!" | tee -a v131_build.log
            echo "🎉 V131 is now available!" | tee -a v131_build.log
        else
            echo "❌ Push failed" | tee -a v131_build.log
        fi
    else
        echo "❌ Build failed" | tee -a v131_build.log
    fi
    
    # Logout
    docker logout
    
) > v131_build.log 2>&1 &

BUILD_PID=$!
echo "🔄 Build running in background (PID: $BUILD_PID)"
echo "📝 Monitor with: tail -f v131_build.log"
echo "🛑 Stop with: kill $BUILD_PID"

# Wait a bit and show initial progress
sleep 5
echo ""
echo "📊 Initial build progress:"
tail -n 10 v131_build.log