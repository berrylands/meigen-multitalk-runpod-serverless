#!/bin/bash
# Build v58 with Docker wait

echo "🚀 MultiTalk V58 Build Script"
echo "============================"

# Wait for Docker to be ready
echo "Waiting for Docker to start..."
max_wait=300  # 5 minutes
waited=0

while [ $waited -lt $max_wait ]; do
    if docker info >/dev/null 2>&1; then
        echo ""
        echo "✅ Docker is ready!"
        break
    fi
    echo -n "."
    sleep 2
    waited=$((waited + 2))
done

if [ $waited -ge $max_wait ]; then
    echo ""
    echo "❌ Docker failed to start after 5 minutes"
    echo "Please start Docker Desktop manually and run this script again"
    exit 1
fi

# Docker is ready, proceed with build
echo ""
echo "Starting build of berrylands/multitalk-v58:device-fix..."
echo ""

# Build the image
docker build -f Dockerfile.v58-minimal -t berrylands/multitalk-v58:device-fix . || {
    echo "❌ Build failed!"
    exit 1
}

echo ""
echo "✅ Build successful!"

# Push to DockerHub
echo "Pushing to DockerHub..."
docker push berrylands/multitalk-v58:device-fix || {
    echo "❌ Push failed! Make sure you're logged in: docker login"
    exit 1
}

echo ""
echo "🎉 Successfully built and pushed berrylands/multitalk-v58:device-fix"
echo ""
echo "The v58 device fix is now available on DockerHub!"