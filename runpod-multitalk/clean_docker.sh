#!/bin/bash
# Clean Docker completely to free up space

echo "🧹 Docker Deep Clean Script"
echo "=========================="
echo "Current Docker.raw size: $(du -sh ~/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw 2>/dev/null | awk '{print $1}')"
echo ""

# Step 1: Quit Docker Desktop
echo "Step 1: Quitting Docker Desktop..."
osascript -e 'quit app "Docker"' 2>/dev/null || true
sleep 5

# Step 2: Check if Docker is still running
if pgrep -x "Docker" > /dev/null; then
    echo "Docker still running, force killing..."
    killall -9 Docker 2>/dev/null || true
    killall -9 com.docker.backend 2>/dev/null || true
    sleep 3
fi

# Step 3: Remove Docker.raw (this contains all images/containers)
echo "Step 2: Removing Docker.raw file (109GB)..."
rm -f ~/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw

# Step 4: Clean other Docker data
echo "Step 3: Cleaning other Docker data..."
rm -rf ~/Library/Containers/com.docker.docker/Data/vms/0/data/containerd
rm -rf ~/Library/Containers/com.docker.docker/Data/vms/0/data/buildkit

# Step 5: Show space saved
echo ""
echo "✅ Docker cleaned!"
echo "Space after cleanup:"
df -h | grep -E "(Filesystem|/$)"

echo ""
echo "You can now:"
echo "1. Restart Docker Desktop"
echo "2. Run: ./build_v58.sh"