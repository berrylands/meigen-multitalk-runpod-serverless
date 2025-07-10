#!/bin/bash

# Quick test image builder for RunPod

set -e

DOCKERHUB_USERNAME=${1:-berrylands}
IMAGE_NAME="multitalk-test"
TAG="latest"

echo "Building test image for RunPod..."
echo "DockerHub username: $DOCKERHUB_USERNAME"

# Create temporary Dockerfile
cat > /tmp/Dockerfile.test << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Install RunPod and basic dependencies
RUN pip install runpod pillow requests

# Create a simple test handler
RUN cat > handler.py << 'HANDLER_EOF'
import runpod
import os
import sys
import json

def handler(job):
    try:
        job_input = job.get('input', {})
        
        # Health check
        if job_input.get('health_check'):
            return {
                "status": "healthy",
                "message": "MultiTalk test handler is working!",
                "python_version": sys.version,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown'),
                "volume_mounted": os.path.exists('/runpod-volume'),
                "model_path_exists": os.path.exists('/runpod-volume/models'),
                "environment": {
                    "MODEL_PATH": os.environ.get('MODEL_PATH', 'Not set'),
                    "RUNPOD_DEBUG_LEVEL": os.environ.get('RUNPOD_DEBUG_LEVEL', 'Not set')
                }
            }
        
        # Echo test
        return {
            "message": "MultiTalk test handler successful!",
            "echo": job_input,
            "server_info": {
                "python_version": sys.version,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown'),
                "volume_available": os.path.exists('/runpod-volume')
            }
        }
        
    except Exception as e:
        return {"error": f"Handler failed: {str(e)}"}

if __name__ == "__main__":
    print("Starting MultiTalk test handler...")
    runpod.serverless.start({"handler": handler})
HANDLER_EOF

CMD ["python", "-u", "handler.py"]
EOF

# Build the image
echo "Building Docker image..."
docker build -f /tmp/Dockerfile.test -t ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG} .

if [ $? -eq 0 ]; then
    echo "✓ Image built successfully: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
    
    # Test locally
    echo "Testing image locally..."
    docker run --rm ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG} python -c "print('Test successful')" || echo "Local test failed"
    
    # Ask if user wants to push
    read -p "Push to DockerHub? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Pushing to DockerHub..."
        docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}
        
        if [ $? -eq 0 ]; then
            echo "✓ Image pushed successfully!"
            echo ""
            echo "🎉 Ready for RunPod deployment!"
            echo "Image: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
            echo ""
            echo "Next steps:"
            echo "1. Go to RunPod dashboard: https://www.runpod.io/console/serverless"
            echo "2. Click 'New Endpoint'"
            echo "3. Use image: ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}"
            echo "4. Select GPU: RTX 4090"
            echo "5. Add network volume: meigen-multitalk -> /runpod-volume"
            echo "6. Test with: {\"input\": {\"health_check\": true}}"
        else
            echo "✗ Push failed"
        fi
    fi
else
    echo "✗ Build failed"
fi

# Cleanup
rm -f /tmp/Dockerfile.test