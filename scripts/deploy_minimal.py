#!/usr/bin/env python3
"""
Deploy minimal working version to test RunPod deployment
"""

import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

DOCKERHUB_USERNAME = os.getenv("DOCKERHUB_USERNAME", "berrylands")  # Default fallback
IMAGE_NAME = "multitalk-minimal"
TAG = "latest"

def build_image():
    """Build the minimal Docker image."""
    print("Building minimal Docker image...")
    
    cmd = [
        "docker", "build",
        "-f", "runpod-multitalk/Dockerfile.minimal",
        "-t", f"{DOCKERHUB_USERNAME}/{IMAGE_NAME}:{TAG}",
        "runpod-multitalk"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Image built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def push_image():
    """Push image to DockerHub."""
    print("Pushing image to DockerHub...")
    
    # Login first (interactive)
    print("Please log in to DockerHub:")
    login_cmd = ["docker", "login"]
    subprocess.run(login_cmd)
    
    # Push
    push_cmd = ["docker", "push", f"{DOCKERHUB_USERNAME}/{IMAGE_NAME}:{TAG}"]
    
    try:
        result = subprocess.run(push_cmd, check=True, capture_output=True, text=True)
        print("✓ Image pushed successfully")
        print(f"Image available at: {DOCKERHUB_USERNAME}/{IMAGE_NAME}:{TAG}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Push failed: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def test_local():
    """Test the image locally."""
    print("Testing image locally...")
    
    cmd = [
        "docker", "run", "--rm",
        "-e", "RUNPOD_DEBUG_LEVEL=INFO",
        f"{DOCKERHUB_USERNAME}/{IMAGE_NAME}:{TAG}",
        "python", "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Local test passed")
        print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Local test failed: {e}")
        return False

def main():
    print("Minimal MultiTalk Deployment")
    print("=" * 50)
    print(f"DockerHub username: {DOCKERHUB_USERNAME}")
    print(f"Image name: {IMAGE_NAME}")
    
    # Build
    if not build_image():
        sys.exit(1)
    
    # Test locally
    if not test_local():
        print("Warning: Local test failed, but continuing...")
    
    # Ask if user wants to push
    push = input("\nPush to DockerHub? (y/n): ").lower().strip()
    if push == 'y':
        if push_image():
            print("\n" + "=" * 50)
            print("Deployment ready!")
            print(f"Use this image in RunPod: {DOCKERHUB_USERNAME}/{IMAGE_NAME}:{TAG}")
            print("\nNext steps:")
            print("1. Go to RunPod dashboard")
            print("2. Create new serverless endpoint")
            print("3. Use the image above")
            print("4. Set GPU to RTX 4090")
            print("5. Test with health check: {\"input\": {\"health_check\": true}}")
    
if __name__ == "__main__":
    main()