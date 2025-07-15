#!/usr/bin/env python3
"""
Deploy S3 Update to RunPod
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def main():
    """Deploy the S3 update."""
    print("MultiTalk S3 Update Deployment")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("runpod-multitalk/Dockerfile.complete").exists():
        print("Error: Please run this script from the meigen-multitalk directory")
        return
    
    print("\nStep 1: Build and Push Docker Image")
    print("-" * 40)
    print("This will build the image with S3 support and push to DockerHub.")
    print("\nCommand to run:")
    print("cd runpod-multitalk")
    print("docker buildx build --platform linux/amd64 -t berrylands/multitalk-complete:v2.1.0 -f Dockerfile.complete --push .")
    
    input("\nPress Enter to continue (or Ctrl+C to cancel)...")
    
    # Build the Docker image
    print("\nBuilding Docker image...")
    cmd = [
        "docker", "buildx", "build",
        "--platform", "linux/amd64",
        "-t", "berrylands/multitalk-complete:v2.1.0",
        "-f", "runpod-multitalk/Dockerfile.complete",
        "--push",
        "runpod-multitalk"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Docker image built and pushed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Docker build failed: {e}")
        print("\nYou can run the command manually:")
        print(" ".join(cmd))
        return
    except KeyboardInterrupt:
        print("\n\nBuild cancelled.")
        return
    
    print("\nStep 2: Update RunPod Endpoint")
    print("-" * 40)
    print("Now we need to update the endpoint to use the new image.")
    print("\nYou have two options:")
    print("1. Update via RunPod dashboard (recommended)")
    print("2. Update via API (if template update is working)")
    
    choice = input("\nChoice (1 or 2): ")
    
    if choice == "1":
        print("\nManual Update Instructions:")
        print("1. Go to: https://www.runpod.io/console/serverless")
        print("2. Click on your endpoint (ID: kkx3cfy484jszl)")
        print("3. Click the 'Edit' button")
        print("4. Change the Docker image to: berrylands/multitalk-complete:v2.1.0")
        print("5. Click 'Save Changes'")
        print("\nThe endpoint will automatically update with the new image.")
        
    else:
        print("\nRunning automatic update...")
        os.system("python update_endpoint_via_template.py")
    
    print("\nStep 3: Test S3 Integration")
    print("-" * 40)
    print("Once the endpoint is updated, you can test S3 integration:")
    print("\n1. First run the test script to check status:")
    print("   python test_s3_integration.py")
    print("\n2. Then test with your S3 audio file:")
    print("   python test_s3_integration.py s3://your-bucket/your-audio.wav")
    
    print("\n" + "=" * 60)
    print("Deployment Summary:")
    print("- Docker Image: berrylands/multitalk-complete:v2.1.0")
    print("- New Features: S3 input/output support")
    print("- Endpoint ID: kkx3cfy484jszl")
    print("=" * 60)


if __name__ == "__main__":
    main()