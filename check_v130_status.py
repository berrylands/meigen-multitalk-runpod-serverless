#!/usr/bin/env python3
"""
Check V130 Build and Deployment Status
"""

import subprocess
import time
import json

def check_docker_image():
    """Check if V130 image is available on Docker Hub"""
    print("🔍 Checking Docker Hub for V130 image...")
    
    cmd = [
        "curl", "-s",
        "https://hub.docker.com/v2/repositories/berrylands/multitalk-runpod/tags/v130"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and "v130" in result.stdout:
            data = json.loads(result.stdout)
            if "last_updated" in data:
                print(f"✅ Image found: berrylands/multitalk-runpod:v130")
                print(f"   Last updated: {data['last_updated']}")
                return True
        else:
            print("⏳ Image not yet available on Docker Hub")
            return False
    except Exception as e:
        print(f"❌ Error checking Docker Hub: {e}")
        return False

def update_runpod_template():
    """Instructions to update RunPod template"""
    print("\n📋 To update RunPod template:")
    print("1. Go to RunPod dashboard")
    print("2. Navigate to Templates")
    print("3. Find template ID: 5y1gyg4n78kqwz")
    print("4. Update Docker image to: berrylands/multitalk-runpod:v130")
    print("\n✅ Then run test_v130_s3.py to verify the fix!")

def main():
    print("=" * 60)
    print("🚀 V130 BUILD STATUS CHECK")
    print("=" * 60)
    
    # Check GitHub Actions
    print("\n📊 GitHub Actions Build:")
    print("Monitor at: https://github.com/berrylands/meigen-multitalk-runpod-serverless/actions")
    
    # Check Docker Hub
    if check_docker_image():
        print("\n🎉 V130 is ready for deployment!")
        update_runpod_template()
    else:
        print("\n⏳ Build still in progress...")
        print("Check GitHub Actions for build status")
        print("This script will check Docker Hub availability")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()