#!/usr/bin/env python3
"""
Check what's actually running in RunPod
"""

import runpod
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check the actual RunPod environment"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 Checking RunPod Environment")
    print("=" * 60)
    
    # Create a diagnostic job
    diagnostic_job = {
        "action": "debug",
        "command": "environment_check"
    }
    
    # First try the debug action
    print("1. Trying debug action...")
    try:
        job = endpoint.run(diagnostic_job)
        result = job.output(timeout=30)
        print(f"Debug result: {result}")
    except:
        print("Debug action not available")
    
    # Check health with more details
    print("\n2. Checking health endpoint...")
    job = endpoint.run({"action": "health"})
    result = job.output(timeout=30)
    
    if result:
        print("\nHealth check result:")
        for key, value in result.items():
            if key == "build_id":
                print(f"  Build ID: {value}")
            elif key == "image_tag":
                print(f"  Image Tag: {value}")
            elif key == "version":
                print(f"  Version: {value}")
    
    # Try to check PyTorch
    print("\n3. Checking PyTorch availability...")
    check_pytorch_job = {
        "action": "check_pytorch"
    }
    
    try:
        job = endpoint.run(check_pytorch_job)
        result = job.output(timeout=30)
        print(f"PyTorch check: {result}")
    except:
        pass
    
    print("\n" + "=" * 60)
    print("\n🤔 Possible issues:")
    print("1. The handler code might have a hardcoded 'Test implementation' message")
    print("2. The actual inference code might not be implemented")
    print("3. The models might not be loading properly")
    print("4. The container might be cached - try forcing a restart")
    
    print("\n💡 To force RunPod to pull the latest image:")
    print("1. In RunPod console, try adding a version tag:")
    print("   berrylands/multitalk-pytorch:latest")
    print("2. Or use the digest directly:")
    print("   berrylands/multitalk-pytorch@sha256:0a91dfbee7450302f2242eaffb0ac3d55f78a959332a1fa1d202d8a85762e6bd")
    print("3. You might need to pause/unpause the endpoint to force a refresh")

if __name__ == "__main__":
    check_environment()