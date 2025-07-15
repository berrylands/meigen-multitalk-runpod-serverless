#!/usr/bin/env python3
"""
Test V105 deployment with xfuser verification
"""

import runpod
import os
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_v105():
    """Test V105 deployment"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🚀 Testing V105 with xfuser verification...")
    
    # First, check model status to verify xfuser
    print("\n1️⃣ Checking model status...")
    model_check = endpoint.run({"action": "model_check"})
    print(f"Job ID: {model_check.job_id}")
    
    # Wait for completion
    for i in range(30):
        status = model_check.status()
        print(f"Status: {status}", end="\r")
        if status in ["COMPLETED", "FAILED"]:
            break
        time.sleep(2)
    
    print(f"\nFinal status: {status}")
    
    if status == "COMPLETED":
        output = model_check.output()
        if output and isinstance(output, dict):
            print("\n✅ Model check results:")
            if "model_info" in output:
                info = output["model_info"]
                print(f"  - CUDA available: {info.get('cuda_available')}")
                print(f"  - PyTorch version: {info.get('pytorch_version')}")
                print(f"  - xfuser available: {info.get('xfuser_available')}")
                print(f"  - xfuser version: {info.get('xfuser_version')}")
            else:
                print(json.dumps(output, indent=2))
    else:
        print(f"❌ Model check failed: {model_check.output()}")
        return
    
    # Test generation if xfuser is available
    print("\n2️⃣ Testing generation...")
    
    job_input = {
        "action": "generate",
        "audio_1": "audio_1.wav",
        "condition_image": "image_1.png",
        "prompt": "A person talking naturally",
        "output_format": "base64"
    }
    
    gen_job = endpoint.run(job_input)
    print(f"Job ID: {gen_job.job_id}")
    
    # Wait for completion
    for i in range(60):
        status = gen_job.status()
        print(f"Status: {status}", end="\r")
        if status in ["COMPLETED", "FAILED"]:
            break
        time.sleep(2)
    
    print(f"\nFinal status: {status}")
    
    if status == "COMPLETED":
        output = gen_job.output()
        if output and isinstance(output, dict):
            if "video_base64" in output:
                print("✅ Video generated successfully!")
                print(f"   Message: {output.get('message')}")
                print(f"   Video data length: {len(output['video_base64'])} chars")
            else:
                print("✅ Generation completed:")
                print(json.dumps(output, indent=2))
    else:
        print(f"❌ Generation failed: {gen_job.output()}")

if __name__ == "__main__":
    test_v105()