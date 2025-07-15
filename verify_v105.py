#!/usr/bin/env python3
"""
Verify V105 deployment using direct API calls
"""

import os
import json
import time
import requests
from pathlib import Path

def verify_v105():
    """Verify V105 deployment"""
    
    # Get API key from environment
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        # Try loading from .env file
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("RUNPOD_API_KEY="):
                        api_key = line.strip().split("=", 1)[1]
                        break
    
    if not api_key:
        print("❌ RUNPOD_API_KEY not found")
        return
    
    endpoint_id = "kkx3cfy484jszl"
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("🚀 Verifying V105 deployment...")
    
    # 1. Check endpoint status
    print("\n1️⃣ Checking endpoint status...")
    response = requests.get(f"{base_url}/health", headers=headers)
    if response.status_code == 200:
        print("✅ Endpoint is healthy")
    else:
        print(f"❌ Endpoint health check failed: {response.status_code}")
        return
    
    # 2. Submit model check job
    print("\n2️⃣ Submitting model check job...")
    
    job_input = {"input": {"action": "model_check"}}
    
    response = requests.post(f"{base_url}/run", headers=headers, json=job_input)
    if response.status_code != 200:
        print(f"❌ Failed to submit job: {response.status_code}")
        print(response.text)
        return
    
    job_data = response.json()
    job_id = job_data.get("id")
    print(f"Job ID: {job_id}")
    
    # 3. Wait for job completion
    print("\n3️⃣ Waiting for job completion...")
    max_attempts = 30
    for i in range(max_attempts):
        response = requests.get(f"{base_url}/status/{job_id}", headers=headers)
        if response.status_code == 200:
            status_data = response.json()
            status = status_data.get("status")
            print(f"Status: {status}", end="\r")
            
            if status == "COMPLETED":
                output = status_data.get("output")
                if output:
                    print("\n\n✅ Model check completed successfully!")
                    
                    # Check for xfuser
                    if isinstance(output, dict) and "model_info" in output:
                        info = output["model_info"]
                        print(f"\n📊 System Information:")
                        print(f"  - CUDA available: {info.get('cuda_available')}")
                        print(f"  - PyTorch version: {info.get('pytorch_version')}")
                        print(f"  - xfuser available: {info.get('xfuser_available')}")
                        print(f"  - xfuser version: {info.get('xfuser_version')}")
                        
                        if info.get('xfuser_available'):
                            print("\n🎉 SUCCESS: V105 is running with real xfuser!")
                        else:
                            print("\n⚠️  WARNING: xfuser is not available in V105")
                    else:
                        print(f"\nOutput: {json.dumps(output, indent=2)}")
                break
            
            elif status == "FAILED":
                error = status_data.get("error") or status_data.get("output")
                print(f"\n\n❌ Job failed: {error}")
                break
        
        time.sleep(2)
    else:
        print("\n\n⏱️  Timeout waiting for job completion")
    
    # 4. Test generation (optional)
    print("\n\n4️⃣ Testing generation with sample inputs...")
    
    gen_input = {
        "input": {
            "action": "generate",
            "audio_1": "audio_1.wav",
            "condition_image": "image_1.png",
            "prompt": "A person talking naturally",
            "output_format": "base64"
        }
    }
    
    response = requests.post(f"{base_url}/run", headers=headers, json=gen_input)
    if response.status_code == 200:
        job_data = response.json()
        job_id = job_data.get("id")
        print(f"Generation job ID: {job_id}")
        print("(Job submitted, check RunPod dashboard for results)")
    else:
        print(f"⚠️  Generation test failed: {response.status_code}")

if __name__ == "__main__":
    verify_v105()