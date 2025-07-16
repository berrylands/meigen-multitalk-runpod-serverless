#!/usr/bin/env python3
"""
Test V122 Handler Fix with S3 files
"""

import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
ENDPOINT_ID = "zu0ik6c8yukyl6"
API_KEY = os.environ.get('RUNPOD_API_KEY')
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

def test_v122():
    """Test V122 handler with both input formats"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("Testing V122 Handler Fix")
    print("=" * 80)
    
    # Test 1: V76 format (should be auto-converted)
    print("\n1. Testing V76 format (audio_s3_key/image_s3_key)...")
    v76_input = {
        "input": {
            "audio_s3_key": "1.wav",
            "image_s3_key": "multi1.png",
            "num_frames": 81,
            "sampling_steps": 30,
            "turbo": True,
            "output_format": "url",
            "prompt": "A person talking naturally with expressive facial movements"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/runsync",
        json=v76_input,
        headers=headers,
        timeout=300
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ V76 format test: {result.get('status', 'UNKNOWN')}")
        if 'output' in result:
            print(f"   Output: {json.dumps(result['output'], indent=2)}")
    else:
        print(f"❌ V76 format test failed: {response.status_code}")
        print(f"   Response: {response.text}")
    
    # Test 2: New format
    print("\n2. Testing new format (audio_1/condition_image)...")
    new_input = {
        "input": {
            "action": "generate",
            "audio_1": "1.wav",
            "condition_image": "multi1.png"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/runsync",
        json=new_input,
        headers=headers,
        timeout=300
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ New format test: {result.get('status', 'UNKNOWN')}")
        if 'output' in result:
            print(f"   Output: {json.dumps(result['output'], indent=2)}")
    else:
        print(f"❌ New format test failed: {response.status_code}")
        print(f"   Response: {response.text}")
    
    print("\n" + "=" * 80)
    print("V122 Handler Fix test complete")
    print("Check RunPod logs for detailed execution info")

if __name__ == "__main__":
    test_v122()