#!/usr/bin/env python3
"""
Test RunPod endpoint automatically
"""
import os
import sys
import time
import json
import requests
import base64
from pathlib import Path

# RunPod endpoint configuration
ENDPOINT_ID = "kkx3cfy484jszl"
API_KEY = os.environ.get("RUNPOD_API_KEY", "CKRTDIOF0IGFFSI4A11KTVP569QQAKQ4NK091965")
API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

def test_endpoint(version="latest"):
    """Test the RunPod endpoint with standard test files"""
    print(f"🧪 Testing MultiTalk endpoint (version: {version})")
    
    # Test payload
    payload = {
        "input": {
            "action": "generate",
            "audio_1": "1.wav",
            "condition_image": "multi1.png",
            "output_format": "s3",
            "s3_output_key": f"multitalk-out/test-{version}-{int(time.time())}.mp4"
        }
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"📤 Sending request to: {API_URL}")
    print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Send request
        start_time = time.time()
        response = requests.post(API_URL, json=payload, headers=headers, timeout=300)
        elapsed = time.time() - start_time
        
        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"📥 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(json.dumps(result, indent=2))
            
            # Check for output
            if result.get("output") and result["output"].get("video_url"):
                print(f"🎬 Video generated: {result['output']['video_url']}")
                return True
            else:
                print("❌ No video URL in output")
                return False
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_endpoint_status():
    """Check if the endpoint is ready"""
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    try:
        response = requests.get(status_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(f"🏥 Endpoint status: {data}")
            return True
        else:
            print(f"❌ Endpoint not ready: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to check status: {e}")
        return False

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "latest"
    
    print("=" * 60)
    print(f"MultiTalk RunPod Endpoint Test")
    print(f"Endpoint ID: {ENDPOINT_ID}")
    print(f"Version: {version}")
    print("=" * 60)
    
    # Check status first
    if check_endpoint_status():
        # Run test
        success = test_endpoint(version)
        sys.exit(0 if success else 1)
    else:
        print("❌ Endpoint not ready")
        sys.exit(1)