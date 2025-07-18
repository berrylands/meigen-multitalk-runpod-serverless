#!/usr/bin/env python3
"""
Test V149 MultiTalk with actual video generation
"""

import runpod
import requests
import json
import time
import os
from pathlib import Path

# RunPod endpoint URL
ENDPOINT_URL = "https://api.runpod.ai/v2/zu0ik6c8yukyl6/runsync"

# Get RunPod API key
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("❌ RUNPOD_API_KEY environment variable not set")
    exit(1)

def test_multitalk_generation():
    """Test actual MultiTalk video generation"""
    
    # Test payload - using S3 files we know exist
    payload = {
        "input": {
            "audio_file": "1.wav",  # Known S3 file
            "image_file": "multi1.png",  # Known S3 file  
            "model_id": "multitalk-480",
            "turbo": False,  # Disable turbo for V149
            "debug": True
        }
    }
    
    print("🎬 Testing V149 MultiTalk video generation...")
    print(f"📝 Payload: {json.dumps(payload, indent=2)}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    start_time = time.time()
    
    try:
        print("📡 Sending request to RunPod endpoint...")
        response = requests.post(
            ENDPOINT_URL, 
            headers=headers,
            json=payload,
            timeout=300  # 5 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"⏱️  Request completed in {elapsed_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response received successfully!")
            print(f"📄 Full response: {json.dumps(result, indent=2)}")
            
            # Check if we got a video URL
            if "output" in result and "video_url" in result["output"]:
                video_url = result["output"]["video_url"]
                print(f"🎥 Video generated successfully: {video_url}")
                return True
            else:
                print("❌ No video_url in response")
                return False
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"📄 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error during request: {e}")
        return False

def test_endpoint_health():
    """Test if the endpoint is healthy"""
    
    health_payload = {
        "input": {
            "test": "health_check"
        }
    }
    
    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    try:
        print("🏥 Testing endpoint health...")
        response = requests.post(
            ENDPOINT_URL,
            headers=headers,
            json=health_payload,
            timeout=60
        )
        
        print(f"📊 Health check status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Endpoint is healthy!")
            print(f"📄 Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Health check failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing V149 MultiTalk Implementation")
    print("=" * 50)
    
    # Test endpoint health first
    if not test_endpoint_health():
        print("❌ Endpoint health check failed, cannot proceed")
        exit(1)
    
    print("\n" + "=" * 50)
    
    # Test actual video generation
    success = test_multitalk_generation()
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 V149 MultiTalk test PASSED!")
        print("✅ Video generation is working")
    else:
        print("❌ V149 MultiTalk test FAILED!")
        print("⚠️  Video generation is not working")
        
    exit(0 if success else 1)