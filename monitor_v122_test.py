#!/usr/bin/env python3
"""
Monitor V122 test progress with async job handling
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

def submit_and_monitor(input_data: dict, test_name: str):
    """Submit job and monitor until completion"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    # Submit job
    response = requests.post(
        f"{BASE_URL}/run",
        json=input_data,
        headers=headers,
        timeout=30
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to submit job: {response.status_code}")
        print(f"   Response: {response.text}")
        return None
    
    job_data = response.json()
    job_id = job_data.get('id')
    print(f"✅ Job submitted: {job_id}")
    
    # Monitor job
    for attempt in range(60):  # 10 minutes max
        time.sleep(10)
        
        status_response = requests.get(
            f"{BASE_URL}/status/{job_id}",
            headers=headers,
            timeout=30
        )
        
        if status_response.status_code == 200:
            result = status_response.json()
            status = result.get('status', 'UNKNOWN')
            
            print(f"[{attempt+1:2d}] Status: {status}")
            
            if status == 'COMPLETED':
                output = result.get('output', {})
                print(f"✅ Job completed!")
                
                if 'error' in output:
                    print(f"❌ Error: {output['error']}")
                    if 'traceback' in output:
                        print(f"Traceback (last 5 lines):")
                        traceback_lines = output['traceback'].split('\n')
                        for line in traceback_lines[-5:]:
                            if line.strip():
                                print(f"   {line}")
                elif output.get('success'):
                    print(f"✅ Success!")
                    if 'video_url' in output:
                        print(f"🎥 Video URL: {output['video_url']}")
                    print(f"📊 Implementation: {output.get('implementation', 'Unknown')}")
                else:
                    print(f"⚠️ Completed but no success flag")
                    print(f"Output: {json.dumps(output, indent=2)}")
                
                return output
                
            elif status == 'FAILED':
                print(f"❌ Job failed!")
                error = result.get('error', 'Unknown error')
                print(f"Error: {error}")
                return None
                
            elif status in ['IN_PROGRESS', 'IN_QUEUE']:
                delay = result.get('delayTime', 0)
                worker = result.get('workerId', 'N/A')
                if status == 'IN_PROGRESS':
                    print(f"    Processing... (worker: {worker})")
                else:
                    print(f"    In queue... (delay: {delay}ms)")
    
    print("⏰ Timeout reached")
    return None

def test_v122():
    """Test V122 handler with both formats"""
    print("V122 Handler Fix Test Suite")
    print("Testing with S3 files: 1.wav and multi1.png")
    
    # Test 1: V76 format
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
    
    result1 = submit_and_monitor(v76_input, "V76 Format (audio_s3_key/image_s3_key)")
    
    # Test 2: New format
    new_input = {
        "input": {
            "action": "generate",
            "audio_1": "1.wav",
            "condition_image": "multi1.png"
        }
    }
    
    result2 = submit_and_monitor(new_input, "New Format (audio_1/condition_image)")
    
    # Summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    print(f"V76 format test: {'✅ PASS' if result1 and result1.get('success') else '❌ FAIL'}")
    print(f"New format test: {'✅ PASS' if result2 and result2.get('success') else '❌ FAIL'}")
    
    if result1 and result1.get('success') and result2 and result2.get('success'):
        print("\n🎉 All tests passed! V122 handler fix is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    test_v122()