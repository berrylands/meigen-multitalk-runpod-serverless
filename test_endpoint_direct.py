#!/usr/bin/env python3
"""
Direct test of RunPod endpoint
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "pdz7evo425qwmz"

def test_endpoint():
    """Test the deployed endpoint."""
    
    base_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    print(f"Testing endpoint: {ENDPOINT_ID}")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1. Health Check Test")
    test_data = {"input": {"health_check": True}}
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/runsync",
            json=test_data,
            headers=headers,
            timeout=120
        )
        elapsed = time.time() - start_time
        
        print(f"   Status: {response.status_code}")
        print(f"   Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result:
                output = result["output"]
                print(f"   ✓ Health Status: {output.get('status', 'Unknown')}")
                print(f"   ✓ Message: {output.get('message', 'No message')}")
                print(f"   ✓ Volume Mounted: {output.get('volume_mounted', False)}")
                print(f"   ✓ Model Path Exists: {output.get('model_path_exists', False)}")
                print(f"   ✓ Worker ID: {output.get('worker_id', 'Unknown')}")
                if 'environment' in output:
                    print(f"   ✓ Environment: {output['environment']}")
            else:
                print(f"   Response: {result}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Test 2: Echo Test
    print("\n2. Echo Test")
    test_data = {"input": {"test": "echo", "message": "Hello RunPod!", "timestamp": time.time()}}
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/runsync",
            json=test_data,
            headers=headers,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        print(f"   Status: {response.status_code}")
        print(f"   Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            if "output" in result:
                output = result["output"]
                print(f"   ✓ Message: {output.get('message', 'No message')}")
                if 'echo' in output:
                    print(f"   ✓ Echo received: {output['echo']}")
                if 'server_info' in output:
                    print(f"   ✓ Server info: {output['server_info']}")
            else:
                print(f"   Response: {result}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    test_endpoint()