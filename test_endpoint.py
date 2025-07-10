#!/usr/bin/env python3
"""
Test your RunPod endpoint
"""

import os
import sys
import time
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

def test_endpoint(endpoint_id):
    """Test the deployed endpoint."""
    
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    print(f"Testing endpoint: {endpoint_id}")
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
                print(f"   ✓ Echo received: {output.get('echo', {})}")
            else:
                print(f"   Response: {result}")
        else:
            print(f"   ✗ Error: {response.text}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("\nNext steps:")
    print("1. If tests pass, your endpoint is working!")
    print("2. Check if model_path_exists is True")
    print("3. If not, we need to download models to the network volume")

def main():
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY not found in .env file")
        sys.exit(1)
    
    endpoint_id = input("Enter your RunPod endpoint ID: ").strip()
    
    if not endpoint_id:
        print("Error: Endpoint ID is required")
        print("You can find it in the RunPod dashboard after deployment")
        sys.exit(1)
    
    test_endpoint(endpoint_id)

if __name__ == "__main__":
    main()