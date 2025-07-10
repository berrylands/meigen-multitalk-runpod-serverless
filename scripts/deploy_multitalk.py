#!/usr/bin/env python3
"""
Deploy MultiTalk to RunPod Serverless
"""

import os
import sys
import time
import runpod
from dotenv import load_dotenv

load_dotenv()

# Set API key
runpod.api_key = os.getenv("RUNPOD_API_KEY")

def create_multitalk_endpoint():
    """Create a MultiTalk serverless endpoint."""
    
    # First, let's use a simple test image to verify deployment works
    endpoint_config = {
        "name": "multitalk-test-v1",
        "image": "runpod/pytorch:2.2.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "gpu_type": "NVIDIA GeForce RTX 4090",
        "min_workers": 0,
        "max_workers": 1,
        "idle_timeout": 60,
        "network_volume_id": "pth5bf7dey",  # Your meigen-multitalk volume
        "env": {
            "MODEL_PATH": "/runpod-volume/models",
            "RUNPOD_DEBUG_LEVEL": "INFO",
            "PYTHONUNBUFFERED": "1"
        }
    }
    
    print("Creating MultiTalk test endpoint...")
    print(f"Name: {endpoint_config['name']}")
    print(f"Image: {endpoint_config['image']}")
    print(f"GPU: {endpoint_config['gpu_type']}")
    print(f"Network Volume: {endpoint_config['network_volume_id']}")
    
    try:
        endpoint = runpod.create_endpoint(**endpoint_config)
        print(f"✓ Endpoint created successfully!")
        print(f"  ID: {endpoint['id']}")
        print(f"  Name: {endpoint['name']}")
        
        return endpoint
        
    except Exception as e:
        print(f"✗ Failed to create endpoint: {e}")
        return None

def wait_for_endpoint_ready(endpoint_id, timeout=300):
    """Wait for endpoint to be ready."""
    print(f"\nWaiting for endpoint {endpoint_id} to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            endpoints = runpod.get_endpoints()
            endpoint = next((ep for ep in endpoints if ep['id'] == endpoint_id), None)
            
            if endpoint:
                status = endpoint.get('status', 'unknown')
                ready_workers = endpoint.get('readyWorkerCount', 0)
                
                print(f"  Status: {status}, Ready workers: {ready_workers}")
                
                if status in ['RUNNING', 'READY'] and ready_workers > 0:
                    print("✓ Endpoint is ready!")
                    return True
                elif status == 'FAILED':
                    print("✗ Endpoint failed to start")
                    return False
            
        except Exception as e:
            print(f"  Error checking status: {e}")
        
        time.sleep(10)
    
    print("✗ Timeout waiting for endpoint")
    return False

def test_endpoint(endpoint_id):
    """Test the endpoint."""
    import requests
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {"Authorization": f"Bearer {runpod.api_key}"}
    
    test_data = {
        "input": {
            "health_check": True
        }
    }
    
    print(f"\nTesting endpoint {endpoint_id}...")
    try:
        response = requests.post(url, json=test_data, headers=headers, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Test successful!")
            print(f"  Response: {result}")
            return True
        else:
            print(f"✗ Test failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False

def main():
    if not runpod.api_key:
        print("Error: RUNPOD_API_KEY not found")
        sys.exit(1)
    
    print("MultiTalk RunPod Deployment")
    print("=" * 50)
    
    # Create endpoint
    endpoint = create_multitalk_endpoint()
    if not endpoint:
        sys.exit(1)
    
    endpoint_id = endpoint['id']
    
    # Wait for ready
    if not wait_for_endpoint_ready(endpoint_id):
        print("Endpoint failed to become ready")
        sys.exit(1)
    
    # Test endpoint
    if test_endpoint(endpoint_id):
        print("\n" + "=" * 50)
        print("🎉 Deployment successful!")
        print(f"Endpoint ID: {endpoint_id}")
        print(f"Test URL: https://api.runpod.ai/v2/{endpoint_id}/runsync")
        print("\nNext steps:")
        print("1. Verify network volume has models")
        print("2. Deploy full MultiTalk image")
        print("3. Test video generation")
    else:
        print("Deployment completed but test failed")

if __name__ == "__main__":
    main()