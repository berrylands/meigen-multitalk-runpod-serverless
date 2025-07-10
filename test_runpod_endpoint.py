#!/usr/bin/env python3
"""
Test RunPod endpoint with proper queue handling
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "pdz7evo425qwmz"

def test_sync_endpoint():
    """Test the endpoint with synchronous calls."""
    base_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    print(f"Testing endpoint: {ENDPOINT_ID}")
    print("=" * 60)
    
    # Health Check Test
    print("\n1. Health Check Test")
    test_data = {"input": {"health_check": True}}
    
    print("   Sending request...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{base_url}/runsync",
            json=test_data,
            headers=headers,
            timeout=300  # 5 minute timeout for cold start
        )
        
        elapsed = time.time() - start_time
        print(f"   Response received in {elapsed:.2f}s")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if job is queued
            if result.get("status") == "IN_QUEUE":
                print("   Job is in queue, waiting for worker to start...")
                job_id = result.get("id")
                
                # Poll for completion
                while True:
                    time.sleep(2)
                    status_response = requests.get(
                        f"{base_url}/status/{job_id}",
                        headers=headers
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        current_status = status_data.get("status")
                        
                        print(f"   Status: {current_status}")
                        
                        if current_status == "COMPLETED":
                            if "output" in status_data:
                                output = status_data["output"]
                                print_health_check_results(output)
                            break
                        elif current_status == "FAILED":
                            print(f"   ✗ Job failed: {status_data.get('error', 'Unknown error')}")
                            break
                        elif current_status not in ["IN_QUEUE", "IN_PROGRESS"]:
                            print(f"   Unknown status: {current_status}")
                            print(f"   Full response: {status_data}")
                            break
                            
            elif "output" in result:
                # Direct response
                print_health_check_results(result["output"])
            else:
                print(f"   Unexpected response: {result}")
                
        else:
            print(f"   ✗ Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"   ✗ Request timed out after {elapsed:.2f}s")
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    # Echo Test
    print("\n2. Echo Test")
    test_data = {"input": {"test": "echo", "message": "Hello RunPod!", "timestamp": time.time()}}
    
    try:
        response = requests.post(
            f"{base_url}/runsync",
            json=test_data,
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("status") == "IN_QUEUE":
                print("   Job queued, endpoint is warm now so should be quick...")
            elif "output" in result:
                output = result["output"]
                print(f"   ✓ Response received immediately")
                print(f"   ✓ Message: {output.get('message', 'No message')}")
                if 'echo' in output:
                    print(f"   ✓ Echo: {output['echo']}")
        else:
            print(f"   ✗ Error: {response.text}")
            
    except Exception as e:
        print(f"   ✗ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")

def print_health_check_results(output):
    """Print health check results."""
    print(f"   ✓ Health Status: {output.get('status', 'Unknown')}")
    print(f"   ✓ Message: {output.get('message', 'No message')}")
    print(f"   ✓ Volume Mounted: {output.get('volume_mounted', False)}")
    print(f"   ✓ Model Path Exists: {output.get('model_path_exists', False)}")
    print(f"   ✓ Worker ID: {output.get('worker_id', 'Unknown')}")
    print(f"   ✓ Python Version: {output.get('python_version', 'Unknown')}")
    if 'environment' in output:
        print(f"   ✓ Environment:")
        for key, value in output['environment'].items():
            print(f"      - {key}: {value}")

if __name__ == "__main__":
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY not found")
        exit(1)
    
    test_sync_endpoint()