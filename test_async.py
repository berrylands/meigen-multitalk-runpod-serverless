#!/usr/bin/env python3
"""
Test RunPod endpoint with async/queue handling
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "pdz7evo425qwmz"

def run_async_job():
    """Submit job and poll for result."""
    base_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    # Submit job
    print("Submitting health check job...")
    response = requests.post(
        f"{base_url}/run",  # Using async endpoint
        json={"input": {"health_check": True}},
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Error submitting job: {response.status_code} - {response.text}")
        return
    
    job_data = response.json()
    job_id = job_data['id']
    print(f"Job submitted: {job_id}")
    
    # Poll for result
    print("Polling for result (this may take 1-2 minutes for cold start)...")
    start_time = time.time()
    
    while True:
        response = requests.get(
            f"{base_url}/status/{job_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            status_data = response.json()
            status = status_data['status']
            
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Status: {status}")
            
            if status == "COMPLETED":
                print("\n✓ Job completed!")
                if 'output' in status_data:
                    output = status_data['output']
                    print(f"Status: {output.get('status', 'Unknown')}")
                    print(f"Message: {output.get('message', 'No message')}")
                    print(f"Volume Mounted: {output.get('volume_mounted', False)}")
                    print(f"Model Path Exists: {output.get('model_path_exists', False)}")
                    print(f"Worker ID: {output.get('worker_id', 'Unknown')}")
                    if 'environment' in output:
                        print("Environment:")
                        for k, v in output['environment'].items():
                            print(f"  {k}: {v}")
                break
            elif status == "FAILED":
                print(f"\n✗ Job failed: {status_data.get('error', 'Unknown error')}")
                break
            
            time.sleep(5)
        else:
            print(f"Error checking status: {response.status_code}")
            break
        
        if elapsed > 300:  # 5 minute timeout
            print("\n✗ Timeout waiting for job completion")
            break

if __name__ == "__main__":
    run_async_job()