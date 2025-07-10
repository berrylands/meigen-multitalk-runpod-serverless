#!/usr/bin/env python3
"""
Simple test to check if endpoint is responding
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "pdz7evo425qwmz"

print(f"Testing endpoint {ENDPOINT_ID}...")
print("=" * 60)

# Create endpoint client
endpoint = runpod.Endpoint(ENDPOINT_ID)

# Submit a simple health check job
print("\nSubmitting health check job...")
try:
    job = endpoint.run({"health_check": True})
    print(f"Job ID: {job.job_id}")
    print(f"Status: {job.status()}")
    
    # Wait for completion
    print("\nWaiting for job completion...")
    start_time = time.time()
    
    while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
        elapsed = time.time() - start_time
        print(f"[{elapsed:.1f}s] Status: {job.status()}")
        time.sleep(5)
        
        if elapsed > 300:  # 5 minute timeout
            print("\nTimeout waiting for job")
            break
    
    # Get final status and output
    final_status = job.status()
    print(f"\nFinal status: {final_status}")
    
    if final_status == "COMPLETED":
        output = job.output()
        print("\n✓ Job completed successfully!")
        print(f"Output: {output}")
    elif final_status == "FAILED":
        print("\n✗ Job failed")
        print(f"Error: {job.output()}")
        
except Exception as e:
    print(f"\n✗ Error: {e}")

print("\n" + "=" * 60)