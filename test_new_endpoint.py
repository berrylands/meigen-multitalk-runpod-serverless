#!/usr/bin/env python3
"""Test the new endpoint"""
import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

print(f"Testing new endpoint: {ENDPOINT_ID}")
print("=" * 60)

endpoint = runpod.Endpoint(ENDPOINT_ID)

# Check health
print("\nChecking endpoint health...")
try:
    health = endpoint.health()
    print(f"Health: {health}")
except Exception as e:
    print(f"Health check error: {e}")

# Submit test job
print("\nSubmitting test job...")
try:
    job = endpoint.run({"health_check": True})
    print(f"Job ID: {job.job_id}")
    
    # Monitor status
    start_time = time.time()
    last_status = None
    
    while True:
        status = job.status()
        if status != last_status:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Status: {status}")
            last_status = status
            
        if status not in ["IN_QUEUE", "IN_PROGRESS"]:
            break
            
        if time.time() - start_time > 180:  # 3 minute timeout
            print("Timeout waiting for job completion")
            break
            
        time.sleep(5)
    
    # Get final result
    if job.status() == "COMPLETED":
        print(f"\n✓ Job completed successfully!")
        print(f"Output: {job.output()}")
    else:
        print(f"\n✗ Job failed with status: {job.status()}")
        try:
            print(f"Output: {job.output()}")
        except:
            pass
            
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)