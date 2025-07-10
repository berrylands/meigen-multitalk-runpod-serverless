#!/usr/bin/env python3
"""
Debug why jobs are stuck in queue
"""

import os
import time
import requests
import runpod
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "pdz7evo425qwmz"

print("Debugging RunPod Queue Issues")
print("=" * 60)

# Check endpoint health through various methods
runpod.api_key = RUNPOD_API_KEY
endpoint = runpod.Endpoint(ENDPOINT_ID)

# 1. Try health check
print("\n1. Checking endpoint health:")
try:
    health = endpoint.health()
    print(f"   Health response: {health}")
except Exception as e:
    print(f"   Health check error: {e}")

# 2. Submit a minimal test job
print("\n2. Submitting minimal test job:")
try:
    job = endpoint.run({"test": "minimal"})
    print(f"   Job ID: {job.job_id}")
    
    # Check status immediately
    for i in range(5):
        status = job.status()
        print(f"   [{i}s] Status: {status}")
        if status not in ["IN_QUEUE", "IN_PROGRESS"]:
            break
        time.sleep(1)
        
except Exception as e:
    print(f"   Error: {e}")

# 3. Check GPU availability in region
print("\n3. Checking region details:")
headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}

# Try to get datacenter/region info
region_endpoints = [
    "https://api.runpod.ai/v1/datacenters",
    "https://api.runpod.ai/v1/gpu-types",
    "https://api.runpod.ai/v1/regions"
]

for url in region_endpoints:
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"\n   {url}")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Data: {str(data)[:200]}...")
    except Exception as e:
        print(f"   Error: {e}")

# 4. Check template details
print("\n4. Template configuration issues:")
print("   Template ID: x7tcaxtizz")
print("   This template might be overriding your Docker image")
print("   The template could be pointing to a different/broken image")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("1. Check the RunPod dashboard logs for specific errors")
print("2. Consider creating a NEW endpoint without using a template")
print("3. The template might be the root cause of the issues")
print("4. RTX 4090 (ADA_24) might not be available in US-NC-1")

# Cancel the test job
try:
    if 'job' in locals():
        print(f"\nCancelling test job {job.job_id}...")
        cancel_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/cancel/{job.job_id}"
        requests.post(cancel_url, headers=headers)
except:
    pass