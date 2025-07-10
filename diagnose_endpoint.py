#!/usr/bin/env python3
"""
Diagnose RunPod endpoint issues
"""

import os
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "pdz7evo425qwmz"

print("RunPod Endpoint Diagnostics")
print("=" * 60)

# Get all endpoints
endpoints = runpod.get_endpoints()

# Find our endpoint
our_endpoint = None
for ep in endpoints:
    if ep['id'] == ENDPOINT_ID:
        our_endpoint = ep
        break

if not our_endpoint:
    print(f"✗ Endpoint {ENDPOINT_ID} not found!")
else:
    print(f"✓ Endpoint found: {our_endpoint['name']}")
    print(f"\nEndpoint Details:")
    for key, value in our_endpoint.items():
        if key not in ['id', 'userId', 'createdAt']:  # Skip some fields
            print(f"  {key}: {value}")

print("\n" + "=" * 60)
print("Common Issues to Check:")
print("1. In RunPod dashboard, check the endpoint logs for errors")
print("2. Verify the Docker image name is correct")
print("3. Check if GPU type is available in your region")
print("4. Ensure network volume is properly attached")
print("\nNext Steps:")
print("1. Go to: https://www.runpod.io/console/serverless")
print("2. Click on your endpoint")
print("3. Check 'Logs' tab for any errors")
print("4. Check 'Workers' tab to see if any are starting")