#!/usr/bin/env python3
"""
Create endpoint using RunPod SDK approach
"""

import os
import runpod
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
runpod.api_key = RUNPOD_API_KEY

print("Creating new endpoint via SDK...")
print("=" * 60)

# Try to use the internal API methods
try:
    # Method 1: Try direct API call
    response = runpod._request(
        method="POST",
        endpoint="/serverless/endpoints",
        json={
            "name": "multitalk-test-v2",
            "imageName": "berrylands/multitalk-test:latest",
            "gpuIds": ["ADA_24"],
            "networkVolumeId": "pth5bf7dey",
            "locations": ["US-NC-1"],
            "idleTimeout": 60,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
            "workersMin": 0,
            "workersMax": 1,
            "gpuCount": 1,
            "containerDiskInGb": 10,
            "volumeMountPath": "/runpod-volume",
            "env": {
                "MODEL_PATH": "/runpod-volume/models",
                "RUNPOD_DEBUG_LEVEL": "DEBUG",
                "PYTHONUNBUFFERED": "1"
            }
        }
    )
    
    if response:
        print(f"✓ Response received: {response}")
        if "id" in response:
            print(f"✓ Endpoint created!")
            print(f"  ID: {response['id']}")
            with open(".new_endpoint_id", "w") as f:
                f.write(response['id'])
    else:
        print("✗ No response received")
        
except Exception as e:
    print(f"✗ Method 1 failed: {e}")

print("\n" + "=" * 60)
print("\nSince API creation is not working, please create the endpoint manually:")
print("\n1. Go to: https://www.runpod.io/console/serverless")
print("2. Click '+ New Endpoint'")
print("3. IMPORTANT: Click 'Continue' to skip template selection")
print("4. Configure with these settings:")
print("   - Name: multitalk-test-v2")
print("   - Container Image: berrylands/multitalk-test:latest")
print("   - Container Start Command: python -u handler.py")
print("   - GPU: Select 'RTX 4090' or '24 GB'")
print("   - Max Workers: 1")
print("   - Idle Timeout: 60")
print("   - Container Disk: 10 GB")
print("   - Network Volume: Select 'meigen-multitalk' → mount at '/runpod-volume'")
print("   - Environment Variables:")
print("     MODEL_PATH = /runpod-volume/models")
print("     RUNPOD_DEBUG_LEVEL = DEBUG")
print("     PYTHONUNBUFFERED = 1")
print("\n5. Click 'Deploy'")
print("\n6. Once created, update ENDPOINT_ID in test scripts with the new ID")