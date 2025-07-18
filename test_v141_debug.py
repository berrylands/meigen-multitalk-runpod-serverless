#!/usr/bin/env python3
"""Test V141 debug version"""

import os
import runpod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure RunPod
runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint("zu0ik6c8yukyl6")

print("🚀 Testing V141 Debug...")
print("Endpoint ID: zu0ik6c8yukyl6")

# Simple test input
test_input = {
    "input": {
        "audio_s3_key": "1.wav",
        "image_s3_key": "multi1.png",
        "turbo": True,
        "sampling_steps": 40
    }
}

print("\nSending test request...")
print(f"Input: {test_input}")

try:
    # Send request
    run = endpoint.run(test_input)
    print(f"\n✅ Job submitted: {run.job_id}")
    print("Waiting for worker to start and show debug info...")
    
    # Don't wait for completion, just check status
    import time
    for i in range(10):
        status = run.status()
        print(f"Status: {status}")
        if status != "IN_QUEUE":
            break
        time.sleep(3)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()