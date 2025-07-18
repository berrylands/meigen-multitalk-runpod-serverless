#!/usr/bin/env python3
"""Simple test for V136 connectivity"""
import runpod

# Initialize RunPod client
runpod.api_key = open('.env').read().strip().split('RUNPOD_API_KEY=')[1].split('\n')[0]
endpoint = runpod.Endpoint("zu0ik6c8yukyl6")

print("Testing V136 endpoint connectivity...")
print(f"Endpoint ID: {endpoint.endpoint_id}")

# Simple health check
try:
    health = endpoint.health()
    print(f"Health check: {health}")
except Exception as e:
    print(f"Health check error: {e}")

# Send test request
test_input = {
    "input": {
        "audio_s3_key": "1.wav",
        "image_s3_key": "multi1.png",
        "turbo": True,
        "sampling_steps": 40,
        "output_format": "s3"
    }
}

print("\nSending test request...")
try:
    run = endpoint.run(test_input)
    print(f"Request submitted: {run.job_id}")
    print(f"Initial status: {run.status()}")
except Exception as e:
    print(f"Request error: {e}")