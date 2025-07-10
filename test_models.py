#!/usr/bin/env python3
"""
Test the models available on the endpoint
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

print("Testing MultiTalk Models")
print("=" * 60)

endpoint = runpod.Endpoint(ENDPOINT_ID)

# Test model listing
print("1. Listing available models...")
try:
    job = endpoint.run({"action": "list_models"})
    print(f"Job ID: {job.job_id}")
    
    while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
        time.sleep(2)
    
    if job.status() == "COMPLETED":
        result = job.output()
        models = result.get('models', [])
        total = result.get('total', 0)
        
        print(f"\n✓ Found {total} models on volume:")
        total_size = 0
        for model in models:
            size_mb = model.get('size_mb', 0)
            total_size += size_mb
            print(f"  - {model['name']}: {size_mb:.1f} MB ({model.get('files', 0)} files)")
        
        print(f"\nTotal storage used: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
        
        # Check for key models
        model_names = [m['name'].lower() for m in models]
        print(f"\nModel availability check:")
        print(f"  Wav2Vec2: {'✓' if any('wav2vec2' in name for name in model_names) else '✗'}")
        print(f"  Wan2.1: {'✓' if any('wan2.1' in name for name in model_names) else '✗'}")
        print(f"  GFPGAN: {'✓' if any('gfp' in name for name in model_names) else '✗'}")
        
    else:
        print(f"✗ Failed to list models: {job.output()}")
        
except Exception as e:
    print(f"Error: {e}")

# Test the handler's supported actions
print(f"\n2. Testing handler capabilities...")
try:
    job = endpoint.run({"test": "capabilities"})
    
    while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
        time.sleep(2)
    
    if job.status() == "COMPLETED":
        result = job.output()
        print(f"✓ Handler response: {result.get('message', 'No message')}")
        
        if 'supported_actions' in result:
            actions = result['supported_actions']
            print(f"  Supported actions: {', '.join(actions)}")
        
        if 'example_request' in result:
            print(f"  Example request format available")
            
    else:
        print(f"✗ Handler test failed: {job.output()}")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Model test complete!")
print("\nNext: Test video generation with available models")