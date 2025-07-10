#!/usr/bin/env python3
"""
Download correct models for MultiTalk
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

print("Downloading MultiTalk-compatible models...")
print("=" * 60)

endpoint = runpod.Endpoint(ENDPOINT_ID)

# Correct models for MultiTalk
models_to_download = [
    {
        "name": "Wav2Vec2 Base",
        "repo_id": "facebook/wav2vec2-base-960h",
        "path": "wav2vec2-base-960h"
    },
    {
        "name": "Wav2Vec2 Large",
        "repo_id": "facebook/wav2vec2-large-960h",
        "path": "wav2vec2-large-960h"
    },
    {
        "name": "GFPGAN Face Restoration",
        "repo_id": "tencentarc/gfpgan",
        "path": "gfpgan",
        "files": ["GFPGANv1.4.pth"]  # Specific model file
    }
]

job_input = {
    "action": "download_models",
    "models": models_to_download
}

print(f"\nDownloading {len(models_to_download)} models...")
for model in models_to_download:
    print(f"  - {model['name']}")

try:
    job = endpoint.run(job_input)
    print(f"\nJob submitted: {job.job_id}")
    
    # Monitor progress
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
            
        if time.time() - start_time > 900:  # 15 minute timeout
            print("Timeout")
            break
            
        time.sleep(5)
    
    if job.status() == "COMPLETED":
        output = job.output()
        
        if isinstance(output, dict) and output.get("success"):
            print("\n✓ Models downloaded successfully!")
            
            if "downloaded" in output:
                total_size = 0
                print("\nDownloaded models:")
                for model in output["downloaded"]:
                    size_mb = model.get('size', 0) / (1024 * 1024)
                    total_size += model.get('size', 0)
                    print(f"  - {model.get('name')} ({size_mb:.1f} MB)")
                    print(f"    Path: {model.get('path')}")
                
                total_mb = total_size / (1024 * 1024)
                print(f"\nTotal size: {total_mb:.1f} MB")
                
            if "errors" in output and output["errors"]:
                print("\nErrors:")
                for error in output["errors"]:
                    print(f"  - {error}")
        else:
            print(f"\n✗ Download failed")
            if isinstance(output, dict):
                print(f"Errors: {output.get('errors', [])}")
                
except Exception as e:
    print(f"\nError: {e}")

# Check what's on the volume now
print("\n" + "=" * 60)
print("Checking models on volume...")

try:
    job = endpoint.run({"action": "list_models"})
    while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
        time.sleep(2)
    
    if job.status() == "COMPLETED":
        result = job.output()
        print(f"\nModels on volume: {result.get('total', 0)}")
        if result.get('models'):
            for model in result['models']:
                print(f"  - {model['name']} ({model.get('size_mb', 0):.1f} MB, {model.get('files', 0)} files)")
                
except Exception as e:
    print(f"Error listing models: {e}")

print("\n" + "=" * 60)
print("Next steps:")
print("1. The basic models are now downloaded")
print("2. For the large Wan2.1 GGUF model (11GB+), you would need to:")
print("   - Use a RunPod GPU pod to download it directly")
print("   - Or download locally and upload to the volume")
print("3. Once all models are ready, deploy the full MultiTalk handler")