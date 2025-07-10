#!/usr/bin/env python3
"""
Download models to RunPod network storage
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

print("Starting model download to network storage...")
print("=" * 60)

endpoint = runpod.Endpoint(ENDPOINT_ID)

# First, check current models
print("\n1. Checking current models on volume...")
try:
    job = endpoint.run({"action": "list_models"})
    print(f"Job ID: {job.job_id}")
    
    # Wait for result
    while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
        time.sleep(2)
    
    if job.status() == "COMPLETED":
        result = job.output()
        print(f"Current models: {result.get('total', 0)}")
        if result.get('models'):
            for model in result['models']:
                print(f"  - {model['name']} ({model.get('size_mb', 0):.1f} MB)")
    else:
        print("Failed to list models")
        
except Exception as e:
    print(f"Error: {e}")

# Define models to download
models_to_download = [
    {
        "name": "wav2vec2-base-960h",
        "repo_id": "facebook/wav2vec2-base-960h",
        "path": "wav2vec2"
    },
    {
        "name": "Tortoise TTS",
        "repo_id": "jbetker/tortoise-tts-v2",
        "path": "tortoise-tts",
        "files": ["autoregressive.pth", "diffusion_decoder.pth"]  # Just key files
    },
    {
        "name": "VQVAE",
        "repo_id": "CompVis/stable-diffusion-v1-4",
        "path": "vqvae",
        "files": ["vae/config.json", "vae/diffusion_pytorch_model.bin"]
    }
]

# Note: The large Wan2.1 GGUF model would need special handling
print("\n2. Starting model downloads...")
print(f"Models to download: {len(models_to_download)}")

job_input = {
    "action": "download_models",
    "models": models_to_download
}

try:
    job = endpoint.run(job_input)
    print(f"\nDownload job submitted: {job.job_id}")
    
    # Monitor progress
    start_time = time.time()
    last_status = None
    
    print("\nMonitoring download progress...")
    print("This may take 5-15 minutes depending on model sizes...")
    
    while True:
        status = job.status()
        if status != last_status:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Status: {status}")
            last_status = status
            
        if status not in ["IN_QUEUE", "IN_PROGRESS"]:
            break
            
        if time.time() - start_time > 1200:  # 20 minute timeout
            print("Timeout waiting for download completion")
            break
            
        time.sleep(10)
    
    # Get final result
    print(f"\nFinal status: {job.status()}")
    
    if job.status() == "COMPLETED":
        output = job.output()
        
        if isinstance(output, dict):
            if output.get("success"):
                print("\n✓ Download completed successfully!")
                print(f"  Download time: {output.get('download_time', 'Unknown')}")
                
                if "downloaded" in output:
                    print("\n  Downloaded models:")
                    for model in output["downloaded"]:
                        size_mb = model.get('size', 0) / (1024 * 1024)
                        print(f"    - {model.get('name')} ({size_mb:.1f} MB)")
                        print(f"      Path: {model.get('path')}")
                        
                if "errors" in output and output["errors"]:
                    print("\n  Errors encountered:")
                    for error in output["errors"]:
                        print(f"    - {error}")
            else:
                print(f"\n✗ Download failed: {output.get('error', 'Unknown error')}")
                if "errors" in output:
                    for error in output["errors"]:
                        print(f"  - {error}")
        else:
            print(f"Output: {output}")
    else:
        print(f"\n✗ Job failed with status: {job.status()}")
        try:
            print(f"Error: {job.output()}")
        except:
            pass
            
except Exception as e:
    print(f"\nError: {e}")

print("\n" + "=" * 60)
print("Note: Large models like Wan2.1 GGUF (11GB+) would need:")
print("1. Direct download via RunPod GPU pod with more resources")
print("2. Or manual upload to the network volume")
print("3. Or streaming download with progress tracking")