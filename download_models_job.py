#!/usr/bin/env python3
"""
Submit a job to download models to the network volume
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

print("Submitting model download job...")
print("=" * 60)

endpoint = runpod.Endpoint(ENDPOINT_ID)

# Submit download job
job_input = {
    "action": "download_models",
    "models": [
        {
            "name": "wav2vec2-base-960h",
            "repo_id": "facebook/wav2vec2-base-960h",
            "path": "wav2vec2"
        },
        {
            "name": "Wan2.1-I2V-14B-480P-GGUF",
            "repo_id": "city96/Wan2.1-I2V-14B-480P-gguf",
            "files": ["Wan2.1-I2V-14B-480P_Q4_K_M.gguf"],
            "path": "wan2.1-i2v-14b-480p"
        }
    ]
}

print("Job input:")
print(f"  Action: {job_input['action']}")
print(f"  Models to download: {len(job_input['models'])}")
for model in job_input['models']:
    print(f"    - {model['name']}")

try:
    job = endpoint.run(job_input)
    print(f"\nJob submitted: {job.job_id}")
    
    # Monitor progress
    start_time = time.time()
    last_status = None
    
    print("\nMonitoring download progress (this may take 10-20 minutes)...")
    
    while True:
        status = job.status()
        if status != last_status:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Status: {status}")
            last_status = status
            
        if status not in ["IN_QUEUE", "IN_PROGRESS"]:
            break
            
        if time.time() - start_time > 1800:  # 30 minute timeout
            print("Timeout waiting for download completion")
            break
            
        time.sleep(10)  # Check every 10 seconds
    
    # Get final result
    print(f"\nFinal status: {job.status()}")
    
    if job.status() == "COMPLETED":
        output = job.output()
        print("\n✓ Download completed!")
        
        if isinstance(output, dict):
            if output.get("success"):
                print(f"  Message: {output.get('message', 'Models downloaded')}")
                if "downloaded" in output:
                    print("  Downloaded models:")
                    for model in output["downloaded"]:
                        print(f"    - {model}")
                if "errors" in output:
                    print("  Errors:")
                    for error in output["errors"]:
                        print(f"    - {error}")
            else:
                print(f"  Error: {output.get('error', 'Unknown error')}")
        else:
            print(f"  Output: {output}")
    else:
        print(f"\n✗ Job failed")
        try:
            print(f"Output: {job.output()}")
        except:
            pass
            
except Exception as e:
    print(f"\nError: {e}")

print("\n" + "=" * 60)
print("Note: The handler currently doesn't implement model downloading.")
print("This was a test to verify job submission works.")
print("\nTo actually download models, you would need to:")
print("1. Update the handler.py to implement download_models action")
print("2. Or run the download script directly on a RunPod GPU pod")