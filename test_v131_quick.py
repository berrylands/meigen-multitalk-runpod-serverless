#!/usr/bin/env python3
"""Quick V131 test - NumPy fix verification"""
import subprocess
import json
import os
import time
from pathlib import Path

# Load .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

api_key = os.getenv("RUNPOD_API_KEY")
endpoint_id = "zu0ik6c8yukyl6"

# Submit job
job_data = {
    "input": {
        "audio_s3_key": "1.wav",
        "image_s3_key": "multi1.png",
        "device": "cuda",
        "video_format": "mp4",
        "turbo_mode": True,
        "text_guide_scale": 5.0,
        "fps": 30
    }
}

# Submit job
url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
cmd = [
    "curl", "-s", "-X", "POST", url,
    "-H", "Content-Type: application/json",
    "-H", f"Authorization: Bearer {api_key}",
    "-d", json.dumps(job_data)
]

print("🚀 Submitting V131 test job (NumPy fix)...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0 and result.stdout.strip():
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse response: {result.stdout}")
        exit(1)
    job_id = response.get("id")
    print(f"📋 Job ID: {job_id}")
    
    # Quick status check
    for i in range(30):
        time.sleep(2)
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
        status_cmd = ["curl", "-s", "-H", f"Authorization: Bearer {api_key}", status_url]
        status_result = subprocess.run(status_cmd, capture_output=True, text=True)
        
        if status_result.returncode == 0:
            status_data = json.loads(status_result.stdout)
            status = status_data.get("status")
            print(f"[{i*2}s] Status: {status}")
            
            if status == "COMPLETED":
                print("\n✅ SUCCESS! V131 completed!")
                print("🎉 NumPy issue fixed - Numba is working!")
                break
            elif status == "FAILED":
                print("\n❌ FAILED!")
                error = status_data.get("output", {})
                print(json.dumps(error, indent=2))
                
                # Check if it's still a NumPy error
                if "numba" in str(error).lower() and "numpy" in str(error).lower():
                    print("\n⚠️  NumPy issue persists - need further investigation")
                break
else:
    print(f"Failed to submit job")
    print(f"Return code: {result.returncode}")
    print(f"Stdout: {result.stdout}")
    print(f"Stderr: {result.stderr}")