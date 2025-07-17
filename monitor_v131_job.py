#!/usr/bin/env python3
"""Monitor V131 job status"""
import subprocess
import json
import os
import time
import sys
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

if len(sys.argv) > 1:
    job_id = sys.argv[1]
else:
    print("Usage: python monitor_v131_job.py <job_id>")
    exit(1)

print(f"🔍 Monitoring job: {job_id}")
print("=" * 60)

start_time = time.time()
last_status = None

while True:
    elapsed = int(time.time() - start_time)
    
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    status_cmd = ["curl", "-s", "-H", f"Authorization: Bearer {api_key}", status_url]
    status_result = subprocess.run(status_cmd, capture_output=True, text=True)
    
    if status_result.returncode == 0:
        try:
            status_data = json.loads(status_result.stdout)
            status = status_data.get("status")
            
            if status != last_status:
                print(f"\n[{elapsed}s] Status changed: {last_status} → {status}")
                last_status = status
            else:
                print(f"\r[{elapsed}s] Status: {status}", end="", flush=True)
            
            if status == "COMPLETED":
                print("\n\n✅ SUCCESS! V131 completed!")
                print("🎉 NumPy issue fixed - Numba is working!")
                output = status_data.get("output", {})
                print("\nOutput:", json.dumps(output, indent=2))
                break
            elif status == "FAILED":
                print("\n\n❌ FAILED!")
                error = status_data.get("output", {})
                print(json.dumps(error, indent=2))
                
                # Check if it's still a NumPy error
                error_str = str(error).lower()
                if "numba" in error_str and "numpy" in error_str:
                    print("\n⚠️  NumPy issue persists - need further investigation")
                elif "importerror" in error_str:
                    print("\n⚠️  Still have import errors")
                break
                
        except json.JSONDecodeError:
            print(f"\n[{elapsed}s] Failed to parse status response")
    
    time.sleep(2)
    
    if elapsed > 600:  # 10 minute timeout
        print("\n\n⏰ Timeout after 10 minutes")
        break