#!/usr/bin/env python3
"""Check job status for V136"""
import runpod
import time
import requests

# Initialize RunPod client
runpod.api_key = open('.env').read().strip().split('RUNPOD_API_KEY=')[1].split('\n')[0]

job_id = "9ca06d3a-d922-42f8-b7de-bb700ffaa711-e1"
endpoint_id = "zu0ik6c8yukyl6"
print(f"Checking job: {job_id}")

# Direct API call to check status
api_key = runpod.api_key
headers = {"Authorization": f"Bearer {api_key}"}
url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

# Poll status
for i in range(60):  # Check for up to 2 minutes
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        status = data.get('status', 'UNKNOWN')
        print(f"[{i*2}s] Status: {status}")
        
        if status in ["COMPLETED", "FAILED"]:
            if status == "COMPLETED":
                output = data.get('output', {})
                print(f"\nResult: {output}")
                if 'error' in output:
                    print("\n🔍 Error details:")
                    print(output.get('error', 'Unknown error'))
                    if 'traceback' in output:
                        print("\nTraceback preview:")
                        print(output['traceback'][:1000])
            else:
                print(f"\nJob failed: {data}")
            break
            
    except Exception as e:
        print(f"Error checking status: {e}")
        
    time.sleep(2)