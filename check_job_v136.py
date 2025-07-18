#!/usr/bin/env python3
"""Check job status for V136"""
import runpod
import time

# Initialize RunPod client
runpod.api_key = open('.env').read().strip().split('RUNPOD_API_KEY=')[1].split('\n')[0]
endpoint = runpod.Endpoint("zu0ik6c8yukyl6")

job_id = "9ca06d3a-d922-42f8-b7de-bb700ffaa711-e1"
print(f"Checking job: {job_id}")

# Get job handle
run = endpoint.run_sync(job_id)

# Poll status
for i in range(60):  # Check for up to 2 minutes
    status = run.status()
    print(f"[{i*2}s] Status: {status}")
    
    if status in ["COMPLETED", "FAILED"]:
        result = run.output()
        print(f"\nFinal status: {status}")
        print(f"Result: {result}")
        
        if isinstance(result, dict) and 'error' in result:
            print("\n🔍 Error details:")
            print(result.get('error', 'Unknown error'))
            if 'traceback' in result:
                print("\nTraceback:")
                print(result['traceback'][:1000])
        break
        
    time.sleep(2)