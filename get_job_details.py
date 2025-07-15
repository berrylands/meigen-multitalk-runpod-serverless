#!/usr/bin/env python3
"""
Get detailed job information
"""

import runpod
import os
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_job_details():
    """Get detailed job information"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    
    # Submit a simple test job
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 Submitting test job to get detailed error...")
    
    job_input = {
        "action": "generate",
        "audio": "s3://760572149-framepack/1.wav",
        "duration": 5.0,
        "width": 480,
        "height": 480
    }
    
    job = endpoint.run(job_input)
    print(f"Job ID: {job.job_id}")
    
    # Wait a bit
    print("Waiting for job to process...")
    time.sleep(5)
    
    # Get all possible details
    print("\n📊 Job Details:")
    print(f"Status: {job.status()}")
    
    # Try different ways to get the output
    try:
        output = job.output()
        print(f"\nOutput type: {type(output)}")
        
        if isinstance(output, dict):
            print("\nOutput contents:")
            for key, value in output.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}... ({len(value)} chars)")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"Output: {output}")
            
    except Exception as e:
        print(f"Error getting output: {e}")
    
    # Try the stream method which might have more details
    try:
        print("\n📜 Attempting to get job stream...")
        if hasattr(job, 'stream'):
            for event in job.stream():
                print(f"Stream event: {event}")
                if event.get('status') == 'FAILED':
                    break
    except:
        pass
    
    # Make a direct API call to get job status
    print("\n🔧 Direct API check:")
    
    import requests
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Try to get job status
    job_status_url = f"https://api.runpod.ai/v2/kkx3cfy484jszl/status/{job.job_id}"
    
    try:
        response = requests.get(job_status_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2))
        else:
            print(f"Status code: {response.status_code}")
    except Exception as e:
        print(f"API error: {e}")

if __name__ == "__main__":
    get_job_details()