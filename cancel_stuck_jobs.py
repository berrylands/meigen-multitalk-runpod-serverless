#!/usr/bin/env python3
"""
Cancel stuck jobs in the queue
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "pdz7evo425qwmz"

def cancel_jobs():
    """Cancel all stuck jobs."""
    
    # Get recent jobs
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/requests"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        requests_list = data.get("requests", [])
        
        for req in requests_list:
            if req.get("status") in ["IN_QUEUE", "IN_PROGRESS"]:
                job_id = req.get("id")
                print(f"Cancelling job: {job_id}")
                
                # Cancel the job
                cancel_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/cancel/{job_id}"
                cancel_response = requests.post(cancel_url, headers=headers)
                
                if cancel_response.status_code == 200:
                    print(f"  ✓ Cancelled successfully")
                else:
                    print(f"  ✗ Failed to cancel: {cancel_response.status_code}")
    else:
        print(f"Error getting jobs: {response.status_code}")

if __name__ == "__main__":
    print("Cancelling stuck jobs...")
    cancel_jobs()