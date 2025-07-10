#!/usr/bin/env python3
"""
Check RunPod endpoint workers and their status
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "pdz7evo425qwmz"

def check_workers():
    """Check worker status for the endpoint."""
    
    # Get worker status
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/workers"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        workers = response.json().get("workers", [])
        
        if not workers:
            print("No workers found for this endpoint")
        else:
            for i, worker in enumerate(workers):
                print(f"\nWorker {i+1}:")
                print(f"  ID: {worker.get('id', 'Unknown')}")
                print(f"  Status: {worker.get('status', 'Unknown')}")
                print(f"  GPU: {worker.get('gpu', 'Unknown')}")
                print(f"  Created: {worker.get('createdAt', 'Unknown')}")
                
                # Check for logs
                if worker.get('logs'):
                    print(f"  Logs: {worker['logs'][:200]}...")  # First 200 chars
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    # Also check recent jobs
    print("\n\nRecent Jobs:")
    print("-" * 60)
    
    # Try to get job history
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/requests"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        requests_list = data.get("requests", [])
        
        if not requests_list:
            print("No recent jobs found")
        else:
            for req in requests_list[:5]:  # Last 5 jobs
                print(f"\nJob ID: {req.get('id', 'Unknown')}")
                print(f"  Status: {req.get('status', 'Unknown')}")
                print(f"  Created: {req.get('createdAt', 'Unknown')}")
                print(f"  Completed: {req.get('completedAt', 'N/A')}")
                if req.get('error'):
                    print(f"  Error: {req['error']}")

if __name__ == "__main__":
    check_workers()