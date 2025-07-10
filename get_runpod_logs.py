#!/usr/bin/env python3
"""
Attempt to get RunPod logs via API
"""

import os
import requests
import runpod
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "pdz7evo425qwmz"

print("Attempting to access RunPod logs via API...")
print("=" * 60)

# Method 1: Try through the RunPod Python SDK
try:
    runpod.api_key = RUNPOD_API_KEY
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Try to get endpoint info
    print("\n1. Endpoint Info:")
    print(f"   Endpoint ID: {endpoint.endpoint_id}")
    
except Exception as e:
    print(f"   SDK Error: {e}")

# Method 2: Try direct API calls for logs
headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}

# Try various log endpoints
log_endpoints = [
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/logs",
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/workers/logs",
    f"https://api.runpod.ai/v1/endpoints/{ENDPOINT_ID}/logs",
    f"https://api.runpod.ai/graphql"  # For GraphQL query
]

print("\n2. Trying various log endpoints:")
for url in log_endpoints[:-1]:  # Skip GraphQL for now
    print(f"\n   Trying: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data}")
        elif response.status_code != 404:
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   Error: {e}")

# Method 3: Try GraphQL for logs
print("\n3. Trying GraphQL query for logs:")
graphql_query = """
query getEndpointLogs($endpointId: String!) {
    endpoint(id: $endpointId) {
        id
        name
        logs {
            timestamp
            message
            level
        }
        workers {
            id
            status
            logs
        }
    }
}
"""

try:
    response = requests.post(
        "https://api.runpod.ai/graphql",
        json={
            "query": graphql_query,
            "variables": {"endpointId": ENDPOINT_ID}
        },
        headers=headers
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        if "errors" in result:
            print(f"   GraphQL Errors: {result['errors']}")
        else:
            print(f"   GraphQL Response: {result}")
    else:
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"   Error: {e}")

# Method 4: Check job-specific logs
print("\n4. Checking recent job logs:")
jobs_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/requests"
try:
    response = requests.get(jobs_url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        jobs = data.get("requests", [])
        
        for job in jobs[:2]:  # Check first 2 jobs
            job_id = job.get("id")
            print(f"\n   Job {job_id}:")
            
            # Try to get job-specific logs
            job_log_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/logs/{job_id}"
            log_response = requests.get(job_log_url, headers=headers)
            print(f"   Log Status: {log_response.status_code}")
            if log_response.status_code == 200:
                print(f"   Logs: {log_response.json()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("Note: RunPod API may not expose detailed logs.")
print("Dashboard access might be required for full log visibility.")