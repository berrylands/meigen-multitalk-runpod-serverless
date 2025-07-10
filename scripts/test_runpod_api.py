#!/usr/bin/env python3
"""
Test RunPod API connection
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

print(f"API Key: {RUNPOD_API_KEY[:10]}...{RUNPOD_API_KEY[-5:]}")

# Test different API endpoints
endpoints = [
    "https://api.runpod.io/graphql",
    "https://api.runpod.ai/graphql",
    "https://api.runpod.io/v2/graphql",
]

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {RUNPOD_API_KEY}"
}

simple_query = """
{
    myself {
        id
        email
    }
}
"""

for endpoint in endpoints:
    print(f"\nTesting endpoint: {endpoint}")
    try:
        response = requests.post(
            endpoint,
            json={"query": simple_query},
            headers=headers,
            timeout=10
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {response.json()}")
            print("  ✓ SUCCESS!")
            break
        else:
            print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

# Also test the REST API
print("\nTesting REST API endpoints:")
rest_endpoints = [
    "https://api.runpod.io/v2/user",
    "https://api.runpod.ai/v2/user",
]

for endpoint in rest_endpoints:
    print(f"\nTesting: {endpoint}")
    try:
        response = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            timeout=10
        )
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")