#!/usr/bin/env python3
"""
Test RunPod REST API
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

print(f"Testing with API key: {RUNPOD_API_KEY[:15]}...")

# Test the REST API endpoints
rest_base = "https://api.runpod.io"

headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

# Try different endpoints
endpoints = [
    "/v1/user",
    "/v2/user", 
    "/v2/user/me",
    "/v2/graphql",
    "/user",
    "/me"
]

for endpoint in endpoints:
    url = rest_base + endpoint
    print(f"\nTesting: {url}")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"  Status: {response.status_code}")
        if response.status_code < 500:
            print(f"  Response: {response.text[:300]}")
            if response.status_code == 200:
                print("  ✓ SUCCESS!")
    except Exception as e:
        print(f"  Error: {e}")

# Also test different auth methods
print("\n" + "="*50)
print("Testing different auth methods:")

# Method 1: Query parameter
url = "https://api.runpod.io/graphql"
print(f"\nMethod 1: Query parameter")
try:
    response = requests.post(
        f"{url}?api_key={RUNPOD_API_KEY}",
        json={"query": "{ myself { id } }"},
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.text[:200]}")
except Exception as e:
    print(f"  Error: {e}")

# Method 2: Header with different formats
auth_formats = [
    f"Bearer {RUNPOD_API_KEY}",
    f"ApiKey {RUNPOD_API_KEY}",
    f"Token {RUNPOD_API_KEY}",
    RUNPOD_API_KEY
]

for i, auth in enumerate(auth_formats):
    print(f"\nMethod {i+2}: Header '{auth[:20]}...'")
    try:
        response = requests.post(
            url,
            json={"query": "{ myself { id } }"},
            headers={
                "Content-Type": "application/json",
                "Authorization": auth
            },
            timeout=10
        )
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text[:100]}")
        if response.status_code == 200:
            print("  ✓ SUCCESS!")
    except Exception as e:
        print(f"  Error: {e}")