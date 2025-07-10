#!/usr/bin/env python3
"""
Check the template configuration
"""

import os
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

# The template ID from the endpoint
TEMPLATE_ID = "x7tcaxtizz"

print("Checking template configuration...")

# Get templates
try:
    # This might not work with the current API, but let's try
    response = runpod._request(
        method="GET",
        endpoint="/templates"
    )
    
    if response and "templates" in response:
        for template in response["templates"]:
            if template.get("id") == TEMPLATE_ID:
                print(f"\nFound template: {template.get('name')}")
                print(f"Docker Image: {template.get('imageName')}")
                print(f"Container Disk: {template.get('containerDiskInGb')}GB")
                break
    else:
        print("Could not retrieve templates")
        
except Exception as e:
    print(f"Error: {e}")

# Let's also check if we can get endpoint details through runpod SDK
print("\n\nChecking endpoint through SDK...")
endpoint = runpod.Endpoint("pdz7evo425qwmz")
print(f"Endpoint ID: {endpoint.endpoint_id}")

# Try to check health
try:
    health = endpoint.health()
    print(f"Health: {health}")
except Exception as e:
    print(f"Health check error: {e}")