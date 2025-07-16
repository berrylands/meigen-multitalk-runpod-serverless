#!/usr/bin/env python3
"""
Update RunPod endpoint to use V121 template
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
ENDPOINT_ID = "zu0ik6c8yukyl6"
TEMPLATE_ID = "slyfdvoag8"  # V121 template
API_KEY = os.environ.get('RUNPOD_API_KEY')

def update_endpoint():
    """Update endpoint to use V121 template."""
    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/template"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "templateId": TEMPLATE_ID
    }
    
    print(f"Updating endpoint {ENDPOINT_ID} to use template {TEMPLATE_ID}")
    
    response = requests.patch(url, headers=headers, json=payload)
    
    print(f"Response status: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("✅ Endpoint updated successfully!")
        return True
    else:
        print("❌ Failed to update endpoint")
        return False

if __name__ == "__main__":
    success = update_endpoint()
    if success:
        print("Ready to test V121 on RunPod!")
    else:
        print("Please check the endpoint update manually.")