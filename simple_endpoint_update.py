#!/usr/bin/env python3
"""
Simple endpoint update using RunPod REST API
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
ENDPOINT_ID = "zu0ik6c8yukyl6"
TEMPLATE_ID = "slyfdvoag8"
API_KEY = os.environ.get('RUNPOD_API_KEY')

def update_endpoint():
    """Update endpoint using REST API."""
    
    # Try different API endpoints
    urls_to_try = [
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}",
        f"https://api.runpod.ai/v2/endpoints/{ENDPOINT_ID}",
        f"https://api.runpod.ai/v1/endpoints/{ENDPOINT_ID}"
    ]
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "templateId": TEMPLATE_ID
    }
    
    for url in urls_to_try:
        print(f"Trying: {url}")
        
        try:
            response = requests.patch(url, headers=headers, json=payload)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
            if response.status_code == 200:
                print("✅ Success!")
                return True
                
        except Exception as e:
            print(f"Error: {str(e)}")
            
    return False

if __name__ == "__main__":
    print("Attempting to update endpoint...")
    success = update_endpoint()
    
    if not success:
        print("\n⚠️ Automatic update failed")
        print("Manual steps:")
        print("1. Go to RunPod console")
        print("2. Find endpoint zu0ik6c8yukyl6")
        print("3. Update template to: multitalk-v121-mock-xfuser")
        print("4. Save changes")
    else:
        print("\n✅ Endpoint updated successfully!")