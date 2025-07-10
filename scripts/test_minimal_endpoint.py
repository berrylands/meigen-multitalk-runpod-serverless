#!/usr/bin/env python3
"""
Test minimal endpoint deployment
"""

import os
import sys
import time
import base64
import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

def create_test_image():
    """Create a simple test image."""
    img = Image.new('RGB', (100, 100), color='blue')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_endpoint(endpoint_id, api_key):
    """Test the minimal endpoint."""
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    tests = [
        {
            "name": "Health Check",
            "data": {"input": {"health_check": True}},
            "timeout": 30
        },
        {
            "name": "Echo Test",
            "data": {"input": {"type": "echo", "message": "Hello from test!", "number": 42}},
            "timeout": 30
        },
        {
            "name": "Image Test",
            "data": {"input": {"image_base64": create_test_image(), "echo_text": "Image processed"}},
            "timeout": 60
        }
    ]
    
    print(f"Testing endpoint: {endpoint_id}")
    print("=" * 60)
    
    for test in tests:
        print(f"\n{test['name']}:")
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/runsync",
                json=test["data"],
                headers=headers,
                timeout=test["timeout"]
            )
            elapsed = time.time() - start_time
            
            print(f"  Status: {response.status_code}")
            print(f"  Time: {elapsed:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    print(f"  ✗ Error: {result['error']}")
                else:
                    print(f"  ✓ Success!")
                    # Print relevant parts of response
                    if "message" in result:
                        print(f"  Message: {result['message']}")
                    if "cuda_available" in result:
                        print(f"  CUDA: {result['cuda_available']}")
                    if "cuda_device_name" in result:
                        print(f"  GPU: {result['cuda_device_name']}")
                    if "echo" in result:
                        print(f"  Echo: {result['echo']}")
            else:
                print(f"  ✗ HTTP Error: {response.text}")
                
        except Exception as e:
            print(f"  ✗ Exception: {e}")

def main():
    # Get credentials
    api_key = os.getenv("RUNPOD_API_KEY")
    endpoint_id = input("Enter your RunPod endpoint ID: ").strip()
    
    if not api_key:
        print("Error: RUNPOD_API_KEY not found in .env file")
        sys.exit(1)
    
    if not endpoint_id:
        print("Error: Endpoint ID is required")
        sys.exit(1)
    
    test_endpoint(endpoint_id, api_key)

if __name__ == "__main__":
    main()