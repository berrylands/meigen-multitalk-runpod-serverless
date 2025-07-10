#!/usr/bin/env python3
"""
Test RunPod deployment
"""

import os
import sys
import json
import base64
import time
from io import BytesIO
from PIL import Image
import numpy as np
import requests

def create_test_image():
    """Create a test image and return as base64."""
    # Create a simple test image
    img = Image.new('RGB', (640, 480), color='blue')
    
    # Add some variation
    pixels = np.array(img)
    pixels[100:200, 100:200] = [255, 0, 0]  # Red square
    pixels[300:400, 300:400] = [0, 255, 0]  # Green square
    
    img = Image.fromarray(pixels)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64

def create_test_audio():
    """Create a test audio and return as base64."""
    import wave
    
    # Generate a simple sine wave
    sample_rate = 16000
    duration = 2.0
    frequency = 440.0
    
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Scale to 16-bit
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write to WAV in memory
    buffer = BytesIO()
    with wave.open(buffer, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    # Convert to base64
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return audio_base64

def get_latest_endpoint(api_key):
    """Get the latest deployed endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    query = """
    query {
        myself {
            serverlessEndpoints {
                id
                name
                status
                dockerImage
                readyWorkerCount
            }
        }
    }
    """
    
    response = requests.post(
        "https://api.runpod.ai/graphql",
        json={"query": query},
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Error getting endpoints: {response.text}")
        return None
    
    data = response.json()
    endpoints = data.get("data", {}).get("myself", {}).get("serverlessEndpoints", [])
    
    # Find the latest multitalk endpoint
    multitalk_endpoints = [e for e in endpoints if "multitalk" in e["name"].lower()]
    
    if not multitalk_endpoints:
        print("No MultiTalk endpoints found")
        return None
    
    # Sort by name (which includes run number) and get the latest
    latest = sorted(multitalk_endpoints, key=lambda x: x["name"])[-1]
    
    print(f"Found endpoint: {latest['name']} (ID: {latest['id']})")
    print(f"Status: {latest['status']}, Ready workers: {latest['readyWorkerCount']}")
    
    return latest["id"]

def test_health_check(endpoint_id, api_key):
    """Test the health check endpoint."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    print("\nTesting health check...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        print("✓ Health check passed")
        return True
    else:
        print(f"✗ Health check failed: {response.status_code} - {response.text}")
        return False

def test_inference(endpoint_id, api_key):
    """Test the inference endpoint."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("\nTesting inference...")
    
    # Create test data
    test_image = create_test_image()
    test_audio = create_test_audio()
    
    request_data = {
        "input": {
            "reference_image": test_image,
            "audio_1": test_audio,
            "prompt": "A person speaking in a test video",
            "num_frames": 50,
            "seed": 42,
            "turbo": True,
            "sampling_steps": 10
        }
    }
    
    print("Sending inference request...")
    start_time = time.time()
    
    response = requests.post(url, json=request_data, headers=headers, timeout=300)
    
    elapsed_time = time.time() - start_time
    print(f"Request completed in {elapsed_time:.2f} seconds")
    
    if response.status_code != 200:
        print(f"✗ Inference failed: {response.status_code} - {response.text}")
        return False
    
    try:
        result = response.json()
        
        if "error" in result:
            print(f"✗ Inference error: {result['error']}")
            return False
        
        if "output" in result:
            output = result["output"]
            if "video_base64" in output or "video_url" in output:
                print("✓ Inference successful - video generated")
                return True
            else:
                print(f"✗ Unexpected output format: {output}")
                return False
        else:
            print(f"✗ No output in response: {result}")
            return False
            
    except Exception as e:
        print(f"✗ Error parsing response: {e}")
        return False

def main():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    
    # Get endpoint ID from environment or find latest
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    
    if not endpoint_id:
        endpoint_id = get_latest_endpoint(api_key)
        
    if not endpoint_id:
        print("No endpoint ID found")
        sys.exit(1)
    
    print(f"Testing endpoint: {endpoint_id}")
    
    # Run tests
    tests_passed = 0
    tests_total = 2
    
    if test_health_check(endpoint_id, api_key):
        tests_passed += 1
    
    if test_inference(endpoint_id, api_key):
        tests_passed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()