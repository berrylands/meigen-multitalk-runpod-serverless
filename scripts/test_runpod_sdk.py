#!/usr/bin/env python3
"""
Test RunPod SDK
"""

import os
import runpod
from dotenv import load_dotenv

load_dotenv()

# Set API key
runpod.api_key = os.getenv("RUNPOD_API_KEY")

print(f"Testing RunPod SDK with API key: {runpod.api_key[:15]}...")

try:
    # Try to get user info
    print("\nTesting SDK functions:")
    
    # Test getting pods
    try:
        pods = runpod.get_pods()
        print(f"✓ get_pods() worked - found {len(pods) if pods else 0} pods")
        if pods:
            for pod in pods[:3]:  # Show first 3
                print(f"  - {pod.get('name', 'Unnamed')} ({pod.get('id', 'No ID')})")
    except Exception as e:
        print(f"✗ get_pods() failed: {e}")
    
    # Test getting endpoints
    try:
        endpoints = runpod.get_endpoints()
        print(f"✓ get_endpoints() worked - found {len(endpoints) if endpoints else 0} endpoints")
        if endpoints:
            for ep in endpoints[:3]:  # Show first 3
                print(f"  - {ep.get('name', 'Unnamed')} ({ep.get('id', 'No ID')})")
    except Exception as e:
        print(f"✗ get_endpoints() failed: {e}")
    
    # Test GPU types
    try:
        gpus = runpod.get_gpu_types()
        print(f"✓ get_gpu_types() worked - found {len(gpus) if gpus else 0} GPU types")
        if gpus:
            rtx_gpus = [g for g in gpus if 'rtx' in g.get('displayName', '').lower() or '4090' in g.get('displayName', '')]
            if rtx_gpus:
                print("  RTX GPUs available:")
                for gpu in rtx_gpus[:3]:
                    print(f"    - {gpu.get('displayName', 'Unknown')} ({gpu.get('memoryInGb', 0)}GB)")
    except Exception as e:
        print(f"✗ get_gpu_types() failed: {e}")

except Exception as e:
    print(f"SDK test failed: {e}")

# Test if we can create a simple endpoint
print("\n" + "="*50)
print("Testing endpoint creation (dry run):")

try:
    # Just test the configuration, don't actually create
    endpoint_config = {
        "name": "test-multitalk",
        "image_name": "runpod/pytorch:2.2.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "gpu_type_id": "NVIDIA RTX A4000",
        "env": {
            "TEST": "true"
        }
    }
    print(f"✓ Configuration ready: {endpoint_config['name']}")
    print(f"  Image: {endpoint_config['image_name']}")
    print(f"  GPU: {endpoint_config['gpu_type_id']}")
    
except Exception as e:
    print(f"✗ Configuration test failed: {e}")

print("\nSDK test complete!")