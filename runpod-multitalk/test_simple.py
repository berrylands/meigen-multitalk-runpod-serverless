#!/usr/bin/env python3
import os
import runpod

# Test if we can use RunPod API at all
runpod.api_key = os.environ.get("RUNPOD_API_KEY", "CKRTDIOF0IGFFSI4A11KTVP569QQAKQ4NK091965")

try:
    # List endpoints
    endpoints = runpod.get_endpoints()
    print(f"Found {len(endpoints)} endpoints")
    
    for ep in endpoints:
        print(f"- {ep['name']} (ID: {ep['id']})")
        
except Exception as e:
    print(f"Error: {e}")