#!/usr/bin/env python3
"""
Verify which handler code is actually running in RunPod
"""

import runpod
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify_handler():
    """Check which handler implementation is running"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 Verifying Handler Implementation")
    print("=" * 60)
    
    # Test 1: Health check
    print("1. Health check to see version info...")
    job = endpoint.run({"health_check": True})
    result = job.output(timeout=30)
    
    if result:
        print(f"   Version: {result.get('version', 'unknown')}")
        print(f"   Image tag: {result.get('image_tag', 'unknown')}")
        print(f"   Build ID: {result.get('build_id', 'unknown')}")
        
        # Check for MultiTalk inference info
        if 'multitalk_inference' in result:
            print(f"   ✅ NEW HANDLER - MultiTalk inference info present")
            print(f"   MultiTalk: {result['multitalk_inference']}")
        else:
            print(f"   ❌ OLD HANDLER - No MultiTalk inference info")
    
    # Test 2: Check default response
    print("\n2. Default handler response...")
    job = endpoint.run({})
    result = job.output(timeout=30)
    
    if result:
        if 'multitalk_inference' in result:
            print(f"   ✅ NEW HANDLER - Has multitalk_inference field")
        else:
            print(f"   ❌ OLD HANDLER - Missing multitalk_inference field")
        
        print(f"   Message: {result.get('message', 'N/A')}")
        print(f"   Version: {result.get('version', 'N/A')}")
    
    # Test 3: Small video generation to check processing note
    print("\n3. Testing video generation to check processing note...")
    test_job = {
        "action": "generate",
        "audio": "1.wav",
        "duration": 1.0,  # Very short
        "width": 128,     # Very small
        "height": 128,
        "fps": 10
    }
    
    job = endpoint.run(test_job)
    result = job.output(timeout=60)
    
    if result and result.get('success'):
        video_info = result.get('video_info', {})
        processing_note = video_info.get('processing_note', 'N/A')
        
        print(f"   Processing note: {processing_note}")
        
        if "Test implementation" in processing_note:
            print(f"   ❌ OLD HANDLER - Still using test implementation")
        elif "Real MultiTalk" in processing_note:
            print(f"   ✅ NEW HANDLER - Using real MultiTalk inference!")
        elif "Fallback" in processing_note:
            print(f"   ⚠️  NEW HANDLER - Using fallback (models not loaded)")
        else:
            print(f"   ❓ UNKNOWN - Unexpected processing note")
    
    print("\n" + "=" * 60)
    print("\n📋 DIAGNOSIS:")
    print("If you see OLD HANDLER indicators, RunPod is using cached code.")
    print("You need to:")
    print("1. Use a completely new image tag")
    print("2. Or restart/recreate the endpoint")
    print("3. Or add a unique version suffix to force update")

if __name__ == "__main__":
    verify_handler()