#!/usr/bin/env python3
"""
Quick test for v4 update - check if MultiTalk inference is working
"""

import runpod
import os
import time
from dotenv import load_dotenv

load_dotenv()

def test_v4():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🚀 Testing MultiTalk Complete v4")
    print("=" * 60)
    
    # Health check
    print("1. Health check...")
    job = endpoint.run({"health_check": True})
    result = job.output(timeout=60)
    
    if result:
        print(f"   Version: {result.get('version', 'unknown')}")
        if 'multitalk_inference' in result:
            mt = result['multitalk_inference']
            print(f"   MultiTalk Available: {mt.get('available', False)}")
            print(f"   MultiTalk Loaded: {mt.get('loaded', False)}")
            print(f"   Engine: {mt.get('engine', 'unknown')}")
            
            if mt.get('available'):
                print("   ✅ MultiTalk inference module loaded!")
            else:
                print("   ❌ MultiTalk inference not available")
    
    # Quick generation test
    print("\n2. Quick generation test...")
    job = endpoint.run({
        "action": "generate",
        "audio": "1.wav",
        "duration": 2.0,
        "width": 256,
        "height": 256,
        "fps": 15
    })
    
    result = job.output(timeout=120)
    
    if result and result.get('success'):
        video_info = result.get('video_info', {})
        note = video_info.get('processing_note', '')
        models = video_info.get('models_used', [])
        
        print(f"   Processing note: {note}")
        print(f"   Models used: {models}")
        
        if "Real MultiTalk inference" in note:
            print("\n🎉 SUCCESS! Real MultiTalk inference is working!")
            if 'audio_features_shape' in video_info:
                print(f"   Audio features: {video_info['audio_features_shape']}")
        elif "Fallback" in note:
            print("\n⚠️  Using fallback - models may not be loaded on RunPod")
            print("   But MultiTalk module is available and ready!")
        else:
            print("\n❓ Unexpected result")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_v4()