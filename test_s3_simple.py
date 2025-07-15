#!/usr/bin/env python3
"""
Test S3 with simple filename - debug version
"""

import runpod
import os
from dotenv import load_dotenv

load_dotenv()

def test_s3_simple():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 Testing S3 Simple Filename Support")
    print("=" * 60)
    
    # Test with ONLY audio, no reference image
    test_job = {
        "action": "generate",
        "audio": "1.wav",  # Simple filename
        "duration": 3.0,
        "width": 256,
        "height": 256,
        "fps": 25
        # NO reference_image - this might be causing issues
    }
    
    print(f"Sending job: {test_job}")
    print("\nNote: NOT including reference_image to avoid that error")
    
    job = endpoint.run(test_job)
    result = job.output(timeout=120)
    
    if result:
        if result.get('success'):
            params = result.get('parameters', {})
            video_info = result.get('video_info', {})
            
            print(f"\n✅ Success!")
            print(f"   Audio size: {params.get('audio_size', 0):,} bytes")
            print(f"   Processing note: {video_info.get('processing_note', 'N/A')}")
            print(f"   Models used: {video_info.get('models_used', [])}")
            
            if params.get('audio_size', 0) > 1000:
                print("\n🎉 Real audio file was loaded from S3!")
            elif params.get('audio_size', 0) == 3:
                print("\n❌ Still getting 3 bytes - base64 decode issue")
                print("   Need to update to v5 image!")
        else:
            print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_s3_simple()