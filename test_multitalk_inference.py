#!/usr/bin/env python3
"""
Test the real MultiTalk inference implementation
"""

import runpod
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_multitalk_inference():
    """Test the complete MultiTalk implementation with real inference"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🚀 Testing Real MultiTalk Inference")
    print("=" * 60)
    
    # First check health
    print("1. Checking health...")
    job = endpoint.run({"health_check": True})
    result = job.output(timeout=60)
    
    if result:
        print(f"✅ Health check passed")
        print(f"   Version: {result.get('version', 'unknown')}")
        print(f"   Image: {result.get('image_tag', 'unknown')}")
        if 'multitalk_inference' in result:
            mt_info = result['multitalk_inference']
            print(f"   MultiTalk Available: {mt_info.get('available', False)}")
            print(f"   MultiTalk Loaded: {mt_info.get('loaded', False)}")
            print(f"   Engine: {mt_info.get('engine', 'unknown')}")
        print(f"   GPU: {result.get('gpu_info', {})}")
    
    # Test video generation with S3 input
    print("\n2. Testing video generation with S3 audio...")
    test_job = {
        "action": "generate",
        "audio": "1.wav",  # S3 file
        "duration": 3.0,   # Shorter for testing
        "width": 256,      # Smaller for faster testing
        "height": 256,
        "fps": 25,
        "output_format": "s3",
        "s3_output_key": f"test_outputs/multitalk_test_{int(time.time())}.mp4"
    }
    
    print(f"   Request: {test_job}")
    start_time = time.time()
    
    job = endpoint.run(test_job)
    result = job.output(timeout=300)  # 5 minutes timeout
    
    elapsed = time.time() - start_time
    
    if result and result.get("success"):
        print(f"\n✅ Video generation successful!")
        print(f"   Processing time: {elapsed:.1f}s")
        print(f"   Video output: {result.get('video', 'N/A')[:100]}...")
        print(f"   Output type: {result.get('output_type', 'unknown')}")
        if result.get('output_type') == 's3_url':
            print(f"   S3 key: {result.get('s3_key', 'N/A')}")
            print(f"   S3 bucket: {result.get('s3_bucket', 'N/A')}")
        
        video_info = result.get('video_info', {})
        print(f"\n   Video Info:")
        print(f"   - Processing note: {video_info.get('processing_note', 'N/A')}")
        print(f"   - Models used: {video_info.get('models_used', [])}")
        print(f"   - Duration: {video_info.get('duration', 0)}s")
        print(f"   - Resolution: {video_info.get('resolution', 'N/A')}")
        print(f"   - Frames: {video_info.get('frames', 0)}")
        print(f"   - Video size: {video_info.get('video_size', 0):,} bytes")
        
        if 'audio_features_shape' in video_info:
            print(f"   - Audio features shape: {video_info['audio_features_shape']}")
        
        if 'audio_features' in result:
            print(f"\n   Audio Features:")
            print(f"   - Shape: {result['audio_features'].get('shape', 'N/A')}")
            print(f"   - Extracted: {result['audio_features'].get('extracted', False)}")
    else:
        print(f"\n❌ Video generation failed!")
        print(f"   Error: {result.get('error', 'Unknown error') if result else 'No response'}")
        if result and 'traceback' in result:
            print(f"\n   Traceback:\n{result['traceback']}")
    
    print("\n" + "=" * 60)
    
    # Check what's actually being used
    if result and result.get('success'):
        if 'Real MultiTalk' in str(video_info.get('processing_note', '')):
            print("\n🎉 SUCCESS! Real MultiTalk inference is working!")
            print("   The models are loading and generating video properly.")
        elif 'Fallback' in str(video_info.get('processing_note', '')):
            print("\n⚠️  Using fallback implementation")
            print("   This means MultiTalk models aren't loading properly.")
            print("   Check worker logs for model loading errors.")
        else:
            print("\n❓ Unable to determine which implementation was used")

if __name__ == "__main__":
    test_multitalk_inference()