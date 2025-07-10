#!/usr/bin/env python3
"""
Test video generation directly using the existing handler
"""

import os
import time
import runpod
import base64
import numpy as np
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

def test_video_generation():
    """Test video generation with the current handler."""
    
    print("Testing Video Generation Pipeline")
    print("=" * 60)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Create test audio data
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_signal = np.sin(2 * np.pi * 440 * t)  # 440Hz tone
    audio_int16 = (audio_signal * 32767).astype(np.int16)
    audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
    
    print(f"Created test audio: {duration}s, {sample_rate}Hz")
    print(f"Audio data size: {len(audio_b64)} characters (base64)")
    
    # Test video generation with current handler
    job_input = {
        "action": "generate",
        "audio": audio_b64,
        "duration": 3.0,
        "fps": 24,
        "width": 480,
        "height": 480
    }
    
    print(f"\nSubmitting video generation job...")
    print(f"Parameters: {duration}s, 24fps, 480x480")
    
    try:
        job = endpoint.run(job_input)
        print(f"Job ID: {job.job_id}")
        
        start_time = time.time()
        last_status = None
        
        while True:
            status = job.status()
            elapsed = time.time() - start_time
            
            if status != last_status:
                print(f"[{elapsed:.1f}s] Status: {status}")
                last_status = status
            
            if status not in ["IN_QUEUE", "IN_PROGRESS"]:
                break
                
            if elapsed > 300:  # 5 minute timeout
                print("Job taking longer than expected...")
                break
                
            time.sleep(5)
        
        print(f"\nFinal status: {job.status()}")
        
        if job.status() == "COMPLETED":
            result = job.output()
            
            if result and result.get("success"):
                print(f"✅ VIDEO GENERATION SUCCESS!")
                print(f"   Processing time: {result.get('processing_time', 'N/A')}")
                print(f"   Models used: {', '.join(result.get('models_used', []))}")
                
                if 'video' in result:
                    video_b64 = result['video']
                    video_data = base64.b64decode(video_b64)
                    
                    # Save the video
                    output_file = "multitalk_test_output.mp4"
                    with open(output_file, "wb") as f:
                        f.write(video_data)
                    
                    print(f"   Video saved: {output_file} ({len(video_data)} bytes)")
                    
                    return True
                else:
                    print(f"   No video data in response")
                    return False
            else:
                print(f"❌ Video generation failed")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                print(f"   Full response: {result}")
                return False
        else:
            print(f"❌ Job failed with status: {job.status()}")
            error_output = job.output()
            print(f"   Error details: {error_output}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during video generation: {e}")
        return False

def test_different_approaches():
    """Try different approaches to trigger video generation."""
    
    print(f"\n" + "=" * 60)
    print("Testing Alternative Video Generation Approaches")
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Approach 1: Simple audio input
    print(f"\n1. Testing with 'audio' key directly...")
    try:
        job = endpoint.run({
            "audio": "dGVzdCBhdWRpbyBkYXRh",  # "test audio data" in base64
            "duration": 2.0
        })
        
        wait_time = 0
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"] and wait_time < 60:
            time.sleep(3)
            wait_time += 3
        
        print(f"   Result: {job.status()}")
        if job.status() == "COMPLETED":
            output = job.output()
            print(f"   Response: {output}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Approach 2: Check what actions are supported
    print(f"\n2. Testing handler capabilities...")
    try:
        job = endpoint.run({})  # Empty request to see default response
        
        wait_time = 0
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"] and wait_time < 30:
            time.sleep(2)
            wait_time += 2
        
        if job.status() == "COMPLETED":
            result = job.output()
            print(f"   Handler info: {result}")
            
            if 'supported_actions' in result:
                print(f"   Supported actions: {result['supported_actions']}")
            
    except Exception as e:
        print(f"   Error: {e}")

def main():
    print("MultiTalk Video Generation Testing")
    print("Testing with current handler and available models")
    
    # Test video generation
    success = test_video_generation()
    
    if not success:
        print(f"\nPrimary test failed - trying alternative approaches...")
        test_different_approaches()
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print("✅ MultiTalk video generation is working!")
        print("✅ End-to-end pipeline operational")
        print("✅ Ready for production use")
    else:
        print(f"\n⚠️  Testing revealed issues to address")
        print("Need to iterate on the implementation")
    
    return success

if __name__ == "__main__":
    main()