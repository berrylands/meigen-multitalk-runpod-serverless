#!/usr/bin/env python3
"""Test V131 with correct S3 format"""
import json
import time
import runpod
from datetime import datetime

# Initialize RunPod client
runpod.api_key = open('.env').read().strip().split('RUNPOD_API_KEY=')[1].split('\n')[0]
endpoint = runpod.Endpoint("zu0ik6c8yukyl6")

def test_v131():
    """Test V131 with fixed NumPy compatibility and correct S3 format"""
    print("🧪 Testing V131 - Fixed NumPy/Numba Compatibility")
    print("=" * 60)
    print("✅ NumPy 1.26.4 (forced reinstall)")
    print("✅ Removed distvae dependency conflict")
    print("✅ PyTorch 2.1.0 + CUDA 11.8")
    print("=" * 60)
    
    # Test payload with S3 files - using correct format from previous tests
    test_input = {
        "input": {
            "audio_s3_key": "1.wav",           # Just filename, no prefix
            "image_s3_key": "multi1.png",      # Just filename, no prefix
            "device": "cuda",
            "video_format": "mp4",
            "turbo_mode": True,
            "text_guide_scale": 5.0,
            "fps": 30
        }
    }
    
    print(f"\n📤 Sending request at {datetime.now().strftime('%H:%M:%S')}")
    print(f"🎵 Audio: {test_input['input']['audio_s3_key']}")
    print(f"🖼️  Image: {test_input['input']['image_s3_key']}")
    print(f"⚡ Turbo: {test_input['input']['turbo_mode']}")
    
    try:
        run = endpoint.run(test_input)
        print(f"✅ Request ID: {run.job_id}")
        
        # Poll for result
        print("\n⏳ Waiting for result...")
        start_time = time.time()
        last_status = None
        
        while True:
            status = run.status()
            if status != last_status:
                elapsed = int(time.time() - start_time)
                print(f"[{elapsed}s] Status: {status}")
                last_status = status
            
            if status == "COMPLETED":
                result = run.output()
                print("\n✅ GENERATION COMPLETE!")
                print(f"Total time: {int(time.time() - start_time)} seconds")
                
                if isinstance(result, dict):
                    if 'error' in result:
                        print(f"\n❌ ERROR: {result['error']}")
                        if 'traceback' in result:
                            print("\n🔍 Traceback:")
                            print(result['traceback'])
                    else:
                        print(f"\n📊 Result type: {type(result)}")
                        if 'video_url' in result:
                            print(f"🎬 Video URL: {result['video_url']}")
                        if 'video_base64' in result:
                            print(f"📦 Video base64 length: {len(result.get('video_base64', ''))}")
                        if 'job_id' in result:
                            print(f"📋 Job ID: {result['job_id']}")
                        
                        # Save full result for analysis
                        with open('v131_result.json', 'w') as f:
                            json.dump(result, f, indent=2)
                        print("💾 Full result saved to v131_result.json")
                else:
                    print(f"📊 Result: {result}")
                
                break
                
            elif status == "FAILED":
                print("\n❌ Job failed!")
                error = run.output()
                print(f"Error: {error}")
                break
                
            time.sleep(2)
            
            # Timeout after 5 minutes
            if time.time() - start_time > 300:
                print("\n⏰ Timeout after 5 minutes")
                run.cancel()
                break
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    return status == "COMPLETED" and 'error' not in str(run.output())

if __name__ == "__main__":
    print("🚀 V131 Test - NumPy/Numba Fix Verification")
    print("🐳 Image: berrylands/multitalk-runpod:v131")
    print("🔧 Template: joospbpdol (multitalk-v131-fixed)")
    print("📂 S3 Bucket: ai-models-datalake/tests")
    print()
    
    success = test_v131()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ V131 TEST SUCCESSFUL!")
        print("🎉 NumPy/Numba compatibility fixed!")
        print("🎥 Video generation working with S3 files!")
    else:
        print("❌ V131 test failed - check error output")
    print("=" * 60)