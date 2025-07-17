#\!/usr/bin/env python3
"""Test V131 with NumPy/Numba fixes"""
import json
import time
import runpod
from datetime import datetime

# Initialize RunPod client
runpod.api_key = open('.env').read().strip().split('RUNPOD_API_KEY=')[1].split('\n')[0]
endpoint = runpod.Endpoint("zu0ik6c8yukyl6")

def test_v131():
    """Test V131 with fixed NumPy compatibility"""
    print("🧪 Testing V131 - Fixed NumPy/Numba Compatibility")
    print("=" * 60)
    print("✅ NumPy 1.26.4 (forced reinstall)")
    print("✅ Removed distvae dependency conflict")
    print("✅ PyTorch 2.1.0 + CUDA 11.8")
    print("=" * 60)
    
    # Test payload with S3 files
    test_input = {
        "input": {
            "audio_url": "s3://ai-models-datalake/tests/1.wav",
            "image_url": "s3://ai-models-datalake/tests/multi1.png",
            "turbo": True
        }
    }
    
    print(f"\n📤 Sending request at {datetime.now().strftime('%H:%M:%S')}")
    print(f"🎵 Audio: 1.wav")
    print(f"🖼️  Image: multi1.png")
    print(f"⚡ Turbo: True")
    
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
                print("\n✅ GENERATION COMPLETE\!")
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
                        
                        # Save full result for analysis
                        with open('v131_result.json', 'w') as f:
                            json.dump(result, f, indent=2)
                        print("💾 Full result saved to v131_result.json")
                else:
                    print(f"📊 Result: {result}")
                
                break
                
            elif status == "FAILED":
                print("\n❌ Job failed\!")
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
    print()
    
    success = test_v131()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ V131 TEST SUCCESSFUL!")
        print("🎉 NumPy/Numba compatibility fixed!")
    else:
        print("❌ V131 test failed - check error output")
    print("=" * 60)