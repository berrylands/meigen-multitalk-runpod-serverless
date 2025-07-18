#!/usr/bin/env python3
"""Test V136 Reference Implementation"""
import json
import time
import runpod
from datetime import datetime

# Initialize RunPod client
runpod.api_key = open('.env').read().strip().split('RUNPOD_API_KEY=')[1].split('\n')[0]
endpoint = runpod.Endpoint("zu0ik6c8yukyl6")

def test_v136():
    """Test V136 with reference implementation"""
    print("🧪 Testing V136 - Reference Implementation")
    print("=" * 60)
    print("✅ Using cog-MultiTalk reference (no optimum-quanto)")
    print("✅ Direct Python API calls")
    print("✅ All S3/RunPod functionality preserved")
    print("✅ Based on PyTorch 2.1.0 CUDA 11.8")
    print("=" * 60)
    
    # Test payload
    test_input = {
        "input": {
            "audio_s3_key": "1.wav",
            "image_s3_key": "multi1.png",
            "turbo": True,
            "sampling_steps": 40,
            "output_format": "s3"
        }
    }
    
    print(f"\n📤 Sending request at {datetime.now().strftime('%H:%M:%S')}")
    print(f"🎵 Audio: {test_input['input']['audio_s3_key']}")
    print(f"🖼️  Image: {test_input['input']['image_s3_key']}")
    print(f"⚡ Turbo: {test_input['input']['turbo']}")
    
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
                            print("\n🔍 Traceback preview:")
                            # Show first 500 chars of traceback
                            tb = result['traceback']
                            print(tb[:500] + "..." if len(tb) > 500 else tb)
                    else:
                        print(f"\n✅ SUCCESS! Result keys: {list(result.keys())}")
                        
                        if 'video_url' in result:
                            print(f"🎬 Video URL: {result['video_url']}")
                        if 'video_s3_key' in result:
                            print(f"📦 Video S3 Key: {result['video_s3_key']}")
                        if 'duration' in result:
                            print(f"⏱️  Duration: {result['duration']:.2f}s")
                        if 'frames' in result:
                            print(f"🎞️  Frames: {result['frames']}")
                        if 'size_mb' in result:
                            print(f"💾 Size: {result['size_mb']:.2f} MB")
                        
                        # Save result
                        with open('v136_result.json', 'w') as f:
                            json.dump(result, f, indent=2)
                        print("💾 Full result saved to v136_result.json")
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
    print("🚀 V136 Test - Reference Implementation")
    print("🐳 Image: berrylands/multitalk-runpod:v136-reference")
    print("🔧 Endpoint: zu0ik6c8yukyl6")
    print("📂 S3 Bucket: ai-models-datalake/tests")
    print()
    
    success = test_v136()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ V136 TEST SUCCESSFUL!")
        print("🎉 Reference implementation working!")
        print("🎥 Video generation complete!")
        print("🚀 No optimum-quanto needed!")
    else:
        print("❌ V136 test failed - check error output")
        print("📝 Review traceback for missing dependencies")
    print("=" * 60)