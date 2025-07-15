#!/usr/bin/env python3
"""
Full S3 test with video generation
"""

import runpod
import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_s3_full():
    """Test full S3 workflow"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🎬 Full S3 MultiTalk Test")
    print("=" * 60)
    
    # Test with S3 audio
    print("\n1. Submitting job with S3 audio...")
    
    job_input = {
        "action": "generate",
        "audio": "s3://760572149-framepack/1.wav",
        "duration": 5.0,
        "width": 480,
        "height": 480,
        "fps": 30,
        "output_format": "base64"  # Get result as base64
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"✅ Job submitted: {job.job_id}")
        
        # Wait for result
        print("\n2. Processing...")
        start_time = time.time()
        last_status = None
        
        while True:
            status = job.status()
            elapsed = time.time() - start_time
            
            if status != last_status:
                print(f"[{elapsed:.1f}s] Status: {status}")
                last_status = status
            
            if status in ["COMPLETED", "FAILED"]:
                break
                
            if elapsed > 300:  # 5 minutes timeout
                print("❌ Timeout after 5 minutes")
                break
                
            time.sleep(2)
        
        # Get result
        result = job.output()
        elapsed = time.time() - start_time
        
        print(f"\n3. Job finished in {elapsed:.1f} seconds")
        
        if status == "COMPLETED":
            print("✅ Job completed successfully!")
            
            if isinstance(result, dict):
                # Check what's in the result
                print("\nResult keys:", list(result.keys()) if result else "None")
                
                if 'video' in result:
                    video_data = result['video']
                    if isinstance(video_data, str):
                        if video_data.startswith('data:'):
                            print(f"📹 Video generated (data URL): {len(video_data)} chars")
                        else:
                            print(f"📹 Video generated (base64): {len(video_data)} chars")
                        
                        # Save video
                        print("\n4. Saving video...")
                        try:
                            import base64
                            
                            # Remove data URL prefix if present
                            if video_data.startswith('data:'):
                                video_data = video_data.split(',')[1]
                            
                            video_bytes = base64.b64decode(video_data)
                            
                            filename = f"multitalk_output_{int(time.time())}.mp4"
                            with open(filename, 'wb') as f:
                                f.write(video_bytes)
                            
                            print(f"✅ Video saved as: {filename}")
                            print(f"   Size: {len(video_bytes):,} bytes")
                            
                        except Exception as e:
                            print(f"❌ Failed to save video: {e}")
                    else:
                        print(f"📹 Video data type: {type(video_data)}")
                
                # Show other result data
                for key, value in result.items():
                    if key != 'video':
                        print(f"{key}: {value}")
                        
            else:
                print(f"Result type: {type(result)}")
                print(f"Result: {result}")
                
        else:
            print(f"\n❌ Job failed with status: {status}")
            print(f"Error details: {json.dumps(result, indent=2)}")
            
            # Common error analysis
            if isinstance(result, dict) and 'error' in result:
                error = result['error']
                if "PyTorch not available" in error:
                    print("\n⚠️  PyTorch is not installed in the container")
                elif "CUDA" in error:
                    print("\n⚠️  GPU/CUDA issue")
                elif "memory" in error.lower():
                    print("\n⚠️  Out of memory")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with S3 output
    print("\n" + "=" * 60)
    print("\n5. Testing S3 output...")
    
    job_input = {
        "action": "generate",
        "audio": "s3://760572149-framepack/1.wav",
        "duration": 3.0,
        "output_format": "s3",
        "s3_output_key": f"outputs/multitalk_{int(time.time())}.mp4"
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"✅ Job submitted: {job.job_id}")
        
        result = job.output(timeout=120)
        
        if job.status() == "COMPLETED":
            print("✅ Job completed!")
            if isinstance(result, dict) and 'video' in result:
                print(f"📹 Video uploaded to S3: {result['video']}")
            else:
                print(f"Result: {result}")
        else:
            print(f"❌ Job failed: {result}")
            
    except Exception as e:
        print(f"❌ S3 output test error: {e}")

if __name__ == "__main__":
    test_s3_full()