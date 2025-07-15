#!/usr/bin/env python3
"""
Test with filename only (no bucket prefix)
"""

import runpod
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_filename_only():
    """Test with just the filename"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🎯 Testing S3 with filename only")
    print("=" * 60)
    print("Since the default bucket is already configured as '760572149-framepack'")
    print("we should use just the filename without the s3:// prefix")
    print("=" * 60)
    
    # Test with just filename
    test_cases = [
        ("Filename only", "1.wav"),
        ("Filename in test folder", "test/1.wav"),
        ("Another file", "2.wav"),
    ]
    
    for test_name, audio_path in test_cases:
        print(f"\n📌 {test_name}: {audio_path}")
        
        job_input = {
            "action": "generate",
            "audio": audio_path,
            "duration": 3.0,
            "width": 480,
            "height": 480
        }
        
        try:
            job = endpoint.run(job_input)
            print(f"Job ID: {job.job_id}")
            
            # Quick check
            time.sleep(5)
            status = job.status()
            
            if status == "COMPLETED":
                print("✅ SUCCESS! This format works!")
                result = job.output()
                if isinstance(result, dict) and 'video' in result:
                    print(f"📹 Video generated: {len(result['video'])} chars")
            elif status == "FAILED":
                result = job.output()
                print(f"❌ Failed: {result}")
            else:
                print(f"⏳ Status: {status}")
                # Wait a bit more
                result = job.output(timeout=30)
                final_status = job.status()
                if final_status == "COMPLETED":
                    print("✅ SUCCESS after waiting!")
                else:
                    print(f"❌ Final status: {final_status}")
                    
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("\n📝 Summary:")
    print("\nFor S3 files, use one of these formats:")
    print("1. Just the filename: '1.wav' (for files in bucket root)")
    print("2. Relative path: 'test/1.wav' (for files in subdirectories)")
    print("\nDo NOT use:")
    print("❌ s3://bucket-name/filename")
    print("❌ Full S3 URLs")
    print("\nThe S3 handler uses the default bucket configured in RunPod")

if __name__ == "__main__":
    test_filename_only()