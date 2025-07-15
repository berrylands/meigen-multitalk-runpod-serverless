#!/usr/bin/env python3
"""
Test S3 with file in test folder
"""

import runpod
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_s3_test_folder():
    """Test S3 with file in test folder"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🎯 Testing S3 with file in test/ folder")
    print("=" * 60)
    
    job_input = {
        "action": "generate",
        "audio": "s3://760572149-framepack/test/1.wav",
        "duration": 5.0,
        "width": 480,
        "height": 480
    }
    
    print(f"Testing: {job_input['audio']}")
    
    job = endpoint.run(job_input)
    print(f"Job ID: {job.job_id}")
    
    # Wait for result
    start_time = time.time()
    while True:
        status = job.status()
        elapsed = time.time() - start_time
        
        if status in ["COMPLETED", "FAILED"]:
            break
            
        print(f"[{elapsed:.1f}s] Status: {status}")
        time.sleep(2)
        
        if elapsed > 120:
            print("Timeout")
            break
    
    result = job.output()
    
    if status == "COMPLETED":
        print("\n✅ SUCCESS! S3 is working!")
        print(f"The issue was permissions - RunPod can access s3://760572149-framepack/test/*")
        print(f"but not s3://760572149-framepack/*")
        
        if isinstance(result, dict) and 'video' in result:
            print(f"\n📹 Video generated: {len(result['video'])} chars")
    else:
        print(f"\n❌ Failed: {result}")
    
    print("\n📝 Solution:")
    print("Either:")
    print("1. Update IAM permissions to allow GetObject on the root bucket")
    print("2. Store your input files in the test/ folder")
    print("3. Use presigned URLs for files in the root")

if __name__ == "__main__":
    test_s3_test_folder()