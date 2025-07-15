#!/usr/bin/env python3
"""
Create a test file in S3 to verify access
"""

import runpod
import os
import base64
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_s3_write():
    """Test S3 write access"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🧪 Testing S3 Write Access")
    print("=" * 60)
    
    # Create a simple test audio
    test_audio = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="  # Minimal WAV
    
    # Test 1: Generate video and save to S3
    print("\n1. Testing S3 write by generating video...")
    
    job_input = {
        "action": "generate",
        "audio": test_audio,
        "duration": 1.0,
        "output_format": "s3",
        "s3_output_key": f"test/multitalk_test_{int(time.time())}.mp4"
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"Job submitted: {job.job_id}")
        
        result = job.output(timeout=60)
        status = job.status()
        
        if status == "COMPLETED":
            print("✅ S3 write successful!")
            if isinstance(result, dict) and 'video' in result:
                s3_url = result['video']
                print(f"📹 Video saved to: {s3_url}")
                
                # Now try to read it back
                print("\n2. Testing S3 read of generated file...")
                
                # Extract the key from the S3 URL
                if s3_url.startswith('s3://'):
                    bucket_and_key = s3_url[5:]  # Remove 's3://'
                    parts = bucket_and_key.split('/', 1)
                    if len(parts) == 2:
                        bucket, key = parts
                        print(f"Bucket: {bucket}")
                        print(f"Key: {key}")
                        
                        # Try to use this as input
                        test_job = endpoint.run({
                            "action": "generate",
                            "audio": test_audio,  # Use minimal audio
                            "reference_image": s3_url,  # Try using the S3 file
                            "duration": 1.0
                        })
                        
                        test_result = test_job.output(timeout=30)
                        if test_job.status() == "COMPLETED":
                            print("✅ S3 read successful!")
                        else:
                            print(f"❌ S3 read failed: {test_result}")
                            
        else:
            print(f"❌ S3 write failed: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: List bucket contents (if possible)
    print("\n3. Checking S3 bucket access patterns...")
    
    # Try different key patterns
    test_patterns = [
        "1.wav",
        "1.WAV",
        "/1.wav",
        "audio/1.wav",
        "input/1.wav",
        "test/1.wav",
        "wav/1.wav",
        "*.wav"  # Wildcard
    ]
    
    print("\nPossible S3 paths to check:")
    for pattern in test_patterns:
        print(f"  • s3://760572149-framepack/{pattern}")
    
    print("\n📝 Recommendations:")
    print("1. Check the exact file path in your S3 bucket")
    print("2. Verify the file is in the root of the bucket or specify the full path")
    print("3. Check S3 bucket permissions allow GetObject")
    print("4. Ensure the bucket is in eu-west-1 region (as configured)")
    print("\nYou can list your bucket contents with:")
    print("  aws s3 ls s3://760572149-framepack/")
    print("\nOr check a specific file:")
    print("  aws s3 ls s3://760572149-framepack/1.wav")

if __name__ == "__main__":
    test_s3_write()