#!/usr/bin/env python3
"""
Test S3 permissions issue
"""

import runpod
import os
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_permissions():
    """Test S3 permissions"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 S3 Permissions Test")
    print("=" * 60)
    
    # First, let's upload the local file to a test location
    print("\n1. Uploading local 1.wav to S3 test location...")
    
    try:
        # Read the local file
        with open("/tmp/test-1.wav", "rb") as f:
            audio_data = f.read()
        
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        print(f"✅ Loaded local file: {len(audio_data)} bytes")
        
        # Generate a video using this audio and save to S3
        job_input = {
            "action": "generate",
            "audio": audio_base64,
            "duration": 3.0,
            "output_format": "s3",
            "s3_output_key": "test/copy-of-1.wav.mp4"
        }
        
        job = endpoint.run(job_input)
        result = job.output(timeout=60)
        
        if job.status() == "COMPLETED":
            print("✅ Successfully processed and saved to S3")
            s3_url = result.get('video', '')
            print(f"Output: {s3_url}")
        else:
            print(f"❌ Failed: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Now test with presigned URL
    print("\n2. Testing with presigned URL...")
    print("\nTo generate a presigned URL for your file:")
    print("aws s3 presign s3://760572149-framepack/1.wav --expires-in 3600")
    
    # Test explicit bucket/key format
    print("\n3. Testing explicit bucket/key specification...")
    
    test_configs = [
        {
            "action": "generate", 
            "audio": "s3://760572149-framepack/1.wav",
            "duration": 3.0
        },
        {
            "action": "generate",
            "audio": {
                "type": "s3",
                "bucket": "760572149-framepack",
                "key": "1.wav"
            },
            "duration": 3.0
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}: {config}")
        try:
            job = endpoint.run(config)
            result = job.output(timeout=30)
            status = job.status()
            
            if status == "COMPLETED":
                print("✅ Success!")
            else:
                print(f"❌ Failed: {result}")
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("\n📝 Analysis:")
    print("\nThe issue appears to be that the RunPod S3 credentials")
    print("don't have GetObject permission for the root of the bucket.")
    print("\nPossible solutions:")
    
    print("\n1. **Update IAM permissions** for the RunPod user to include:")
    print("   {")
    print('     "Effect": "Allow",')
    print('     "Action": ["s3:GetObject"],')
    print('     "Resource": "arn:aws:s3:::760572149-framepack/*"')
    print("   }")
    
    print("\n2. **Use presigned URLs** which include temporary credentials:")
    print("   aws s3 presign s3://760572149-framepack/1.wav")
    
    print("\n3. **Copy files to the test/ folder** where write permissions exist:")
    print("   aws s3 cp s3://760572149-framepack/1.wav s3://760572149-framepack/test/1.wav")
    
    print("\n4. **Upload files through the API** as base64")

if __name__ == "__main__":
    test_permissions()