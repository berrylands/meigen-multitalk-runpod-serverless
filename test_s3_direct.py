#!/usr/bin/env python3
"""
Test S3 access directly from RunPod
"""

import runpod
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_s3_access():
    """Test S3 access through RunPod"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 Testing S3 Access from RunPod")
    print("=" * 60)
    
    # Test 1: Check S3 configuration
    print("\n1. Checking S3 configuration...")
    job = endpoint.run({"action": "health"})
    result = job.output(timeout=30)
    
    if result:
        s3_info = result.get('s3_integration', {})
        print(f"S3 Available: {s3_info.get('available', False)}")
        print(f"S3 Enabled: {s3_info.get('enabled', False)}")
        print(f"Default Bucket: {s3_info.get('default_bucket', 'N/A')}")
    
    # Test 2: Try to list S3 bucket contents
    print("\n2. Testing S3 bucket access...")
    debug_job_input = {
        "action": "debug_s3",
        "operation": "list_bucket",
        "bucket": "760572149-framepack"
    }
    
    try:
        job = endpoint.run(debug_job_input)
        result = job.output(timeout=30)
        print("Debug result:", result)
    except Exception as e:
        print(f"Debug action not available: {e}")
    
    # Test 3: Try different S3 URL formats
    print("\n3. Testing different S3 URL formats...")
    
    test_urls = [
        "s3://760572149-framepack/1.wav",
        "s3://760572149-framepack/1.WAV",
        "s3://760572149-framepack/audio/1.wav",
        "s3://760572149-framepack/test/1.wav",
        "https://760572149-framepack.s3.eu-west-1.amazonaws.com/1.wav",
        "https://s3.eu-west-1.amazonaws.com/760572149-framepack/1.wav"
    ]
    
    for url in test_urls:
        print(f"\nTesting: {url}")
        
        job_input = {
            "action": "generate",
            "audio": url,
            "duration": 3.0
        }
        
        try:
            job = endpoint.run(job_input)
            result = job.output(timeout=30)
            
            if isinstance(result, dict) and 'error' in result:
                error = result['error']
                if "not found" in error.lower():
                    print(f"  ❌ File not found")
                elif "access denied" in error.lower():
                    print(f"  ❌ Access denied")
                else:
                    print(f"  ❌ Error: {error}")
            else:
                print(f"  ✅ Success! Use this URL: {url}")
                return url
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    print("\n📝 Troubleshooting:")
    print("1. The bucket name might be incorrect")
    print("2. The file might be in a subdirectory")
    print("3. There might be a permissions issue")
    print("4. The AWS region might be different")
    
    # Test 4: Check if it's a region issue
    print("\n4. Testing region-specific URLs...")
    regions = ['us-east-1', 'eu-west-1', 'eu-central-1', 'us-west-2']
    
    for region in regions:
        url = f"https://760572149-framepack.s3.{region}.amazonaws.com/1.wav"
        print(f"\nTesting region {region}: {url}")
        
        job_input = {
            "action": "generate",
            "audio": url,
            "duration": 3.0
        }
        
        try:
            job = endpoint.run(job_input)
            result = job.output(timeout=20)
            
            if isinstance(result, dict) and 'error' in result:
                if "not found" in result['error'].lower():
                    print(f"  ℹ️  Bucket might exist in {region} but file not found")
                else:
                    print(f"  ❌ {result['error']}")
            else:
                print(f"  ✅ Success! Bucket is in {region}")
                return url
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")

if __name__ == "__main__":
    test_s3_access()