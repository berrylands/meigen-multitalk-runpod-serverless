#!/usr/bin/env python3
"""
Test different S3 URL formats
"""

import runpod
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_url_formats():
    """Test different S3 URL formats"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 Testing S3 URL Formats")
    print("=" * 60)
    print("When you 'Copy URL' from S3 console, it typically gives you an HTTPS URL")
    print("that looks like one of these formats:")
    print("=" * 60)
    
    # Test different URL formats
    test_urls = [
        # S3 URI format (Copy S3 URI)
        ("S3 URI format", "s3://760572149-framepack/1.wav"),
        
        # HTTPS Virtual-hosted-style (most common from Copy URL)
        ("HTTPS Virtual-hosted (eu-west-1)", "https://760572149-framepack.s3.eu-west-1.amazonaws.com/1.wav"),
        ("HTTPS Virtual-hosted (no region)", "https://760572149-framepack.s3.amazonaws.com/1.wav"),
        
        # HTTPS Path-style
        ("HTTPS Path-style (eu-west-1)", "https://s3.eu-west-1.amazonaws.com/760572149-framepack/1.wav"),
        ("HTTPS Path-style (no region)", "https://s3.amazonaws.com/760572149-framepack/1.wav"),
        
        # With different regions
        ("HTTPS us-east-1", "https://760572149-framepack.s3.us-east-1.amazonaws.com/1.wav"),
        ("HTTPS us-west-2", "https://760572149-framepack.s3.us-west-2.amazonaws.com/1.wav"),
        
        # Presigned URL pattern (if it includes query parameters)
        ("Presigned URL example", "https://760572149-framepack.s3.eu-west-1.amazonaws.com/1.wav?X-Amz-Algorithm=..."),
    ]
    
    successful_url = None
    
    for url_type, url in test_urls:
        print(f"\n📌 Testing {url_type}:")
        print(f"   URL: {url}")
        
        if "X-Amz-Algorithm" in url:
            print("   ⚠️  If using presigned URL, paste the full URL with all parameters")
            continue
        
        job_input = {
            "action": "generate",
            "audio": url,
            "duration": 3.0
        }
        
        try:
            job = endpoint.run(job_input)
            
            # Quick check
            time.sleep(3)
            status = job.status()
            
            if status == "COMPLETED":
                print("   ✅ SUCCESS! This format works!")
                successful_url = url
                break
            elif status == "FAILED":
                result = job.output()
                if isinstance(result, dict) and 'error' in result:
                    error = result.get('error', 'Unknown error')
                    if "not found" in error.lower():
                        print("   ❌ File not found with this URL format")
                    else:
                        print(f"   ❌ Error: {error}")
                else:
                    print("   ❌ Failed")
            else:
                print(f"   ⏳ Status: {status}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print("\n" + "=" * 60)
    
    if successful_url:
        print(f"\n✅ Use this URL format: {successful_url}")
    else:
        print("\n📝 Troubleshooting Steps:")
        print("\n1. In S3 Console, click on your file '1.wav'")
        print("2. Click 'Copy URL' (not 'Copy S3 URI')")
        print("3. The URL should look like:")
        print("   https://YOUR-BUCKET.s3.REGION.amazonaws.com/1.wav")
        print("\n4. If the file is public, this URL should work")
        print("5. If the file is private, you might need:")
        print("   - Presigned URL (includes authentication parameters)")
        print("   - Or ensure your RunPod S3 credentials have GetObject permission")
        
        print("\n🔑 Permission Check:")
        print("Your S3 credentials need at least these permissions:")
        print("- s3:GetObject on arn:aws:s3:::760572149-framepack/*")
        print("- s3:PutObject on arn:aws:s3:::760572149-framepack/* (for output)")
        
        print("\n🌍 Region Check:")
        print("Your RunPod is configured for region: eu-west-1")
        print("Make sure your bucket is in the same region")

if __name__ == "__main__":
    test_url_formats()