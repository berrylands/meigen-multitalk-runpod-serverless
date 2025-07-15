#!/usr/bin/env python3
"""
Test if RunPod has the correct AWS_REGION set
"""

import runpod
import os
from dotenv import load_dotenv

load_dotenv()

def test_region():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 Checking RunPod AWS Configuration")
    print("=" * 60)
    
    # Health check to see environment
    job = endpoint.run({"health_check": True})
    result = job.output(timeout=30)
    
    if result and 's3_integration' in result:
        s3_info = result['s3_integration']
        print(f"S3 Enabled: {s3_info.get('enabled', False)}")
        print(f"Default Bucket: {s3_info.get('default_bucket', 'Not set')}")
        
        # The issue might be here - check if region is shown
        if 'region' in s3_info:
            print(f"Region: {s3_info.get('region', 'Not shown')}")
    
    # Test with explicit S3 URL to force region
    print("\n" + "=" * 60)
    print("\nTESTING: Your bucket is in eu-west-1")
    print("RunPod might be defaulting to us-east-1")
    print("\nTry setting this RunPod secret:")
    print("AWS_REGION = eu-west-1")
    print("\nOr use full S3 URLs with region:")
    print("s3://760572149-framepack.s3.eu-west-1.amazonaws.com/1.wav")

if __name__ == "__main__":
    test_region()