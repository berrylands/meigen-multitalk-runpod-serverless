#!/usr/bin/env python3
"""
Test the S3 debug handler to diagnose credential issues
"""

import runpod
import os
import json
import time

def test_s3_debug():
    """Test S3 debug functionality"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        print("Set it with: export RUNPOD_API_KEY='your-api-key'")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 S3 Debug Test")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Debug Image: berrylands/multitalk-s3-debug:latest")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Health Check (detailed S3 status)...")
    try:
        job = endpoint.run({"action": "health"})
        result = job.output(timeout=30)
        
        if result:
            print(json.dumps(result, indent=2))
            
            # Analyze results
            print("\n📊 Analysis:")
            
            # Check S3 availability
            if result.get("s3_available"):
                print("✅ S3 handler loaded successfully")
            else:
                print("❌ S3 handler failed to load")
                if result.get("s3_import_error"):
                    print(f"   Error: {result['s3_import_error']}")
            
            # Check environment
            env = result.get("environment", {})
            print("\n🔑 Environment Variables:")
            for key, value in env.items():
                if "EMPTY_STRING" in key:
                    print(f"❌ {key.replace('_IS_EMPTY_STRING', '')} is an empty string!")
                elif value == "NOT_SET":
                    print(f"❌ {key} is not set")
                elif value == "SET":
                    print(f"✅ {key} is set")
                else:
                    print(f"ℹ️  {key}: {value}")
            
            # Check S3 handler status
            s3h = result.get("s3_handler", {})
            if s3h:
                print("\n🪣 S3 Handler Status:")
                print(f"   Enabled: {s3h.get('enabled')}")
                print(f"   Default Bucket: {s3h.get('default_bucket')}")
                print(f"   Has Client: {s3h.get('has_client')}")
            
            # Check S3 access
            s3a = result.get("s3_access", {})
            if s3a:
                print("\n🔐 S3 Access Test:")
                print(f"   Can List Bucket: {s3a.get('can_list_bucket')}")
                print(f"   Can Write: {s3a.get('can_write')}")
                if s3a.get("error"):
                    print(f"   Error: {s3a['error']}")
                    
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test 2: Debug environment
    print("\n\n2. Debug Environment (raw values)...")
    try:
        job = endpoint.run({"action": "debug_env"})
        result = job.output(timeout=30)
        
        if result:
            raw_env = result.get("raw_env", {})
            if raw_env:
                print("\n🔍 Raw Environment Variables:")
                for k, v in raw_env.items():
                    if v == "":
                        print(f"   {k}: [EMPTY STRING]")
                    else:
                        print(f"   {k}: {v}")
                        
    except Exception as e:
        print(f"❌ Debug env failed: {e}")
    
    # Test 3: Test S3 URL parsing
    print("\n\n3. Test S3 URL Parsing...")
    try:
        job = endpoint.run({
            "action": "test_s3",
            "test_url": "s3://760572149-framepack/1.wav"
        })
        result = job.output(timeout=30)
        
        if result:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"❌ S3 test failed: {e}")
    
    # Recommendations
    print("\n\n📋 Recommendations:")
    print("1. Update RunPod endpoint to use: berrylands/multitalk-s3-debug:latest")
    print("2. Check the output above for missing or empty environment variables")
    print("3. In RunPod, ensure these secrets are set:")
    print("   - AWS_ACCESS_KEY_ID (your access key)")
    print("   - AWS_SECRET_ACCESS_KEY (your secret key)")
    print("   - AWS_REGION (e.g., 'us-east-1')")
    print("   - AWS_S3_BUCKET_NAME (e.g., '760572149-framepack')")
    print("4. Do NOT set BUCKET_ENDPOINT_URL unless using non-AWS S3")
    print("\n5. If variables show as [EMPTY STRING], delete and recreate them in RunPod")

if __name__ == "__main__":
    test_s3_debug()