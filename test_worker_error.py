#!/usr/bin/env python3
"""
Test script to help diagnose worker exit code 1
"""

import runpod
import os
import sys
import time

def test_deployment():
    """Test the deployment and diagnose issues"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint_id = "kkx3cfy484jszl"
    
    print("🔍 Diagnosing Worker Exit Code 1")
    print("=" * 50)
    print(f"Endpoint ID: {endpoint_id}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        endpoint = runpod.Endpoint(endpoint_id)
        
        # First, try a simple health check
        print("\n1. Testing health check...")
        try:
            job = endpoint.run({
                "action": "health"
            })
            print(f"Job ID: {job.job_id}")
            print(f"Status: {job.status()}")
            
            # Wait for result
            result = job.output(timeout=30)
            print(f"Result: {result}")
            
            if result and "s3_available" in result:
                print(f"✓ S3 Available: {result.get('s3_available')}")
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            
            # Try to get job details
            if hasattr(e, 'response'):
                print(f"Response: {e.response}")
        
        # Check recent jobs for error messages
        print("\n2. Checking recent job errors...")
        # Note: This is a workaround since RunPod SDK doesn't have direct job history access
        print("Please check your RunPod dashboard for worker logs")
        print("Look for:")
        print("- Import errors (missing modules)")
        print("- File not found errors")
        print("- Permission errors")
        print("- Environment variable issues")
        
        print("\n3. Common causes of worker exit code 1:")
        print("- Missing Python dependencies (boto3, etc.)")
        print("- Handler file not found at expected path")
        print("- Syntax errors in handler.py")
        print("- Import errors in s3_handler.py")
        print("- Missing environment variables")
        print("- Container startup script issues")
        
        print("\n4. Recommended actions:")
        print("a) Try the debug image first:")
        print("   Image: berrylands/multitalk-s3-fix:latest")
        print("   This has better error logging")
        
        print("\nb) Check worker logs in RunPod dashboard:")
        print("   https://www.runpod.io/console/serverless")
        print(f"   Click on endpoint {endpoint_id}")
        print("   Go to 'Workers' tab")
        print("   Check logs for the failed worker")
        
        print("\nc) Try running with debug handler:")
        print("   Update CMD in Dockerfile to:")
        print("   CMD [\"python\", \"-u\", \"/app/debug_handler.py\"]")
        
    except Exception as e:
        print(f"❌ Error connecting to endpoint: {e}")

if __name__ == "__main__":
    test_deployment()