#!/usr/bin/env python3
"""
Debug S3 credential issues in RunPod
"""

import os
import runpod
import base64

def create_debug_job():
    """Create a job to debug S3 credentials"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 Debugging S3 Credentials in RunPod")
    print("=" * 50)
    
    # Create a debug job that prints environment variables
    debug_code = '''
import os
import json

# Check environment variables
env_vars = {
    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", "NOT_SET"),
    "AWS_SECRET_ACCESS_KEY": "SET" if os.environ.get("AWS_SECRET_ACCESS_KEY") else "NOT_SET",
    "AWS_REGION": os.environ.get("AWS_REGION", "NOT_SET"),
    "AWS_S3_BUCKET_NAME": os.environ.get("AWS_S3_BUCKET_NAME", "NOT_SET"),
    "BUCKET_ENDPOINT_URL": os.environ.get("BUCKET_ENDPOINT_URL", "NOT_SET"),
}

# Check if values are empty strings
for key, value in env_vars.items():
    if key != "AWS_SECRET_ACCESS_KEY" and value == "":
        env_vars[key] = "EMPTY_STRING"

print(json.dumps({
    "env_vars": env_vars,
    "s3_should_work": env_vars["AWS_ACCESS_KEY_ID"] not in ["NOT_SET", "EMPTY_STRING"] 
                      and env_vars["AWS_SECRET_ACCESS_KEY"] == "SET"
}, indent=2))
'''
    
    try:
        # First try health check
        print("\n1. Testing health endpoint...")
        job = endpoint.run({"action": "health"})
        result = job.output(timeout=30)
        
        if result:
            print(f"Health check result:")
            print(f"  S3 Available: {result.get('s3_available', 'N/A')}")
            print(f"  S3 Status: {result.get('s3_status', {})}")
        
        # Now check environment
        print("\n2. Checking environment variables...")
        print("\nIn RunPod, ensure these secrets are configured:")
        print("- AWS_ACCESS_KEY_ID (should have a value)")
        print("- AWS_SECRET_ACCESS_KEY (should have a value)")
        print("- AWS_REGION (e.g., 'us-east-1' or 'eu-west-1')")
        print("- AWS_S3_BUCKET_NAME (your bucket name)")
        print("- BUCKET_ENDPOINT_URL (leave empty for standard AWS S3)")
        
        print("\n3. Common issues:")
        print("- Empty string values: Make sure values aren't just empty strings")
        print("- BUCKET_ENDPOINT_URL: Should be empty for AWS S3, only set for S3-compatible services")
        print("- Typos in secret names: Double-check exact spelling")
        
        print("\n4. Your S3 URL:")
        print("s3://760572149-framepack/1.wav")
        print("This suggests your bucket name is: 760572149-framepack")
        print("Make sure AWS_S3_BUCKET_NAME is set to: 760572149-framepack")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_debug_job()