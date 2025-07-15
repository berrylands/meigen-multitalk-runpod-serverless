#!/usr/bin/env python3
"""
Minimal test to check S3 access in RunPod
"""

import runpod
import os
from dotenv import load_dotenv

load_dotenv()

# Create a minimal test job that just tries to list S3 buckets
test_code = '''
import boto3
import os

# Get AWS credentials from RunPod environment
access_key = os.environ.get('AWS_ACCESS_KEY_ID', 'NOT_SET')
secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY', 'NOT_SET')
region = os.environ.get('AWS_REGION', 'NOT_SET')
bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', 'NOT_SET')

print(f"AWS_ACCESS_KEY_ID: {access_key[:10]}..." if access_key != 'NOT_SET' else "NOT_SET")
print(f"AWS_REGION: {region}")
print(f"AWS_S3_BUCKET_NAME: {bucket_name}")

if access_key != 'NOT_SET' and secret_key != 'NOT_SET':
    try:
        # Create client
        s3 = boto3.client('s3', 
                         aws_access_key_id=access_key,
                         aws_secret_access_key=secret_key,
                         region_name=region if region != 'NOT_SET' else 'us-east-1')
        
        # Try to list buckets
        buckets = s3.list_buckets()
        print(f"Can list buckets: {len(buckets['Buckets'])}")
        
        # Try to access the specific file
        try:
            obj = s3.get_object(Bucket='760572149-framepack', Key='1.wav')
            data = obj['Body'].read()
            print(f"✅ Downloaded 1.wav: {len(data)} bytes")
        except Exception as e:
            print(f"❌ Failed to download 1.wav: {e}")
            
    except Exception as e:
        print(f"Failed to create S3 client: {e}")
else:
    print("AWS credentials not found in environment")
'''

def test_s3_access():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 Testing Raw S3 Access in RunPod")
    print("=" * 60)
    
    # Send a job that just tests S3
    job = endpoint.run({
        "action": "debug",
        "code": test_code
    })
    
    result = job.output(timeout=60)
    print(result)

if __name__ == "__main__":
    test_s3_access()