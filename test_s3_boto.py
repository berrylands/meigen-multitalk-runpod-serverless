#!/usr/bin/env python3
"""
Test S3 access with boto3 to diagnose RunPod issue
"""

import boto3
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()

# Get credentials
aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_region = os.environ.get("AWS_REGION", "us-east-1")

print("🔍 Testing S3 Access")
print("=" * 60)
print(f"Region from env: {aws_region}")
print(f"Bucket: 760572149-framepack")
print(f"File: 1.wav")

# Test with different region configurations
regions_to_test = [aws_region, 'eu-west-1', 'us-east-1']

for region in regions_to_test:
    print(f"\n📍 Testing with region: {region}")
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )
        
        # Try to download
        response = s3_client.get_object(Bucket='760572149-framepack', Key='1.wav')
        data = response['Body'].read()
        print(f"   ✅ SUCCESS! Downloaded {len(data)} bytes")
        print(f"   → RunPod should use AWS_REGION={region}")
        break
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"   ❌ Failed: {error_code}")
        if error_code == 'NoSuchKey':
            print("   → File not found (but bucket accessible)")
        elif error_code == 'AccessDenied':
            print("   → Access denied")
        elif error_code == 'PermanentRedirect':
            print("   → Wrong region!")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)