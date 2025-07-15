#!/usr/bin/env python3
"""
Debug S3 path issues
"""

import os
import boto3
from dotenv import load_dotenv

load_dotenv()

# Get credentials
aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
aws_region = os.environ.get("AWS_REGION", "us-east-1")
bucket_name = "760572149-framepack"

if not aws_access_key or not aws_secret_key:
    print("❌ AWS credentials not set")
    exit(1)

# Create S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name=aws_region
)

print(f"🔍 Debugging S3 paths in bucket: {bucket_name}")
print("=" * 60)

# List files in the bucket
try:
    print("\n1. Listing files in bucket root:")
    response = s3_client.list_objects_v2(
        Bucket=bucket_name,
        MaxKeys=20
    )
    
    if 'Contents' in response:
        for obj in response['Contents']:
            print(f"   - {obj['Key']} ({obj['Size']} bytes)")
    else:
        print("   No files found")
        
    # Check specific file
    print("\n2. Checking for '1.wav':")
    try:
        obj = s3_client.head_object(Bucket=bucket_name, Key='1.wav')
        print(f"   ✅ Found: 1.wav ({obj['ContentLength']} bytes)")
        print(f"   Last modified: {obj['LastModified']}")
    except:
        print("   ❌ Not found at root level")
        
    # Try to download
    print("\n3. Attempting to download '1.wav':")
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key='1.wav')
        data = response['Body'].read()
        print(f"   ✅ Downloaded successfully: {len(data)} bytes")
        
        # Check if it's a valid WAV file
        if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
            print("   ✅ Valid WAV file detected")
        else:
            print(f"   ⚠️  File header: {data[:12].hex()}")
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("\n💡 If the file exists but RunPod can't find it:")
print("1. Check if RunPod's AWS_S3_BUCKET_NAME matches this bucket")
print("2. Verify the file is in the root of the bucket (not in a subfolder)")
print("3. Check S3 permissions for the RunPod IAM user")