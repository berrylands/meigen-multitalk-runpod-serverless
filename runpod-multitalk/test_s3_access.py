#!/usr/bin/env python3
"""Test S3 access directly"""
import os
import boto3
from botocore.exceptions import ClientError

# Get environment
bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
region = os.environ.get('AWS_REGION', 'eu-west-1')

print(f"Testing S3 access to bucket: {bucket_name}")
print(f"Region: {region}")

# Create client
s3 = boto3.client('s3', region_name=region)

# Test files
test_files = ['1.wav', 'multi1.png']

for filename in test_files:
    try:
        response = s3.head_object(Bucket=bucket_name, Key=filename)
        print(f"✅ {filename}: {response['ContentLength']} bytes, LastModified: {response['LastModified']}")
    except ClientError as e:
        print(f"❌ {filename}: {e.response['Error']['Code']} - {e.response['Error']['Message']}")

# List first 5 files
print("\nFirst 5 files in bucket:")
try:
    response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=5)
    if 'Contents' in response:
        for obj in response['Contents']:
            print(f"  - {obj['Key']} ({obj['Size']} bytes)")
    else:
        print("  No objects found")
except ClientError as e:
    print(f"  Error listing bucket: {e}")