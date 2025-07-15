#!/usr/bin/env python3
"""
Check where the S3 files actually are
"""

import boto3
import os
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

print(f"🔍 Searching for files in bucket: {bucket_name}")
print("=" * 60)

# List all objects
try:
    print("Listing all objects in bucket:")
    paginator = s3_client.get_paginator('list_objects_v2')
    
    wav_files = []
    png_files = []
    jpg_files = []
    all_files = []
    
    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                all_files.append(key)
                
                if key.endswith('.wav'):
                    wav_files.append(key)
                elif key.endswith('.png'):
                    png_files.append(key)
                elif key.endswith('.jpg') or key.endswith('.jpeg'):
                    jpg_files.append(key)
    
    print(f"\nTotal files: {len(all_files)}")
    
    # Show audio files
    print(f"\n📢 WAV files found ({len(wav_files)}):")
    for f in wav_files[:10]:  # Show first 10
        print(f"   - {f}")
    if len(wav_files) > 10:
        print(f"   ... and {len(wav_files) - 10} more")
    
    # Show image files
    print(f"\n🖼️  PNG files found ({len(png_files)}):")
    for f in png_files[:10]:  # Show first 10
        print(f"   - {f}")
    if len(png_files) > 10:
        print(f"   ... and {len(png_files) - 10} more")
    
    print(f"\n📷 JPG files found ({len(jpg_files)}):")
    for f in jpg_files[:10]:  # Show first 10
        print(f"   - {f}")
    if len(jpg_files) > 10:
        print(f"   ... and {len(jpg_files) - 10} more")
    
    # Look for specific files
    print("\n🔎 Looking for specific files:")
    target_files = ['1.wav', 'multi1.png']
    
    for target in target_files:
        found = False
        for f in all_files:
            if f.endswith(target) or f == target:
                print(f"   ✅ Found '{target}' at: {f}")
                found = True
                break
        if not found:
            print(f"   ❌ '{target}' not found")
            # Try to find similar
            similar = [f for f in all_files if target.split('.')[0] in f]
            if similar:
                print(f"      Similar files: {similar[:3]}")
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("\n💡 If your files are in a subdirectory, you need to:")
print("1. Use the full path: 'subfolder/1.wav'")
print("2. Or move them to the bucket root")
print("3. Or update your job to use the correct path")