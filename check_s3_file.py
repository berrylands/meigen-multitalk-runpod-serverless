#!/usr/bin/env python3
"""
Check if S3 file exists and list bucket contents
"""

import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_s3_file():
    """Check S3 file and list bucket contents"""
    
    # Get credentials from environment
    access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    region = os.environ.get('AWS_REGION', 'eu-west-1')
    bucket_name = '760572149-framepack'
    
    if not access_key or not secret_key:
        print("❌ AWS credentials not found in environment")
        print("Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        return
    
    print("🔍 Checking S3 Bucket Contents")
    print("=" * 60)
    print(f"Bucket: {bucket_name}")
    print(f"Region: {region}")
    print(f"Looking for: 1.wav")
    print("=" * 60)
    
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # List objects in the bucket
        print("\n📂 Listing bucket contents:")
        
        try:
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                MaxKeys=100
            )
            
            if 'Contents' in response:
                print(f"\nFound {len(response['Contents'])} objects:\n")
                
                wav_files = []
                for obj in response['Contents']:
                    key = obj['Key']
                    size = obj['Size']
                    modified = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    print(f"  • {key} ({size:,} bytes) - {modified}")
                    
                    if key.endswith('.wav'):
                        wav_files.append(key)
                
                print(f"\n🎵 WAV files found: {len(wav_files)}")
                for wav in wav_files:
                    print(f"  • s3://{bucket_name}/{wav}")
                    
                # Check specifically for 1.wav
                if '1.wav' not in [obj['Key'] for obj in response['Contents']]:
                    print("\n⚠️  '1.wav' not found in bucket root")
                    print("\nPossible issues:")
                    print("1. File might be in a subdirectory")
                    print("2. File might have a different name")
                    print("3. File might not have been uploaded yet")
                    
            else:
                print("❌ Bucket is empty")
                
        except s3_client.exceptions.NoSuchBucket:
            print(f"❌ Bucket '{bucket_name}' does not exist")
            
        except Exception as e:
            print(f"❌ Error listing bucket: {e}")
            
            # Try to check if we have access
            print("\n🔍 Checking bucket access...")
            try:
                s3_client.head_bucket(Bucket=bucket_name)
                print(f"✅ Bucket '{bucket_name}' exists and is accessible")
            except Exception as e2:
                print(f"❌ Cannot access bucket: {e2}")
        
        # Check if file exists in any common locations
        print("\n🔍 Checking common file locations:")
        common_paths = [
            '1.wav',
            'audio/1.wav',
            'inputs/1.wav',
            'test/1.wav',
            'samples/1.wav'
        ]
        
        for path in common_paths:
            try:
                s3_client.head_object(Bucket=bucket_name, Key=path)
                print(f"✅ Found at: s3://{bucket_name}/{path}")
                return f"s3://{bucket_name}/{path}"
            except:
                pass
        
        print("\n❌ Could not find 1.wav in common locations")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
    print("\n📝 Next steps:")
    print("1. Upload your WAV file to the bucket")
    print("2. Use the correct S3 path when calling the API")
    print("3. Make sure the file is in the bucket root or update the path")

if __name__ == "__main__":
    check_s3_file()