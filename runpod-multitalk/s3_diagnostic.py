import os
import boto3
from botocore.exceptions import ClientError

print("[S3_DIAG] Starting S3 diagnostic")
print(f"[S3_DIAG] AWS_ACCESS_KEY_ID: {os.environ.get('AWS_ACCESS_KEY_ID', 'NOT_SET')[:10]}...")
print(f"[S3_DIAG] AWS_REGION: {os.environ.get('AWS_REGION', 'NOT_SET')}")
print(f"[S3_DIAG] AWS_S3_BUCKET_NAME: {os.environ.get('AWS_S3_BUCKET_NAME', 'NOT_SET')}")

# Test S3 access directly
if os.environ.get('AWS_ACCESS_KEY_ID'):
    try:
        s3 = boto3.client('s3',
                         aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                         aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                         region_name=os.environ.get('AWS_REGION', 'eu-west-1'))
        
        # Try to list buckets
        buckets = s3.list_buckets()
        print(f"[S3_DIAG] ✅ Can list buckets: {len(buckets['Buckets'])}")
        
        # Try the actual file
        try:
            obj = s3.get_object(Bucket='760572149-framepack', Key='1.wav')
            print(f"[S3_DIAG] ✅ Can access 1.wav: {obj['ContentLength']} bytes")
        except ClientError as e:
            print(f"[S3_DIAG] ❌ Cannot access 1.wav: {e.response['Error']['Code']}")
            print(f"[S3_DIAG] Full error: {e}")
            
    except Exception as e:
        print(f"[S3_DIAG] ❌ S3 client creation failed: {e}")
else:
    print("[S3_DIAG] ❌ No AWS credentials found")

# Continue with normal imports
from handler import *