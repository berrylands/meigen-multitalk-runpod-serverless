import os
print("[ENV_CHECK] Environment variables at startup:")
for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'AWS_S3_BUCKET_NAME']:
    value = os.environ.get(key, 'NOT_SET')
    if key.endswith('_ID') and value != 'NOT_SET':
        value = value[:10] + '...'
    elif key.endswith('_KEY') and value != 'NOT_SET':
        value = '***'
    print(f"[ENV_CHECK] {key}={value}")

# Test S3 access with the environment
import boto3
try:
    s3 = boto3.client('s3')
    response = s3.head_object(Bucket='760572149-framepack', Key='1.wav')
    print(f"[ENV_CHECK] ✅ Can access 1.wav via default client: {response['ContentLength']} bytes")
except Exception as e:
    print(f"[ENV_CHECK] ❌ Cannot access 1.wav via default client: {e}")

# Import the handler to continue
from handler import *