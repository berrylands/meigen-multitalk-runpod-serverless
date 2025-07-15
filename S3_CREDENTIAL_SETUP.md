# S3 Credential Setup for RunPod

## Issue Summary
The S3 handler is not detecting AWS credentials properly, showing:
- "Failed to initialize S3 client: Invalid endpoint:"
- "S3 integration disabled (no credentials)"
- "S3 integration is not enabled. Missing AWS credentials."

## Root Cause
1. The `BUCKET_ENDPOINT_URL` is set but empty, causing the "Invalid endpoint:" error
2. AWS credentials might not be properly configured in RunPod secrets

## Solution

### 1. Update to Fixed Image
Use: `berrylands/multitalk-s3-endpoint-fix:latest`

This image fixes the empty endpoint URL issue.

### 2. Configure RunPod Secrets

Go to your RunPod endpoint settings and add these secrets:

| Secret Name | Value | Notes |
|------------|-------|-------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key | Required |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | Required |
| `AWS_REGION` | `us-east-1` (or your region) | Required |
| `AWS_S3_BUCKET_NAME` | `760572149-framepack` | Based on your S3 URL |
| `BUCKET_ENDPOINT_URL` | (leave empty) | Only set for S3-compatible services |

### 3. Important Notes

- **Don't use empty strings**: Make sure each value has actual content
- **BUCKET_ENDPOINT_URL**: Leave this completely unset for standard AWS S3
- **Bucket name**: Based on your URL `s3://760572149-framepack/1.wav`, your bucket is `760572149-framepack`

### 4. Verify Setup

After updating:

```bash
python debug_s3_credentials.py
```

This will show if S3 is properly configured.

### 5. Test S3 Access

```python
import runpod
runpod.api_key = "your-api-key"
endpoint = runpod.Endpoint("kkx3cfy484jszl")

# Test with your S3 file
job = endpoint.run({
    "action": "generate",
    "audio": "s3://760572149-framepack/1.wav",
    "duration": 5.0
})
print(job.output())
```

## Troubleshooting

If still not working:

1. **Check RunPod Secrets**: Go to endpoint settings → Environment Variables
2. **Remove BUCKET_ENDPOINT_URL**: If it exists, delete it entirely
3. **Check AWS Permissions**: Ensure your AWS credentials have S3 read/write access
4. **Verify Bucket Region**: Make sure AWS_REGION matches where your bucket is located

## Expected Success Log

When working correctly, you should see:
```
[INFO] S3 handler imported successfully
[INFO] S3 integration enabled. Default bucket: 760572149-framepack, Region: us-east-1
[INFO] ✓ S3 integration enabled (bucket: 760572149-framepack)
```