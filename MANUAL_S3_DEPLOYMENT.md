# Manual S3 Deployment Steps

Since the automated build is timing out, here are the manual steps to deploy S3 support:

## Option 1: Build Locally (Recommended)

1. **Open a terminal** (not through Claude) to avoid timeouts:

```bash
cd /Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk

# Build with a unique tag
docker build --platform linux/amd64 \
  -t berrylands/multitalk-complete:v2.1.0-s3 \
  -f Dockerfile.complete .
```

This will take 10-15 minutes due to PyTorch installation.

2. **Push to DockerHub**:

```bash
docker push berrylands/multitalk-complete:v2.1.0-s3
```

3. **Update RunPod Endpoint**:
   - Go to https://www.runpod.io/console/serverless
   - Click on your endpoint (ID: `kkx3cfy484jszl`)
   - Click Edit/Settings
   - Change Docker image to: `berrylands/multitalk-complete:v2.1.0-s3`
   - Save changes

## Option 2: Use Pre-built Image

If you want to skip building, you can use this approach:

1. **Create a simple update script** that adds S3 to your existing deployment:

```bash
# Create a temporary directory
mkdir -p /tmp/multitalk-s3-patch
cd /tmp/multitalk-s3-patch

# Copy the necessary files
cp /Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk/complete_multitalk_handler.py .
cp /Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk/s3_handler.py .

# Create a minimal Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir runpod boto3 torch transformers numpy
COPY *.py ./
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/runpod-volume/models
CMD ["python", "-u", "handler.py"]
EOF

# Build and push
docker buildx build --platform linux/amd64 -t berrylands/multitalk-s3:latest --push .
```

## Option 3: Test First

Before building, you can test if your current deployment already has S3 support:

```bash
python test_s3_integration.py
```

## Configuration Steps (After Deployment)

1. **Add S3 Environment Variables in RunPod**:
   - Go to your endpoint settings
   - Add these environment variables:
     - `AWS_ACCESS_KEY_ID` = your-access-key
     - `AWS_SECRET_ACCESS_KEY` = your-secret-key
     - `AWS_REGION` = us-east-1 (or your region)
     - `AWS_S3_BUCKET_NAME` = your-bucket-name

2. **Test S3 Integration**:

```bash
# Check S3 status
python test_s3_integration.py

# Test with your S3 file
python test_s3_integration.py s3://your-bucket/your-audio.wav
```

## What's Included

The S3 update adds:
- ✅ `s3_handler.py` - Complete S3 integration module
- ✅ Updated handler with S3 URL detection
- ✅ boto3 for AWS S3 operations
- ✅ Support for both S3 input and output
- ✅ Backward compatibility with base64

## Quick Test

After deployment, test with:

```python
import runpod
runpod.api_key = "your-api-key"
endpoint = runpod.Endpoint("kkx3cfy484jszl")

# Test S3 input
job = endpoint.run({
    "action": "generate",
    "audio": "s3://your-bucket/audio.wav",
    "duration": 5.0
})
```

## Troubleshooting

If S3 isn't working:
1. Check the health endpoint shows S3 as enabled
2. Verify AWS credentials are set in RunPod
3. Check boto3 is installed in the container
4. Look at worker logs for import errors