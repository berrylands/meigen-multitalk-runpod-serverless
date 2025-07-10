# Manual RunPod Deployment Guide

Since we're having API key authorization issues, let's deploy manually through the RunPod web interface.

## Step 1: Build and Push Docker Image

### Option A: Use pre-built test image
I recommend using a simple test image first. Create this Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN pip install runpod

COPY << 'EOF' /app/handler.py
import runpod
import os
import sys

def handler(job):
    job_input = job.get('input', {})
    
    if job_input.get('health_check'):
        return {
            "status": "healthy",
            "message": "Test handler working!",
            "python_version": sys.version,
            "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown')
        }
    
    return {
        "message": "Handler successful!",
        "echo": job_input
    }

runpod.serverless.start({"handler": handler})
EOF

CMD ["python", "-u", "handler.py"]
```

### Build and push:
```bash
# Build the image
docker build -t YOUR_DOCKERHUB_USERNAME/multitalk-test:latest .

# Login to DockerHub
docker login

# Push the image
docker push YOUR_DOCKERHUB_USERNAME/multitalk-test:latest
```

## Step 2: Create RunPod Serverless Endpoint Manually

1. **Go to RunPod Dashboard**: https://www.runpod.io/console/serverless

2. **Click "New Endpoint"**

3. **Configure the endpoint**:
   - **Name**: `multitalk-test`
   - **Container Image**: `YOUR_DOCKERHUB_USERNAME/multitalk-test:latest`
   - **Container Registry Credentials**: None (for public images)
   - **Container Disk**: 5 GB
   - **GPU Type**: RTX 4000 Ada (cheaper) or RTX 4090 (faster)
   - **Min Workers**: 0
   - **Max Workers**: 1
   - **Idle Timeout**: 60 seconds
   - **Flash Boot**: Enabled

4. **Environment Variables** (Optional):
   - `RUNPOD_DEBUG_LEVEL`: `INFO`

5. **Click "Deploy"**

## Step 3: Test the Endpoint

Once deployed, you'll get an endpoint ID. Test it with:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"health_check": true}}'
```

Expected response:
```json
{
  "status": "healthy",
  "message": "Test handler working!",
  "python_version": "...",
  "worker_id": "..."
}
```

## Step 4: Advanced Configuration (Once Basic Works)

### Add Network Volume (for models):
1. Go to **Storage → Network Volumes**
2. Create new volume: 100GB, same region as endpoint
3. In endpoint settings, add volume mount:
   - **Volume**: Select your volume
   - **Mount Path**: `/runpod-volume`

### Use the Full MultiTalk Image:
Once the test works, replace with our full image:
- `berrylands/meigen-multitalk-runpod-serverless:latest`

## Step 5: Test MultiTalk Functionality

Test with image and audio:
```python
import requests
import base64

# Read test image
with open("test_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Read test audio  
with open("test_audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "input": {
            "reference_image": image_b64,
            "audio_1": audio_b64,
            "prompt": "A person speaking",
            "num_frames": 30,
            "turbo": True
        }
    }
)
```

## Troubleshooting

1. **Endpoint fails to start**: Check logs in RunPod dashboard
2. **Image not found**: Verify DockerHub image is public
3. **Timeout errors**: Increase GPU type or reduce workload
4. **Memory errors**: Use smaller models or more VRAM

## What's Your DockerHub Username?

Please let me know your DockerHub username so I can update the image references accordingly.

## Next Steps

Once we have a working basic endpoint, we can:
1. Add the full MultiTalk models
2. Set up automated deployments
3. Optimize performance
4. Add monitoring