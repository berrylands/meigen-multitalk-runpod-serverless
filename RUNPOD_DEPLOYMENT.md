# RunPod Deployment Instructions

## Image Ready! 🎉

Your test image is now available on DockerHub:
- **Image**: `berrylands/multitalk-test:latest`

## Deploy on RunPod Web Interface

### Step 1: Go to RunPod Dashboard
1. Visit: https://www.runpod.io/console/serverless
2. Click **"+ New Endpoint"**

### Step 2: Configure Endpoint
Fill in these settings:

**Basic Configuration:**
- **Endpoint Name**: `multitalk-test`
- **Select Template**: Skip (click "Continue")
- **Container Image**: `berrylands/multitalk-test:latest`
- **Container Registry**: Leave empty (DockerHub public)
- **Container Disk**: `5 GB`

**Worker Configuration:**
- **Min Workers**: `0`
- **Max Workers**: `1`
- **Idle Timeout**: `60` seconds
- **Flash Boot**: `Enabled`

**GPU Configuration:**
- **Select GPU**: `RTX 4090` (24GB VRAM)
- **GPU Count**: `1`

**Advanced Options:**
- **Network Volume**: Select `meigen-multitalk` (100GB)
- **Mount Path**: `/runpod-volume`

**Environment Variables:**
- Add: `MODEL_PATH` = `/runpod-volume/models`
- Add: `RUNPOD_DEBUG_LEVEL` = `INFO`

### Step 3: Deploy
Click **"Deploy"** button at the bottom.

## Test Your Endpoint

Once deployed, you'll get an endpoint ID. Save it!

### Test Script
Save your endpoint ID to test: