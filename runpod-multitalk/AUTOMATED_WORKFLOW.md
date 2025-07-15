# Automated MultiTalk RunPod Workflow

## Overview

This workflow provides a self-sufficient system for building, deploying, and testing MultiTalk on RunPod.

## Components

### 1. RunPod Template
- **Template ID**: `joospbpdol`
- **Template Name**: `multitalk-v80-auto-update`
- **Auto-updates**: Yes, via GitHub Actions

### 2. GitHub Actions Workflow
- **File**: `.github/workflows/docker-build.yml`
- **Triggers**: 
  - Push to master branch (Dockerfile changes)
  - Manual workflow dispatch
- **Auto-builds**: Latest version
- **Auto-pushes**: To Docker Hub

### 3. Version Management
- **Naming**: `Dockerfile.v80`, `Dockerfile.v81`, etc.
- **Latest**: Always tagged as `latest`
- **Specific**: Tagged with version number

## Workflow Steps

### 1. Create New Version
```bash
# Copy from previous version
cp Dockerfile.v80 Dockerfile.v81

# Edit to fix issues
vim Dockerfile.v81
```

### 2. Push to GitHub
```bash
git add Dockerfile.v81
git commit -m "Create v81 with <fixes>"
git push
```

GitHub Actions will automatically:
- Build the Docker image
- Push to Docker Hub
- Update the template (if configured)

### 3. Update Template (if needed)
```bash
python update_template.py v81
```

### 4. Test Endpoint
```bash
python test_runpod_endpoint.py v81
```

## Key Fixes in V80

1. **xfuser imports**: Properly stubbed out with replacement functions
2. **HuggingFace cache**: Runtime initialization ensures directories exist
3. **Permissions**: All cache directories set to 777
4. **Runtime script**: `/app/init_runtime.sh` handles initialization

## Environment Variables

Set in RunPod secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_REGION`
- `AWS_S3_BUCKET_NAME`
- `AWS_SECRET_ACCESS_KEY`
- `BUCKET_ENDPOINT_URL`

## Debugging

### Check logs
```bash
# Via RunPod dashboard or API
curl -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.ai/v2/$ENDPOINT_ID/logs
```

### Test locally
```bash
docker run -it \
  -e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -e AWS_S3_BUCKET_NAME=$AWS_S3_BUCKET_NAME \
  -e AWS_REGION=$AWS_REGION \
  -e BUCKET_ENDPOINT_URL=$BUCKET_ENDPOINT_URL \
  -v /path/to/models:/runpod-volume/models \
  berrylands/multitalk-runpod:v80 \
  /bin/bash
```

## Creating Next Version

When V80 has issues:

1. Check logs to identify the problem
2. Create `Dockerfile.v81` with fixes
3. Push to GitHub
4. Test with `python test_runpod_endpoint.py v81`

## Common Issues and Fixes

### xfuser import errors
- Fixed in V80 by stubbing out functions
- If persists, check `/app/multitalk_official/wan/utils/multitalk_utils.py`

### HuggingFace cache permissions
- Fixed in V80 with runtime initialization
- If persists, check `/runpod-volume/huggingface` permissions

### Missing dependencies
- Add to pip install in Dockerfile
- Ensure single RUN layer for all pip installs

### Model not found
- Check `/runpod-volume/models` contents
- Verify MODEL_PATH environment variable