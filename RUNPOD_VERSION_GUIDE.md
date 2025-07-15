# RunPod Version Management Guide

## The Problem
RunPod can cache Docker images aggressively, sometimes using an older cached version even when you've pushed a new image with the same tag.

## Solutions (In Order of Reliability)

### 1. 🏆 Use Unique Timestamp Tags (RECOMMENDED)

**Never reuse tags!** Always use unique tags with timestamps:

```bash
# Use the deploy script
cd runpod-multitalk
./deploy.sh

# This creates tags like:
# berrylands/multitalk-complete:v2.1.0-20240110-153045
```

**Benefits:**
- RunPod MUST pull the new image (tag doesn't exist in cache)
- You can track exactly which version is running
- Easy rollback to previous versions

### 2. 🔒 Use Docker Image Digest (MOST RELIABLE)

Use the SHA256 digest for absolute certainty:

```bash
# After pushing your image
docker push berrylands/multitalk-complete:v2.1.0

# Get the digest
docker inspect berrylands/multitalk-complete:v2.1.0 --format='{{index .RepoDigests 0}}'

# Use in RunPod (example):
berrylands/multitalk-complete@sha256:8d4c79f34379f926dd12c87c7e8f77f3261e9d7a429c5ec1e2b2e662f9c3abcd
```

### 3. 🔄 Force Cache Refresh

If you must reuse a tag:

1. **Temporary Tag Switch:**
   - Update endpoint to `berrylands/multitalk-complete:temp-invalid`
   - Wait for it to fail
   - Update back to your desired tag
   - RunPod will pull fresh

2. **Add --no-cache to Docker Build:**
   ```bash
   docker buildx build --no-cache --platform linux/amd64 ...
   ```

### 4. 💣 Nuclear Option: New Endpoint

1. Delete the current endpoint
2. Create a new one with the same settings
3. Use your new image tag

**Note:** This changes your endpoint ID!

## Verification Methods

### 1. Check Build Info in Health Check

The updated handler now returns build information:

```python
# Test with:
python test_s3_integration.py

# Look for:
{
  "version": "2.1.0",
  "build_time": "2024-01-10 15:30:45",
  "build_id": "1704901845",
  "container_id": "runpod-worker-xyz"
}
```

### 2. Add Version File to Image

Create a VERSION file during build:

```dockerfile
RUN echo "Built: $(date)" > /app/VERSION
RUN echo "Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')" >> /app/VERSION
```

### 3. Log Version on Startup

Add to handler initialization:

```python
print(f"Starting MultiTalk v2.1.0 - Built: {os.environ.get('BUILD_TIME', 'unknown')}")
```

## Best Practices

1. **Always use timestamp tags for production deployments**
2. **Save deployment information:**
   ```bash
   echo "Deployed: berrylands/multitalk-complete:v2.1.0-20240110-153045" > last_deployment.txt
   ```

3. **Test immediately after deployment:**
   ```bash
   python test_s3_integration.py
   ```

4. **Monitor RunPod worker logs** during first request to see initialization messages

5. **Use environment variables** to pass version info:
   ```yaml
   env:
     DEPLOYMENT_VERSION: "v2.1.0-20240110-153045"
   ```

## Quick Deployment Commands

### Option 1: Use the Deploy Script
```bash
cd runpod-multitalk
./deploy.sh
# Follow the instructions it provides
```

### Option 2: Manual with Timestamp
```bash
TAG="v2.1.0-$(date +%Y%m%d-%H%M%S)"
docker buildx build --platform linux/amd64 \
  -t berrylands/multitalk-complete:$TAG \
  -f Dockerfile.complete \
  --push \
  --no-cache .
  
echo "Update RunPod to use: berrylands/multitalk-complete:$TAG"
```

### Option 3: Python Script
```bash
python force_update_runpod.py
# Choose option 1 and follow prompts
```

## Troubleshooting

### Still Getting Old Version?

1. **Check RunPod Worker Status:**
   - Ensure all workers have restarted
   - May take 2-3 minutes for new workers to spin up

2. **Verify Image Was Pushed:**
   ```bash
   docker pull berrylands/multitalk-complete:your-tag
   docker inspect berrylands/multitalk-complete:your-tag
   ```

3. **Check RunPod Logs:**
   - Look for "Pulling image" messages
   - Check for any pull errors

4. **Force Worker Restart:**
   - Scale workers to 0
   - Wait 30 seconds
   - Scale back to desired count

### Emergency Rollback

Keep track of working versions:
```bash
# Save before deploying new version
echo "Last working: berrylands/multitalk-complete:v2.1.0-20240110-150000" > last_working.txt
```

## Summary

**For S3 deployment, run:**
```bash
cd runpod-multitalk
./deploy.sh
```

This will:
1. Build with a unique timestamp tag
2. Push to DockerHub
3. Give you the exact image name to use in RunPod
4. Include build info in the health check

No more cache issues! 🎉