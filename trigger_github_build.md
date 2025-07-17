# Trigger GitHub Actions Build

## Current Status
- ✅ GitHub Actions workflow is properly configured
- ✅ Docker Hub authentication is working (berrylands account)
- ❌ Recent builds failed due to "No space left on device" in GitHub runners
- ❌ Need to trigger a fresh build with a clean runner

## Manual Trigger Steps

1. **Go to GitHub Actions**: https://github.com/berrylands/meigen-multitalk-runpod-serverless/actions

2. **Select Workflow**: Click on "Build and Push MultiTalk" workflow

3. **Run Workflow**:
   - Click "Run workflow" dropdown
   - Branch: `master`
   - Version: `v131-fixed`
   - Click "Run workflow" button

4. **Monitor Progress**: The build should create `berrylands/multitalk-runpod:v131`

## What V131-Fixed Contains
- PyTorch 2.1.0 + CUDA 11.8 compatibility
- NumPy 1.26.4 for Numba compatibility
- All dependencies properly aligned

## Expected Outcome
- Success: `berrylands/multitalk-runpod:v131` available on Docker Hub
- RunPod template already configured to use v131
- Ready to test with existing S3 files

## If Build Fails Again
- GitHub Actions runners sometimes have space issues
- Try triggering another build (fresh runner)
- The Dockerfile.v131-fixed is correct based on previous analysis