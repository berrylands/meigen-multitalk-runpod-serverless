# MultiTalk V73 Deployment Instructions

## Quick Deploy V73 Minimal Fix

### 1. Build V73 Minimal (Recommended)
```bash
cd runpod-multitalk
docker build -f Dockerfile.v73-minimal -t berrylands/multitalk-v73:minimal-fix .
docker push berrylands/multitalk-v73:minimal-fix
```

**RunPod Image**: `berrylands/multitalk-v73:minimal-fix`

### 2. Alternative: Build V73 Complete
```bash
cd runpod-multitalk
docker build -f Dockerfile.v73-runtime-deps -t berrylands/multitalk-v73:runtime-deps .
docker push berrylands/multitalk-v73:runtime-deps
```

**RunPod Image**: `berrylands/multitalk-v73:runtime-deps`

## V73 Key Fixes

### Environment Variables Added
- `CC=gcc` - C compiler for Triton runtime compilation
- `CXX=g++` - C++ compiler
- `CUDA_HOME=/usr/local/cuda` - CUDA installation path

### Files Updated
- `handler_v73.py` - Version 73.0.0 with updated logging
- `multitalk_v73_official_wrapper.py` - V73 wrapper with runtime dependency fixes

## Expected Results

V73 should resolve the V72 errors:
1. ✅ **C Compiler Error**: Fixed with `CC=gcc` environment variable
2. ✅ **XFormers Warning**: Should still warn but not fail fatally
3. ✅ **Triton Compilation**: Should work with proper compiler environment

## Test Command

After deployment on RunPod:
```bash
# Test with empty input (will use default files)
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {}}'
```

## Fallback Plan

If V73 still has issues, we have these options:
1. **Environment-only fix**: Set `CC=gcc` in RunPod environment variables
2. **Disable problematic features**: Set `DISABLE_TRITON=1` 
3. **Use CPU fallback**: Force CPU execution for problematic operations

## Confidence Level: High

V72 proved the complete official implementation works. V73 addresses the specific runtime environment issues identified in the logs.

## Success Criteria

✅ Handler starts without C compiler errors  
✅ Official MultiTalk script executes  
✅ Video generation completes (even if with warnings)  
✅ S3 upload of generated video succeeds