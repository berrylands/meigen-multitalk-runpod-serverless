# V115 Deployment Guide

## Overview
This guide covers the deployment and testing of MultiTalk V115 with strict MeiGen-MultiTalk requirements.

## V115 Implementation Status

### ✅ Completed
- **Implementation**: `multitalk_v115_implementation.py` - Proper MeiGen-MultiTalk integration
- **Handler**: `handler_v115.py` - Strict requirements enforcement
- **Docker**: `Dockerfile.v115` - Container configuration
- **Build Script**: `build_v115.sh` - Build automation
- **Tests**: Multiple test scripts for validation

### 🔧 Key Features
- **No Fallback Mechanisms**: Fails cleanly without proper MeiGen-MultiTalk
- **Strict Error Messages**: Clear requirements for missing components
- **Proper Imports**: Uses `wan.MultiTalkPipeline` and `Wav2Vec2Model`
- **Component Validation**: Requires all 5 models + 3 components loaded

## Structure Validation

### ✅ Verified Components
```
📁 Files Structure: ✅ All required files present
🚫 No Fallback Code: ✅ Fallback mechanisms removed
💬 Strict Error Handling: ✅ Clear error messages
📦 Proper MeiGen Imports: ✅ wan.MultiTalkPipeline, Wav2Vec2Model
🔧 Handler Strict Mode: ✅ Enforces 5/5 models, 3/3 components
🐳 Dockerfile Structure: ✅ Proper build configuration
```

## Deployment Steps

### Step 1: Build V115 Image
```bash
cd /Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk
./build_v115.sh
```

### Step 2: Push to Registry
```bash
docker push berrylands/multitalk-v115:proper-meigen-multitalk
```

### Step 3: Update Endpoint
Update existing endpoint `zu0ik6c8yukyl6` to use V115 image:
- Container Image: `berrylands/multitalk-v115:proper-meigen-multitalk`
- Container Disk: 20 GB
- Volume Disk: 100 GB (existing)
- Volume Mount: `/runpod-volume`

## Testing Procedures

### 1. Structure Test (No API Key Required)
```bash
python3 test_v115_structure.py
```

**Expected Result**: All 6 tests pass
- File Structure ✅
- No Fallback Code ✅
- Strict Error Handling ✅
- Proper MeiGen Imports ✅
- Handler Strict Mode ✅
- Dockerfile Structure ✅

### 2. Current Endpoint Test (Requires API Key)
```bash
export RUNPOD_API_KEY=your_api_key
python3 test_current_endpoint.py
```

**Expected Results**:
- If V115 deployed: Shows V115 implementation details
- If V112 or older: Shows older version, needs update

### 3. V115 Health Check Test (Requires API Key)
```bash
python3 test_v115_implementation.py
```

**Expected Results**:
- **With MeiGen-MultiTalk**: Full functionality
- **Without MeiGen-MultiTalk**: Clear error messages

### 4. Video Generation Test (Requires API Key)
```bash
python3 test_video_generation.py
```

**Expected Results**:
- **V115 with MeiGen-MultiTalk**: Proper video generation
- **V115 without MeiGen-MultiTalk**: Clear error about missing components
- **V112 or older**: "Failed to generate video" error

## Error Scenarios

### V115 Without MeiGen-MultiTalk Models
```json
{
  "output": {
    "status": "error",
    "error": "MultiTalk V115 not initialized - MeiGen-MultiTalk components required"
  }
}
```

### V115 With Missing Components
```json
{
  "output": {
    "status": "error",
    "error": "Required MeiGen-MultiTalk audio components missing: No module named 'wan'"
  }
}
```

### V115 Ready State
```json
{
  "output": {
    "status": "healthy",
    "version": "V115",
    "implementation": "MeiGen-MultiTalk",
    "multitalk_available": true,
    "multitalk_loaded": true,
    "model_info": {
      "models_available": {"multitalk": true, "wan21": true, "wav2vec": true, "vae": true, "clip": true},
      "models_loaded": {"wan_i2v": true, "audio_encoder": true, "feature_extractor": true}
    }
  }
}
```

## Troubleshooting

### Build Issues
- **Problem**: Docker build fails
- **Solution**: Check `build_v115.log` for errors
- **Common**: Missing dependencies, network issues

### Deployment Issues
- **Problem**: Container fails to start
- **Solution**: Check RunPod logs for import errors
- **Common**: Missing MeiGen-MultiTalk repository

### Model Loading Issues
- **Problem**: "MeiGen-MultiTalk components missing"
- **Solution**: Ensure models are properly downloaded to network volume
- **Check**: `/runpod-volume/models/` contains all required models

### Video Generation Issues
- **Problem**: "Failed to generate video"
- **Solution**: With V115, this means MeiGen-MultiTalk components are missing
- **Action**: Check model availability and component loading

## Success Criteria

### V115 Deployment Success
- ✅ Health check returns V115 version
- ✅ All 5 models available
- ✅ All 3 components loaded
- ✅ MeiGen-MultiTalk pipeline active

### V115 Error Handling Success
- ✅ Clear error messages when components missing
- ✅ No fallback or placeholder videos
- ✅ Proper component validation
- ✅ Strict requirements enforced

## API Key Required Commands

All actual endpoint testing requires the RUNPOD_API_KEY:

```bash
# Set API key
export RUNPOD_API_KEY=your_actual_api_key

# Test current endpoint
python3 test_current_endpoint.py

# Test V115 implementation
python3 test_v115_implementation.py

# Test video generation
python3 test_video_generation.py

# Diagnose issues
python3 diagnose_video_generation.py
```

## Next Steps

1. **Build V115**: Complete Docker build
2. **Deploy V115**: Update endpoint to use V115 image
3. **Test with API**: Run endpoint tests with API key
4. **Verify Functionality**: Ensure proper video generation or clear errors

The V115 implementation is ready for deployment and will provide either:
- ✅ **Perfect video generation** with proper MeiGen-MultiTalk
- ❌ **Clear error messages** without proper MeiGen-MultiTalk

No ambiguous states or fallback functionality.