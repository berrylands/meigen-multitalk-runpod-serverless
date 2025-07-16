# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a serverless implementation of MeiGen MultiTalk for RunPod, enabling audio-driven multi-person conversational video generation without paying for idle servers.

## Project Structure

```
meigen-multitalk/
└── runpod-multitalk/
    ├── src/
    │   ├── handler.py          # RunPod serverless handler
    │   └── multitalk_inference.py  # Model inference wrapper
    ├── scripts/
    │   ├── download_models.py  # One-time model download
    │   ├── build_and_push.sh   # Docker build script
    │   └── entrypoint.sh       # Container entrypoint
    ├── examples/
    │   └── client.py           # Python client example
    ├── tests/
    │   └── test_local.py       # Local testing
    ├── Dockerfile              # RunPod container definition
    ├── requirements.txt        # Python dependencies
    └── README.md              # Comprehensive documentation
```

## Development Commands

### Local Testing
```bash
cd runpod-multitalk
python tests/test_local.py
```

### Build Docker Image
```bash
cd runpod-multitalk
docker build -t multitalk-runpod .
```

### Deploy to DockerHub
```bash
./scripts/build_and_push.sh YOUR_DOCKERHUB_USERNAME
```

## Architecture Overview

1. **Serverless Handler**: Uses RunPod's serverless API to process video generation requests
2. **Model Storage**: Models stored on 100GB RunPod network volume (not in container)
3. **Inference Pipeline**: 
   - Load models from network volume on cold start
   - Process audio with Wav2Vec2
   - Generate video with Wan2.1 + MultiTalk
   - Return video URL or base64

## Key Implementation Notes

1. **Model Loading**: The actual Wan2.1 GGUF model loading is not fully implemented - needs GGUF loader integration
2. **Video Generation**: Current implementation creates placeholder videos - needs full MultiTalk integration
3. **Storage Optimization**: Uses quantized models to fit in 100GB instead of multi-TB storage
4. **Cold Start**: Expect 30-60 second cold starts due to model loading

## Environment Variables

- `MODEL_PATH`: Path to models directory (default: `/runpod-volume/models`)
- `RUNPOD_DEBUG_LEVEL`: Debug logging level
- `S3_BUCKET`: Optional S3 bucket for output storage
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: S3 credentials

## Testing Approach

1. Local testing with mock models
2. Docker testing with GPU
3. RunPod staging deployment
4. Production deployment with real models

## MeiGen-MultiTalk Implementation

**IMPORTANT**: Always refer to `MEIGEN_MULTITALK_REFERENCE.md` for the correct implementation details from the working codebase.

Key points:
- Uses `wan.MultiTalkPipeline` from the official MeiGen-MultiTalk implementation
- Requires specific model loading sequence: Wav2Vec2 → MultiTalk → WAN 2.1 → VAE → CLIP
- Uses size bucketing ("multitalk-480") for resolution management
- Supports turbo mode for faster generation
- Proper audio embedding extraction with `Wav2Vec2Model`

## Important TODOs

1. ✅ ~~Implement actual GGUF model loading for Wan2.1~~ Use `wan.MultiTalkPipeline` instead
2. ✅ ~~Integrate real MultiTalk video generation pipeline~~ Use proper MeiGen-MultiTalk implementation
3. Add proper error handling and retry logic
4. Implement S3 upload functionality
5. Add monitoring and metrics
6. **PRIORITY**: Deploy V114 to replace broken V112 on existing endpoint