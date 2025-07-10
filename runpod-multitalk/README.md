# RunPod Serverless MultiTalk

A serverless implementation of MeiGen MultiTalk for RunPod, enabling audio-driven multi-person conversational video generation without paying for idle servers.

## Overview

This implementation packages MeiGen MultiTalk as a serverless function on RunPod, using a 100GB network volume for model storage instead of expensive multi-TB storage solutions.

### Key Features
- ✅ True serverless - pay only for GPU time used
- ✅ Optimized storage - only ~60-70GB needed
- ✅ Fast cold starts with model caching
- ✅ Support for single and multi-person video generation
- ✅ 480p and 720p output options

## Architecture

```
User Request → RunPod API → Serverless Worker (RTX 4090)
                                    ↓
                            Load Models from Network Volume
                                    ↓
                            Generate Video with MultiTalk
                                    ↓
                            Return Video URL/Base64
```

## Prerequisites

- RunPod account with API access
- DockerHub account
- 100GB RunPod network volume
- (Optional) AWS S3 bucket for output storage

## Setup Instructions

### 1. Create RunPod Network Volume

1. Log into RunPod dashboard
2. Go to Storage → Network Volumes
3. Create new volume:
   - Name: `multitalk-models`
   - Size: 100GB
   - Region: Choose your preferred region

### 2. Download Models to Network Volume

1. Create a temporary GPU pod with the network volume attached
2. SSH into the pod and run:

```bash
# Install Python and git
apt update && apt install -y python3-pip git

# Clone this repository
git clone https://github.com/yourusername/meigen-multitalk.git
cd meigen-multitalk/runpod-multitalk

# Install dependencies
pip install huggingface-hub

# Run model download script
python scripts/download_models.py --model-path /runpod-volume/models
```

This will download ~60-70GB of models. The process may take 30-60 minutes.

### 3. Build and Push Docker Image

On your local machine:

```bash
# Clone repository
git clone https://github.com/yourusername/meigen-multitalk.git
cd meigen-multitalk/runpod-multitalk

# Build and push to DockerHub
chmod +x scripts/build_and_push.sh
./scripts/build_and_push.sh YOUR_DOCKERHUB_USERNAME
```

### 4. Create Serverless Endpoint

1. In RunPod dashboard, go to Serverless → Endpoints
2. Click "New Endpoint"
3. Configure:
   - Container Image: `YOUR_DOCKERHUB_USERNAME/multitalk-runpod:latest`
   - GPU Type: RTX 4090 (24GB VRAM)
   - Min Workers: 0
   - Max Workers: 3
   - Idle Timeout: 60 seconds
   - Container Disk: 10GB
4. Add Network Volume:
   - Select your `multitalk-models` volume
   - Mount Path: `/runpod-volume`
5. Add Environment Variables:
   - `MODEL_PATH`: `/runpod-volume/models`
   - (Optional) S3 credentials if using S3 storage

## Usage

### API Request Format

```python
import requests
import base64

# Your RunPod API key and endpoint
api_key = "YOUR_RUNPOD_API_KEY"
endpoint_id = "YOUR_ENDPOINT_ID"

# Prepare request
with open("reference_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

with open("audio1.wav", "rb") as f:
    audio1_base64 = base64.b64encode(f.read()).decode()

request_data = {
    "input": {
        "reference_image": image_base64,
        "audio_1": audio1_base64,
        "prompt": "Two people having a friendly conversation",
        "num_frames": 100,
        "seed": 42,
        "sampling_steps": 20
    }
}

# Send request
response = requests.post(
    f"https://api.runpod.ai/v2/{endpoint_id}/runsync",
    json=request_data,
    headers={"Authorization": f"Bearer {api_key}"}
)

result = response.json()
```

### Input Parameters

- `reference_image` (required): Base64 encoded image or URL
- `audio_1` (required): Base64 encoded audio file or URL
- `audio_2` (optional): Second audio file for multi-person conversation
- `prompt` (required): Description of the conversation
- `num_frames` (optional): Number of frames to generate (default: 100, max: 201)
- `seed` (optional): Random seed for reproducibility (default: 42)
- `turbo` (optional): Enable turbo mode for faster generation (default: false)
- `sampling_steps` (optional): Number of sampling steps (default: 20, range: 2-100)
- `guidance_scale` (optional): Guidance scale for generation (default: 7.5)
- `fps` (optional): Frames per second for output video (default: 8)

### Response Format

```json
{
  "video_url": "https://s3.amazonaws.com/bucket/output.mp4"
}
```

Or if S3 is not configured:

```json
{
  "video_base64": "base64_encoded_video_data"
}
```

## Cost Estimation

- **Storage**: $7/month (100GB network volume)
- **GPU Time**: ~$0.74/hour for RTX 4090
- **Per Video Cost**: 
  - 10-second generation: ~$0.10-0.30
  - 15-second generation: ~$0.20-0.50

## Performance

- **Cold Start**: 30-60 seconds (model loading)
- **Warm Generation**: 10-30 seconds per video
- **Concurrent Requests**: Up to 3 workers (configurable)

## Troubleshooting

### Models not found
- Ensure network volume is mounted at `/runpod-volume`
- Check that model download completed successfully
- Verify `download_complete.txt` exists in models directory

### Out of memory errors
- Reduce `num_frames` or use `turbo` mode
- Consider using quantized models
- Ensure using GPU with 24GB+ VRAM

### Slow generation
- Enable `turbo` mode for faster inference
- Reduce `sampling_steps` (min: 2)
- Use smaller frame counts

## Advanced Configuration

### Using S3 for Output Storage

Add these environment variables to your endpoint:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `S3_BUCKET`

### Custom Model Paths

Modify `MODEL_PATH` environment variable to use different model locations.

## Development

### Local Testing

```bash
# Build Docker image
docker build -t multitalk-local .

# Run with GPU
docker run --gpus all -v $(pwd)/models:/runpod-volume/models multitalk-local
```

### Adding New Features

1. Modify `src/multitalk_inference.py` for model changes
2. Update `src/handler.py` for API changes
3. Rebuild and push Docker image

## License

This wrapper is provided under MIT License. MeiGen MultiTalk is licensed under Apache 2.0.

## Credits

- MeiGen MultiTalk: https://github.com/MeiGen-AI/MultiTalk
- RunPod: https://runpod.io