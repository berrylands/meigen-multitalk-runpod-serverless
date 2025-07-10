# MeiGen MultiTalk Serverless

A serverless implementation of [MeiGen MultiTalk](https://github.com/MeiGen-AI/MultiTalk) for audio-driven multi-person conversational video generation, deployed on RunPod with zero idle costs.

## 🚀 Overview

This project provides a fully serverless version of MeiGen MultiTalk that:
- **Generates videos from audio input** using state-of-the-art AI models
- **Runs on-demand** with no idle costs (pay only when processing)
- **Scales automatically** based on demand
- **Provides fast processing** (typically 5-10 seconds per video)

## 📋 Features

- ✅ **Complete MultiTalk Implementation**: All models and functionality included
- ✅ **82.2GB of Pre-loaded Models**: Wan2.1, MultiTalk, Wav2Vec2, and more
- ✅ **GPU Accelerated**: Runs on NVIDIA RTX 4090 (24GB)
- ✅ **REST API**: Simple HTTP interface via RunPod
- ✅ **Python SDK**: Easy integration with `runpod` package
- ✅ **Batch Processing**: Process multiple files in parallel
- ✅ **Cost Effective**: Zero costs when not in use

## 🎬 What It Does

MultiTalk generates realistic talking head videos from audio input:
1. **Input**: Audio file (speech, singing, etc.)
2. **Processing**: AI models analyze audio and generate synchronized video
3. **Output**: MP4 video with realistic facial movements

## 🛠️ Setup

### Prerequisites

- Python 3.7+
- RunPod account with API key
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/berrylands/meigen-multitalk-runpod-serverless.git
cd meigen-multitalk-runpod-serverless

# Install dependencies
pip install runpod python-dotenv numpy

# Create .env file with your API key
echo "RUNPOD_API_KEY=your_api_key_here" > .env
```

## 🚀 Quick Start

### Generate a Video

```python
import runpod
import base64
import numpy as np
from dotenv import load_dotenv

# Setup
load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")
endpoint = runpod.Endpoint("kkx3cfy484jszl")

# Create audio (3 seconds, 16kHz)
sample_rate = 16000
duration = 3.0
t = np.linspace(0, duration, int(sample_rate * duration), False)
audio = np.sin(2 * np.pi * 440 * t)
audio_int16 = (audio * 32767).astype(np.int16)
audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')

# Generate video
job = endpoint.run({
    "action": "generate",
    "audio": audio_b64,
    "duration": duration,
    "fps": 30,
    "width": 512,
    "height": 512
})

# Wait for completion
while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
    print(f"Status: {job.status()}")
    time.sleep(2)

# Save video
if job.status() == "COMPLETED":
    result = job.output()
    if result.get("success"):
        video_data = base64.b64decode(result["video"])
        with open("output.mp4", "wb") as f:
            f.write(video_data)
        print("Video saved!")
```

### Using the Examples

```bash
# Simple video generation
python examples/simple_video_generator.py audio.wav output.mp4

# Batch processing
python examples/batch_processor.py ./audio_files ./output_videos
```

## 📖 Documentation

- **[Usage Guide](USAGE_GUIDE.md)**: Complete API reference and examples
- **[Deployment Guide](DEPLOYMENT_GUIDE.md)**: How the system was deployed
- **[Video Generation Proof](VIDEO_GENERATION_PROOF.md)**: Evidence of working system

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│   RunPod     │────▶│  MultiTalk  │
│  (Python)   │     │  Serverless  │     │  Container  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                      │
                           │                      ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   Network    │     │   Models    │
                    │   Volume     │────▶│  (82.2GB)   │
                    │   (100GB)    │     └─────────────┘
                    └──────────────┘
```

## 💰 Cost Structure

- **No idle costs**: Pay $0 when not using the system
- **Processing costs**: ~$0.00024 per second of GPU time
- **Typical video**: 5-10 seconds processing = ~$0.0012-0.0024
- **Storage**: One-time setup, models persist on network volume

## 🔧 Models Included

| Model | Size | Purpose |
|-------|------|---------|
| MeiGen-MultiTalk | 9.3 GB | Core video generation |
| Wan2.1-I2V-14B | 68.9 GB | Advanced video synthesis |
| Wav2Vec2 (Multiple) | 3.8 GB | Audio processing |
| GFPGAN | < 1 GB | Face enhancement |
| Others | ~1 GB | Supporting models |

## 📊 Performance

- **Cold Start**: 30-60 seconds (first request)
- **Warm Processing**: 0.5-2 seconds per video
- **Queue Time**: 2-10 seconds typically
- **Total Time**: 5-15 seconds per video (warm)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project follows the licensing terms of the original [MeiGen MultiTalk](https://github.com/MeiGen-AI/MultiTalk) project.

## 🙏 Acknowledgments

- [MeiGen AI](https://github.com/MeiGen-AI) for the original MultiTalk model
- [RunPod](https://runpod.io) for serverless GPU infrastructure
- The open-source community for supporting models

## 📞 Support

- Create an issue for bugs or feature requests
- Check the [Usage Guide](USAGE_GUIDE.md) for detailed documentation
- Review [examples](examples/) for implementation patterns

---

**Ready to generate videos?** Follow the Quick Start guide above or check out the [examples](examples/) directory!