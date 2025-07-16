# MeiGen-MultiTalk Implementation Reference

## Overview
This document contains the distilled implementation details from the working MeiGen-MultiTalk codebase at https://github.com/zsxkib/cog-MultiTalk

## Key Architecture Components

### 1. Model Structure
```
MeiGen-MultiTalk.tar          # Main MultiTalk model (~9.5GB)
Wan2.1-I2V-14B-480P.tar      # WAN 2.1 Image-to-Video model (~14GB)
chinese-wav2vec2-base.tar     # Audio processing model (~1.2GB)
```

### 2. Core Pipeline Classes

#### Main Pipeline
```python
from wan import MultiTalkPipeline
from wan.configs import WAN_CONFIGS

# Initialize pipeline
self.wan_i2v = wan.MultiTalkPipeline(
    config=WAN_CONFIGS["multitalk-14B"],
    checkpoint_dir=self.ckpt_dir,
    device_id=0
)
```

#### Audio Processing
```python
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor

# Load audio encoder
self.audio_encoder = Wav2Vec2Model.from_pretrained(
    self.wav2vec_dir, 
    local_files_only=True
)
```

### 3. Video Generation Method

#### Core Generation Call
```python
video = self.wan_i2v.generate(
    input_data,
    size_buckget="multitalk-480",    # Resolution bucket
    motion_frame=25,                 # Motion frame rate
    frame_num=num_frames,            # Total frames (default: 81)
    sampling_steps=sampling_steps,   # Diffusion steps (default: 40)
    text_guide_scale=3.0,           # Text guidance scale
    seed=seed,                       # Random seed
    turbo=turbo                      # Turbo mode for faster generation
)
```

#### Input Data Structure
```python
input_data = {
    "image": reference_image,        # PIL Image or tensor
    "audio_embeddings": audio_emb,   # Processed audio features
    "prompt": prompt,                # Text prompt
    "num_frames": num_frames,        # Frame count
    "motion_frame": 25               # Motion frame rate
}
```

### 4. Audio Processing Pipeline

#### Audio Embedding Extraction
```python
# Load audio file
audio_data, sr = sf.read(audio_path)

# Extract features using Wav2Vec2
audio_features = self.audio_encoder.extract_features(audio_data, sr)

# Process for MultiTalk
audio_embeddings = self.audio_encoder.get_embeddings(audio_features)
```

#### Multi-Person Audio Support
```python
# For multiple speakers
if second_audio:
    audio_emb_1 = process_audio(first_audio)
    audio_emb_2 = process_audio(second_audio)
    combined_emb = combine_audio_embeddings(audio_emb_1, audio_emb_2)
```

### 5. Model Loading Sequence

#### Required Models
1. **Wav2Vec2**: Audio feature extraction
2. **MultiTalk**: Motion generation from audio
3. **WAN 2.1**: Image-to-video diffusion model
4. **VAE**: Video encoding/decoding
5. **CLIP**: Text/image embedding

#### Loading Order
```python
def setup(self):
    # 1. Load audio models first
    self.load_wav2vec_models()
    
    # 2. Load MultiTalk pipeline
    self.load_multitalk_pipeline()
    
    # 3. Apply GPU optimizations
    self.apply_gpu_optimizations()
    
    # 4. Warm up models
    self.warmup_models()
```

### 6. Key Configuration Parameters

#### Generation Parameters
```python
DEFAULT_PARAMS = {
    "num_frames": 81,           # Video length (3.24s at 25fps)
    "sampling_steps": 40,       # Diffusion sampling steps
    "motion_frame": 25,         # Motion frame rate
    "text_guide_scale": 3.0,    # Text guidance strength
    "size_buckget": "multitalk-480",  # Resolution (480p)
    "turbo": True              # Fast generation mode
}
```

#### Model Paths
```python
MODEL_PATHS = {
    "multitalk": "/models/MeiGen-MultiTalk",
    "wan21": "/models/Wan2.1-I2V-14B-480P", 
    "wav2vec": "/models/chinese-wav2vec2-base",
    "vae": "/models/Wan2.1_VAE.pth",
    "clip": "/models/clip_model.pth"
}
```

### 7. Error Handling & Fallbacks

#### Common Issues
1. **CUDA Memory**: Use gradient checkpointing
2. **Model Loading**: Check file existence first
3. **Audio Processing**: Validate sample rate (16kHz)
4. **Video Generation**: Monitor GPU memory usage

#### Fallback Strategy
```python
try:
    # Try full quality generation
    video = self.wan_i2v.generate(input_data, **full_params)
except torch.cuda.OutOfMemoryError:
    # Fallback to lower quality
    video = self.wan_i2v.generate(input_data, **reduced_params)
```

### 8. Output Format

#### Video Output
```python
# Generated video is typically:
# - Format: MP4
# - Resolution: 480x480 (configurable)
# - Frame rate: 25 FPS
# - Duration: Based on num_frames (81 frames = ~3.24s)
```

### 9. Dependencies

#### Core ML Libraries
```python
import torch
import numpy as np
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor
from PIL import Image
import torchvision.transforms as transforms
```

#### MeiGen-Specific Imports
```python
import wan
from wan.configs import WAN_CONFIGS
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
```

## Implementation Notes

1. **Model Size**: Total ~25GB of models required
2. **Memory**: Minimum 16GB GPU memory recommended
3. **Performance**: ~30-60 seconds per video on A100
4. **Quality**: 480p resolution, expandable to higher resolutions
5. **Audio**: Supports single or multi-person audio inputs

## Key Differences from Our Current Implementation

1. **Uses `wan.MultiTalkPipeline`** instead of custom implementations
2. **Proper audio embedding extraction** with Wav2Vec2Model
3. **Size bucketing** for different resolutions
4. **Motion frame rate** separate from video frame rate
5. **Turbo mode** for faster generation

## Next Steps for Integration

1. Update model loading to use `wan.MultiTalkPipeline`
2. Implement proper audio embedding extraction
3. Add size bucketing support
4. Integrate turbo mode for faster generation
5. Add multi-person audio support