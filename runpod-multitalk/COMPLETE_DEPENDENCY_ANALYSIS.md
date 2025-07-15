# Complete MeiGen MultiTalk Dependency Analysis

## Overview
This document provides a comprehensive analysis of all dependencies required to run generate_multitalk.py and its related modules.

## 1. Core Import Tree from generate_multitalk.py

### Standard Library Imports
- argparse
- logging
- os
- sys
- json
- warnings
- random
- subprocess
- re
- datetime

### Third-Party Package Imports
- torch
- torch.distributed
- PIL (pillow)
- librosa
- pyloudnorm
- numpy
- einops
- soundfile
- transformers (specifically Wav2Vec2FeatureExtractor)

### Local Package Imports
1. **wan package**
   - wan (main package)
   - wan.configs (SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS)
   - wan.utils.utils (cache_image, cache_video, str2bool)
   - wan.utils.multitalk_utils (save_video_ffmpeg)

2. **kokoro package**
   - kokoro.KPipeline

3. **src package**
   - src.audio_analysis.wav2vec2.Wav2Vec2Model

## 2. Dependency Chain Analysis

### src.audio_analysis.wav2vec2
**Dependencies:**
- transformers.Wav2Vec2Config
- transformers.Wav2Vec2Model
- transformers.modeling_outputs.BaseModelOutput
- src.audio_analysis.torch_utils.linear_interpolation

### src.audio_analysis.torch_utils
**Dependencies:**
- torch
- torch.nn.functional

### wan.multitalk.MultiTalkPipeline
**Dependencies:**
- Standard: gc, logging, json, math, importlib, os, random, sys, types, contextlib, functools
- Third-party: numpy, torch, torchvision.transforms, tqdm, safetensors.torch, optimum.quanto
- Local wan modules:
  - wan.distributed.fsdp
  - wan.modules.clip.CLIPModel
  - wan.modules.multitalk_model (WanModel, WanLayerNorm, WanRMSNorm)
  - wan.modules.t5 (T5EncoderModel, T5LayerNorm, T5RelativeEmbedding)
  - wan.modules.vae (WanVAE, CausalConv3d, RMS_norm, Upsample)
  - wan.utils.multitalk_utils (MomentumBuffer, adaptive_projected_guidance, match_and_blend_colors)
  - wan.wan_lora.WanLoraWrapper
- Local src modules:
  - src.vram_management (AutoWrappedQLinear, AutoWrappedLinear, AutoWrappedModule, enable_vram_management)

### kokoro.KPipeline
**Dependencies:**
- loguru (for logging)
- Local kokoro modules (model, pipeline)

## 3. Complete File Structure Required

```
MultiTalk/
├── generate_multitalk.py
├── requirements.txt
├── src/
│   ├── audio_analysis/
│   │   ├── __init__.py (may be empty)
│   │   ├── wav2vec2.py
│   │   └── torch_utils.py
│   ├── vram_management/
│   │   └── (various vram management modules)
│   └── utils.py
├── wan/
│   ├── __init__.py
│   ├── first_last_frame2video.py
│   ├── image2video.py
│   ├── multitalk.py
│   ├── text2video.py
│   ├── vace.py
│   ├── wan_lora.py
│   ├── configs/
│   │   ├── __init__.py
│   │   ├── shared_config.py
│   │   ├── wan_i2v_14B.py
│   │   ├── wan_multitalk_14B.py
│   │   ├── wan_t2v_14B.py
│   │   └── wan_t2v_1_3B.py
│   ├── distributed/
│   │   └── fsdp.py (and other distributed modules)
│   ├── modules/
│   │   ├── clip.py
│   │   ├── multitalk_model.py
│   │   ├── t5.py
│   │   └── vae.py
│   └── utils/
│       ├── __init__.py
│       ├── fm_solvers.py
│       ├── fm_solvers_unipc.py
│       ├── multitalk_utils.py
│       ├── prompt_extend.py
│       ├── qwen_vl_utils.py
│       ├── utils.py
│       └── vace_processor.py
└── kokoro/
    ├── __init__.py
    ├── __main__.py
    ├── custom_stft.py
    ├── istftnet.py
    ├── model.py
    ├── modules.py
    └── pipeline.py
```

## 4. Third-Party Dependencies (from requirements.txt)

```
opencv-python>=4.9.0.80
diffusers>=0.31.0
transformers>=4.49.0
tokenizers>=0.20.3
accelerate>=1.1.1
tqdm
imageio
easydict
ftfy
dashscope
imageio-ffmpeg
scikit-image
loguru
gradio>=5.0.0
numpy>=1.23.5,<2
xfuser>=0.4.1
pyloudnorm
optimum-quanto==0.2.6
```

## 5. Additional Dependencies Found in Code
- einops
- soundfile
- librosa
- Pillow (PIL)
- safetensors
- torchvision

## 6. Missing from Current Setup

The current setup is missing:
1. The entire `src/audio_analysis/` directory structure
2. The `src/vram_management/` modules
3. Many `wan/modules/` files
4. The `wan/distributed/` directory
5. Several utility files in `wan/utils/`
6. Most of the `kokoro/` package files

## 7. Critical Missing Components

The most critical missing component causing the current error is:
- `src/audio_analysis/__init__.py` (even if empty)
- `src/audio_analysis/wav2vec2.py`
- `src/audio_analysis/torch_utils.py`