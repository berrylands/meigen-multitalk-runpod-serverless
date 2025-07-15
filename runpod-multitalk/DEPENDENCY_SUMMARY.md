# MeiGen MultiTalk Dependency Summary

## Key Findings

### 1. Missing Audio Analysis Module
The immediate error `ModuleNotFoundError: No module named 'src.audio_analysis'` is caused by missing files:
- `src/__init__.py` (can be empty)
- `src/audio_analysis/__init__.py` (can be empty)
- `src/audio_analysis/wav2vec2.py`
- `src/audio_analysis/torch_utils.py`

### 2. Complete Package Structure
The MultiTalk repository has a complex structure with these main packages:
- **src/**: Contains audio_analysis, vram_management, and utils
- **wan/**: The main Wan model package with configs, modules, utils, and distributed components
- **kokoro/**: Audio processing pipeline

### 3. Third-Party Dependencies
From requirements.txt and code analysis, these packages are needed:
```
# From requirements.txt
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

# Additional from code
einops
soundfile
librosa
Pillow
safetensors
torchvision
```

### 4. VRAM Management
The repository includes sophisticated VRAM management in `src/vram_management/`:
- `layers.py`: Contains AutoWrappedModule, AutoWrappedQLinear, AutoWrappedLinear
- Supports low-VRAM mode with `--num_persistent_param_in_dit 0`

### 5. Quick Fix
To fix the immediate issue, run:
```bash
python fix_audio_analysis.py
```

### 6. Complete Setup
For a complete setup with all dependencies:
```bash
python setup_complete_multitalk.py
```

### 7. Model Weights
Note that model weights are NOT included in the GitHub repository and must be downloaded separately from Hugging Face or other sources.

## Recommended Steps

1. **Quick Fix** (for immediate error):
   ```bash
   python fix_audio_analysis.py
   ```

2. **Complete Setup** (for full functionality):
   ```bash
   python setup_complete_multitalk.py
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install einops soundfile librosa Pillow safetensors torchvision
   ```

4. **Download Model Weights** (separate process):
   - Wan2.1 models
   - Wav2Vec2 models
   - Other required weights

## Important Notes

- The repository uses optimum.quanto for quantization to reduce VRAM usage
- TeaCache acceleration can speed up inference by 2-3x
- Low VRAM mode allows running on RTX 4090 with 480P video generation
- Multi-person generation requires significantly more VRAM than single-person