"""
MultiTalk Reference Wrapper - Direct API implementation
Based on cog-MultiTalk reference implementation
NO mock, dummy, placeholder, demo or fallback code
"""

import os
import sys
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import numpy as np
from PIL import Image
import soundfile as sf
import librosa
import pyloudnorm as pyln
from einops import rearrange

# Add reference implementation to path
sys.path.insert(0, '/app/cog_multitalk_reference')

# Import reference implementation components
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.multitalk_utils import save_video_ffmpeg
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTalkReferenceWrapper:
    """Direct API wrapper for MultiTalk reference implementation"""
    
    def __init__(self):
        logger.info("🔍 Initializing MultiTalk Reference Wrapper")
        
        # Model paths on RunPod network storage
        self.model_base = "/runpod-volume/models"
        self.ckpt_dir = f"{self.model_base}/wan2.1-i2v-14b-480p"
        self.wav2vec_dir = f"{self.model_base}/wav2vec2"
        
        # Validate models exist
        self._validate_models()
        
        # Initialize device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self._load_models()
        
        logger.info("✅ MultiTalk Reference Wrapper initialized")
    
    def _validate_models(self):
        """Validate model paths exist"""
        paths_to_check = {
            "Wan2.1 checkpoint": self.ckpt_dir,
            "Wav2Vec2 model": self.wav2vec_dir
        }
        
        for name, path in paths_to_check.items():
            if os.path.exists(path):
                logger.info(f"✅ {name} found: {path}")
            else:
                raise RuntimeError(f"❌ {name} not found: {path}")
    
    def _load_models(self):
        """Load all required models"""
        # Determine audio device based on VRAM
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"🔍 Detected {vram_gb:.1f}GB VRAM")
        
        # Load wav2vec models
        logger.info("Loading Wav2Vec2 models...")
        audio_device = self.device if vram_gb > 40 else 'cpu'
        logger.info(f"Loading audio encoder on: {audio_device}")
        
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            self.wav2vec_dir,
            local_files_only=True,
            attn_implementation="eager"
        ).to(audio_device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.wav2vec_dir,
            local_files_only=True
        )
        self.audio_device = audio_device
        
        # Load MultiTalk pipeline
        logger.info("Loading MultiTalk pipeline...")
        self.cfg = WAN_CONFIGS["multitalk-14B"]
        self.wan_i2v = wan.MultiTalkPipeline(
            config=self.cfg,
            checkpoint_dir=self.ckpt_dir,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False  # Keep T5 on GPU for speed
        )
        
        # Enable GPU optimizations
        if torch.cuda.is_available():
            if vram_gb > 40:  # High VRAM setup
                logger.info("🚀 High-VRAM detected: Enabling maximum performance optimizations")
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.cuda.empty_cache()
            else:
                logger.info("🔧 Standard GPU optimizations enabled")
                torch.backends.cuda.enable_flash_sdp(True)
                torch.cuda.empty_cache()
    
    def loudness_norm(self, audio_array, sr=16000, lufs=-23):
        """Normalize audio loudness"""
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio
    
    def audio_prepare_single(self, audio_path, sample_rate=16000):
        """Prepare single audio file"""
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = self.loudness_norm(human_speech_array, sr)
        return human_speech_array
    
    def get_embedding(self, speech_array, sr=16000):
        """Extract audio embeddings"""
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * 25  # Assume 25 fps
        
        # Extract audio features
        inputs = self.wav2vec_feature_extractor(
            speech_array, 
            sampling_rate=sr, 
            return_tensors="pt", 
            padding="longest"
        )
        
        # Extract embeddings
        input_values = inputs['input_values'].to(self.audio_device)
        audio_embedding = self.audio_encoder(
            input_values, 
            video_length=video_length, 
            return_dict=False
        )[0]
        
        # Process embeddings
        audio_embedding = audio_embedding.permute((0, 2, 1))
        audio_embedding = rearrange(audio_embedding, 'b d n -> b n d')
        
        return audio_embedding
    
    def generate(self, audio_path: str, image_path: str, output_path: str,
                 prompt: str = "A person talking naturally with expressive lip sync",
                 num_frames: Optional[int] = None,
                 sampling_steps: int = 40,
                 text_guidance: float = 7.5,
                 audio_guidance: float = 3.5,
                 seed: int = 42,
                 turbo: bool = True) -> str:
        """Generate video using direct API calls to reference implementation"""
        try:
            logger.info("="*80)
            logger.info("🎬 Starting video generation with reference implementation")
            logger.info(f"Audio: {audio_path}")
            logger.info(f"Image: {image_path}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Turbo mode: {turbo}")
            logger.info("="*80)
            
            # Set random seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Load and prepare audio
            logger.info("🎵 Processing audio...")
            audio_array = self.audio_prepare_single(audio_path)
            audio_duration = len(audio_array) / 16000
            
            # Calculate frames if not specified
            if num_frames is None:
                # 25 fps, round to nearest 4n+1
                raw_frames = int(audio_duration * 25)
                num_frames = ((raw_frames + 2) // 4) * 4 + 1
                num_frames = max(25, min(num_frames, 201))
                logger.info(f"Auto-calculated {num_frames} frames for {audio_duration:.2f}s audio")
            
            # Get audio embedding
            audio_embedding = self.get_embedding(audio_array)
            
            # Load reference image
            logger.info("🖼️ Loading reference image...")
            image = Image.open(image_path).convert("RGB")
            
            # Generate video
            logger.info(f"🎬 Generating video with {sampling_steps} steps...")
            
            # Prepare inputs for pipeline
            text_inputs = [prompt]
            images_inputs = [image]
            audio_embeddings = [audio_embedding]
            
            # Set size based on pipeline configuration
            size = "multitalk-480"
            
            # TeaCache settings
            enable_teacache = turbo
            
            # Run generation
            with torch.no_grad():
                results = self.wan_i2v.generate(
                    prompt=text_inputs,
                    negative_prompt=[""],
                    image=images_inputs,
                    audio_embeddings=audio_embeddings,
                    height=None,
                    width=None,
                    size=size,
                    custom_resolution=None,
                    num_frames=num_frames,
                    num_inference_steps=sampling_steps,
                    video_guidance_scale=text_guidance,
                    audio_guidance_scale=audio_guidance,
                    generator=torch.Generator(device=self.device).manual_seed(seed),
                    output_type="tensor",
                    save_memory=False,
                    cpu_offloading=False,
                    inference_multigpu=False,
                    enable_teacache=enable_teacache,
                    save_path=None,
                    output_format="gif",
                    log_time=True,
                )
            
            # Extract video tensor
            video_tensor = results['video'][0]  # [F, C, H, W]
            
            # Convert to numpy and prepare for saving
            video_np = video_tensor.cpu().numpy()
            video_np = (video_np * 255).astype(np.uint8)
            video_np = video_np.transpose(0, 2, 3, 1)  # [F, H, W, C]
            
            # Save video using ffmpeg
            logger.info(f"💾 Saving video to {output_path}...")
            save_video_ffmpeg(
                video_np,
                output_path,
                fps=25,
                crf=18
            )
            
            # Verify output
            if not os.path.exists(output_path):
                raise RuntimeError(f"Failed to save video to {output_path}")
            
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ Video saved: {output_path} ({file_size/1024/1024:.2f} MB)")
            
            return output_path
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def generate_with_options(self, audio_path: str, image_path: str, output_path: str,
                            prompt: str = "A person talking naturally with expressive lip sync",
                            num_frames: Optional[int] = None,
                            sample_steps: int = 40,
                            mode: str = "clip",
                            size: str = "multitalk-480",
                            teacache: bool = True,
                            text_guidance: float = 7.5,
                            audio_guidance: float = 3.5,
                            seed: int = 42) -> str:
        """Compatibility method for existing handler interface"""
        return self.generate(
            audio_path=audio_path,
            image_path=image_path,
            output_path=output_path,
            prompt=prompt,
            num_frames=num_frames,
            sampling_steps=sample_steps,
            text_guidance=text_guidance,
            audio_guidance=audio_guidance,
            seed=seed,
            turbo=teacache
        )