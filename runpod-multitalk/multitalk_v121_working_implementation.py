#!/usr/bin/env python3
"""
MultiTalk V121 Working Implementation
Based on the proven working code from https://github.com/zsxkib/cog-MultiTalk

This implementation exactly matches the working cog-MultiTalk implementation
"""

import os
import sys
import json
import torch
import numpy as np
import tempfile
import logging
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import traceback
import time
from PIL import Image
import librosa
import pyloudnorm as pyln
from einops import rearrange

# Add paths for MeiGen-MultiTalk
sys.path.insert(0, '/app/multitalk_official')
sys.path.insert(0, '/app')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def loudness_norm(audio_array, sr=16000, target_loudness=-20.0):
    """Normalize audio loudness to target level"""
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    
    if loudness == -np.inf:
        return audio_array
    
    loudness_normalized_audio = pyln.normalize.loudness(audio_array, loudness, target_loudness)
    return loudness_normalized_audio

def audio_prepare_single(audio_path: str, sr: int = 16000, target_loudness: float = -20.0) -> np.ndarray:
    """Prepare single audio file for processing"""
    logger.info(f"Loading audio: {audio_path}")
    
    # Load audio
    audio, orig_sr = librosa.load(audio_path, sr=None)
    
    # Resample if needed
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    
    # Normalize loudness
    audio = loudness_norm(audio, sr=sr, target_loudness=target_loudness)
    
    logger.info(f"Audio prepared: {len(audio)} samples at {sr}Hz")
    return audio

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    """Extract audio embeddings optimized for GPU processing"""
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25  # Assume the video fps is 25

    # Extract audio features
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # Generate embeddings on appropriate device
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    return audio_emb.cpu().detach()

def validate_frame_count(num_frames: int) -> int:
    """Validate and correct frame count to 4n+1 format"""
    if (num_frames - 1) % 4 != 0:
        # Find the nearest valid values
        n_lower = (num_frames - 1) // 4
        n_upper = n_lower + 1
        
        frames_lower = 4 * n_lower + 1
        frames_upper = 4 * n_upper + 1
        
        # Choose the closer one
        if abs(num_frames - frames_lower) <= abs(num_frames - frames_upper):
            corrected_frames = frames_lower
        else:
            corrected_frames = frames_upper
        
        logger.info(f"Frame count corrected from {num_frames} to {corrected_frames} (4n+1 format)")
        return corrected_frames
    
    return num_frames

class MultiTalkV121Working:
    """
    Working MeiGen-MultiTalk Implementation V121
    Based on the proven cog-MultiTalk implementation
    """
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model paths (matching working implementation)
        self.ckpt_dir = self.model_path / "Wan2.1-I2V-14B-480P"
        self.wav2vec_dir = self.model_path / "chinese-wav2vec2-base"
        self.multitalk_dir = self.model_path / "MeiGen-MultiTalk"
        
        # Model components
        self.wan_i2v = None
        self.wav2vec_feature_extractor = None
        self.audio_encoder = None
        self.cfg = None
        
        # Check available models
        self.models_available = self._check_models()
        
        # GPU optimizations
        self._setup_gpu_optimizations()
        
        logger.info(f"MultiTalk V121 Working initialized on {self.device}")
        logger.info(f"Models available: {self.models_available}")
    
    def _check_models(self) -> Dict[str, bool]:
        """Check which models are available"""
        models = {
            "wan2.1": self.ckpt_dir.exists(),
            "wav2vec": self.wav2vec_dir.exists(),
            "multitalk": self.multitalk_dir.exists()
        }
        return models
    
    def _setup_gpu_optimizations(self):
        """Setup GPU optimizations like the working implementation"""
        if torch.cuda.is_available():
            # Get VRAM amount
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"VRAM available: {vram_gb:.1f}GB")
            
            # High VRAM optimizations
            if vram_gb > 40:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✓ High VRAM optimizations enabled")
            
            # Enable memory efficient attention
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("✓ Flash attention enabled")
            except:
                logger.info("Flash attention not available")
    
    def load_models(self) -> bool:
        """Load models using the exact working implementation approach"""
        try:
            logger.info("Loading models using working MeiGen-MultiTalk implementation...")
            
            # Step 1: Load WAV2VEC2 components
            self._load_wav2vec_models()
            
            # Step 2: Load MultiTalk pipeline
            self._load_multitalk_pipeline()
            
            logger.info("✓ All models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            traceback.print_exc()
            return False
    
    def _load_wav2vec_models(self):
        """Load WAV2VEC2 models exactly as in working implementation"""
        try:
            logger.info("Loading WAV2VEC2 models...")
            
            # Import required components
            from transformers import Wav2Vec2FeatureExtractor
            from src.audio_analysis.wav2vec2 import Wav2Vec2Model
            
            # Load feature extractor
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                str(self.wav2vec_dir),
                local_files_only=True
            )
            
            # Load audio encoder
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                str(self.wav2vec_dir),
                local_files_only=True
            ).to(self.device)
            
            logger.info("✓ WAV2VEC2 models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load WAV2VEC2 models: {e}")
            raise
    
    def _load_multitalk_pipeline(self):
        """Load MultiTalk pipeline exactly as in working implementation"""
        try:
            logger.info("Loading MultiTalk pipeline...")
            
            # Import required components
            import wan
            from wan.configs import WAN_CONFIGS
            
            # Get configuration
            self.cfg = WAN_CONFIGS["multitalk-14B"]
            
            # Initialize pipeline exactly as working implementation
            self.wan_i2v = wan.MultiTalkPipeline(
                config=self.cfg,
                checkpoint_dir=str(self.ckpt_dir),
                device_id=0,
                rank=0,
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=False  # Keep T5 on GPU for speed
            )
            
            logger.info("✓ MultiTalk pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load MultiTalk pipeline: {e}")
            raise
    
    def generate_video(
        self,
        audio_path: str,
        image_path: str,
        prompt: str = "A person talking naturally with expressive facial movements",
        num_frames: int = 81,
        sampling_steps: int = 30,
        seed: Optional[int] = None,
        turbo: bool = True,
        text_guide_scale: float = 3.0,
        audio_guide_scale: float = 3.0,
        shift: float = 5.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using the exact working implementation approach
        """
        try:
            logger.info(f"Generating video with Working MultiTalk V121...")
            logger.info(f"Audio: {audio_path}")
            logger.info(f"Image: {image_path}")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Frames: {num_frames}, Steps: {sampling_steps}")
            
            # Validate models are loaded
            if not self.wan_i2v or not self.audio_encoder or not self.wav2vec_feature_extractor:
                raise Exception("Models not loaded - call load_models() first")
            
            # Set random seed
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Validate and correct frame count
            num_frames = validate_frame_count(num_frames)
            
            # Prepare audio
            audio_array = audio_prepare_single(audio_path, sr=16000)
            
            # Create temp directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save processed audio
                audio_file = temp_path / "processed_audio.wav"
                sf.write(str(audio_file), audio_array, 16000)
                
                # Extract audio embeddings
                logger.info("Extracting audio embeddings...")
                audio_emb = get_embedding(
                    audio_array,
                    self.wav2vec_feature_extractor,
                    self.audio_encoder,
                    sr=16000,
                    device=self.device
                )
                
                # Save embeddings
                emb_file = temp_path / "audio_embeddings.pt"
                torch.save(audio_emb, str(emb_file))
                
                # Prepare input data (exact format from working implementation)
                input_data = {
                    "prompt": prompt,
                    "cond_image": str(image_path),
                    "cond_audio": {
                        "person1": str(emb_file)
                    },
                    "video_audio": str(audio_file)
                }
                
                # Set turbo mode parameters
                if turbo:
                    teacache_thresh = 0.8
                    text_guide_scale = 3.0
                    audio_guide_scale = 3.0
                    shift = 5.0
                    offload_model = False
                else:
                    teacache_thresh = 0.0
                    offload_model = True
                
                # Extra generation arguments
                extra_args = {
                    "teacache_thresh": teacache_thresh
                }
                
                logger.info("Generating video with MultiTalk pipeline...")
                
                # Generate video using exact working implementation call
                video = self.wan_i2v.generate(
                    input_data,
                    size_buckget="multitalk-480",  # Important: size bucketing
                    motion_frame=25,
                    frame_num=num_frames,
                    shift=shift,
                    sampling_steps=sampling_steps,
                    text_guide_scale=text_guide_scale,
                    audio_guide_scale=audio_guide_scale,
                    seed=seed,
                    offload_model=offload_model,
                    max_frames_num=num_frames,
                    extra_args=extra_args
                )
                
                # Save video using MultiTalk utils
                from wan.utils.multitalk_utils import save_video_ffmpeg
                
                output_path = tempfile.mktemp(suffix='.mp4')
                save_video_ffmpeg(video, output_path)
                
                # Verify video was created
                if not os.path.exists(output_path):
                    raise Exception(f"Video file not created: {output_path}")
                
                video_size = os.path.getsize(output_path)
                logger.info(f"✓ Video generated successfully: {output_path} ({video_size} bytes)")
                
                return {
                    "success": True,
                    "video_path": output_path,
                    "video_size": video_size,
                    "num_frames": num_frames,
                    "sampling_steps": sampling_steps,
                    "resolution": "480x480",
                    "implementation": "Working MeiGen-MultiTalk V121",
                    "turbo_mode": turbo,
                    "audio_embeddings_shape": list(audio_emb.shape)
                }
                
        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "implementation": "Working MeiGen-MultiTalk V121"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "version": "V121-Working",
            "implementation": "Working MeiGen-MultiTalk (based on cog-MultiTalk)",
            "device": str(self.device),
            "models_available": self.models_available,
            "models_loaded": {
                "wan_i2v": self.wan_i2v is not None,
                "audio_encoder": self.audio_encoder is not None,
                "wav2vec_feature_extractor": self.wav2vec_feature_extractor is not None
            },
            "model_paths": {
                "wan2.1": str(self.ckpt_dir),
                "wav2vec": str(self.wav2vec_dir),
                "multitalk": str(self.multitalk_dir)
            }
        }
    
    def initialize_models(self) -> bool:
        """Initialize all models (convenience method)"""
        return self.load_models()

# Alias for compatibility
MultiTalkV121 = MultiTalkV121Working