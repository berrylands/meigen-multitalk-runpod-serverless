"""
MultiTalk Reference Wrapper V150 - Enhanced with graceful error handling
Based on cog-MultiTalk reference implementation with proper dependency management
"""

import os
import sys
import logging
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test imports with graceful error handling
def test_imports():
    """Test and report on all required imports"""
    logger.info("🔍 Testing imports for MultiTalk V150...")
    
    # Essential imports
    try:
        import torch
        import numpy as np
        from PIL import Image
        import soundfile as sf
        import librosa
        import pyloudnorm as pyln
        from einops import rearrange
        logger.info("✅ Essential imports successful")
    except Exception as e:
        logger.error(f"❌ Essential imports failed: {e}")
        raise ImportError(f"Essential imports failed: {e}")
    
    # Add reference implementation to path
    ref_path = '/app/cog_multitalk_reference'
    if not os.path.exists(ref_path):
        raise RuntimeError(f"Reference implementation not found: {ref_path}")
    
    if ref_path not in sys.path:
        sys.path.insert(0, ref_path)
        logger.info(f"✅ Added reference path: {ref_path}")
    
    # Test reference implementation imports
    try:
        import wan
        from wan.configs import WAN_CONFIGS
        from wan.utils.multitalk_utils import save_video_ffmpeg
        logger.info("✅ WAN imports successful")
    except Exception as e:
        logger.error(f"❌ WAN imports failed: {e}")
        raise ImportError(f"WAN imports failed: {e}")
    
    # Test audio processing imports
    try:
        from transformers import Wav2Vec2FeatureExtractor
        from src.audio_analysis.wav2vec2 import Wav2Vec2Model
        logger.info("✅ Audio processing imports successful")
    except Exception as e:
        logger.error(f"❌ Audio processing imports failed: {e}")
        raise ImportError(f"Audio processing imports failed: {e}")
    
    # Test optional optimization imports
    optional_imports = []
    try:
        import xformers
        optional_imports.append("xformers")
        logger.info("✅ XFormers available")
    except:
        logger.warning("⚠️  XFormers not available")
    
    try:
        import xfuser
        optional_imports.append("xfuser")
        logger.info("✅ XFuser available")
    except:
        logger.warning("⚠️  XFuser not available")
    
    try:
        import flash_attn
        optional_imports.append("flash_attn")
        logger.info("✅ Flash Attention available")
    except:
        logger.warning("⚠️  Flash Attention not available")
    
    logger.info(f"Optional optimizations available: {optional_imports}")
    return True

class MultiTalkReferenceWrapper:
    """Enhanced MultiTalk wrapper with graceful error handling"""
    
    def __init__(self):
        logger.info("🔍 Initializing MultiTalk Reference Wrapper V150")
        
        # Test all imports first
        test_imports()
        
        # Import modules after testing
        import torch
        import numpy as np
        from PIL import Image
        import soundfile as sf
        import librosa
        import pyloudnorm as pyln
        from einops import rearrange
        
        # Import reference implementation
        import wan
        from wan.configs import WAN_CONFIGS
        from wan.utils.multitalk_utils import save_video_ffmpeg
        from transformers import Wav2Vec2FeatureExtractor
        from src.audio_analysis.wav2vec2 import Wav2Vec2Model
        
        # Store imports as instance variables
        self.torch = torch
        self.np = np
        self.Image = Image
        self.sf = sf
        self.librosa = librosa
        self.pyln = pyln
        self.rearrange = rearrange
        self.wan = wan
        self.WAN_CONFIGS = WAN_CONFIGS
        self.save_video_ffmpeg = save_video_ffmpeg
        self.Wav2Vec2FeatureExtractor = Wav2Vec2FeatureExtractor
        self.Wav2Vec2Model = Wav2Vec2Model
        
        # Model paths on RunPod network storage
        self.model_base = "/runpod-volume/models"
        self.ckpt_dir = f"{self.model_base}/wan2.1-i2v-14b-480p"
        self.wav2vec_dir = f"{self.model_base}/wav2vec2"
        
        # Validate models exist
        self._validate_models()
        
        # Initialize device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models with error handling
        self._load_models()
        
        logger.info("✅ MultiTalk Reference Wrapper V150 initialized successfully")
    
    def _validate_models(self):
        """Validate model paths exist"""
        logger.info("🔍 Validating model paths...")
        
        paths_to_check = {
            "Network volume": "/runpod-volume",
            "Models base": self.model_base,
            "Wan2.1 checkpoint": self.ckpt_dir,
            "Wav2Vec2 model": self.wav2vec_dir
        }
        
        missing_paths = []
        for name, path in paths_to_check.items():
            if os.path.exists(path):
                logger.info(f"✅ {name} found: {path}")
            else:
                logger.error(f"❌ {name} not found: {path}")
                missing_paths.append(f"{name}: {path}")
        
        if missing_paths:
            error_msg = f"Missing model paths: {', '.join(missing_paths)}"
            raise RuntimeError(error_msg)
    
    def _load_models(self):
        """Load all required models with enhanced error handling"""
        logger.info("🔄 Loading models with enhanced error handling...")
        
        try:
            # Determine audio device based on VRAM
            vram_gb = 0
            if self.torch.cuda.is_available():
                vram_gb = self.torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"🔍 Detected {vram_gb:.1f}GB VRAM")
            
            # Load wav2vec models
            logger.info("Loading Wav2Vec2 models...")
            audio_device = self.device if vram_gb > 40 else 'cpu'
            logger.info(f"Loading audio encoder on: {audio_device}")
            
            try:
                self.audio_encoder = self.Wav2Vec2Model.from_pretrained(
                    self.wav2vec_dir,
                    local_files_only=True,
                    attn_implementation="eager"
                ).to(audio_device)
                self.audio_encoder.feature_extractor._freeze_parameters()
                logger.info("✅ Wav2Vec2 model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load Wav2Vec2 model: {e}")
                raise RuntimeError(f"Failed to load Wav2Vec2 model: {e}")
            
            try:
                self.wav2vec_feature_extractor = self.Wav2Vec2FeatureExtractor.from_pretrained(
                    self.wav2vec_dir,
                    local_files_only=True
                )
                self.audio_device = audio_device
                logger.info("✅ Wav2Vec2 feature extractor loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load Wav2Vec2 feature extractor: {e}")
                raise RuntimeError(f"Failed to load Wav2Vec2 feature extractor: {e}")
            
            # Load MultiTalk pipeline
            logger.info("Loading MultiTalk pipeline...")
            try:
                self.cfg = self.WAN_CONFIGS["multitalk-14B"]
                self.wan_i2v = self.wan.MultiTalkPipeline(
                    config=self.cfg,
                    checkpoint_dir=self.ckpt_dir,
                    device_id=0,
                    rank=0,
                    t5_fsdp=False,
                    dit_fsdp=False,
                    use_usp=False,
                    t5_cpu=False  # Keep T5 on GPU for speed
                )
                logger.info("✅ MultiTalk pipeline loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load MultiTalk pipeline: {e}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Failed to load MultiTalk pipeline: {e}")
            
            # Enable GPU optimizations if available
            if self.torch.cuda.is_available():
                try:
                    if vram_gb > 40:  # High VRAM setup
                        logger.info("🚀 High-VRAM detected: Enabling maximum performance optimizations")
                        self.torch.backends.cuda.enable_flash_sdp(True)
                        self.torch.backends.cudnn.benchmark = True
                        self.torch.backends.cuda.matmul.allow_tf32 = True
                        self.torch.backends.cudnn.allow_tf32 = True
                    else:
                        logger.info("🔧 Standard GPU optimizations enabled")
                        self.torch.backends.cuda.enable_flash_sdp(True)
                    
                    self.torch.cuda.empty_cache()
                    logger.info("✅ GPU optimizations enabled")
                except Exception as e:
                    logger.warning(f"⚠️  GPU optimizations failed: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def loudness_norm(self, audio_array, sr=16000, lufs=-23):
        """Normalize audio loudness with error handling"""
        try:
            meter = self.pyln.Meter(sr)
            loudness = meter.integrated_loudness(audio_array)
            if abs(loudness) > 100:
                logger.warning("⚠️  Audio loudness out of range, skipping normalization")
                return audio_array
            normalized_audio = self.pyln.normalize.loudness(audio_array, loudness, lufs)
            return normalized_audio
        except Exception as e:
            logger.warning(f"⚠️  Audio normalization failed: {e}, using original audio")
            return audio_array
    
    def audio_prepare_single(self, audio_path, sample_rate=16000):
        """Prepare single audio file with error handling"""
        try:
            human_speech_array, sr = self.librosa.load(audio_path, sr=sample_rate)
            human_speech_array = self.loudness_norm(human_speech_array, sr)
            return human_speech_array
        except Exception as e:
            logger.error(f"❌ Audio preparation failed: {e}")
            raise RuntimeError(f"Audio preparation failed: {e}")
    
    def get_embedding(self, speech_array, sr=16000):
        """Extract audio embeddings with error handling"""
        try:
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
            audio_embedding = self.rearrange(audio_embedding, 'b d n -> b n d')
            
            return audio_embedding
        except Exception as e:
            logger.error(f"❌ Audio embedding extraction failed: {e}")
            raise RuntimeError(f"Audio embedding extraction failed: {e}")
    
    def generate(self, audio_path: str, image_path: str, output_path: str,
                 prompt: str = "A person talking naturally with expressive lip sync",
                 num_frames: Optional[int] = None,
                 sampling_steps: int = 40,
                 text_guidance: float = 7.5,
                 audio_guidance: float = 3.5,
                 seed: int = 42,
                 turbo: bool = False) -> str:
        """Generate video with comprehensive error handling"""
        try:
            logger.info("="*80)
            logger.info("🎬 Starting video generation with V150 enhanced wrapper")
            logger.info(f"Audio: {audio_path}")
            logger.info(f"Image: {image_path}")
            logger.info(f"Output: {output_path}")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Turbo mode: {turbo}")
            logger.info("="*80)
            
            # Set random seeds
            self.torch.manual_seed(seed)
            self.np.random.seed(seed)
            
            # Load and prepare audio
            logger.info("🎵 Processing audio...")
            audio_array = self.audio_prepare_single(audio_path)
            audio_duration = len(audio_array) / 16000
            
            # Calculate frames if not specified
            if num_frames is None:
                raw_frames = int(audio_duration * 25)
                num_frames = ((raw_frames + 2) // 4) * 4 + 1
                num_frames = max(25, min(num_frames, 201))
                logger.info(f"Auto-calculated {num_frames} frames for {audio_duration:.2f}s audio")
            
            # Get audio embedding
            logger.info("🔊 Extracting audio embeddings...")
            audio_embedding = self.get_embedding(audio_array)
            
            # Load reference image
            logger.info("🖼️ Loading reference image...")
            image = self.Image.open(image_path).convert("RGB")
            
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
            
            # Run generation with error handling
            try:
                with self.torch.no_grad():
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
                        generator=self.torch.Generator(device=self.device).manual_seed(seed),
                        output_type="tensor",
                        save_memory=False,
                        cpu_offloading=False,
                        inference_multigpu=False,
                        enable_teacache=enable_teacache,
                        save_path=None,
                        output_format="gif",
                        log_time=True,
                    )
                logger.info("✅ Video generation completed successfully")
            except Exception as e:
                logger.error(f"❌ Video generation failed: {e}")
                raise RuntimeError(f"Video generation failed: {e}")
            
            # Extract video tensor
            try:
                video_tensor = results['video'][0]  # [F, C, H, W]
                
                # Convert to numpy and prepare for saving
                video_np = video_tensor.cpu().numpy()
                video_np = (video_np * 255).astype(self.np.uint8)
                video_np = video_np.transpose(0, 2, 3, 1)  # [F, H, W, C]
                logger.info("✅ Video tensor processed successfully")
            except Exception as e:
                logger.error(f"❌ Video tensor processing failed: {e}")
                raise RuntimeError(f"Video tensor processing failed: {e}")
            
            # Save video using ffmpeg
            try:
                logger.info(f"💾 Saving video to {output_path}...")
                self.save_video_ffmpeg(
                    video_np,
                    output_path,
                    fps=25,
                    crf=18
                )
                logger.info("✅ Video saved successfully")
            except Exception as e:
                logger.error(f"❌ Video saving failed: {e}")
                raise RuntimeError(f"Video saving failed: {e}")
            
            # Verify output
            if not os.path.exists(output_path):
                raise RuntimeError(f"Video file not created: {output_path}")
            
            file_size = os.path.getsize(output_path)
            logger.info(f"✅ Video generation complete: {output_path} ({file_size/1024/1024:.2f} MB)")
            
            return output_path
            
        except Exception as e:
            error_msg = f"V150 generation failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def generate_with_options(self, audio_path: str, image_path: str, output_path: str,
                            prompt: str = "A person talking naturally with expressive lip sync",
                            num_frames: Optional[int] = None,
                            sample_steps: int = 40,
                            mode: str = "clip",
                            size: str = "multitalk-480",
                            teacache: bool = False,
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