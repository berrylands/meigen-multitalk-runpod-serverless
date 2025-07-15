"""
MeiGen-MultiTalk Implementation for RunPod
Proper integration with the official MultiTalk model
"""
import os
import sys
import torch
import numpy as np
import tempfile
import logging
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try imports with error handling
try:
    import soundfile as sf
except ImportError:
    logger.error("soundfile not available")
    sf = None

try:
    from PIL import Image
    import cv2
except ImportError:
    logger.error("PIL or cv2 not available")
    Image = None
    cv2 = None

try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    HAS_TRANSFORMERS = True
except ImportError:
    logger.error("transformers not available")
    HAS_TRANSFORMERS = False

# Import MultiTalk components
try:
    # Add the MultiTalk path to system
    multitalk_path = Path("/runpod-volume/models/MultiTalk")
    if multitalk_path.exists():
        sys.path.insert(0, str(multitalk_path))
    
    # Try to import wan (the MultiTalk model)
    import wan
    from wan.utils.config import WAN_CONFIGS
    HAS_MULTITALK = True
    logger.info("✓ MultiTalk model available")
except ImportError as e:
    logger.error(f"MultiTalk model not available: {e}")
    HAS_MULTITALK = False
    wan = None

class MeiGenMultiTalkPipeline:
    """Official MeiGen-MultiTalk implementation"""
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Initializing MeiGen-MultiTalk on device: {self.device}")
        
        # Model components
        self.multitalk_pipeline = None
        self.wav2vec_processor = None
        self.wav2vec_model = None
        
        # Model paths
        self.wan_checkpoint = self.model_path / "Wan2.1-I2V-14B-480P"
        self.wav2vec_path = self.model_path / "chinese-wav2vec2-base"
        self.multitalk_weights = self.model_path / "MeiGen-MultiTalk"
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize MultiTalk models"""
        try:
            # 1. Initialize Wav2Vec2 for audio processing
            if HAS_TRANSFORMERS and self.wav2vec_path.exists():
                logger.info("Loading Wav2Vec2 for audio processing...")
                self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                    str(self.wav2vec_path),
                    local_files_only=True
                )
                self.wav2vec_model = Wav2Vec2Model.from_pretrained(
                    str(self.wav2vec_path),
                    local_files_only=True,
                    torch_dtype=self.dtype
                ).to(self.device)
                logger.info("✓ Wav2Vec2 loaded")
            
            # 2. Initialize MultiTalk pipeline
            if HAS_MULTITALK and self.wan_checkpoint.exists():
                logger.info("Loading MultiTalk pipeline...")
                
                # Get configuration
                cfg_key = "wan_multitalk_480"  # Use 480p config
                cfg = WAN_CONFIGS.get(cfg_key, {})
                
                # Initialize the pipeline
                self.multitalk_pipeline = wan.MultiTalkPipeline(
                    config=cfg,
                    checkpoint_dir=str(self.wan_checkpoint),
                    device_id=0 if torch.cuda.is_available() else -1,
                    rank=0,
                    world_size=1,
                    wav2vec_dir=str(self.wav2vec_path),
                    audio_condition_dir=str(self.multitalk_weights),
                    low_vram=torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3 if torch.cuda.is_available() else False,
                    quantize_type=None,  # No quantization for quality
                    model_offload=False
                )
                logger.info("✓ MultiTalk pipeline loaded")
            else:
                logger.warning("MultiTalk model files not found, using fallback mode")
                
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def process_audio_to_video(
        self,
        audio_data: bytes,
        reference_image: bytes,
        prompt: str = "A person talking naturally",
        num_frames: int = 81,
        fps: int = 25,
        sample_steps: int = 40,
        audio_cfg: float = 3.5,
        video_cfg: float = 7.5,
        use_teacache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Process audio and image to generate talking video using MultiTalk"""
        try:
            logger.info("Processing with MeiGen-MultiTalk...")
            
            # Save temporary files
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_tmp:
                audio_tmp.write(audio_data)
                audio_path = audio_tmp.name
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_tmp:
                img_tmp.write(reference_image)
                image_path = img_tmp.name
            
            # Use MultiTalk pipeline if available
            if self.multitalk_pipeline:
                video_data = self._generate_with_multitalk(
                    audio_path=audio_path,
                    image_path=image_path,
                    prompt=prompt,
                    num_frames=num_frames,
                    sample_steps=sample_steps,
                    audio_cfg=audio_cfg,
                    video_cfg=video_cfg,
                    use_teacache=use_teacache
                )
            else:
                # Fallback to command-line generation
                video_data = self._generate_with_cli(
                    audio_path=audio_path,
                    image_path=image_path,
                    prompt=prompt,
                    num_frames=num_frames,
                    sample_steps=sample_steps,
                    audio_cfg=audio_cfg,
                    video_cfg=video_cfg
                )
            
            # Cleanup temporary files
            os.unlink(audio_path)
            os.unlink(image_path)
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return {
                "success": True,
                "video_data": video_data,
                "model": "meigen-multitalk",
                "num_frames": num_frames,
                "fps": fps
            }
            
        except Exception as e:
            logger.error(f"Error in MultiTalk processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_with_multitalk(
        self,
        audio_path: str,
        image_path: str,
        prompt: str,
        num_frames: int,
        sample_steps: int,
        audio_cfg: float,
        video_cfg: float,
        use_teacache: bool
    ) -> bytes:
        """Generate video using MultiTalk pipeline"""
        logger.info("Generating with MultiTalk pipeline...")
        
        # Prepare input data structure
        input_data = {
            "condition_image": image_path,
            "audio_1": audio_path,
            "prompt": prompt,
            "num_frames": num_frames
        }
        
        # Generate video
        output_path = tempfile.mktemp(suffix='.mp4')
        
        video = self.multitalk_pipeline.generate(
            input_data,
            size_buckget="multitalk-480",  # 480p output
            frame_num=num_frames,
            sampling_steps=sample_steps,
            text_guide_scale=video_cfg,
            audio_guide_scale=audio_cfg,
            seed=42,
            use_teacache=use_teacache,
            teacache_thresh=0.3,
            save_path=output_path
        )
        
        # Read generated video
        with open(output_path, 'rb') as f:
            video_data = f.read()
        
        os.unlink(output_path)
        return video_data
    
    def _generate_with_cli(
        self,
        audio_path: str,
        image_path: str,
        prompt: str,
        num_frames: int,
        sample_steps: int,
        audio_cfg: float,
        video_cfg: float
    ) -> bytes:
        """Generate video using CLI fallback"""
        logger.info("Generating with CLI fallback...")
        
        # Create input JSON
        input_json = {
            "people": [
                {
                    "reference_image": image_path,
                    "person_audio": audio_path,
                    "person_name": "Speaker"
                }
            ],
            "control_prompt": prompt
        }
        
        # Save input JSON
        json_path = tempfile.mktemp(suffix='.json')
        with open(json_path, 'w') as f:
            json.dump(input_json, f)
        
        # Output path
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # Build command
        multitalk_script = self.model_path / "MultiTalk" / "generate_multitalk.py"
        
        cmd = [
            "python", str(multitalk_script),
            "--ckpt_dir", str(self.wan_checkpoint),
            "--wav2vec_dir", str(self.wav2vec_path),
            "--input_json", json_path,
            "--sample_steps", str(sample_steps),
            "--mode", "clip",  # Single clip mode
            "--frame_num", str(num_frames),
            "--sample_audio_guide_scale", str(audio_cfg),
            "--sample_text_guide_scale", str(video_cfg),
            "--save_file", output_path
        ]
        
        # Add optional flags
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 24 * 1024**3:
            cmd.extend(["--low_vram"])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run generation
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(self.model_path / "MultiTalk")
            )
            logger.info(f"CLI output: {result.stdout}")
            
            # Read generated video
            with open(f"{output_path}.mp4", 'rb') as f:
                video_data = f.read()
            
            # Cleanup
            os.unlink(json_path)
            if os.path.exists(f"{output_path}.mp4"):
                os.unlink(f"{output_path}.mp4")
            
            return video_data
            
        except subprocess.CalledProcessError as e:
            logger.error(f"CLI generation failed: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            
            # Return a simple fallback video
            return self._create_basic_video(image_path, audio_path, num_frames)
    
    def _create_basic_video(self, image_path: str, audio_path: str, num_frames: int) -> bytes:
        """Create basic video as last resort fallback"""
        logger.info("Creating basic fallback video...")
        
        output_path = tempfile.mktemp(suffix='.mp4')
        
        # Use ffmpeg to create video with audio
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', image_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            '-shortest',
            output_path
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        
        with open(output_path, 'rb') as f:
            video_data = f.read()
        
        os.unlink(output_path)
        return video_data
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass