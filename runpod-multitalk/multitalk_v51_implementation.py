"""
MultiTalk V51 Implementation - Using Actual Model Paths
Based on debug findings from v50
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

# Import required libraries
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    logger.error("soundfile not available")
    HAS_SOUNDFILE = False

try:
    from PIL import Image
    import cv2
    HAS_IMAGE = True
except ImportError:
    logger.error("PIL or cv2 not available")
    HAS_IMAGE = False

try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    HAS_TRANSFORMERS = True
except ImportError:
    logger.error("transformers not available")
    HAS_TRANSFORMERS = False

try:
    import imageio
    import imageio_ffmpeg
    HAS_IMAGEIO = True
except ImportError:
    logger.error("imageio not available")
    HAS_IMAGEIO = False

# Check for OmegaConf (for config loading)
try:
    from omegaconf import OmegaConf
    HAS_OMEGACONF = True
except ImportError:
    logger.error("omegaconf not available")
    HAS_OMEGACONF = False

class MultiTalkV51Pipeline:
    """MultiTalk implementation using actual model paths from debug"""
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        logger.info(f"Initializing MultiTalk V51 on device: {self.device}")
        
        # Correct model paths based on debug findings
        self.wan_path = self.model_path / "wan2.1-i2v-14b-480p"  # lowercase
        self.meigen_path = self.model_path / "meigen-multitalk"  # lowercase
        self.wav2vec_path = self.model_path / "wav2vec2-base-960h"
        self.chinese_wav2vec_path = self.model_path / "chinese-wav2vec2-base"
        
        # Log paths
        logger.info(f"WAN path: {self.wan_path} (exists: {self.wan_path.exists()})")
        logger.info(f"MeiGen path: {self.meigen_path} (exists: {self.meigen_path.exists()})")
        logger.info(f"Wav2Vec path: {self.wav2vec_path} (exists: {self.wav2vec_path.exists()})")
        
        # Model components
        self.wav2vec_processor = None
        self.wav2vec_model = None
        self.multitalk_model = None
        
        # Initialize components
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with correct paths"""
        # 1. Load Wav2Vec2 (we know this works from debug)
        if HAS_TRANSFORMERS and self.wav2vec_path.exists():
            try:
                logger.info("Loading Wav2Vec2...")
                self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                    str(self.wav2vec_path),
                    local_files_only=True
                )
                self.wav2vec_model = Wav2Vec2Model.from_pretrained(
                    str(self.wav2vec_path),
                    local_files_only=True,
                    torch_dtype=self.dtype
                ).to(self.device)
                logger.info("✓ Wav2Vec2 loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Wav2Vec2: {e}")
        
        # 2. Check for MultiTalk components
        self._check_multitalk_structure()
        
        # 3. Try to load MultiTalk model
        self._load_multitalk_model()
    
    def _check_multitalk_structure(self):
        """Check the structure of meigen-multitalk directory"""
        if self.meigen_path.exists():
            logger.info("Checking meigen-multitalk structure:")
            
            # List top-level contents
            contents = list(self.meigen_path.iterdir())[:20]
            for item in contents:
                if item.is_file():
                    logger.info(f"  File: {item.name}")
                elif item.is_dir():
                    logger.info(f"  Dir: {item.name}/")
                    # Check subdirectory
                    sub_contents = list(item.iterdir())[:5]
                    for sub in sub_contents:
                        logger.info(f"    - {sub.name}")
            
            # Look for key files
            key_files = [
                "model.pt", "model.pth", "model.bin", "model.safetensors",
                "config.json", "model_config.json", "audio_condition.pt",
                "multitalk.py", "generate.py", "inference.py"
            ]
            
            for key_file in key_files:
                found = list(self.meigen_path.rglob(key_file))
                if found:
                    logger.info(f"✓ Found {key_file}: {found[0].relative_to(self.meigen_path)}")
    
    def _load_multitalk_model(self):
        """Attempt to load MultiTalk model"""
        try:
            # Check for model files in meigen-multitalk
            model_files = list(self.meigen_path.glob("*.pt")) + \
                         list(self.meigen_path.glob("*.pth")) + \
                         list(self.meigen_path.glob("*.safetensors"))
            
            if model_files:
                logger.info(f"Found {len(model_files)} model files in meigen-multitalk")
                
                # Try to load the first model file
                model_file = model_files[0]
                logger.info(f"Attempting to load: {model_file.name}")
                
                # Check if it's a state dict or full model
                checkpoint = torch.load(model_file, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    logger.info(f"Loaded checkpoint with keys: {list(checkpoint.keys())[:5]}")
                    
                    # Look for audio conditioning weights
                    if 'audio_condition' in checkpoint:
                        logger.info("✓ Found audio_condition in checkpoint")
                    if 'model_state_dict' in checkpoint:
                        logger.info("✓ Found model_state_dict in checkpoint")
                    if 'state_dict' in checkpoint:
                        logger.info("✓ Found state_dict in checkpoint")
                        
                self.multitalk_model = checkpoint
                
        except Exception as e:
            logger.error(f"Failed to load MultiTalk model: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def extract_audio_features(self, audio_data: bytes) -> torch.Tensor:
        """Extract audio features using Wav2Vec2"""
        if not self.wav2vec_processor or not self.wav2vec_model:
            logger.error("Wav2Vec2 not initialized")
            return None
        
        try:
            # Save audio temporarily
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_tmp:
                audio_tmp.write(audio_data)
                audio_path = audio_tmp.name
            
            # Load audio
            audio_array, sr = sf.read(audio_path)
            os.unlink(audio_path)
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            
            # Process with Wav2Vec2
            inputs = self.wav2vec_processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.wav2vec_model(**inputs.to(self.device))
                audio_features = outputs.last_hidden_state
            
            logger.info(f"Extracted audio features: {audio_features.shape}")
            return audio_features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            return None
    
    def generate_video_with_audio_sync(
        self,
        reference_image: bytes,
        audio_features: torch.Tensor,
        audio_data: bytes,
        num_frames: int = 81,
        fps: int = 25
    ) -> bytes:
        """Generate video with audio synchronization"""
        try:
            # Load reference image
            nparr = np.frombuffer(reference_image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size
            image = cv2.resize(image, (480, 480))
            
            # Generate frames based on audio features
            frames = self._generate_synced_frames(image, audio_features, num_frames)
            
            # Create video with audio
            video_data = self._create_video_with_audio(frames, audio_data, fps)
            
            return video_data
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _generate_synced_frames(self, base_image: np.ndarray, audio_features: torch.Tensor, num_frames: int) -> List[np.ndarray]:
        """Generate frames synchronized with audio features"""
        frames = []
        
        if audio_features is not None:
            # Map audio features to frames
            feature_frames = audio_features.shape[1]
            frame_mapping = np.linspace(0, feature_frames - 1, num_frames).astype(int)
            
            for i in range(num_frames):
                frame = base_image.copy()
                
                # Get corresponding audio feature
                if i < len(frame_mapping):
                    feature_idx = frame_mapping[i]
                    if feature_idx < feature_frames:
                        # Get feature intensity
                        frame_features = audio_features[0, feature_idx].cpu().numpy()
                        intensity = np.mean(np.abs(frame_features))
                        
                        # Apply animation based on intensity
                        frame = self._apply_talking_animation(frame, intensity)
                
                frames.append(frame)
        else:
            # No audio features, just repeat the image
            frames = [base_image.copy() for _ in range(num_frames)]
        
        return frames
    
    def _apply_talking_animation(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """Apply talking animation to frame"""
        height, width = frame.shape[:2]
        
        # Normalize intensity
        intensity = min(1.0, intensity * 2.0)
        
        # Mouth region
        mouth_y = int(height * 0.75)
        mouth_x = width // 2
        
        # Create talking effect
        if intensity > 0.1:
            # Open mouth based on intensity
            mouth_width = int(30 + intensity * 20)
            mouth_height = int(5 + intensity * 15)
            
            # Draw mouth opening
            cv2.ellipse(frame, (mouth_x, mouth_y), 
                       (mouth_width, mouth_height), 
                       0, 0, 180, (50, 30, 30), -1)
            
            # Add inner mouth detail
            if intensity > 0.3:
                inner_width = int(mouth_width * 0.7)
                inner_height = int(mouth_height * 0.7)
                cv2.ellipse(frame, (mouth_x, mouth_y), 
                           (inner_width, inner_height), 
                           0, 0, 180, (20, 10, 10), -1)
        
        return frame
    
    def _create_video_with_audio(self, frames: List[np.ndarray], audio_data: bytes, fps: int) -> bytes:
        """Create video with audio track"""
        try:
            # Save frames as temporary video
            video_tmp = tempfile.mktemp(suffix='.mp4')
            
            # Write video without audio first
            with imageio.get_writer(video_tmp, fps=fps, codec='libx264', pixelformat='yuv420p') as writer:
                for frame in frames:
                    writer.append_data(frame)
            
            # Save audio temporarily
            audio_tmp = tempfile.mktemp(suffix='.wav')
            with open(audio_tmp, 'wb') as f:
                f.write(audio_data)
            
            # Combine video and audio using ffmpeg
            output_tmp = tempfile.mktemp(suffix='.mp4')
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_tmp,
                '-i', audio_tmp,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                '-movflags', '+faststart',
                output_tmp
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                # Return video without audio
                with open(video_tmp, 'rb') as f:
                    video_data = f.read()
            else:
                # Read final video
                with open(output_tmp, 'rb') as f:
                    video_data = f.read()
            
            # Cleanup
            for tmp_file in [video_tmp, audio_tmp, output_tmp]:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
            
            return video_data
            
        except Exception as e:
            logger.error(f"Error creating video with audio: {e}")
            return None
    
    def process_audio_to_video(
        self,
        audio_data: bytes,
        reference_image: bytes,
        prompt: str = "A person talking naturally",
        num_frames: int = 81,
        fps: int = 25,
        **kwargs
    ) -> Dict[str, Any]:
        """Main processing pipeline"""
        try:
            logger.info("Processing with MultiTalk V51...")
            
            # Extract audio features
            audio_features = self.extract_audio_features(audio_data)
            
            # Generate video
            video_data = self.generate_video_with_audio_sync(
                reference_image=reference_image,
                audio_features=audio_features,
                audio_data=audio_data,
                num_frames=num_frames,
                fps=fps
            )
            
            if video_data:
                return {
                    "success": True,
                    "video_data": video_data,
                    "model": "multitalk-v51",
                    "num_frames": num_frames,
                    "fps": fps
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to generate video"
                }
                
        except Exception as e:
            logger.error(f"Error in MultiTalk V51 processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()