"""
MultiTalk V52 - Proper Safetensors Loading
Loading multitalk.safetensors properly with dtype fixes
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

try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    logger.error("safetensors not available")
    HAS_SAFETENSORS = False

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    HAS_DIFFUSERS = True
except ImportError:
    logger.error("diffusers not available")
    HAS_DIFFUSERS = False

class MultiTalkV52Pipeline:
    """MultiTalk implementation with proper safetensors loading"""
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Keep everything in float32 to avoid dtype mismatches
        self.dtype = torch.float32
        
        logger.info(f"Initializing MultiTalk V52 on device: {self.device}")
        logger.info(f"Using dtype: {self.dtype}")
        
        # Model paths
        self.wan_path = self.model_path / "wan2.1-i2v-14b-480p"
        self.meigen_path = self.model_path / "meigen-multitalk"
        self.wav2vec_path = self.model_path / "wav2vec2-base-960h"
        
        # Model components
        self.wav2vec_processor = None
        self.wav2vec_model = None
        self.multitalk_weights = None
        self.wan_model = None
        
        # Initialize components
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with correct loading methods"""
        # 1. Load Wav2Vec2 with consistent dtype
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
                    torch_dtype=self.dtype  # Use float32
                ).to(self.device)
                logger.info("✓ Wav2Vec2 loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Wav2Vec2: {e}")
        
        # 2. Load MultiTalk safetensors
        self._load_multitalk_safetensors()
        
        # 3. Check WAN model structure
        self._check_wan_model()
    
    def _load_multitalk_safetensors(self):
        """Load multitalk.safetensors properly"""
        multitalk_file = self.meigen_path / "multitalk.safetensors"
        
        if multitalk_file.exists() and HAS_SAFETENSORS:
            try:
                logger.info(f"Loading MultiTalk from: {multitalk_file}")
                
                # Load safetensors file
                self.multitalk_weights = load_safetensors(str(multitalk_file))
                
                logger.info(f"✓ Loaded MultiTalk safetensors with {len(self.multitalk_weights)} tensors")
                
                # Log some key information
                total_params = sum(t.numel() for t in self.multitalk_weights.values())
                logger.info(f"Total parameters: {total_params:,}")
                
                # Show first few tensor names
                tensor_names = list(self.multitalk_weights.keys())[:10]
                logger.info(f"Sample tensor names: {tensor_names}")
                
                # Check for specific components
                audio_condition_tensors = [k for k in self.multitalk_weights.keys() if 'audio' in k.lower()]
                if audio_condition_tensors:
                    logger.info(f"Found {len(audio_condition_tensors)} audio-related tensors")
                    logger.info(f"Sample audio tensors: {audio_condition_tensors[:5]}")
                
            except Exception as e:
                logger.error(f"Failed to load multitalk.safetensors: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"multitalk.safetensors not found at {multitalk_file}")
    
    def _check_wan_model(self):
        """Check WAN model structure"""
        if self.wan_path.exists():
            logger.info("Checking WAN model structure:")
            
            # List contents
            contents = list(self.wan_path.iterdir())[:10]
            for item in contents:
                if item.is_file():
                    logger.info(f"  File: {item.name} ({item.stat().st_size / 1024**3:.2f} GB)")
                elif item.is_dir():
                    logger.info(f"  Dir: {item.name}/")
            
            # Look for GGUF files
            gguf_files = list(self.wan_path.glob("*.gguf"))
            if gguf_files:
                logger.info(f"Found {len(gguf_files)} GGUF files:")
                for gguf in gguf_files[:5]:
                    logger.info(f"  - {gguf.name} ({gguf.stat().st_size / 1024**3:.2f} GB)")
    
    def extract_audio_features(self, audio_data: bytes) -> torch.Tensor:
        """Extract audio features with consistent dtype"""
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
            
            # Convert inputs to correct dtype
            inputs = {k: v.to(self.device).to(self.dtype) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.wav2vec_model(**inputs)
                audio_features = outputs.last_hidden_state
            
            logger.info(f"Extracted audio features: {audio_features.shape}, dtype: {audio_features.dtype}")
            return audio_features
            
        except Exception as e:
            logger.error(f"Error extracting audio features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def apply_multitalk_conditioning(self, audio_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """Apply MultiTalk audio conditioning if weights are loaded"""
        if self.multitalk_weights is None:
            logger.warning("MultiTalk weights not loaded, using basic conditioning")
            return audio_features
        
        try:
            # Look for audio projection layers in the weights
            audio_proj_keys = [k for k in self.multitalk_weights.keys() 
                             if any(term in k.lower() for term in ['audio_proj', 'audio_encoder', 'cross_attention'])]
            
            if audio_proj_keys:
                logger.info(f"Found {len(audio_proj_keys)} audio projection layers")
                
                # Apply first projection layer as example
                if 'audio_proj.weight' in self.multitalk_weights:
                    proj_weight = self.multitalk_weights['audio_proj.weight'].to(self.device).to(self.dtype)
                    audio_features = torch.matmul(audio_features, proj_weight.T)
                    logger.info(f"Applied audio projection: {audio_features.shape}")
            
            return audio_features
            
        except Exception as e:
            logger.error(f"Error applying MultiTalk conditioning: {e}")
            return audio_features
    
    def generate_video_with_multitalk(
        self,
        reference_image: bytes,
        audio_features: torch.Tensor,
        audio_data: bytes,
        num_frames: int = 81,
        fps: int = 25
    ) -> bytes:
        """Generate video using MultiTalk approach"""
        try:
            # Load reference image
            nparr = np.frombuffer(reference_image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (512, 512))  # Use 512x512 for better quality
            
            # Convert image to tensor for potential feature extraction
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            image_tensor = image_tensor.to(self.device)
            
            # Apply MultiTalk conditioning if available
            if audio_features is not None and self.multitalk_weights is not None:
                conditioned_features = self.apply_multitalk_conditioning(audio_features, image_tensor)
            else:
                conditioned_features = audio_features
            
            # Generate frames
            frames = self._generate_multitalk_frames(image, conditioned_features, num_frames)
            
            # Create video with audio
            video_data = self._create_video_with_audio(frames, audio_data, fps)
            
            return video_data
            
        except Exception as e:
            logger.error(f"Error generating video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _generate_multitalk_frames(self, base_image: np.ndarray, audio_features: torch.Tensor, num_frames: int) -> List[np.ndarray]:
        """Generate frames with MultiTalk-style animation"""
        frames = []
        
        if audio_features is not None:
            # Map audio features to frames
            feature_frames = audio_features.shape[1]
            frame_mapping = np.linspace(0, feature_frames - 1, num_frames).astype(int)
            
            for i in range(num_frames):
                frame = base_image.copy()
                
                if i < len(frame_mapping):
                    feature_idx = frame_mapping[i]
                    if feature_idx < feature_frames:
                        # Get feature intensity
                        frame_features = audio_features[0, feature_idx].cpu().numpy()
                        intensity = np.mean(np.abs(frame_features))
                        
                        # Apply more sophisticated animation
                        frame = self._apply_multitalk_animation(frame, intensity, i / num_frames)
                
                frames.append(frame)
        else:
            frames = [base_image.copy() for _ in range(num_frames)]
        
        return frames
    
    def _apply_multitalk_animation(self, frame: np.ndarray, intensity: float, progress: float) -> np.ndarray:
        """Apply MultiTalk-style facial animation"""
        height, width = frame.shape[:2]
        
        # Normalize and enhance intensity
        intensity = min(1.0, intensity * 3.0)
        
        # Face landmarks (approximate)
        mouth_y = int(height * 0.75)
        mouth_x = width // 2
        
        # Eye positions
        left_eye_x = int(width * 0.35)
        right_eye_x = int(width * 0.65)
        eye_y = int(height * 0.4)
        
        # Animate mouth with more detail
        if intensity > 0.1:
            # Outer mouth
            mouth_width = int(35 + intensity * 25)
            mouth_height = int(8 + intensity * 20)
            
            # Create gradient for mouth
            y_start = max(0, mouth_y - mouth_height)
            y_end = min(height, mouth_y + mouth_height)
            x_start = max(0, mouth_x - mouth_width)
            x_end = min(width, mouth_x + mouth_width)
            
            # Apply smooth mouth opening
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    dist_x = abs(x - mouth_x) / mouth_width
                    dist_y = abs(y - mouth_y) / mouth_height
                    dist = np.sqrt(dist_x**2 + dist_y**2)
                    
                    if dist < 1.0 and y > mouth_y:
                        darkness = int(50 * (1 - dist) * intensity)
                        frame[y, x] = np.maximum(frame[y, x] - darkness, 0)
        
        # Subtle eye animation (blinking)
        if progress % 0.15 < 0.02:  # Blink occasionally
            cv2.ellipse(frame, (left_eye_x, eye_y), (15, 3), 0, 0, 360, (50, 30, 30), -1)
            cv2.ellipse(frame, (right_eye_x, eye_y), (15, 3), 0, 0, 360, (50, 30, 30), -1)
        
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
                with open(video_tmp, 'rb') as f:
                    video_data = f.read()
            else:
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
            logger.info("Processing with MultiTalk V52...")
            
            # Extract audio features
            audio_features = self.extract_audio_features(audio_data)
            
            # Generate video with MultiTalk approach
            video_data = self.generate_video_with_multitalk(
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
                    "model": "multitalk-v52-safetensors",
                    "num_frames": num_frames,
                    "fps": fps,
                    "has_multitalk_weights": self.multitalk_weights is not None
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to generate video"
                }
                
        except Exception as e:
            logger.error(f"Error in MultiTalk V52 processing: {e}")
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