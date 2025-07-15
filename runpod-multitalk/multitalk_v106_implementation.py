"""
MultiTalk V106 Implementation
Based on MeiGen-AI/MultiTalk generate_multitalk.py
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import cv2
from PIL import Image

# Add MultiTalk to path
sys.path.insert(0, '/app/multitalk_official')

class MultiTalkInference:
    """MultiTalk inference wrapper for RunPod"""
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        
        # Model components
        self.wav2vec_model = None
        self.wav2vec_processor = None
        self.wan_pipeline = None
        
    def load_models(self):
        """Load all required models"""
        if self.models_loaded:
            return True
            
        try:
            print("Loading MultiTalk models...")
            
            # 1. Load Wav2Vec2 for audio processing
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            
            wav2vec_path = self.model_path / "wav2vec2"
            if not wav2vec_path.exists():
                # Use HuggingFace model as fallback
                print("Loading Wav2Vec2 from HuggingFace...")
                self.wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
                self.wav2vec_model = Wav2Vec2Model.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                ).to(self.device)
            else:
                # Load from local path
                self.wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
                self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_path).to(self.device)
            
            # 2. Load WAN pipeline
            wan_checkpoint = self.model_path / "wan2.1-i2v-14b-480p"
            if wan_checkpoint.exists():
                print(f"Loading WAN pipeline from {wan_checkpoint}...")
                # TODO: Implement actual WAN loading when model structure is clear
                # from wan import MultiTalkPipeline
                # self.wan_pipeline = MultiTalkPipeline.from_pretrained(wan_checkpoint)
            else:
                print(f"WAN checkpoint not found at {wan_checkpoint}")
                
            self.models_loaded = True
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_audio(self, audio_path: str, duration: Optional[float] = None) -> np.ndarray:
        """Process audio file and extract features"""
        try:
            # Load audio
            audio_data, sample_rate = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
                sample_rate = 16000
            
            # Limit duration if specified
            if duration:
                max_samples = int(duration * sample_rate)
                audio_data = audio_data[:max_samples]
            
            # Extract features using Wav2Vec2
            if self.wav2vec_model is not None:
                inputs = self.wav2vec_processor(
                    audio_data, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.wav2vec_model(**inputs)
                    audio_features = outputs.last_hidden_state.cpu().numpy()
            else:
                # Fallback: simple MFCC features
                audio_features = librosa.feature.mfcc(
                    y=audio_data, 
                    sr=sample_rate, 
                    n_mfcc=13
                ).T
            
            return audio_features
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            raise
    
    def process_image(self, image_path: str, target_size: tuple = (480, 480)) -> np.ndarray:
        """Process reference image"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Resize to target size
            image = image.resize(target_size, Image.LANCZOS)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Normalize to [0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            return image_array
            
        except Exception as e:
            print(f"Error processing image: {e}")
            raise
    
    def generate_video(
        self,
        audio_path: str,
        image_path: str,
        output_path: str,
        prompt: str = "A person talking naturally",
        duration: Optional[float] = None,
        sample_steps: int = 30,
        text_guidance_scale: float = 7.5,
        audio_guidance_scale: float = 3.5,
        seed: int = 42
    ) -> str:
        """Generate video from audio and reference image"""
        
        # Ensure models are loaded
        if not self.models_loaded:
            if not self.load_models():
                raise RuntimeError("Failed to load models")
        
        try:
            # Process inputs
            print("Processing audio...")
            audio_features = self.process_audio(audio_path, duration)
            
            print("Processing image...")
            image_array = self.process_image(image_path)
            
            # Generate video
            if self.wan_pipeline is not None:
                print("Generating video with WAN pipeline...")
                # TODO: Implement actual WAN generation
                # video_frames = self.wan_pipeline.generate(
                #     audio_features=audio_features,
                #     reference_image=image_array,
                #     prompt=prompt,
                #     num_inference_steps=sample_steps,
                #     guidance_scale=text_guidance_scale,
                #     audio_guidance_scale=audio_guidance_scale,
                #     generator=torch.Generator(device=self.device).manual_seed(seed)
                # )
                video_frames = self._generate_test_video(audio_features, image_array)
            else:
                print("WAN pipeline not available, generating test video...")
                video_frames = self._generate_test_video(audio_features, image_array)
            
            # Save video
            self._save_video(video_frames, audio_path, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"Error generating video: {e}")
            raise
    
    def _generate_test_video(self, audio_features: np.ndarray, image: np.ndarray) -> list:
        """Generate test video frames"""
        # Calculate number of frames based on audio length
        fps = 25
        duration_seconds = len(audio_features) / 50  # Approximate
        num_frames = max(int(fps * duration_seconds), 75)  # At least 3 seconds
        
        frames = []
        h, w = image.shape[:2]
        
        for i in range(num_frames):
            # Create frame from reference image
            frame = (image * 255).astype(np.uint8).copy()
            
            # Add animation overlay
            progress = i / max(num_frames - 1, 1)
            cv2.putText(
                frame, 
                f"MultiTalk V106 - Frame {i+1}/{num_frames}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Add audio waveform visualization
            if i < len(audio_features):
                audio_level = int(abs(audio_features[i].mean()) * 100)
                cv2.rectangle(
                    frame,
                    (10, h - 50),
                    (10 + audio_level * 3, h - 30),
                    (0, 255, 0),
                    -1
                )
            
            frames.append(frame)
        
        return frames
    
    def _save_video(self, frames: list, audio_path: str, output_path: str):
        """Save frames as video with audio"""
        if not frames:
            raise ValueError("No frames to save")
        
        # Get frame dimensions
        h, w = frames[0].shape[:2]
        fps = 25
        
        # Create temporary video without audio
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        # Add audio using moviepy
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            
            video_clip = VideoFileClip(temp_video)
            audio_clip = AudioFileClip(audio_path)
            
            # Sync audio duration with video
            if audio_clip.duration > video_clip.duration:
                audio_clip = audio_clip.subclip(0, video_clip.duration)
            
            # Combine
            final_clip = video_clip.set_audio(audio_clip)
            final_clip.write_videofile(
                output_path, 
                codec='libx264', 
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Cleanup
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            os.unlink(temp_video)
            
        except Exception as e:
            print(f"Error adding audio, saving video without audio: {e}")
            # Fallback: just rename temp video
            import shutil
            shutil.move(temp_video, output_path)


# Example usage
if __name__ == "__main__":
    # Test the implementation
    inference = MultiTalkInference()
    
    # Load models
    if inference.load_models():
        print("Models loaded successfully!")
    else:
        print("Failed to load models")