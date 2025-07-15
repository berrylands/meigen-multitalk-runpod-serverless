"""
MultiTalk V75.0 - Using Correct JSON Input Format
Based on actual command-line interface from generate_multitalk.py
"""

import os
import json
import subprocess
import logging
import traceback
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import soundfile as sf
from typing import Optional, Tuple, Dict, Any
import tempfile
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiTalkV75JsonWrapper:
    """Wrapper using the CORRECT JSON input format for MultiTalk"""
    
    def __init__(self):
        logger.info("🔍 Initializing MultiTalk V75.0 with JSON input format")
        
        # Model paths - matching official structure
        self.model_base = "/runpod-volume/models"
        self.wan_path = f"{self.model_base}/wan2.1-i2v-14b-480p"
        self.wav2vec_path = f"{self.model_base}/wav2vec2"
        self.multitalk_path = "/app/multitalk_official"
        
        # Validate models exist
        self._validate_models()
        
        logger.info("✅ MultiTalk V75.0 initialized with JSON input support")
    
    def _validate_models(self):
        """Validate model paths exist"""
        paths_to_check = {
            "Wan2.1 model": self.wan_path,
            "Wav2Vec2 model": self.wav2vec_path,
            "MultiTalk implementation": self.multitalk_path
        }
        
        for name, path in paths_to_check.items():
            if os.path.exists(path):
                logger.info(f"✅ {name} found: {path}")
            else:
                logger.warning(f"⚠️ {name} not found: {path}")
    
    def generate(self, audio_path: str, image_path: str, output_path: str) -> str:
        """Generate video using the CORRECT JSON input format"""
        try:
            logger.info("="*80)
            logger.info("🎬 Starting video generation with JSON input format")
            logger.info(f"Audio: {audio_path}")
            logger.info(f"Image: {image_path}")
            logger.info(f"Output: {output_path}")
            logger.info("="*80)
            
            # Validate inputs
            self._validate_inputs(audio_path, image_path)
            
            # Generate using JSON input format
            return self._run_with_json_input(audio_path, image_path, output_path)
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def _validate_inputs(self, audio_path: str, image_path: str):
        """Validate input files"""
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        # Log input file details
        audio_info = sf.info(audio_path)
        logger.info(f"Audio: {audio_info.duration:.2f}s, {audio_info.samplerate}Hz")
        
        image = Image.open(image_path)
        logger.info(f"Image: {image.size[0]}x{image.size[1]}, {image.mode}")
    
    def _run_with_json_input(self, audio_path: str, image_path: str, output_path: str) -> str:
        """Run with JSON input file (correct official interface)"""
        
        # Create temporary directory for this generation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create input JSON with proper format
            input_json = {
                "prompt": "A person talking naturally with expressive lip sync",
                "negative_prompt": "",
                "speakers": [
                    {
                        "id": 0,
                        "condition_image": str(image_path),
                        "condition_audio": str(audio_path)
                    }
                ]
            }
            
            # Save JSON file
            json_path = temp_path / "input.json"
            with open(json_path, "w") as f:
                json.dump(input_json, f, indent=2)
            
            logger.info(f"📝 Created input JSON: {json_path}")
            logger.info(f"   Content: {json.dumps(input_json, indent=2)}")
            
            # Calculate frames based on audio duration
            audio_info = sf.info(audio_path)
            fps = 25
            num_frames = int(audio_info.duration * fps)
            logger.info(f"📹 Generating {num_frames} frames at {fps}fps")
            
            # Extract output name without extension
            output_name = Path(output_path).stem
            
            # Build command using CORRECT arguments from the help output
            cmd = [
                "python", "-u",
                f"{self.multitalk_path}/generate_multitalk.py",
                "--task", "multitalk-14B",  # Specific MultiTalk task
                "--ckpt_dir", self.wan_path,
                "--wav2vec_dir", self.wav2vec_path,
                "--input_json", str(json_path),  # JSON input instead of direct paths
                "--save_file", output_name,  # Without extension
                "--frame_num", str(num_frames),
                "--sample_steps", "40",
                "--mode", "clip",  # Single clip generation
                "--size", "multitalk-480",  # Official size option
                "--base_seed", "42",  # Correct seed parameter
                "--use_teacache",  # Enable acceleration
                "--num_persistent_param_in_dit", "0",  # For low VRAM
                "--sample_text_guide_scale", "7.5",  # Text guidance
                "--sample_audio_guide_scale", "3.5",  # Audio guidance
            ]
            
            # Log the command
            logger.info("🚀 Running MultiTalk with JSON input:")
            logger.info("   " + " ".join(cmd))
            
            # Set environment
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["PYTHONPATH"] = f"{self.multitalk_path}:{env.get('PYTHONPATH', '')}"
            
            # Run the command
            logger.info("⏱️ Starting generation...")
            start_time = time.time()
            
            process = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=str(self.multitalk_path),  # Run from MultiTalk directory
                timeout=1800  # 30 minute timeout for large generations
            )
            
            generation_time = time.time() - start_time
            
            if process.returncode == 0:
                logger.info(f"✅ Generation completed successfully in {generation_time:.2f}s!")
                
                # Find the actual output file
                # The script runs with cwd=self.multitalk_path, so check there first
                potential_outputs = [
                    Path(self.multitalk_path) / f"{output_name}.mp4",  # Most likely location
                    Path(self.multitalk_path) / "output_video.mp4",    # In case it ignores save_file
                    output_path,  # Our requested path
                    Path(self.multitalk_path) / "outputs" / f"{output_name}.mp4",
                    Path(temp_dir) / f"{output_name}.mp4",
                    Path("/tmp") / f"{output_name}.mp4",
                    Path("/tmp") / "output_video.mp4"
                ]
                
                actual_output = None
                for potential in potential_outputs:
                    if os.path.exists(potential):
                        actual_output = potential
                        break
                
                if not actual_output:
                    # Search for any MP4 files created
                    logger.info("🔍 Searching for generated video...")
                    for mp4_file in Path(self.multitalk_path).rglob("*.mp4"):
                        if mp4_file.stat().st_mtime > start_time:
                            actual_output = mp4_file
                            logger.info(f"Found generated video: {actual_output}")
                            break
                
                if actual_output and actual_output != output_path:
                    # Move to requested location
                    import shutil
                    shutil.move(str(actual_output), output_path)
                    logger.info(f"📹 Moved output to: {output_path}")
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    logger.info(f"📹 Output video: {output_path} ({file_size:.2f}MB)")
                    return output_path
                else:
                    # Log stdout to help debug
                    logger.warning("Output file not found at expected location")
                    logger.info(f"STDOUT (last 1000 chars):\n{process.stdout[-1000:]}")
                    raise RuntimeError("Generation completed but output file not found")
            else:
                logger.error(f"❌ Generation failed with code {process.returncode}")
                logger.error(f"STDOUT:\n{process.stdout}")
                logger.error(f"STDERR:\n{process.stderr}")
                raise RuntimeError(f"Generation failed with code {process.returncode}")
    
    def generate_with_options(self, audio_path: str, image_path: str, output_path: str,
                            prompt: str = "A person talking naturally",
                            sample_steps: int = 40,
                            mode: str = "clip",
                            size: str = "multitalk-480",
                            use_teacache: bool = True,
                            text_guide_scale: float = 7.5,
                            audio_guide_scale: float = 3.5,
                            seed: int = 42) -> str:
        """Generate with full control over parameters using JSON input"""
        
        logger.info("🎬 Generating with custom options:")
        logger.info(f"  - Prompt: {prompt}")
        logger.info(f"  - Sample steps: {sample_steps}")
        logger.info(f"  - Mode: {mode}")
        logger.info(f"  - Size: {size}")
        logger.info(f"  - TeaCache: {use_teacache}")
        logger.info(f"  - Text guidance: {text_guide_scale}")
        logger.info(f"  - Audio guidance: {audio_guide_scale}")
        logger.info(f"  - Seed: {seed}")
        
        # Validate inputs
        self._validate_inputs(audio_path, image_path)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create input JSON
            input_json = {
                "prompt": prompt,
                "negative_prompt": "",
                "speakers": [
                    {
                        "id": 0,
                        "condition_image": str(image_path),
                        "condition_audio": str(audio_path)
                    }
                ]
            }
            
            json_path = temp_path / "input.json"
            with open(json_path, "w") as f:
                json.dump(input_json, f, indent=2)
            
            # Calculate frames
            audio_info = sf.info(audio_path)
            fps = 25 if "480" in size else 30
            num_frames = int(audio_info.duration * fps)
            
            # Extract output name
            output_name = Path(output_path).stem
            
            # Build command
            cmd = [
                "python", "-u",
                f"{self.multitalk_path}/generate_multitalk.py",
                "--task", "multitalk-14B",
                "--ckpt_dir", self.wan_path,
                "--wav2vec_dir", self.wav2vec_path,
                "--input_json", str(json_path),
                "--save_file", output_name,
                "--frame_num", str(num_frames),
                "--sample_steps", str(sample_steps),
                "--mode", mode,
                "--size", size,
                "--base_seed", str(seed),
                "--sample_text_guide_scale", str(text_guide_scale),
                "--sample_audio_guide_scale", str(audio_guide_scale),
                "--num_persistent_param_in_dit", "0"
            ]
            
            if use_teacache:
                cmd.append("--use_teacache")
            
            # Run command
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["PYTHONPATH"] = f"{self.multitalk_path}:{env.get('PYTHONPATH', '')}"
            
            process = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=str(self.multitalk_path),
                timeout=1800
            )
            
            if process.returncode != 0:
                logger.error(f"Command failed with return code {process.returncode}")
                logger.error(f"STDOUT: {process.stdout}")
                logger.error(f"STDERR: {process.stderr}")
                raise RuntimeError(f"Generation failed: {process.stderr}")
            
            logger.info(f"Command completed successfully")
            logger.info(f"STDOUT: {process.stdout[-500:]}")  # Last 500 chars
            
            # Record process start time for searching recent files
            import time
            end_time = time.time()
            
            # Find the generated video file - search multiple locations
            # The script runs with cwd=self.multitalk_path, so check there first
            potential_outputs = [
                Path(self.multitalk_path) / f"{output_name}.mp4",  # Most likely location
                Path(self.multitalk_path) / "output_video.mp4",    # In case it ignores save_file
                temp_path / f"{output_name}.mp4", 
                Path(self.multitalk_path) / "outputs" / f"{output_name}.mp4",
                Path(self.multitalk_path) / "samples" / f"{output_name}.mp4",
                Path("/tmp") / f"{output_name}.mp4",
                Path("/tmp") / "output_video.mp4",
                Path.cwd() / f"{output_name}.mp4",
                Path.cwd() / "output_video.mp4"
            ]
            
            actual_output = None
            for potential in potential_outputs:
                if potential.exists():
                    actual_output = potential
                    logger.info(f"Found generated video: {actual_output}")
                    break
            
            if not actual_output:
                # Search for any MP4 files created recently (last 2 minutes)
                logger.info("🔍 Searching for recently created MP4 files...")
                search_paths = [
                    Path(self.multitalk_path),
                    temp_path,
                    Path("/tmp"),
                    Path.cwd()
                ]
                
                for search_path in search_paths:
                    if search_path.exists():
                        for mp4_file in search_path.rglob("*.mp4"):
                            try:
                                if mp4_file.stat().st_mtime > (end_time - 120):  # Last 2 minutes
                                    actual_output = mp4_file
                                    logger.info(f"Found recently created video: {actual_output}")
                                    break
                            except OSError:
                                continue
                        if actual_output:
                            break
            
            if actual_output:
                # Copy to requested location
                import shutil
                try:
                    shutil.copy2(str(actual_output), output_path)
                    logger.info(f"✅ Copied output to: {output_path}")
                    
                    # Verify the copy worked
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path) / (1024 * 1024)
                        logger.info(f"📹 Final output: {output_path} ({file_size:.2f}MB)")
                        return output_path
                    else:
                        raise RuntimeError(f"Failed to copy file to {output_path}")
                        
                except Exception as e:
                    logger.error(f"Error copying file: {e}")
                    # Try moving instead of copying
                    try:
                        shutil.move(str(actual_output), output_path)
                        logger.info(f"✅ Moved output to: {output_path}")
                        return output_path
                    except Exception as e2:
                        logger.error(f"Error moving file: {e2}")
                        raise RuntimeError(f"Failed to copy or move output file: {e}, {e2}")
            else:
                # Log directory contents for debugging
                logger.warning("🔍 No output file found. Listing directory contents:")
                for search_path in [Path(self.multitalk_path), temp_path, Path("/tmp")]:
                    if search_path.exists():
                        logger.info(f"Contents of {search_path}:")
                        try:
                            for item in search_path.iterdir():
                                if item.is_file() and item.suffix.lower() in ['.mp4', '.avi', '.mov']:
                                    logger.info(f"  Video file: {item}")
                        except PermissionError:
                            logger.warning(f"  Permission denied to list {search_path}")
                
                raise RuntimeError(f"No output video found. Checked locations: {[str(p) for p in potential_outputs]}")