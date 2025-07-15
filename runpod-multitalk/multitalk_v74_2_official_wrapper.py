"""
MultiTalk V74.2 Official Wrapper
Version: 74.2.0 - Fixed model path verification regression

This wrapper uses the official MeiGen-AI/MultiTalk implementation
by calling their generate_multitalk.py script directly.

Key fixes:
- Model verification is now optional and non-blocking
- More flexible model path discovery
- Better error handling with fallbacks
"""

import os
import sys
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

# Configure logging
logger = logging.getLogger(__name__)

class MultiTalkV74OfficialWrapper:
    """Wrapper for official MultiTalk implementation - V74.2 with flexible model handling."""
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        """Initialize the wrapper with flexible model discovery."""
        self.model_path = Path(model_path)
        logger.info(f"Initializing MultiTalk V74.2 Official Wrapper")
        logger.info(f"Model base path: {self.model_path}")
        
        # Verify gcc is available
        try:
            gcc_version = subprocess.run(['gcc', '--version'], 
                                       capture_output=True, 
                                       text=True, 
                                       check=True)
            gcc_info = gcc_version.stdout.split('\n')[0]
            logger.info(f"✅ GCC available: {gcc_info}")
        except Exception as e:
            logger.warning(f"⚠️ GCC check failed: {str(e)} - Triton compilation may fail")
        
        # Discover available models instead of hardcoding paths
        self._discover_models()
        
    def _discover_models(self):
        """Discover available models on the volume."""
        logger.info("Discovering available models...")
        
        # List what's actually in the model directory
        if self.model_path.exists():
            available_models = list(self.model_path.iterdir())
            logger.info(f"Available items in {self.model_path}:")
            for item in available_models:
                logger.info(f"  - {item.name} {'(dir)' if item.is_dir() else '(file)'}")
        else:
            logger.warning(f"Model path does not exist: {self.model_path}")
            logger.warning("Creating model path...")
            self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Try to find models with flexible naming
        self.wan_path = self._find_model_path(["wan2.1", "wan-2.1", "wan21", "wan"], "Wan2.1")
        self.wav2vec_path = self._find_model_path(["wav2vec", "wav2vec2", "Wav2Vec2"], "Wav2Vec2")
        
        # Look for the official MultiTalk implementation in the expected locations
        self.multitalk_path = self._find_multitalk_implementation()
        
        # Log what we found
        logger.info("Model discovery results:")
        logger.info(f"  - Wan2.1: {self.wan_path if self.wan_path else 'NOT FOUND'}")
        logger.info(f"  - MultiTalk: {self.multitalk_path if self.multitalk_path else 'NOT FOUND'}")
        logger.info(f"  - Wav2Vec2: {self.wav2vec_path if self.wav2vec_path else 'NOT FOUND'}")
        
        # If critical models are missing, we'll handle it during generation
        if not self.wan_path or not self.multitalk_path:
            logger.warning("⚠️ Some models not found. Will attempt to proceed with available resources.")
        
        # Add MultiTalk to Python path if found
        if self.multitalk_path:
            wan_repo_path = self.multitalk_path / "wan"
            if wan_repo_path.exists():
                sys.path.insert(0, str(self.multitalk_path))
                logger.info(f"✅ Added to Python path: {self.multitalk_path}")
            else:
                # Try adding the model path itself
                sys.path.insert(0, str(self.multitalk_path))
                logger.info(f"✅ Added to Python path: {self.multitalk_path}")
    
    def _find_model_path(self, possible_names: List[str], model_type: str) -> Optional[Path]:
        """Find a model path using various possible names."""
        if not self.model_path.exists():
            return None
            
        # Check each possible name
        for name in possible_names:
            # Check exact match
            path = self.model_path / name
            if path.exists():
                return path
            
            # Check with common suffixes
            for suffix in ["-official", "_official", "-model", "_model", ""]:
                full_name = f"{name}{suffix}"
                path = self.model_path / full_name
                if path.exists():
                    return path
            
            # Case-insensitive search
            for item in self.model_path.iterdir():
                if name.lower() in item.name.lower():
                    return item
        
        return None
    
    def _find_multitalk_implementation(self) -> Optional[Path]:
        """Find the official MultiTalk implementation in expected locations."""
        # Priority order for finding the official implementation
        possible_locations = [
            # First try the app directory where we install it
            Path("/app/multitalk_official"),
            # Then try the volume symlink
            Path("/runpod-volume/models/multitalk-official"),
            # Legacy locations for compatibility
            self.model_path / "multitalk-official",
            self.model_path / "meigen-multitalk"
        ]
        
        for location in possible_locations:
            if location.exists():
                generate_script = location / "generate_multitalk.py"
                if generate_script.exists():
                    logger.info(f"Found official MultiTalk implementation at: {location}")
                    return location
                else:
                    logger.warning(f"Found directory {location} but missing generate_multitalk.py")
        
        logger.error("Official MultiTalk implementation not found in any expected location")
        logger.error("Expected locations checked:")
        for location in possible_locations:
            logger.error(f"  - {location}")
        
        return None
    
    def _setup_model_linking(self):
        """Set up model linking if models are available."""
        if not self.wan_path or not self.multitalk_path:
            logger.warning("Skipping model linking - models not available")
            return
            
        logger.info("Setting up model linking...")
        
        # Paths for linking
        wan_index = self.wan_path / "model_index.json"
        wan_index_old = self.wan_path / "model_index_old.json"
        mt_index = self.multitalk_path / "model_index.json"
        mt_weights = self.multitalk_path / "multitalk.safetensors"
        wan_mt_weights = self.wan_path / "multitalk.safetensors"
        
        # Only set up if not already done
        if wan_mt_weights.exists() and wan_index_old.exists():
            logger.info("✅ Model linking already set up")
            return
        
        try:
            # Backup original index if exists
            if wan_index.exists() and not wan_index_old.exists():
                shutil.move(str(wan_index), str(wan_index_old))
                logger.info("✅ Backed up original model_index.json")
            
            # Copy MultiTalk index
            if mt_index.exists():
                shutil.copy2(str(mt_index), str(wan_index))
                logger.info("✅ Copied MultiTalk model_index.json")
            
            # Create symlink for weights
            if not wan_mt_weights.exists() and mt_weights.exists():
                os.symlink(str(mt_weights.absolute()), str(wan_mt_weights))
                logger.info("✅ Created symlink for multitalk.safetensors")
            
        except Exception as e:
            logger.warning(f"Model linking failed: {str(e)} - Will proceed without linking")
    
    def generate(self, audio_path: str, image_path: str, output_path: str) -> Dict[str, Any]:
        """
        Generate video using official MultiTalk script - no fallbacks allowed.
        
        Args:
            audio_path: Path to input audio file
            image_path: Path to input image file  
            output_path: Path for output video
            
        Returns:
            Dict with generation results
        """
        logger.info("=" * 80)
        logger.info("Starting MultiTalk generation")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Image: {image_path}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)
        
        # Check if we have the official script
        if not self.multitalk_path:
            raise RuntimeError("MultiTalk directory not found. Expected directory structure missing.")
        
        generate_script = self.multitalk_path / "generate_multitalk.py"
        if not generate_script.exists():
            raise RuntimeError(f"Official MultiTalk script not found at {generate_script}. "
                             f"The MultiTalk implementation is incomplete. "
                             f"Required files: generate_multitalk.py, wan/ directory structure, model weights.")
        
        return self._run_official_script(audio_path, image_path, output_path)
    
    def _run_official_script(self, audio_path: str, image_path: str, output_path: str) -> Dict[str, Any]:
        """Run the official MultiTalk generation script."""
        generate_script = self.multitalk_path / "generate_multitalk.py"
        
        # Prepare command
        cmd = [
            sys.executable, str(generate_script),
            "--wan_ckpt_path", str(self.wan_path) if self.wan_path else "/tmp/wan_dummy",
            "--mt_ckpt_path", str(self.multitalk_path),
            "--wav2vec_ckpt_path", str(self.wav2vec_path) if self.wav2vec_path else "/tmp/wav2vec_dummy",
            "--ref_img_path", image_path,
            "--ref_audio_path", audio_path,
            "--save_path", output_path,
            "--num_frames", "96",
            "--fps", "25",
            "--seed", "42",
            "--device", "cuda"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Set environment for subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{self.multitalk_path}:{env.get('PYTHONPATH', '')}"
        env['CUDA_VISIBLE_DEVICES'] = "0"
        env['CC'] = 'gcc'
        env['CXX'] = 'g++'
        env['CUDA_HOME'] = '/usr/local/cuda'
        
        try:
            # Run the official script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                cwd=str(self.multitalk_path)
            )
            
            logger.info("✅ Official script completed successfully")
            logger.info(f"STDOUT:\n{result.stdout}")
            
            if result.stderr:
                logger.warning(f"STDERR:\n{result.stderr}")
            
            # Check if output was created
            if not os.path.exists(output_path):
                raise RuntimeError("Output video not created")
            
            output_size = os.path.getsize(output_path)
            logger.info(f"✅ Video generated: {output_path} ({output_size} bytes)")
            
            return {
                'success': True,
                'output_path': output_path,
                'size': output_size,
                'duration': 96 / 25.0  # frames / fps
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Official script failed with code {e.returncode}")
            logger.error(f"STDOUT:\n{e.stdout}")
            logger.error(f"STDERR:\n{e.stderr}")
            raise RuntimeError(f"Official MultiTalk script execution failed with code {e.returncode}. "
                             f"Check logs for detailed error information.")