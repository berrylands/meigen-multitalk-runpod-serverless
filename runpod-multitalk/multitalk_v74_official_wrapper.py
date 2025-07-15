"""
MultiTalk V74 Official Wrapper
Version: 74.0.0 - With gcc installed for runtime compilation

This wrapper uses the official MeiGen-AI/MultiTalk implementation
by calling their generate_multitalk.py script directly.

Key features:
- Uses official generate_multitalk.py instead of custom implementation
- Proper model weight linking (MultiTalk weights -> Wan2.1 directory)
- Full compatibility with official MultiTalk architecture
- Now with gcc installed for Triton runtime compilation
"""

import os
import sys
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import torch

# Configure logging
logger = logging.getLogger(__name__)

class MultiTalkV74OfficialWrapper:
    """Wrapper for official MultiTalk implementation - V74 with gcc support."""
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        """Initialize the wrapper with model verification."""
        self.model_path = Path(model_path)
        logger.info(f"Initializing MultiTalk V74 Official Wrapper")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"GCC should now be available for runtime compilation")
        
        # Verify gcc is available
        try:
            gcc_version = subprocess.run(['gcc', '--version'], 
                                       capture_output=True, 
                                       text=True, 
                                       check=True)
            logger.info(f"✅ GCC available: {gcc_version.stdout.split('\\n')[0]}")
        except Exception as e:
            logger.error(f"⚠️ GCC check failed: {str(e)}")
        
        # Set up paths
        self.wan_path = self.model_path / "wan2.1-i2v-14b-480p-official"
        self.multitalk_path = self.model_path / "multitalk-official"
        self.wav2vec_path = self.model_path / "wav2vec2-base-960h"
        
        # Verify all models exist
        self._verify_models()
        
        # Set up model linking
        self._setup_model_linking()
        
        # Add wan directory to Python path for imports
        wan_repo_path = self.multitalk_path / "wan"
        if wan_repo_path.exists():
            sys.path.insert(0, str(self.multitalk_path))
            logger.info(f"✅ Added to Python path: {self.multitalk_path}")
        
    def _verify_models(self):
        """Verify all required models are present."""
        logger.info("Verifying models...")
        
        # Check Wan2.1
        if not self.wan_path.exists():
            raise RuntimeError(f"Wan2.1 model not found at {self.wan_path}")
        logger.info(f"✅ Wan2.1 found: {self.wan_path}")
        
        # Check MultiTalk
        if not self.multitalk_path.exists():
            raise RuntimeError(f"MultiTalk model not found at {self.multitalk_path}")
        logger.info(f"✅ MultiTalk found: {self.multitalk_path}")
        
        # Check Wav2Vec2
        if not self.wav2vec_path.exists():
            raise RuntimeError(f"Wav2Vec2 model not found at {self.wav2vec_path}")
        logger.info(f"✅ Wav2Vec2 found: {self.wav2vec_path}")
        
        # Verify generate_multitalk.py exists
        generate_script = self.multitalk_path / "generate_multitalk.py"
        if not generate_script.exists():
            raise RuntimeError(f"generate_multitalk.py not found at {generate_script}")
        logger.info(f"✅ Official script found: {generate_script}")
        
        # Verify wan subdirectories
        wan_dir = self.multitalk_path / "wan"
        required_subdirs = ["configs", "distributed", "modules", "utils"]
        for subdir in required_subdirs:
            subdir_path = wan_dir / subdir
            if not subdir_path.exists():
                raise RuntimeError(f"Missing wan/{subdir} directory")
            logger.info(f"✅ Found wan/{subdir}")
    
    def _setup_model_linking(self):
        """Set up model linking as per official implementation."""
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
            logger.error(f"❌ Model linking failed: {str(e)}")
            raise
    
    def generate(self, audio_path: str, image_path: str, output_path: str) -> Dict[str, Any]:
        """
        Generate video using official MultiTalk script.
        
        Args:
            audio_path: Path to input audio file
            image_path: Path to input image file  
            output_path: Path for output video
            
        Returns:
            Dict with generation results
        """
        logger.info("=" * 80)
        logger.info("Starting official MultiTalk generation")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Image: {image_path}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)
        
        # Prepare command
        generate_script = self.multitalk_path / "generate_multitalk.py"
        
        cmd = [
            sys.executable, str(generate_script),
            "--wan_ckpt_path", str(self.wan_path),
            "--mt_ckpt_path", str(self.multitalk_path),
            "--wav2vec_ckpt_path", str(self.wav2vec_path),
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
        
        # Ensure CC is set for subprocess
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
            raise RuntimeError(f"MultiTalk generation failed: {e.stderr}")
        except Exception as e:
            logger.error(f"❌ Generation error: {str(e)}")
            raise