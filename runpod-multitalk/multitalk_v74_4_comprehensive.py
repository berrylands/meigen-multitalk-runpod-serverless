#!/usr/bin/env python3
"""
MultiTalk V74.4 Comprehensive Wrapper
Complete official implementation with strict validation - no dummy/mock paths
"""

import os
import sys
import logging
import time
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import psutil
import shutil

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiTalkV74ComprehensiveWrapper:
    """
    Comprehensive wrapper for official MultiTalk implementation.
    NO DUMMY/MOCK PATHS - Fails fast if requirements not met.
    """
    
    def __init__(self):
        """Initialize with comprehensive validation."""
        logger.info("🔍 Starting MultiTalk V74.4 Comprehensive Validation")
        
        # Set model base path
        self.model_base = Path(os.environ.get('MODEL_PATH', '/runpod-volume/models'))
        
        # Perform comprehensive validation
        self._validate_system()
        
        logger.info("✅ All validations passed - ready for inference")
    
    def _validate_system(self):
        """Perform comprehensive system validation."""
        # Validate dependencies
        self._validate_system_dependencies()
        
        # Validate resources
        self._validate_system_resources()
        
        # Validate environment
        self._validate_environment()
        
        # Validate and discover models - STRICT MODE
        self._discover_and_validate_models()
    
    def _validate_system_dependencies(self):
        """Validate all system dependencies and versions."""
        logger.info("🔍 Validating system dependencies...")
        
        # Check NumPy/SciPy compatibility
        try:
            import numpy as np
            logger.info(f"NumPy version: {np.__version__}")
            
            import scipy
            logger.info(f"SciPy version: {scipy.__version__}")
            
            # Test the problematic function
            from scipy.spatial.distance import cdist
            test_array = np.array([[1, 2], [3, 4]])
            cdist(test_array, test_array)
            logger.info("✅ NumPy/SciPy compatibility verified")
        except Exception as e:
            raise RuntimeError(f"NumPy/SciPy incompatibility detected: {e}")
        
        # Check PyTorch and CUDA
        try:
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
        except Exception as e:
            logger.warning(f"PyTorch/CUDA check failed: {e}")
        
        # Check GCC
        try:
            gcc_result = subprocess.run(['gcc', '--version'], capture_output=True, text=True)
            if gcc_result.returncode == 0:
                gcc_version = gcc_result.stdout.split('\n')[0]
                logger.info(f"✅ GCC available: {gcc_version}")
            else:
                raise RuntimeError("GCC not found - required for xformers compilation")
        except Exception as e:
            raise RuntimeError(f"GCC check failed: {e}")
        
        # Check FFmpeg
        try:
            ffmpeg_result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if ffmpeg_result.returncode == 0:
                logger.info("✅ FFmpeg available")
            else:
                raise RuntimeError("FFmpeg not found - required for video processing")
        except Exception as e:
            raise RuntimeError(f"FFmpeg check failed: {e}")
    
    def _validate_system_resources(self):
        """Validate system resources."""
        logger.info("🔍 Validating system resources...")
        
        # Check RAM
        mem = psutil.virtual_memory()
        available_ram_gb = mem.available / (1024**3)
        logger.info(f"Available RAM: {available_ram_gb:.1f}GB")
        if available_ram_gb < 16:
            logger.warning(f"⚠️ Low RAM: {available_ram_gb:.1f}GB available (16GB+ recommended)")
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU memory: {gpu_mem:.1f}GB")
            if gpu_mem < 16:
                logger.warning(f"⚠️ Low GPU memory: {gpu_mem:.1f}GB (16GB+ recommended)")
        
        # Check disk space
        if os.path.exists('/tmp'):
            tmp_space = shutil.disk_usage('/tmp').free / (1024**3)
            logger.info(f"Available disk space (/tmp): {tmp_space:.1f}GB")
            if tmp_space < 5:
                logger.warning(f"⚠️ Low disk space: {tmp_space:.1f}GB free in /tmp")
        
        # Check model volume space
        if self.model_base.exists():
            model_space = shutil.disk_usage(self.model_base).free / (1024**3)
            logger.info(f"Model volume free space: {model_space:.1f}GB")
    
    def _validate_environment(self):
        """Validate environment configuration."""
        logger.info("🔍 Validating environment configuration...")
        
        # Check required environment variables
        required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            logger.warning(f"⚠️ Missing S3 environment variables: {missing_vars}")
        
        optional_vars = ['S3_BUCKET', 'AWS_DEFAULT_REGION']
        for var in optional_vars:
            value = os.environ.get(var)
            if value:
                logger.info(f"✅ {var}: {value}")
            else:
                logger.warning(f"⚠️ {var} not set")
        
        # Check CUDA environment
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        if os.path.exists(cuda_home):
            logger.info(f"✅ CUDA_HOME exists: {cuda_home}")
        else:
            logger.warning(f"⚠️ CUDA_HOME not found: {cuda_home}")
    
    def _discover_and_validate_models(self):
        """Discover and validate required models - STRICT MODE."""
        logger.info("🔍 Discovering and validating models...")
        
        # List available models
        logger.info(f"Available items in {self.model_base}:")
        for item in sorted(self.model_base.iterdir()):
            logger.info(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        # Find models with flexible discovery
        self.wan_path = None
        self.multitalk_path = None
        self.wav2vec_path = None
        
        # 1. Find Wan2.1 model - REQUIRED
        wan_candidates = [
            self.model_base / "wan2.1-i2v-14b-480p",
            self.model_base / "wan2.1-i2v-14b-480p-official",
            self.model_base / "Wan2.1-I2V-14B-480P",
            self.model_base / "wan"
        ]
        
        for candidate in wan_candidates:
            if candidate.exists() and candidate.is_dir():
                # Check for common Wan2.1 model files/directories
                # Different model formats may have different structures
                wan_indicators = [
                    'model.safetensors',  # Single file format
                    'model.safetensors.index.json',  # Sharded format
                    'pytorch_model.bin',  # PyTorch format
                    'pytorch_model.bin.index.json',  # Sharded PyTorch
                    'vae',  # VAE directory
                    'transformer',  # Transformer directory
                    'unet',  # UNet directory
                    'diffusion_pytorch_model.safetensors'  # Diffusers format
                ]
                
                found_indicators = []
                for indicator in wan_indicators:
                    if (candidate / indicator).exists():
                        found_indicators.append(indicator)
                
                if found_indicators:
                    logger.info(f"Wan model at {candidate} contains: {found_indicators}")
                    self.wan_path = candidate
                    logger.info(f"✅ Valid Wan2.1 model found: {candidate}")
                    break
                else:
                    logger.warning(f"Wan model at {candidate} doesn't contain expected files")
        
        if not self.wan_path:
            raise RuntimeError("❌ Wan2.1 model not found! Required at one of: " + 
                             ", ".join(str(c) for c in wan_candidates))
        
        # 2. Find Wav2Vec2 model - REQUIRED
        wav2vec_candidates = [
            self.model_base / "wav2vec2",
            self.model_base / "chinese-wav2vec2-base",
            self.model_base / "wav2vec2-base-960h",
            self.model_base / "wav2vec2-large-960h"
        ]
        
        for candidate in wav2vec_candidates:
            if candidate.exists() and candidate.is_dir():
                # Basic validation
                if (candidate / "config.json").exists() or (candidate / "model.safetensors").exists():
                    self.wav2vec_path = candidate
                    logger.info(f"✅ Valid Wav2Vec2 model found: {candidate}")
                    break
        
        if not self.wav2vec_path:
            raise RuntimeError("❌ Wav2Vec2 model not found! Required at one of: " + 
                             ", ".join(str(c) for c in wav2vec_candidates))
        
        # 3. Find MultiTalk implementation - REQUIRED
        multitalk_candidates = [
            Path("/app/multitalk_official"),
            Path("/app/MultiTalk"),
            self.model_base / "MultiTalk",
            self.model_base / "meigen-multitalk"
        ]
        
        for candidate in multitalk_candidates:
            if candidate.exists() and candidate.is_dir():
                # Check for generate_multitalk.py
                if (candidate / "generate_multitalk.py").exists():
                    # Validate critical directories
                    required_dirs = ['wan', 'src']
                    missing_dirs = []
                    for req_dir in required_dirs:
                        if not (candidate / req_dir).exists():
                            missing_dirs.append(req_dir)
                    
                    if missing_dirs:
                        logger.warning(f"MultiTalk at {candidate} missing directories: {missing_dirs}")
                        continue
                    
                    self.multitalk_path = candidate
                    logger.info(f"✅ Valid MultiTalk implementation found: {candidate}")
                    break
        
        if not self.multitalk_path:
            raise RuntimeError("❌ MultiTalk implementation not found! Required at one of: " + 
                             ", ".join(str(c) for c in multitalk_candidates))
        
        # Log discovery results
        logger.info("Model discovery results:")
        logger.info(f"  - Wan2.1: {self.wan_path}")
        logger.info(f"  - MultiTalk: {self.multitalk_path}")
        logger.info(f"  - Wav2Vec2: {self.wav2vec_path}")
        
        # Add MultiTalk to Python path
        if str(self.multitalk_path) not in sys.path:
            sys.path.insert(0, str(self.multitalk_path))
            logger.info(f"✅ Added to Python path: {self.multitalk_path}")
    
    def _validate_input_files(self, audio_path: str, image_path: str):
        """Validate input files."""
        logger.info("🔍 Validating input files...")
        
        # Validate audio file
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        audio_size = os.path.getsize(audio_path)
        if audio_size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        # Validate audio format
        try:
            import soundfile as sf
            data, samplerate = sf.read(audio_path)
            logger.info(f"Audio: {len(data)} samples at {samplerate}Hz")
            
            if len(data) == 0:
                raise ValueError("Audio file has no samples")
            
            if samplerate < 16000:
                logger.warning(f"Low sample rate: {samplerate}Hz (16000Hz+ recommended)")
            
            logger.info("✅ Audio file validated")
        except Exception as e:
            raise ValueError(f"Invalid audio file: {e}")
        
        # Validate image file
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        image_size = os.path.getsize(image_path)
        if image_size == 0:
            raise ValueError(f"Image file is empty: {image_path}")
        
        # Validate image format
        try:
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            logger.info(f"Image: {img.size} pixels, mode: {img.mode}")
            
            if width < 256 or height < 256:
                raise ValueError(f"Image too small: {width}x{height} (min 256x256)")
            
            if img.mode not in ['RGB', 'RGBA']:
                logger.warning(f"Converting image from {img.mode} to RGB")
                img = img.convert('RGB')
                img.save(image_path)
            
            logger.info("✅ Image file validated")
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")
    
    def generate(self, audio_path: str, image_path: str, output_path: str) -> Dict[str, Any]:
        """Generate video using official MultiTalk implementation."""
        logger.info("=" * 80)
        logger.info("Starting MultiTalk generation with comprehensive validation")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Image: {image_path}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)
        
        try:
            # Validate inputs
            self._validate_input_files(audio_path, image_path)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Run official script with monitoring
            return self._run_official_script_with_monitoring(audio_path, image_path, output_path)
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _run_official_script_with_monitoring(self, audio_path: str, image_path: str, output_path: str) -> Dict[str, Any]:
        """Run the official script with comprehensive monitoring - NO DUMMY PATHS."""
        generate_script = self.multitalk_path / "generate_multitalk.py"
        
        if not generate_script.exists():
            raise RuntimeError(f"generate_multitalk.py not found at {generate_script}")
        
        # Prepare command - NO DUMMY PATHS
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
        env['CC'] = 'gcc'
        env['CXX'] = 'g++'
        env['CUDA_HOME'] = '/usr/local/cuda'
        
        # Run with timeout and monitoring
        try:
            start_time = time.time()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(self.multitalk_path)
            )
            
            # Monitor process with timeout
            timeout = 600  # 10 minutes
            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.error(f"Process timed out after {timeout}s")
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                    raise TimeoutError(f"Generation timed out after {timeout}s")
                
                # Check system resources
                if elapsed % 30 == 0:  # Every 30 seconds
                    mem = psutil.virtual_memory()
                    logger.info(f"Progress: {elapsed:.0f}s, RAM: {mem.percent}%, "
                              f"CPU: {psutil.cpu_percent()}%")
                
                time.sleep(1)
            
            # Get output
            stdout, stderr = process.communicate()
            elapsed = time.time() - start_time
            
            if process.returncode != 0:
                logger.error(f"❌ Official script failed with code {process.returncode}")
                logger.error(f"STDOUT:\n{stdout}")
                logger.error(f"STDERR:\n{stderr}")
                raise RuntimeError(f"Official MultiTalk script execution failed with code {process.returncode}")
            
            # Verify output was created
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output video not created at {output_path}")
            
            output_size = os.path.getsize(output_path)
            if output_size == 0:
                raise RuntimeError(f"Output video is empty")
            
            logger.info(f"✅ Video generated successfully in {elapsed:.1f}s")
            logger.info(f"Output size: {output_size / (1024*1024):.1f}MB")
            
            return {
                'success': True,
                'output_path': output_path,
                'size': output_size,
                'generation_time': elapsed,
                'stdout': stdout,
                'stderr': stderr
            }
            
        except subprocess.TimeoutExpired:
            process.kill()
            raise TimeoutError("Generation process timed out")
        except Exception as e:
            if 'process' in locals():
                process.kill()
            raise RuntimeError(f"Script execution failed: {e}")

if __name__ == "__main__":
    # Test initialization
    try:
        wrapper = MultiTalkV74ComprehensiveWrapper()
        logger.info("✅ MultiTalk V74.4 wrapper initialized successfully")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)