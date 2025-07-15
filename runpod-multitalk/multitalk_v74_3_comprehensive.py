"""
MultiTalk V74.3 Comprehensive Validation Wrapper
Version: 74.3.0 - Proactive Validation Framework

Addresses all potential failure points:
- Dependency validation (NumPy/SciPy/PyTorch compatibility)
- Resource validation (GPU memory, disk space, RAM)
- Model structure validation (complete file structure)
- Input validation (audio/image format and content)
- Configuration validation (paths, environment, S3)
- Runtime monitoring (timeouts, process health)
"""

import os
import sys
import json
import logging
import subprocess
import shutil
import signal
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch

# Configure logging
logger = logging.getLogger(__name__)

class MultiTalkV74ComprehensiveWrapper:
    """Comprehensive MultiTalk wrapper with proactive validation."""
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        """Initialize with comprehensive validation."""
        self.model_path = Path(model_path)
        
        logger.info("🔍 Starting MultiTalk V74.3 Comprehensive Validation")
        
        # Run comprehensive validation
        self._validate_system_dependencies()
        self._validate_resources()
        self._validate_environment()
        self._discover_and_validate_models()
        
        logger.info("✅ All validations passed - ready for inference")
    
    def _validate_system_dependencies(self):
        """Validate all system dependencies and versions."""
        logger.info("🔍 Validating system dependencies...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 10:
            raise RuntimeError(f"Python 3.10+ required, found: {python_version.major}.{python_version.minor}")
        
        # Check critical packages and versions
        try:
            import numpy as np
            logger.info(f"NumPy version: {np.__version__}")
            
            # Check for NumPy 2.x incompatibility
            if np.__version__.startswith('2.'):
                raise RuntimeError(f"NumPy 2.x not supported. Found: {np.__version__}. Need NumPy < 2.0.0")
            
            import scipy
            logger.info(f"SciPy version: {scipy.__version__}")
            
            # Validate NumPy/SciPy compatibility
            try:
                from scipy.spatial.distance import cdist
                test_array = np.array([[1, 2], [3, 4]])
                cdist(test_array, test_array)  # Test the problematic function
                logger.info("✅ NumPy/SciPy compatibility verified")
            except Exception as e:
                raise RuntimeError(f"NumPy/SciPy incompatibility detected: {e}")
            
            import torch
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"GPU count: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
            
        except ImportError as e:
            raise RuntimeError(f"Missing critical dependency: {e}")
        
        # Check gcc for Triton compilation
        try:
            result = subprocess.run(['gcc', '--version'], capture_output=True, text=True, check=True)
            gcc_version = result.stdout.split('\n')[0]
            logger.info(f"✅ GCC available: {gcc_version}")
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError("GCC not found - required for Triton compilation")
        
        # Check ffmpeg for audio processing
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("✅ FFmpeg available")
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("⚠️ FFmpeg not found - may cause audio processing issues")
    
    def _validate_resources(self):
        """Validate system resources."""
        logger.info("🔍 Validating system resources...")
        
        # Check RAM
        memory = psutil.virtual_memory()
        available_ram_gb = memory.available / (1024**3)
        logger.info(f"Available RAM: {available_ram_gb:.1f}GB")
        
        if available_ram_gb < 8:
            raise RuntimeError(f"Insufficient RAM: {available_ram_gb:.1f}GB available, 8GB+ required")
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU memory: {gpu_memory:.1f}GB")
            
            if gpu_memory < 16:
                logger.warning(f"⚠️ Limited GPU memory: {gpu_memory:.1f}GB. May cause OOM errors.")
        else:
            raise RuntimeError("CUDA not available - GPU required for MultiTalk")
        
        # Check disk space
        disk_usage = psutil.disk_usage('/tmp')
        free_space_gb = disk_usage.free / (1024**3)
        logger.info(f"Available disk space (/tmp): {free_space_gb:.1f}GB")
        
        if free_space_gb < 5:
            raise RuntimeError(f"Insufficient disk space: {free_space_gb:.1f}GB available, 5GB+ required")
        
        # Check model volume space
        if self.model_path.exists():
            volume_usage = psutil.disk_usage(str(self.model_path))
            volume_free_gb = volume_usage.free / (1024**3)
            logger.info(f"Model volume free space: {volume_free_gb:.1f}GB")
    
    def _validate_environment(self):
        """Validate environment configuration."""
        logger.info("🔍 Validating environment configuration...")
        
        # Check S3 configuration
        s3_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'S3_BUCKET']
        missing_vars = [var for var in s3_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.warning(f"⚠️ Missing S3 environment variables: {missing_vars}")
        else:
            logger.info("✅ S3 environment variables configured")
            
            # Test S3 connectivity
            try:
                import boto3
                s3_client = boto3.client('s3')
                bucket = os.environ.get('S3_BUCKET')
                s3_client.head_bucket(Bucket=bucket)
                logger.info(f"✅ S3 connectivity verified: {bucket}")
            except Exception as e:
                logger.warning(f"⚠️ S3 connectivity test failed: {e}")
        
        # Check CUDA environment
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        if Path(cuda_home).exists():
            logger.info(f"✅ CUDA_HOME: {cuda_home}")
        else:
            logger.warning(f"⚠️ CUDA_HOME not found: {cuda_home}")
    
    def _discover_and_validate_models(self):
        """Discover and validate model structure."""
        logger.info("🔍 Discovering and validating models...")
        
        # List available items
        if self.model_path.exists():
            available_items = list(self.model_path.iterdir())
            logger.info(f"Available items in {self.model_path}:")
            for item in available_items:
                logger.info(f"  - {item.name} {'(dir)' if item.is_dir() else '(file)'}")
        else:
            logger.warning(f"Model path does not exist: {self.model_path}")
            self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Find models with validation
        self.wan_path = self._find_and_validate_wan_model()
        self.wav2vec_path = self._find_and_validate_wav2vec_model()
        self.multitalk_path = self._find_and_validate_multitalk_implementation()
        
        # Log discovery results
        logger.info("Model discovery results:")
        logger.info(f"  - Wan2.1: {self.wan_path if self.wan_path else 'NOT FOUND'}")
        logger.info(f"  - MultiTalk: {self.multitalk_path if self.multitalk_path else 'NOT FOUND'}")
        logger.info(f"  - Wav2Vec2: {self.wav2vec_path if self.wav2vec_path else 'NOT FOUND'}")
        
        # Validate critical paths
        if not self.multitalk_path:
            raise RuntimeError("MultiTalk implementation not found. Required for inference.")
        
        # Add to Python path
        sys.path.insert(0, str(self.multitalk_path))
        logger.info(f"✅ Added to Python path: {self.multitalk_path}")
    
    def _find_and_validate_wan_model(self) -> Optional[Path]:
        """Find and validate Wan2.1 model."""
        possible_names = ["wan2.1", "wan-2.1", "wan21", "wan"]
        
        for name in possible_names:
            for suffix in ["-i2v-14b-480p", "-official", "_official", "-model", "_model", ""]:
                full_name = f"{name}{suffix}"
                path = self.model_path / full_name
                if path.exists() and path.is_dir():
                    # Validate Wan model structure
                    expected_files = ["model_index.json"]
                    missing_files = [f for f in expected_files if not (path / f).exists()]
                    
                    if missing_files:
                        logger.warning(f"Wan model at {path} missing files: {missing_files}")
                    else:
                        logger.info(f"✅ Valid Wan model found: {path}")
                        return path
        
        logger.warning("⚠️ Wan2.1 model not found or incomplete")
        return None
    
    def _find_and_validate_wav2vec_model(self) -> Optional[Path]:
        """Find and validate Wav2Vec2 model."""
        possible_names = ["wav2vec", "wav2vec2", "chinese-wav2vec2-base", "wav2vec2-base-960h"]
        
        for name in possible_names:
            path = self.model_path / name
            if path.exists() and path.is_dir():
                # Check for common Wav2Vec2 files
                common_files = ["config.json", "pytorch_model.bin"]
                has_files = any((path / f).exists() for f in common_files)
                
                if has_files:
                    logger.info(f"✅ Valid Wav2Vec2 model found: {path}")
                    return path
                else:
                    logger.warning(f"Wav2Vec2 model at {path} appears incomplete")
        
        logger.warning("⚠️ Wav2Vec2 model not found")
        return None
    
    def _find_and_validate_multitalk_implementation(self) -> Optional[Path]:
        """Find and validate MultiTalk implementation."""
        # Priority order for finding the official implementation
        possible_locations = [
            Path("/app/multitalk_official"),  # From setup script
            self.model_path / "multitalk-official",
            self.model_path / "meigen-multitalk"
        ]
        
        for location in possible_locations:
            if location.exists():
                # Validate required files
                required_files = [
                    "generate_multitalk.py",
                    "wan/__init__.py",
                    "wan/modules/__init__.py",
                    "wan/configs/__init__.py",
                    "wan/utils/__init__.py"
                ]
                
                missing_files = []
                for file_path in required_files:
                    full_path = location / file_path
                    if not full_path.exists():
                        missing_files.append(file_path)
                    elif full_path.stat().st_size == 0:
                        missing_files.append(f"{file_path} (empty)")
                
                if missing_files:
                    logger.warning(f"MultiTalk at {location} missing files: {missing_files}")
                    continue
                else:
                    logger.info(f"✅ Valid MultiTalk implementation found: {location}")
                    return location
        
        logger.error("❌ Complete MultiTalk implementation not found")
        logger.error("Expected locations checked:")
        for location in possible_locations:
            logger.error(f"  - {location}")
        
        return None
    
    def validate_inputs(self, audio_path: str, image_path: str):
        """Validate input files before processing."""
        logger.info("🔍 Validating input files...")
        
        # Validate audio file
        try:
            audio_path_obj = Path(audio_path)
            if not audio_path_obj.exists():
                raise RuntimeError(f"Audio file not found: {audio_path}")
            
            if audio_path_obj.stat().st_size == 0:
                raise RuntimeError("Audio file is empty")
            
            # Try to load with librosa if available
            try:
                import librosa
                audio, sr = librosa.load(audio_path, sr=None)
                logger.info(f"Audio: {len(audio)} samples at {sr}Hz")
                
                if len(audio) == 0:
                    raise RuntimeError("Audio file contains no audio data")
                if sr < 8000:
                    raise RuntimeError(f"Audio sample rate too low: {sr}Hz")
                
                logger.info("✅ Audio file validated")
            except ImportError:
                logger.warning("⚠️ librosa not available - skipping detailed audio validation")
                
        except Exception as e:
            raise RuntimeError(f"Audio file validation failed: {e}")
        
        # Validate image file
        try:
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                raise RuntimeError(f"Image file not found: {image_path}")
            
            if image_path_obj.stat().st_size == 0:
                raise RuntimeError("Image file is empty")
            
            # Try to load with PIL
            try:
                from PIL import Image
                image = Image.open(image_path)
                logger.info(f"Image: {image.size} pixels, mode: {image.mode}")
                
                if image.mode not in ['RGB', 'RGBA', 'L']:
                    logger.warning(f"⚠️ Unusual image mode: {image.mode}")
                
                if min(image.size) < 64:
                    raise RuntimeError(f"Image too small: {image.size}")
                
                logger.info("✅ Image file validated")
            except ImportError:
                logger.warning("⚠️ PIL not available - skipping detailed image validation")
                
        except Exception as e:
            raise RuntimeError(f"Image file validation failed: {e}")
    
    def generate(self, audio_path: str, image_path: str, output_path: str) -> Dict[str, Any]:
        """Generate video with comprehensive monitoring."""
        logger.info("=" * 80)
        logger.info("Starting MultiTalk generation with comprehensive validation")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Image: {image_path}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)
        
        # Pre-generation validation
        self.validate_inputs(audio_path, image_path)
        
        # Check if we have the official script
        if not self.multitalk_path:
            raise RuntimeError("MultiTalk implementation not available")
        
        generate_script = self.multitalk_path / "generate_multitalk.py"
        if not generate_script.exists():
            raise RuntimeError(f"Official MultiTalk script not found at {generate_script}")
        
        return self._run_official_script_with_monitoring(audio_path, image_path, output_path)
    
    def _run_official_script_with_monitoring(self, audio_path: str, image_path: str, output_path: str) -> Dict[str, Any]:
        """Run the official script with comprehensive monitoring."""
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
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise RuntimeError(f"Video generation timed out after {timeout}s")
                
                time.sleep(1)
            
            # Get results
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ Official script completed successfully")
                if stdout:
                    logger.info(f"STDOUT:\n{stdout}")
                if stderr:
                    logger.warning(f"STDERR:\n{stderr}")
                
                # Validate output
                if not os.path.exists(output_path):
                    raise RuntimeError("Output video not created")
                
                output_size = os.path.getsize(output_path)
                if output_size == 0:
                    raise RuntimeError("Output video is empty")
                
                logger.info(f"✅ Video generated: {output_path} ({output_size} bytes)")
                
                return {
                    'success': True,
                    'output_path': output_path,
                    'size': output_size,
                    'duration': 96 / 25.0,  # frames / fps
                    'generation_time': time.time() - start_time
                }
            else:
                logger.error(f"❌ Official script failed with code {process.returncode}")
                logger.error(f"STDOUT:\n{stdout}")
                logger.error(f"STDERR:\n{stderr}")
                raise RuntimeError(f"Official MultiTalk script execution failed with code {process.returncode}")
                
        except Exception as e:
            logger.error(f"❌ Script execution failed: {str(e)}")
            raise