#!/usr/bin/env python3
"""
MultiTalk V74.5 Adaptive Wrapper
Adaptive model detection with diagnostic capabilities for successful lip-sync generation
"""

import os
import sys
import logging
import time
import subprocess
import traceback
import json
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

class MultiTalkV74AdaptiveWrapper:
    """
    Adaptive wrapper for official MultiTalk implementation.
    Intelligently detects model formats and ensures high-quality lip-sync generation.
    """
    
    def __init__(self):
        """Initialize with adaptive validation."""
        logger.info("🔍 Starting MultiTalk V74.5 Adaptive Validation")
        
        # Set model base path
        self.model_base = Path(os.environ.get('MODEL_PATH', '/runpod-volume/models'))
        
        # Perform comprehensive validation
        self._validate_system()
        
        logger.info("✅ All validations passed - ready for high-quality lip-sync generation")
    
    def _validate_system(self):
        """Perform comprehensive system validation."""
        # Validate dependencies
        self._validate_system_dependencies()
        
        # Validate resources
        self._validate_system_resources()
        
        # Validate environment
        self._validate_environment()
        
        # Adaptive model discovery and validation
        self._discover_and_validate_models_adaptive()
    
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
    
    def _examine_wan_model_structure(self, wan_path: Path) -> Dict[str, Any]:
        """Examine the structure of a Wan2.1 model directory."""
        logger.info(f"🔍 Examining Wan2.1 model structure: {wan_path}")
        
        analysis = {
            'path': str(wan_path),
            'exists': wan_path.exists(),
            'files': [],
            'directories': [],
            'config_files': {},
            'model_files': [],
            'estimated_format': 'unknown'
        }
        
        if not wan_path.exists():
            return analysis
        
        try:
            # Scan all files and directories
            for item in wan_path.rglob('*'):
                rel_path = str(item.relative_to(wan_path))
                if item.is_file():
                    size = item.stat().st_size
                    analysis['files'].append({
                        'path': rel_path,
                        'size': size,
                        'size_mb': size / (1024 * 1024)
                    })
                    
                    # Identify model files
                    if any(ext in item.suffix.lower() for ext in ['.safetensors', '.bin', '.pt', '.pth', '.ckpt']):
                        analysis['model_files'].append(rel_path)
                    
                    # Read config files
                    if item.suffix.lower() == '.json':
                        try:
                            with open(item, 'r') as f:
                                config = json.load(f)
                            analysis['config_files'][rel_path] = config
                        except Exception as e:
                            logger.warning(f"Could not read JSON file {rel_path}: {e}")
                            
                elif item.is_dir():
                    analysis['directories'].append(rel_path)
            
            # Estimate format based on files found
            if any('model.safetensors' in f['path'] for f in analysis['files']):
                analysis['estimated_format'] = 'safetensors'
            elif any('.safetensors' in f['path'] for f in analysis['files']):
                analysis['estimated_format'] = 'safetensors_sharded'
            elif any('pytorch_model.bin' in f['path'] for f in analysis['files']):
                analysis['estimated_format'] = 'pytorch'
            elif any('model_index.json' in f['path'] for f in analysis['files']):
                analysis['estimated_format'] = 'diffusers'
            elif analysis['model_files']:
                analysis['estimated_format'] = 'custom'
            
            logger.info(f"Model format analysis:")
            logger.info(f"  - Estimated format: {analysis['estimated_format']}")
            logger.info(f"  - Total files: {len(analysis['files'])}")
            logger.info(f"  - Model files: {len(analysis['model_files'])}")
            logger.info(f"  - Config files: {len(analysis['config_files'])}")
            
            # Log some key files
            large_files = [f for f in analysis['files'] if f['size_mb'] > 100]
            if large_files:
                logger.info(f"  - Large model files:")
                for f in sorted(large_files, key=lambda x: x['size_mb'], reverse=True)[:5]:
                    logger.info(f"    • {f['path']} ({f['size_mb']:.1f}MB)")
                    
        except Exception as e:
            logger.error(f"Error examining model structure: {e}")
        
        return analysis
    
    def _discover_and_validate_models_adaptive(self):
        """Adaptively discover and validate required models."""
        logger.info("🔍 Discovering and validating models adaptively...")
        
        # List available models
        logger.info(f"Available items in {self.model_base}:")
        for item in sorted(self.model_base.iterdir()):
            logger.info(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        # Find models with adaptive discovery
        self.wan_path = None
        self.multitalk_path = None
        self.wav2vec_path = None
        
        # 1. Find Wan2.1 model - ADAPTIVE DETECTION
        wan_candidates = [
            self.model_base / "wan2.1-i2v-14b-480p",
            self.model_base / "wan2.1-i2v-14b-480p-official", 
            self.model_base / "Wan2.1-I2V-14B-480P",
            self.model_base / "wan"
        ]
        
        for candidate in wan_candidates:
            if candidate.exists() and candidate.is_dir():
                # Perform detailed analysis
                analysis = self._examine_wan_model_structure(candidate)
                
                # Accept if it has any model files or reasonable structure
                if (analysis['model_files'] or 
                    len(analysis['files']) > 10 or  # Has substantial content
                    analysis['estimated_format'] != 'unknown'):
                    
                    self.wan_path = candidate
                    logger.info(f"✅ Wan2.1 model found and analyzed: {candidate}")
                    logger.info(f"   Format: {analysis['estimated_format']}")
                    logger.info(f"   Files: {len(analysis['files'])}, Model files: {len(analysis['model_files'])}")
                    break
                else:
                    logger.warning(f"Wan model at {candidate} appears empty or incomplete")
        
        if not self.wan_path:
            logger.error("❌ No valid Wan2.1 model found")
            logger.info("Available directories for debugging:")
            for candidate in wan_candidates:
                if candidate.exists():
                    analysis = self._examine_wan_model_structure(candidate)
                    logger.info(f"  {candidate}: {len(analysis['files'])} files, format: {analysis['estimated_format']}")
            raise RuntimeError("❌ Wan2.1 model not found or invalid! Check model installation.")
        
        # 2. Find Wav2Vec2 model - REQUIRED
        wav2vec_candidates = [
            self.model_base / "wav2vec2",
            self.model_base / "chinese-wav2vec2-base",
            self.model_base / "wav2vec2-base-960h",
            self.model_base / "wav2vec2-large-960h"
        ]
        
        for candidate in wav2vec_candidates:
            if candidate.exists() and candidate.is_dir():
                # Basic validation - just check it has some files
                files = list(candidate.rglob('*'))
                if len(files) > 5:  # Has some content
                    self.wav2vec_path = candidate
                    logger.info(f"✅ Wav2Vec2 model found: {candidate}")
                    break
        
        if not self.wav2vec_path:
            raise RuntimeError("❌ Wav2Vec2 model not found! Required for audio processing.")
        
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
                    logger.info(f"✅ MultiTalk implementation found: {candidate}")
                    break
        
        if not self.multitalk_path:
            raise RuntimeError("❌ MultiTalk implementation not found! Required for video generation.")
        
        # Log discovery results
        logger.info("🎯 Final model discovery results:")
        logger.info(f"  - Wan2.1: {self.wan_path}")
        logger.info(f"  - MultiTalk: {self.multitalk_path}")
        logger.info(f"  - Wav2Vec2: {self.wav2vec_path}")
        
        # Add MultiTalk to Python path
        if str(self.multitalk_path) not in sys.path:
            sys.path.insert(0, str(self.multitalk_path))
            logger.info(f"✅ Added to Python path: {self.multitalk_path}")
    
    def _validate_input_files(self, audio_path: str, image_path: str):
        """Validate input files for optimal lip-sync generation."""
        logger.info("🔍 Validating input files for lip-sync generation...")
        
        # Validate audio file
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        audio_size = os.path.getsize(audio_path)
        if audio_size == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")
        
        # Validate audio format and quality
        try:
            import soundfile as sf
            data, samplerate = sf.read(audio_path)
            duration = len(data) / samplerate
            
            logger.info(f"Audio analysis:")
            logger.info(f"  - Samples: {len(data)}")
            logger.info(f"  - Sample rate: {samplerate}Hz")
            logger.info(f"  - Duration: {duration:.2f}s")
            logger.info(f"  - Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
            
            if len(data) == 0:
                raise ValueError("Audio file has no samples")
            
            if duration < 0.5:
                logger.warning(f"⚠️ Very short audio: {duration:.2f}s (may affect lip-sync quality)")
            elif duration > 30:
                logger.warning(f"⚠️ Long audio: {duration:.2f}s (processing may take time)")
            
            if samplerate < 16000:
                logger.warning(f"⚠️ Low sample rate: {samplerate}Hz (22050Hz+ recommended for better quality)")
            
            logger.info("✅ Audio file validated for lip-sync generation")
        except Exception as e:
            raise ValueError(f"Invalid audio file: {e}")
        
        # Validate image file
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        
        image_size = os.path.getsize(image_path)
        if image_size == 0:
            raise ValueError(f"Image file is empty: {image_path}")
        
        # Validate image format and quality
        try:
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            
            logger.info(f"Image analysis:")
            logger.info(f"  - Size: {width}x{height} pixels")
            logger.info(f"  - Mode: {img.mode}")
            logger.info(f"  - Format: {img.format}")
            
            if width < 256 or height < 256:
                raise ValueError(f"Image too small: {width}x{height} (min 256x256 for quality)")
            
            if width > 2048 or height > 2048:
                logger.warning(f"⚠️ Large image: {width}x{height} (may be downscaled)")
            
            # Check for face detection hint
            aspect_ratio = width / height
            if 0.7 <= aspect_ratio <= 1.3:
                logger.info("✅ Good aspect ratio for portrait/face images")
            else:
                logger.warning(f"⚠️ Unusual aspect ratio: {aspect_ratio:.2f} (portraits work best)")
            
            if img.mode not in ['RGB', 'RGBA']:
                logger.info(f"Converting image from {img.mode} to RGB")
                img = img.convert('RGB')
                img.save(image_path)
            
            logger.info("✅ Image file validated for lip-sync generation")
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")
    
    def generate(self, audio_path: str, image_path: str, output_path: str) -> Dict[str, Any]:
        """Generate high-quality lip-synced video using official MultiTalk implementation."""
        logger.info("=" * 80)
        logger.info("🎬 Starting HIGH-QUALITY LIP-SYNC VIDEO GENERATION")
        logger.info(f"Audio: {audio_path}")
        logger.info(f"Image: {image_path}")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)
        
        try:
            # Validate inputs for optimal quality
            self._validate_input_files(audio_path, image_path)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Run official script with optimized parameters for quality
            return self._run_official_script_with_quality_focus(audio_path, image_path, output_path)
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _run_official_script_with_quality_focus(self, audio_path: str, image_path: str, output_path: str) -> Dict[str, Any]:
        """Run the official script with parameters optimized for high-quality lip-sync."""
        generate_script = self.multitalk_path / "generate_multitalk.py"
        
        if not generate_script.exists():
            raise RuntimeError(f"generate_multitalk.py not found at {generate_script}")
        
        # Calculate optimal frame count based on audio duration
        try:
            import soundfile as sf
            data, samplerate = sf.read(audio_path)
            duration = len(data) / samplerate
            # Use 25fps for smooth motion, ensure minimum 24 frames
            num_frames = max(24, int(duration * 25))
            logger.info(f"Calculated {num_frames} frames for {duration:.2f}s audio (25fps)")
        except:
            num_frames = 96  # Fallback
            logger.warning("Could not calculate duration, using default 96 frames")
        
        # Prepare command with quality-focused parameters
        cmd = [
            sys.executable, str(generate_script),
            "--wan_ckpt_path", str(self.wan_path),
            "--mt_ckpt_path", str(self.multitalk_path), 
            "--wav2vec_ckpt_path", str(self.wav2vec_path),
            "--ref_img_path", image_path,
            "--ref_audio_path", audio_path,
            "--save_path", output_path,
            "--num_frames", str(num_frames),
            "--fps", "25",  # High FPS for smooth motion
            "--seed", "42", # Consistent results
            "--device", "cuda"
        ]
        
        logger.info("🚀 Running MultiTalk with high-quality parameters:")
        logger.info(f"   Command: {' '.join(cmd)}")
        
        # Set environment for optimal performance
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{self.multitalk_path}:{env.get('PYTHONPATH', '')}"
        env['CUDA_VISIBLE_DEVICES'] = "0"
        env['CC'] = 'gcc'
        env['CXX'] = 'g++'
        env['CUDA_HOME'] = '/usr/local/cuda'
        
        # Run with enhanced monitoring
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
            
            # Monitor process with extended timeout for quality generation
            timeout = 900  # 15 minutes for high-quality generation
            logger.info(f"⏱️ Starting generation with {timeout}s timeout...")
            
            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.error(f"❌ Generation timed out after {timeout}s")
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                    raise TimeoutError(f"High-quality generation timed out after {timeout}s")
                
                # Progress updates every 30 seconds
                if elapsed % 30 == 0 and elapsed > 0:
                    mem = psutil.virtual_memory()
                    gpu_mem = "N/A"
                    if torch.cuda.is_available():
                        gpu_mem = f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
                    logger.info(f"🎬 Generation progress: {elapsed:.0f}s, RAM: {mem.percent}%, GPU: {gpu_mem}")
                
                time.sleep(1)
            
            # Get output and analyze results
            stdout, stderr = process.communicate()
            elapsed = time.time() - start_time
            
            if process.returncode != 0:
                logger.error(f"❌ MultiTalk generation failed with code {process.returncode}")
                logger.error(f"STDOUT:\n{stdout}")
                logger.error(f"STDERR:\n{stderr}")
                raise RuntimeError(f"MultiTalk generation failed with code {process.returncode}")
            
            # Verify high-quality output was created
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output video not created at {output_path}")
            
            output_size = os.path.getsize(output_path)
            if output_size == 0:
                raise RuntimeError(f"Output video is empty")
            
            # Analyze output quality
            output_mb = output_size / (1024 * 1024)
            fps = num_frames / duration if 'duration' in locals() else 25
            
            logger.info("🎉 HIGH-QUALITY LIP-SYNC VIDEO GENERATED SUCCESSFULLY!")
            logger.info(f"   ⏱️ Generation time: {elapsed:.1f}s")
            logger.info(f"   📁 Output size: {output_mb:.1f}MB")
            logger.info(f"   🎬 Frames: {num_frames} @ {fps:.1f}fps")
            logger.info(f"   📊 Quality: {output_mb/elapsed:.1f}MB/s processing rate")
            
            return {
                'success': True,
                'output_path': output_path,
                'size': output_size,
                'size_mb': output_mb,
                'generation_time': elapsed,
                'frames': num_frames,
                'fps': fps,
                'quality_metrics': {
                    'processing_rate_mb_per_sec': output_mb/elapsed,
                    'file_size_mb': output_mb,
                    'duration_sec': duration if 'duration' in locals() else None
                },
                'stdout': stdout,
                'stderr': stderr
            }
            
        except subprocess.TimeoutExpired:
            process.kill()
            raise TimeoutError("High-quality generation process timed out")
        except Exception as e:
            if 'process' in locals():
                process.kill()
            raise RuntimeError(f"High-quality generation failed: {e}")

if __name__ == "__main__":
    # Test initialization
    try:
        wrapper = MultiTalkV74AdaptiveWrapper()
        logger.info("✅ MultiTalk V74.5 adaptive wrapper initialized successfully")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)