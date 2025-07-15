import runpod
import os
import sys
import json
import time
import base64
import tempfile
import traceback
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

# Import S3 handler
S3_AVAILABLE = False
s3_handler = None
try:
    from s3_handler import s3_handler, download_input, prepare_output
    S3_AVAILABLE = True
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] S3 handler imported successfully")
except ImportError as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] S3 handler import failed: {e}")
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Unexpected error importing S3 handler: {e}")

# Import S3 utilities
try:
    from s3_utils import process_input_data
    S3_UTILS_AVAILABLE = True
except ImportError:
    S3_UTILS_AVAILABLE = False
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] S3 utils not available, using fallback")

# Import MultiTalk inference
MULTITALK_AVAILABLE = False
multitalk_inference = None
try:
    from multitalk_inference import MultiTalkInference
    MULTITALK_AVAILABLE = True
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] MultiTalk inference imported successfully")
except ImportError as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] MultiTalk inference import failed: {e}")
except Exception as e:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Unexpected error importing MultiTalk inference: {e}")

# Model paths based on our current inventory
MODEL_BASE = Path(os.environ.get("MODEL_PATH", "/runpod-volume/models"))
MODELS = {
    "multitalk": MODEL_BASE / "meigen-multitalk",
    "wan21": MODEL_BASE / "wan2.1-i2v-14b-480p", 
    "chinese_wav2vec": MODEL_BASE / "chinese-wav2vec2-base",
    "kokoro": MODEL_BASE / "kokoro-82m",
    "wav2vec_base": MODEL_BASE / "wav2vec2-base-960h",
    "wav2vec_large": MODEL_BASE / "wav2vec2-large-960h",
    "gfpgan": MODEL_BASE / "gfpgan"
}

# Global model cache
models_cache = {}
MODEL_LOAD_STATUS = {"loaded": False, "loading": False, "error": None}

def log_message(message: str, level: str = "INFO"):
    """Enhanced logging with timestamps."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def check_gpu_availability():
    """Check if GPU is available and get info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
            return {
                "available": True,
                "count": gpu_count,
                "name": gpu_name,
                "memory_gb": round(gpu_memory, 1)
            }
        else:
            return {"available": False, "count": 0}
    except ImportError:
        return {"available": False, "count": 0, "error": "PyTorch not available"}

def load_models():
    """Load all required models into memory."""
    global models_cache, MODEL_LOAD_STATUS, multitalk_inference
    
    if MODEL_LOAD_STATUS["loaded"]:
        return True
    
    if MODEL_LOAD_STATUS["loading"]:
        log_message("Models already loading...")
        return False
    
    MODEL_LOAD_STATUS["loading"] = True
    log_message("Starting model loading process...")
    
    try:
        # Check GPU
        gpu_info = check_gpu_availability()
        log_message(f"GPU Info: {gpu_info}")
        
        # Check which models exist
        available_models = {}
        for name, path in MODELS.items():
            if path.exists():
                log_message(f"✓ Found {name} at {path}")
                available_models[name] = str(path)
            else:
                log_message(f"✗ Missing {name} at {path}")
        
        if not available_models:
            raise Exception("No models found on volume")
        
        # Initialize MultiTalk inference if available
        if MULTITALK_AVAILABLE:
            try:
                log_message("Initializing MultiTalk inference engine...")
                multitalk_inference = MultiTalkInference(MODEL_BASE)
                if multitalk_inference.load_models():
                    log_message("✓ MultiTalk inference engine ready")
                    models_cache["multitalk_engine"] = "loaded"
                else:
                    log_message("⚠️ MultiTalk models partially loaded - using fallback mode")
            except Exception as e:
                log_message(f"Failed to initialize MultiTalk inference: {e}", "WARNING")
                multitalk_inference = None
        else:
            log_message("MultiTalk inference not available - using dummy implementation")
        
        # Store available model paths
        models_cache.update(available_models)
        
        MODEL_LOAD_STATUS["loaded"] = True
        MODEL_LOAD_STATUS["loading"] = False
        MODEL_LOAD_STATUS["error"] = None
        
        log_message(f"Model loading completed. Available: {list(available_models.keys())}")
        return True
        
    except Exception as e:
        MODEL_LOAD_STATUS["loading"] = False
        MODEL_LOAD_STATUS["error"] = str(e)
        log_message(f"Model loading failed: {e}", "ERROR")
        return False

def process_audio_with_wav2vec(audio_data: bytes, model_name: str = "wav2vec_base") -> Dict[str, Any]:
    """Process audio using Wav2Vec2 models."""
    try:
        log_message(f"Processing audio with {model_name}")
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            tmp_audio.write(audio_data)
            audio_path = tmp_audio.name
        
        try:
            # For now, return mock audio features
            # In production, this would load and run the actual Wav2Vec2 model
            audio_features = {
                "duration": 5.0,  # Mock duration
                "sample_rate": 16000,
                "features_shape": [400, 768],  # Typical Wav2Vec2 output shape
                "features_extracted": True,
                "model_used": model_name,
                "processing_time": time.time()
            }
            
            log_message(f"Audio processing completed: {audio_features['duration']}s")
            return audio_features
            
        finally:
            # Cleanup
            os.unlink(audio_path)
            
    except Exception as e:
        log_message(f"Audio processing failed: {e}", "ERROR")
        return {"error": str(e), "features_extracted": False}

def generate_video_with_multitalk(
    audio_data: bytes,
    reference_image: Optional[bytes] = None,
    **kwargs
) -> Dict[str, Any]:
    """Generate video using MultiTalk and Wan2.1 models."""
    global multitalk_inference
    
    try:
        log_message("Starting MultiTalk video generation...")
        
        # Extract parameters
        duration = kwargs.get('duration', 5.0)
        fps = kwargs.get('fps', 30)
        width = kwargs.get('width', 480)
        height = kwargs.get('height', 480)
        
        num_frames = int(duration * fps)
        
        log_message(f"Generation params: {duration}s, {fps}fps, {width}x{height}, {num_frames} frames")
        
        # Use real MultiTalk inference if available
        if MULTITALK_AVAILABLE and multitalk_inference is not None:
            log_message("Using real MultiTalk inference engine...")
            
            # Process with MultiTalk
            result = multitalk_inference.process_audio_to_video(
                audio_data=audio_data,
                reference_image=reference_image,
                duration=duration,
                fps=fps,
                width=width,
                height=height
            )
            
            if result.get("success"):
                video_data = result["video_data"]
                generation_result = {
                    "success": True,
                    "video_size": len(video_data),
                    "duration": result.get("duration", duration),
                    "fps": result.get("fps", fps),
                    "resolution": result.get("resolution", f"{width}x{height}"),
                    "frames": result.get("frames", num_frames),
                    "models_used": result.get("models_used", ["MultiTalk"]),
                    "audio_features_shape": result.get("audio_features_shape", []),
                    "processing_note": "Real MultiTalk inference"
                }
                
                log_message(f"MultiTalk video generation completed: {len(video_data)} bytes")
                return {"video_data": video_data, **generation_result}
            else:
                log_message(f"MultiTalk inference failed: {result.get('error')}", "WARNING")
                # Fall through to dummy implementation
        
        # Fallback to test video with FFmpeg
        log_message("Using fallback FFmpeg test video generation...")
        output_path = tempfile.mktemp(suffix=".mp4")
        
        # Create test pattern video
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration}:size={width}x{height}:rate={fps}",
            "-f", "lavfi", 
            "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            output_path
        ]
        
        log_message("Generating test video with FFmpeg...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        # Read generated video
        with open(output_path, "rb") as f:
            video_data = f.read()
        
        os.unlink(output_path)
        
        generation_result = {
            "success": True,
            "video_size": len(video_data),
            "duration": duration,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "frames": num_frames,
            "models_used": ["FFmpeg"],
            "processing_note": "Fallback test implementation - MultiTalk not available"
        }
        
        log_message(f"Video generation completed: {len(video_data)} bytes")
        return {"video_data": video_data, **generation_result}
        
    except Exception as e:
        log_message(f"Video generation failed: {e}", "ERROR")
        return {"error": str(e), "success": False}

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Complete MultiTalk handler with full video generation pipeline."""
    
    try:
        job_input = job.get('input', {})
        start_time = time.time()
        
        log_message(f"Processing job: {job.get('id', 'unknown')}")
        
        # Health check
        if job_input.get('health_check'):
            gpu_info = check_gpu_availability()
            
            # Check S3 status
            s3_status = {}
            if S3_AVAILABLE:
                s3_status = s3_handler.check_s3_access()
            else:
                s3_status = {"enabled": False, "error": "S3 handler not available"}
            
            return {
                "status": "healthy",
                "message": "Complete MultiTalk handler ready!",
                "version": "2.2.0", "build_id": "1752220071", "image_tag": "v2.2.0-20250711-multitalk",
                "build_time": os.environ.get("BUILD_TIME", "unknown"),
                "build_id": os.environ.get("BUILD_ID", "unknown"),
                "container_id": os.environ.get("HOSTNAME", "unknown"),
                "models_loaded": MODEL_LOAD_STATUS["loaded"],
                "models_available": {name: path.exists() for name, path in MODELS.items()},
                "models_in_cache": len(models_cache),
                "multitalk_inference": {
                    "available": MULTITALK_AVAILABLE,
                    "loaded": "multitalk_engine" in models_cache,
                    "engine": "Real MultiTalk" if multitalk_inference is not None else "Fallback FFmpeg"
                },
                "gpu_info": gpu_info,
                "s3_integration": s3_status,
                "python_version": sys.version,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown'),
                "volume_mounted": os.path.exists('/runpod-volume'),
                "model_path": str(MODEL_BASE),
                "storage_used_gb": round(sum(
                    sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    for path in MODELS.values() if path.exists()
                ) / (1024**3), 1)
            }
        
        # Model loading
        if job_input.get('action') == 'load_models':
            success = load_models()
            return {
                "success": success,
                "models_loaded": MODEL_LOAD_STATUS["loaded"],
                "error": MODEL_LOAD_STATUS["error"],
                "available_models": list(models_cache.keys())
            }
        
        # Model listing (from previous handler)
        if job_input.get('action') == 'list_models':
            models_info = []
            total_size = 0
            
            for name, path in MODELS.items():
                path_obj = Path(path)
                if path_obj.exists():
                    if path_obj.is_file():
                        size = path_obj.stat().st_size
                    else:
                        size = sum(f.stat().st_size for f in path_obj.rglob("*") if f.is_file())
                    
                    total_size += size
                    models_info.append({
                        "name": name,
                        "path": str(path),
                        "size_mb": round(size / (1024 * 1024), 2),
                        "loaded": name in models_cache,
                        "type": "video" if name in ["multitalk", "wan21"] else 
                               "audio" if "wav2vec" in name or name == "kokoro" else
                               "enhancement"
                    })
            
            return {
                "models": models_info,
                "total": len(models_info),
                "total_size_gb": round(total_size / (1024**3), 1),
                "models_loaded": MODEL_LOAD_STATUS["loaded"]
            }
        
        # Complete MultiTalk video generation
        if job_input.get('action') == 'generate' or 'audio' in job_input:
            log_message("Starting complete MultiTalk pipeline...")
            
            # Ensure models are loaded
            if not MODEL_LOAD_STATUS["loaded"]:
                log_message("Loading models first...")
                if not load_models():
                    return {"error": "Failed to load models", "details": MODEL_LOAD_STATUS["error"]}
            
            # Validate input
            audio_input = job_input.get('audio')
            if not audio_input:
                return {"error": "No audio data provided"}
            
            # Process audio input (S3 URL, S3 filename, or base64)
            try:
                log_message(f"Processing audio input. S3_AVAILABLE: {S3_AVAILABLE}, Input type: {type(audio_input)}, Input preview: {str(audio_input)[:100]}...")
                
                if S3_UTILS_AVAILABLE and isinstance(audio_input, str):
                    # Use smart detection
                    audio_data = process_input_data(
                        audio_input, 
                        "audio",
                        s3_handler if S3_AVAILABLE else None,
                        s3_handler.default_bucket if S3_AVAILABLE else None
                    )
                elif isinstance(audio_input, str):
                    # Fallback to original logic
                    if audio_input.startswith('s3://') or 'amazonaws.com' in audio_input:
                        if not S3_AVAILABLE:
                            return {"error": "S3 URL provided but S3 integration is not available."}
                        log_message(f"Detected S3 URL, downloading from: {audio_input}")
                        audio_data = s3_handler.download_from_s3(audio_input)
                    elif S3_AVAILABLE and not audio_input.startswith('/') and len(audio_input) < 100:
                        # Likely a filename - try S3
                        log_message(f"Treating as S3 key in default bucket: {audio_input}")
                        s3_url = f"s3://{s3_handler.default_bucket}/{audio_input}"
                        audio_data = s3_handler.download_from_s3(s3_url)
                    else:
                        # Try base64
                        audio_data = base64.b64decode(audio_input)
                else:
                    audio_data = audio_input
                    
            except Exception as e:
                return {"error": f"Failed to process audio input: {e}", "input_preview": str(audio_input)[:200]}
            
            # Process reference image (S3 URL, S3 filename, or base64)
            reference_image = job_input.get('reference_image')
            if reference_image and isinstance(reference_image, str):
                try:
                    if S3_UTILS_AVAILABLE:
                        # Use smart detection
                        reference_image = process_input_data(
                            reference_image,
                            "reference_image", 
                            s3_handler if S3_AVAILABLE else None,
                            s3_handler.default_bucket if S3_AVAILABLE else None
                        )
                    else:
                        # Fallback logic
                        if reference_image.startswith('s3://') or 'amazonaws.com' in reference_image:
                            log_message(f"Downloading reference image from S3: {reference_image}")
                            reference_image = s3_handler.download_from_s3(reference_image)
                        elif S3_AVAILABLE and not reference_image.startswith('/') and len(reference_image) < 100:
                            # Likely a filename - try S3
                            log_message(f"Treating reference image as S3 key: {reference_image}")
                            s3_url = f"s3://{s3_handler.default_bucket}/{reference_image}"
                            reference_image = s3_handler.download_from_s3(s3_url)
                        else:
                            reference_image = base64.b64decode(reference_image)
                except Exception as e:
                    log_message(f"Failed to process reference image: {e}", "WARNING")
                    reference_image = None
            
            duration = job_input.get('duration', 5.0)
            fps = job_input.get('fps', 30)
            width = job_input.get('width', 480)
            height = job_input.get('height', 480)
            
            # Generate video directly with audio data
            log_message("Processing audio and generating video...")
            video_result = generate_video_with_multitalk(
                audio_data,
                reference_image,
                duration=duration,
                fps=fps,
                width=width,
                height=height
            )
            
            if not video_result.get('success'):
                return {"error": "Video generation failed", "details": video_result.get('error')}
            
            # Step 3: Prepare output
            video_data = video_result.pop('video_data')
            
            # Get output format preferences
            output_format = job_input.get('output_format', 'base64')
            s3_output_key = job_input.get('s3_output_key')
            
            # Handle output based on format
            if output_format == 's3' and S3_AVAILABLE:
                if not s3_output_key:
                    # Generate default key
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    s3_output_key = f"multitalk/videos/output_{timestamp}.mp4"
                
                try:
                    log_message(f"Uploading video to S3 with key: {s3_output_key}")
                    video_output = s3_handler.upload_to_s3(
                        video_data, 
                        s3_output_key, 
                        content_type='video/mp4'
                    )
                    output_type = "s3_url"
                except Exception as e:
                    log_message(f"S3 upload failed, falling back to base64: {e}", "WARNING")
                    video_output = base64.b64encode(video_data).decode('utf-8')
                    output_type = "base64"
            else:
                # Default to base64
                video_output = base64.b64encode(video_data).decode('utf-8')
                output_type = "base64"
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "video": video_output,
                "output_type": output_type,
                "video_info": video_result,
                "processing_time": f"{processing_time:.1f}s",
                "models_used": video_result.get('models_used', []),
                "parameters": {
                    "duration": duration,
                    "fps": fps,
                    "resolution": f"{width}x{height}",
                    "audio_size": len(audio_data),
                    "video_size": len(video_data),
                    "output_format": output_format
                }
            }
            
            # Add audio features if present in video result
            if 'audio_features_shape' in video_result:
                result["audio_features"] = {
                    "shape": video_result["audio_features_shape"],
                    "extracted": True
                }
            
            # Add S3-specific info if applicable
            if output_type == "s3_url":
                result["s3_key"] = s3_output_key
                result["s3_bucket"] = s3_handler.default_bucket
            
            log_message(f"Complete pipeline finished in {processing_time:.1f}s")
            return result
        
        # Model download (from previous handler)
        if job_input.get('action') == 'download_models':
            # Import the download function from the previous handler
            try:
                from handler_with_download import download_models
                models = job_input.get('models', [])
                return download_models(models)
            except ImportError:
                return {"error": "Download functionality not available"}
        
        # Default response
        example_requests = [
            {
                "action": "generate",
                "audio": "<base64_encoded_audio>",
                "reference_image": "<optional_base64_image>",
                "duration": 5.0,
                "fps": 30,
                "width": 480,
                "height": 480
            }
        ]
        
        # Add S3 example if available
        if S3_AVAILABLE and s3_handler.enabled:
            example_requests.append({
                "action": "generate",
                "audio": "s3://bucket/audio/speech.wav",
                "reference_image": "s3://bucket/images/face.jpg",
                "output_format": "s3",
                "s3_output_key": "videos/generated.mp4",
                "duration": 5.0
            })
        
        return {
            "message": "Complete MultiTalk handler ready!",
            "version": "2.2.0", "build_id": "1752220071", "image_tag": "v2.2.0-20250711-multitalk",
            "supported_actions": [
                "health_check",
                "load_models", 
                "list_models",
                "generate",
                "download_models"
            ],
            "s3_integration": {
                "available": S3_AVAILABLE,
                "enabled": S3_AVAILABLE and s3_handler.enabled,
                "default_bucket": s3_handler.default_bucket if S3_AVAILABLE else None
            },
            "multitalk_inference": {
                "available": MULTITALK_AVAILABLE,
                "implementation": "Real MultiTalk" if MULTITALK_AVAILABLE else "Fallback FFmpeg"
            },
            "example_requests": example_requests,
            "models_status": {
                "loaded": MODEL_LOAD_STATUS["loaded"],
                "available": len([p for p in MODELS.values() if p.exists()]),
                "total": len(MODELS)
            }
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        log_message(f"Handler error: {e}", "ERROR")
        log_message(f"Traceback: {error_trace}", "ERROR")
        return {
            "error": f"Handler failed: {str(e)}",
            "traceback": error_trace,
            "job_input": job_input
        }

def initialize():
    """Initialize the complete MultiTalk handler."""
    log_message("=" * 60)
    log_message("Complete MultiTalk Handler Starting...")
    log_message(f"Model path: {MODEL_BASE}")
    log_message(f"Volume mounted: {os.path.exists('/runpod-volume')}")
    
    # Check available models
    available = 0
    total_size = 0
    for name, path in MODELS.items():
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
            else:
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            total_size += size
            available += 1
            log_message(f"✓ {name} ({size / (1024**3):.1f} GB)")
        else:
            log_message(f"✗ {name} (missing)")
    
    log_message(f"Models available: {available}/{len(MODELS)}")
    log_message(f"Total model size: {total_size / (1024**3):.1f} GB")
    
    # Check GPU
    gpu_info = check_gpu_availability()
    log_message(f"GPU: {gpu_info}")
    
    # Check S3 integration
    if S3_AVAILABLE:
        s3_status = s3_handler.check_s3_access()
        if s3_status["enabled"]:
            log_message(f"✓ S3 integration enabled (bucket: {s3_status['default_bucket']})")
        else:
            log_message("✗ S3 integration disabled (no credentials)")
    else:
        log_message("✗ S3 handler not available")
    
    if available >= 3:  # Need at least 3 models for basic functionality
        log_message("✅ Ready for MultiTalk video generation!")
    else:
        log_message("⚠️  Limited functionality - missing critical models")
    
    log_message("=" * 60)

if __name__ == "__main__":
    initialize()
    runpod.serverless.start({"handler": handler})