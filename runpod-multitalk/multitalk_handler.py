import runpod
import os
import sys
import json
import time
import base64
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Model paths
MODEL_BASE = Path(os.environ.get("MODEL_PATH", "/runpod-volume/models"))
MODELS = {
    "wav2vec2": MODEL_BASE / "wav2vec2-base-960h",
    "wan21": MODEL_BASE / "wan2.1-i2v-14b-480p" / "Wan2.1-I2V-14B-480P_Q4_K_M.gguf",
    "face_detection": MODEL_BASE / "face_detection" / "detection_Resnet50_Final.pth",
    "face_parsing": MODEL_BASE / "face_parsing" / "parsing_parsenet.pth",
    "gfpgan": MODEL_BASE / "gfpgan" / "GFPGANv1.4.pth"
}

# Global model cache
models_cache = {}

def load_models():
    """Load models into memory (called on cold start)."""
    global models_cache
    
    print("Loading models...")
    start_time = time.time()
    
    # Check which models exist
    for name, path in MODELS.items():
        if path.exists():
            print(f"✓ Found {name} at {path}")
            # In production, you would actually load the model here
            models_cache[name] = f"Model {name} loaded"
        else:
            print(f"✗ Missing {name} at {path}")
    
    elapsed = time.time() - start_time
    print(f"Model loading completed in {elapsed:.1f}s")
    
    return len(models_cache) > 0

def process_audio(audio_data: bytes) -> Dict[str, Any]:
    """Process audio with Wav2Vec2."""
    # Placeholder for actual audio processing
    # In production, this would:
    # 1. Decode audio data
    # 2. Run through Wav2Vec2
    # 3. Extract features for lip sync
    
    return {
        "duration": 5.0,
        "sample_rate": 16000,
        "features_extracted": True
    }

def generate_video(
    audio_features: Dict[str, Any],
    reference_image: Optional[bytes] = None,
    num_frames: int = 150
) -> bytes:
    """Generate video using Wan2.1 and MultiTalk."""
    
    # Placeholder for actual video generation
    # In production, this would:
    # 1. Process reference image if provided
    # 2. Use audio features for lip sync
    # 3. Generate video frames with Wan2.1
    # 4. Apply face enhancement with GFPGAN
    # 5. Encode to video format
    
    # For now, create a simple test video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        # Create a test pattern video with ffmpeg
        cmd = [
            "ffmpeg", "-f", "lavfi", "-i", 
            f"testsrc=duration={audio_features['duration']}:size=480x480:rate=30",
            "-pix_fmt", "yuv420p", "-y", tmp.name
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            with open(tmp.name, "rb") as f:
                video_data = f.read()
            os.unlink(tmp.name)
            return video_data
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create test video: {e}")

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main RunPod handler for MultiTalk inference."""
    
    try:
        job_input = job.get('input', {})
        
        # Health check
        if job_input.get('health_check'):
            return {
                "status": "healthy",
                "message": "MultiTalk handler ready!",
                "models_loaded": len(models_cache),
                "models_available": {name: path.exists() for name, path in MODELS.items()},
                "python_version": sys.version,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown')
            }
        
        # Model download action (from previous handler)
        if job_input.get('action') == 'download_models':
            from handler_with_download import download_models
            models = job_input.get('models', [])
            return download_models(models)
        
        # List models action
        if job_input.get('action') == 'list_models':
            models_info = []
            for name, path in MODELS.items():
                if path.exists():
                    if path.is_file():
                        size = path.stat().st_size
                    else:
                        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    
                    models_info.append({
                        "name": name,
                        "path": str(path),
                        "size_mb": round(size / (1024 * 1024), 2),
                        "loaded": name in models_cache
                    })
            
            return {
                "models": models_info,
                "total": len(models_info),
                "loaded": len(models_cache)
            }
        
        # MultiTalk inference
        if job_input.get('action') == 'generate' or 'audio' in job_input:
            print("Starting MultiTalk video generation...")
            start_time = time.time()
            
            # Validate input
            audio_input = job_input.get('audio')
            if not audio_input:
                return {"error": "No audio data provided"}
            
            # Decode audio from base64 if needed
            if isinstance(audio_input, str):
                try:
                    audio_data = base64.b64decode(audio_input)
                except Exception as e:
                    return {"error": f"Failed to decode audio: {e}"}
            else:
                audio_data = audio_input
            
            # Get optional parameters
            reference_image = job_input.get('reference_image')
            if reference_image and isinstance(reference_image, str):
                reference_image = base64.b64decode(reference_image)
            
            fps = job_input.get('fps', 30)
            duration = job_input.get('duration', 5.0)
            
            # Process audio
            print("Processing audio...")
            audio_features = process_audio(audio_data)
            
            # Generate video
            print("Generating video...")
            num_frames = int(duration * fps)
            video_data = generate_video(
                audio_features, 
                reference_image,
                num_frames
            )
            
            # Encode video to base64 for response
            video_b64 = base64.b64encode(video_data).decode('utf-8')
            
            elapsed = time.time() - start_time
            
            return {
                "success": True,
                "video": video_b64,
                "duration": duration,
                "fps": fps,
                "frames": num_frames,
                "processing_time": f"{elapsed:.1f}s",
                "models_used": list(models_cache.keys())
            }
        
        # Default response
        return {
            "message": "MultiTalk handler ready",
            "supported_actions": [
                "health_check",
                "download_models", 
                "list_models",
                "generate"
            ],
            "example_request": {
                "action": "generate",
                "audio": "<base64_encoded_audio>",
                "reference_image": "<optional_base64_image>",
                "duration": 5.0,
                "fps": 30
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "error": f"Handler failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

def initialize():
    """Initialize handler on worker start."""
    print("=" * 60)
    print("MultiTalk RunPod Handler Starting...")
    print(f"Model path: {MODEL_BASE}")
    print(f"Volume mounted: {os.path.exists('/runpod-volume')}")
    
    # Try to load models
    if not load_models():
        print("WARNING: No models loaded. Please download models first.")
    
    print("=" * 60)

if __name__ == "__main__":
    initialize()
    runpod.serverless.start({"handler": handler})