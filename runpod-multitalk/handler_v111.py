#!/usr/bin/env python3
"""
MultiTalk V111 Handler - Real WAN Model Implementation
Uses the discovered WAN 2.1 models for actual video generation
"""

import runpod
import os
import sys
import json
import time
import torch
import base64
import boto3
import tempfile
import traceback
import warnings
from pathlib import Path

# Add MultiTalk to path
sys.path.insert(0, '/app/multitalk_official')
sys.path.insert(0, '/app')

print("="*50)
print("V111: MultiTalk Handler - Real WAN Model Implementation")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Test imports with detailed feedback
XFUSER_AVAILABLE = False
try:
    import xfuser
    print(f"✓ xfuser {xfuser.__version__} imported successfully")
    XFUSER_AVAILABLE = True
except ImportError as e:
    print(f"✗ xfuser import failed: {e}")

XFORMERS_AVAILABLE = False
try:
    import xformers
    print(f"✓ xformers {xformers.__version__} imported successfully")
    XFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"✗ xformers import failed: {e}")

SAFETENSORS_AVAILABLE = False
try:
    import safetensors
    print(f"✓ safetensors {safetensors.__version__} imported successfully")
    SAFETENSORS_AVAILABLE = True
except ImportError as e:
    print(f"✗ safetensors import failed: {e}")

# Test transformers
try:
    import transformers
    print(f"✓ transformers {transformers.__version__} imported successfully")
except ImportError as e:
    print(f"✗ transformers import failed: {e}")

print("="*50)

# Import our MultiTalk V111 implementation
MULTITALK_V111_AVAILABLE = False
multitalk_v111 = None

try:
    from multitalk_v111_implementation import MultiTalkV111
    print("✓ MultiTalk V111 implementation imported successfully")
    MULTITALK_V111_AVAILABLE = True
    
    # Initialize MultiTalk V111
    print("\\nInitializing MultiTalk V111...")
    multitalk_v111 = MultiTalkV111()
    
    if multitalk_v111.initialize_models():
        print("✓ MultiTalk V111 initialized successfully")
        
        # Show model info
        model_info = multitalk_v111.get_model_info()
        print(f"Models loaded: {sum(1 for v in model_info['models_loaded'].values() if v)}/6")
        
        for model_name, loaded in model_info['models_loaded'].items():
            status = "✓" if loaded else "✗"
            print(f"  {status} {model_name}: {'loaded' if loaded else 'failed'}")
        
    else:
        print("✗ MultiTalk V111 initialization failed")
        multitalk_v111 = None
        
except ImportError as e:
    print(f"✗ MultiTalk V111 implementation import failed: {e}")
    traceback.print_exc()

print("="*50)

# S3 utilities
def download_from_s3(s3_url):
    """Download file from S3 URL"""
    try:
        if s3_url.startswith("s3://"):
            parts = s3_url[5:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            return None
        
        s3_client = boto3.client('s3')
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        s3_client.download_file(bucket, key, temp_file.name)
        return temp_file.name
        
    except Exception as e:
        print(f"V111: S3 download error: {e}")
        return None

def upload_to_s3(file_path, s3_key):
    """Upload file to S3"""
    try:
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("V111: Missing S3 credentials")
            return None
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        s3_client.upload_file(file_path, bucket_name, s3_key)
        url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_key}"
        return url
        
    except Exception as e:
        print(f"V111: S3 upload error: {e}")
        return None

def process_input_file(input_ref, file_type):
    """Process input that could be S3 URL, base64, or local file"""
    try:
        if isinstance(input_ref, str):
            # S3 URL
            if input_ref.startswith("s3://"):
                return download_from_s3(input_ref)
            
            # Base64 data
            elif input_ref.startswith(f"data:{file_type}"):
                header, data = input_ref.split(',', 1)
                decoded_data = base64.b64decode(data)
                
                ext = ".wav" if file_type == "audio" else ".png"
                temp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                temp_file.write(decoded_data)
                temp_file.close()
                return temp_file.name
            
            # Local file in models directory
            else:
                model_path = Path("/runpod-volume/models")
                
                # Try exact match first
                exact_path = model_path / input_ref
                if exact_path.exists():
                    return str(exact_path)
                
                # Search for file
                matches = list(model_path.rglob(input_ref))
                if matches:
                    return str(matches[0])
                
                # Try without extension
                name_only = Path(input_ref).stem
                for ext in ['wav', 'mp3', 'png', 'jpg', 'jpeg']:
                    matches = list(model_path.rglob(f"{name_only}.{ext}"))
                    if matches:
                        return str(matches[0])
        
        return None
        
    except Exception as e:
        print(f"V111: Error processing input: {e}")
        return None

def explore_network_volume():
    """Explore the contents of the network volume (from V110)"""
    volume_path = Path("/runpod-volume")
    model_path = volume_path / "models"
    
    exploration = {
        "volume_exists": volume_path.exists(),
        "models_dir_exists": model_path.exists(),
        "wan_models": [],
        "wav2vec_models": [],
        "total_size_gb": 0,
        "key_model_files": {}
    }
    
    if not volume_path.exists():
        exploration["error"] = "Network volume not mounted"
        return exploration
    
    if not model_path.exists():
        exploration["error"] = "Models directory not found"
        return exploration
    
    # Check for key model files
    key_files = {
        "multitalk": "wan2.1-i2v-14b-480p/multitalk.safetensors",
        "diffusion": "wan2.1-i2v-14b-480p/diffusion_pytorch_model-00007-of-00007.safetensors",
        "vae": "wan2.1-i2v-14b-480p/Wan2.1_VAE.pth",
        "clip": "wan2.1-i2v-14b-480p/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "wav2vec": "wav2vec2-large-960h/pytorch_model.bin",
        "text_encoder": "wan2.1-i2v-14b-480p/google/umt5-xxl/config.json"
    }
    
    total_size = 0
    for model_name, file_path in key_files.items():
        full_path = model_path / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            total_size += size
            exploration["key_model_files"][model_name] = {
                "exists": True,
                "size_mb": round(size / (1024 * 1024), 2),
                "path": str(file_path)
            }
        else:
            exploration["key_model_files"][model_name] = {
                "exists": False,
                "path": str(file_path)
            }
    
    exploration["total_size_gb"] = round(total_size / (1024 ** 3), 2)
    
    return exploration

def handler(job):
    """V111 Handler - Real WAN Model Implementation"""
    print(f"V111: Received job: {job}")
    
    job_input = job.get("input", {})
    action = job_input.get("action", "generate")
    
    try:
        if action == "volume_explore":
            # Volume exploration
            exploration = explore_network_volume()
            
            return {
                "output": {
                    "status": "completed",
                    "exploration": exploration,
                    "message": "Network volume exploration completed (V111)"
                }
            }
        
        elif action == "model_check":
            # Enhanced model check
            model_info = {
                "network_volume_mounted": os.path.exists("/runpod-volume"),
                "models_directory_exists": os.path.exists("/runpod-volume/models"),
                "s3_configured": all([
                    os.getenv('AWS_ACCESS_KEY_ID'),
                    os.getenv('AWS_SECRET_ACCESS_KEY'),
                    os.getenv('AWS_S3_BUCKET_NAME')
                ]),
                "cuda_available": torch.cuda.is_available(),
                "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                "xfuser_available": XFUSER_AVAILABLE,
                "xformers_available": XFORMERS_AVAILABLE,
                "safetensors_available": SAFETENSORS_AVAILABLE,
                "multitalk_v111_available": MULTITALK_V111_AVAILABLE,
                "multitalk_v111_initialized": multitalk_v111 is not None,
                "pytorch_version": torch.__version__,
                "volume_exploration": explore_network_volume()
            }
            
            # Add MultiTalk V111 specific info
            if multitalk_v111:
                model_info["multitalk_v111_info"] = multitalk_v111.get_model_info()
            
            return {
                "output": {
                    "status": "ready",
                    "message": "V111 MultiTalk handler ready (Real WAN model implementation)",
                    "version": "111",
                    "model_info": model_info
                }
            }
        
        elif action == "generate":
            # Generate video with MultiTalk V111
            audio_1 = job_input.get("audio_1")
            condition_image = job_input.get("condition_image")
            prompt = job_input.get("prompt", "A person talking naturally")
            output_format = job_input.get("output_format", "s3")
            s3_key = job_input.get("s3_output_key", f"multitalk-v111/output-{int(time.time())}.mp4")
            
            # Generation parameters
            duration = job_input.get("duration")
            sample_steps = job_input.get("sample_steps", 30)
            text_guidance_scale = job_input.get("text_guidance_scale", 7.5)
            audio_guidance_scale = job_input.get("audio_guidance_scale", 3.5)
            seed = job_input.get("seed", 42)
            
            if not audio_1 or not condition_image:
                return {
                    "output": {
                        "status": "error",
                        "error": "Missing required inputs: audio_1 and condition_image"
                    }
                }
            
            # Check if MultiTalk V111 is available
            if not multitalk_v111:
                return {
                    "output": {
                        "status": "error",
                        "error": "MultiTalk V111 not available",
                        "details": {
                            "multitalk_v111_available": MULTITALK_V111_AVAILABLE,
                            "models_available": multitalk_v111.models_available if multitalk_v111 else None
                        }
                    }
                }
            
            # Process inputs
            audio_path = process_input_file(audio_1, "audio")
            image_path = process_input_file(condition_image, "image")
            
            if not audio_path or not image_path:
                return {
                    "output": {
                        "status": "error",
                        "error": "Failed to process input files",
                        "details": {
                            "audio_found": audio_path is not None,
                            "image_found": image_path is not None
                        }
                    }
                }
            
            print(f"V111: Processing audio: {audio_path}")
            print(f"V111: Processing image: {image_path}")
            print(f"V111: Prompt: {prompt}")
            
            # Generate video with MultiTalk V111
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            
            video_path = multitalk_v111.generate_video(
                audio_path=audio_path,
                image_path=image_path,
                output_path=temp_output,
                prompt=prompt,
                duration=duration,
                sample_steps=sample_steps,
                text_guidance_scale=text_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                seed=seed
            )
            
            if not video_path or not os.path.exists(video_path):
                return {
                    "output": {
                        "status": "error",
                        "error": "Failed to generate video with MultiTalk V111",
                        "details": {
                            "video_path": video_path,
                            "file_exists": os.path.exists(video_path) if video_path else False
                        }
                    }
                }
            
            print(f"V111: Video generated: {video_path}, size: {os.path.getsize(video_path)} bytes")
            
            # Handle output format
            if output_format == "s3":
                s3_key = s3_key.replace("{version}", "v111")
                s3_key = s3_key.replace("{timestamp}", str(int(time.time())))
                
                video_url = upload_to_s3(video_path, s3_key)
                
                if video_url:
                    result = {
                        "status": "completed",
                        "video_url": video_url,
                        "s3_key": s3_key,
                        "message": f"Video generated successfully with MultiTalk V111 (Real WAN models)",
                        "generation_params": {
                            "prompt": prompt,
                            "sample_steps": sample_steps,
                            "text_guidance_scale": text_guidance_scale,
                            "audio_guidance_scale": audio_guidance_scale,
                            "seed": seed
                        }
                    }
                else:
                    with open(video_path, 'rb') as f:
                        video_base64 = base64.b64encode(f.read()).decode('utf-8')
                    result = {
                        "status": "completed",
                        "video_base64": video_base64,
                        "message": "Video generated with MultiTalk V111 (S3 upload failed)",
                        "generation_params": {
                            "prompt": prompt,
                            "sample_steps": sample_steps,
                            "text_guidance_scale": text_guidance_scale,
                            "audio_guidance_scale": audio_guidance_scale,
                            "seed": seed
                        }
                    }
            else:
                with open(video_path, 'rb') as f:
                    video_base64 = base64.b64encode(f.read()).decode('utf-8')
                result = {
                    "status": "completed",
                    "video_base64": video_base64,
                    "message": f"Video generated successfully with MultiTalk V111 (Real WAN models)",
                    "generation_params": {
                        "prompt": prompt,
                        "sample_steps": sample_steps,
                        "text_guidance_scale": text_guidance_scale,
                        "audio_guidance_scale": audio_guidance_scale,
                        "seed": seed
                    }
                }
            
            # Cleanup
            try:
                os.unlink(video_path)
                if audio_path.startswith('/tmp'):
                    os.unlink(audio_path)
                if image_path.startswith('/tmp'):
                    os.unlink(image_path)
            except:
                pass
            
            return {"output": result}
        
        else:
            return {
                "output": {
                    "status": "error",
                    "error": f"Unknown action: {action}",
                    "available_actions": ["generate", "model_check", "volume_explore"]
                }
            }
            
    except Exception as e:
        print(f"V111: Handler error: {e}")
        traceback.print_exc()
        return {
            "output": {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "version": "111"
            }
        }

# Start handler
print("V111: Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})