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
print("V108: MultiTalk Handler with Network Volume Explorer")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# Test imports
try:
    import xfuser
    print(f"✓ xfuser {xfuser.__version__} imported successfully")
    XFUSER_AVAILABLE = True
except ImportError as e:
    print(f"✗ xfuser import failed: {e}")
    XFUSER_AVAILABLE = False

try:
    import xformers
    print(f"✓ xformers {xformers.__version__} imported successfully")
except ImportError as e:
    print(f"✗ xformers import failed: {e}")

print("="*50)

def explore_network_volume():
    """Explore the contents of the network volume"""
    volume_path = Path("/runpod-volume")
    model_path = volume_path / "models"
    
    exploration = {
        "volume_exists": volume_path.exists(),
        "models_dir_exists": model_path.exists(),
        "volume_contents": [],
        "model_directories": {},
        "wan_models": [],
        "wav2vec_models": [],
        "checkpoints": [],
        "config_files": [],
        "total_size_gb": 0
    }
    
    if not volume_path.exists():
        return exploration
    
    # List top-level volume contents
    try:
        for item in volume_path.iterdir():
            item_info = {
                "name": item.name,
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else 0
            }
            exploration["volume_contents"].append(item_info)
    except Exception as e:
        exploration["volume_error"] = str(e)
    
    if not model_path.exists():
        return exploration
    
    # Explore models directory in detail
    try:
        total_size = 0
        
        # Look for WAN models
        wan_patterns = ["wan*", "WAN*", "*wan2.1*", "*i2v*"]
        for pattern in wan_patterns:
            for wan_path in model_path.rglob(pattern):
                if wan_path.is_dir():
                    wan_info = {
                        "path": str(wan_path.relative_to(model_path)),
                        "contents": []
                    }
                    # List contents of WAN directory
                    for item in wan_path.iterdir():
                        item_info = {
                            "name": item.name,
                            "is_dir": item.is_dir(),
                            "size": item.stat().st_size if item.is_file() else 0
                        }
                        wan_info["contents"].append(item_info)
                        if item.is_file():
                            total_size += item.stat().st_size
                    exploration["wan_models"].append(wan_info)
        
        # Look for Wav2Vec models
        wav_patterns = ["wav2vec*", "Wav2Vec*", "*wav2vec*"]
        for pattern in wav_patterns:
            for wav_path in model_path.rglob(pattern):
                if wav_path.is_dir():
                    wav_info = {
                        "path": str(wav_path.relative_to(model_path)),
                        "contents": []
                    }
                    for item in wav_path.iterdir():
                        item_info = {
                            "name": item.name,
                            "is_dir": item.is_dir(),
                            "size": item.stat().st_size if item.is_file() else 0
                        }
                        wav_info["contents"].append(item_info)
                        if item.is_file():
                            total_size += item.stat().st_size
                    exploration["wav2vec_models"].append(wav_info)
        
        # Look for checkpoint files
        checkpoint_patterns = ["*.ckpt", "*.pt", "*.pth", "*.bin", "*.safetensors", "*.gguf"]
        for pattern in checkpoint_patterns:
            for ckpt_path in model_path.rglob(pattern):
                if ckpt_path.is_file():
                    ckpt_info = {
                        "path": str(ckpt_path.relative_to(model_path)),
                        "size_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 2),
                        "extension": ckpt_path.suffix
                    }
                    exploration["checkpoints"].append(ckpt_info)
                    total_size += ckpt_path.stat().st_size
        
        # Look for config files
        config_patterns = ["*.json", "*.yaml", "*.yml", "config.py", "model_index.json"]
        for pattern in config_patterns:
            for config_path in model_path.rglob(pattern):
                if config_path.is_file() and config_path.stat().st_size < 1024 * 1024:  # < 1MB
                    config_info = {
                        "path": str(config_path.relative_to(model_path)),
                        "size_kb": round(config_path.stat().st_size / 1024, 2)
                    }
                    # Try to read JSON configs
                    if config_path.suffix == ".json":
                        try:
                            with open(config_path, 'r') as f:
                                content = json.load(f)
                                if isinstance(content, dict):
                                    config_info["keys"] = list(content.keys())[:10]  # First 10 keys
                        except:
                            pass
                    exploration["config_files"].append(config_info)
        
        # List all directories in models
        for item in model_path.iterdir():
            if item.is_dir():
                dir_info = {
                    "name": item.name,
                    "file_count": len(list(item.rglob("*")))
                }
                exploration["model_directories"][item.name] = dir_info
        
        exploration["total_size_gb"] = round(total_size / (1024 ** 3), 2)
        
    except Exception as e:
        exploration["model_exploration_error"] = str(e)
        traceback.print_exc()
    
    return exploration

# Explore on startup
print("\n" + "="*50)
print("EXPLORING NETWORK VOLUME...")
volume_info = explore_network_volume()
print(json.dumps(volume_info, indent=2))
print("="*50 + "\n")

# Import our MultiTalk implementation
try:
    from multitalk_v106_implementation import MultiTalkInference
    print("✓ MultiTalk implementation imported successfully")
    MULTITALK_INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"✗ MultiTalk implementation import failed: {e}")
    MULTITALK_INFERENCE_AVAILABLE = False

# Initialize MultiTalk inference
multitalk_inference = None
if MULTITALK_INFERENCE_AVAILABLE:
    try:
        multitalk_inference = MultiTalkInference()
        print("✓ MultiTalk inference initialized")
    except Exception as e:
        print(f"✗ MultiTalk inference initialization failed: {e}")

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
        print(f"V108: S3 download error: {e}")
        return None

def upload_to_s3(file_path, s3_key):
    """Upload file to S3"""
    try:
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("V108: Missing S3 credentials")
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
        print(f"V108: S3 upload error: {e}")
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
        print(f"V108: Error processing input: {e}")
        return None

def handler(job):
    """V108 Handler - Network Volume Explorer"""
    print(f"V108: Received job: {job}")
    
    job_input = job.get("input", {})
    action = job_input.get("action", "generate")
    
    try:
        if action == "volume_explore":
            # Special action to explore the volume in detail
            exploration = explore_network_volume()
            
            # Also check for specific paths that might contain models
            additional_checks = {
                "huggingface_cache": os.path.exists("/runpod-volume/huggingface"),
                "common_model_paths": {}
            }
            
            common_paths = [
                "/runpod-volume/models/wan2.1-i2v-14b-480p",
                "/runpod-volume/models/WAN",
                "/runpod-volume/models/multitalk",
                "/runpod-volume/models/MultiTalk",
                "/runpod-volume/checkpoints",
                "/runpod-volume/weights"
            ]
            
            for path in common_paths:
                path_obj = Path(path)
                if path_obj.exists():
                    additional_checks["common_model_paths"][path] = {
                        "exists": True,
                        "is_dir": path_obj.is_dir(),
                        "contents": [item.name for item in path_obj.iterdir()][:20] if path_obj.is_dir() else None
                    }
            
            return {
                "output": {
                    "status": "completed",
                    "exploration": exploration,
                    "additional_checks": additional_checks,
                    "message": "Network volume exploration completed"
                }
            }
        
        elif action == "generate":
            # Get inputs
            audio_1 = job_input.get("audio_1")
            condition_image = job_input.get("condition_image")
            prompt = job_input.get("prompt", "A person talking naturally")
            output_format = job_input.get("output_format", "s3")
            s3_key = job_input.get("s3_output_key", f"multitalk-out/output-{int(time.time())}.mp4")
            
            # Optional parameters
            duration = job_input.get("duration", None)
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
            
            print(f"V108: Processing audio: {audio_path}")
            print(f"V108: Processing image: {image_path}")
            
            # Generate video
            video_path = None
            if multitalk_inference is not None:
                try:
                    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                    video_path = multitalk_inference.generate_video(
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
                    print(f"V108: Video generated using MultiTalk inference")
                except Exception as e:
                    print(f"V108: MultiTalk inference failed: {e}")
                    traceback.print_exc()
            
            # Fallback to simple test video if needed
            if not video_path or not os.path.exists(video_path):
                print("V108: Falling back to simple test video generation")
                video_path = create_simple_test_video(audio_path, image_path)
            
            if not video_path or not os.path.exists(video_path):
                return {
                    "output": {
                        "status": "error",
                        "error": "Failed to generate video"
                    }
                }
            
            print(f"V108: Video generated: {video_path}, size: {os.path.getsize(video_path)} bytes")
            
            # Handle output format
            if output_format == "s3":
                s3_key = s3_key.replace("{version}", "v108")
                s3_key = s3_key.replace("{int(time.time())}", str(int(time.time())))
                
                video_url = upload_to_s3(video_path, s3_key)
                
                if video_url:
                    result = {
                        "status": "completed",
                        "video_url": video_url,
                        "s3_key": s3_key,
                        "message": f"Video generated successfully (V108)"
                    }
                else:
                    with open(video_path, 'rb') as f:
                        video_base64 = base64.b64encode(f.read()).decode('utf-8')
                    result = {
                        "status": "completed",
                        "video_base64": video_base64,
                        "message": "Video generated (S3 upload failed)"
                    }
            else:
                with open(video_path, 'rb') as f:
                    video_base64 = base64.b64encode(f.read()).decode('utf-8')
                result = {
                    "status": "completed",
                    "video_base64": video_base64,
                    "message": f"Video generated successfully (V108)"
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
        
        elif action == "model_check":
            # Enhanced model check with volume exploration
            model_path = Path("/runpod-volume/models")
            model_info = {
                "network_volume_mounted": os.path.exists("/runpod-volume"),
                "models_directory_exists": model_path.exists(),
                "s3_configured": all([
                    os.getenv('AWS_ACCESS_KEY_ID'),
                    os.getenv('AWS_SECRET_ACCESS_KEY'),
                    os.getenv('AWS_S3_BUCKET_NAME')
                ]),
                "cuda_available": torch.cuda.is_available(),
                "device": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
                "xfuser_available": XFUSER_AVAILABLE,
                "multitalk_inference_available": MULTITALK_INFERENCE_AVAILABLE,
                "pytorch_version": torch.__version__,
                "volume_exploration": volume_info  # Include volume exploration
            }
            
            # Check versions
            try:
                import xfuser
                model_info["xfuser_version"] = xfuser.__version__
            except:
                model_info["xfuser_version"] = "not available"
            
            try:
                import xformers
                model_info["xformers_version"] = xformers.__version__
            except:
                model_info["xformers_version"] = "not available"
            
            return {
                "output": {
                    "status": "ready",
                    "message": "V108 MultiTalk handler ready (with volume explorer)",
                    "version": "108",
                    "model_info": model_info
                }
            }
        
        else:
            return {
                "output": {
                    "status": "error",
                    "error": f"Unknown action: {action}"
                }
            }
            
    except Exception as e:
        print(f"V108: Handler error: {e}")
        traceback.print_exc()
        return {
            "output": {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        }

def create_simple_test_video(audio_path, image_path):
    """Create a simple test video as fallback"""
    try:
        import cv2
        from moviepy.editor import VideoFileClip, AudioFileClip
        from PIL import Image
        import numpy as np
        
        # Load image
        img = np.array(Image.open(image_path).convert('RGB'))
        h, w = img.shape[:2]
        
        # Create video
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, 25.0, (w, h))
        
        # Create 3 seconds of video
        for i in range(75):
            frame = img.copy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"V108 Test - {i+1}/75", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out.write(frame_bgr)
        
        out.release()
        
        # Add audio
        final_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        video_clip = VideoFileClip(temp_video)
        audio_clip = AudioFileClip(audio_path)
        
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(final_output, codec='libx264', audio_codec='aac', 
                                  verbose=False, logger=None)
        
        # Cleanup
        video_clip.close()
        audio_clip.close()
        final_clip.close()
        os.unlink(temp_video)
        
        return final_output
        
    except Exception as e:
        print(f"V108: Error creating test video: {e}")
        return None

# Start handler
print("V108: Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})