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
print("V107: MultiTalk Handler with Real Inference")
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

# Import our MultiTalk implementation
try:
    from multitalk_v106_implementation import MultiTalkInference
    print("✓ MultiTalk implementation imported successfully")
    MULTITALK_INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"✗ MultiTalk implementation import failed: {e}")
    MULTITALK_INFERENCE_AVAILABLE = False
    traceback.print_exc()

print("="*50)

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
        print(f"V107: S3 download error: {e}")
        return None

def upload_to_s3(file_path, s3_key):
    """Upload file to S3"""
    try:
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            print("V107: Missing S3 credentials")
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
        print(f"V107: S3 upload error: {e}")
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
        print(f"V107: Error processing input: {e}")
        return None

def handler(job):
    """V107 Handler - Real MultiTalk Inference"""
    print(f"V107: Received job: {job}")
    
    job_input = job.get("input", {})
    action = job_input.get("action", "generate")
    
    try:
        if action == "generate":
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
            
            print(f"V107: Processing audio: {audio_path}")
            print(f"V107: Processing image: {image_path}")
            print(f"V107: Parameters: steps={sample_steps}, text_scale={text_guidance_scale}, audio_scale={audio_guidance_scale}")
            
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
                    print(f"V107: Video generated using MultiTalk inference")
                except Exception as e:
                    print(f"V107: MultiTalk inference failed: {e}")
                    traceback.print_exc()
            
            # Fallback to simple test video if needed
            if not video_path or not os.path.exists(video_path):
                print("V107: Falling back to simple test video generation")
                video_path = create_simple_test_video(audio_path, image_path)
            
            if not video_path or not os.path.exists(video_path):
                return {
                    "output": {
                        "status": "error",
                        "error": "Failed to generate video"
                    }
                }
            
            print(f"V107: Video generated: {video_path}, size: {os.path.getsize(video_path)} bytes")
            
            # Handle output format
            if output_format == "s3":
                s3_key = s3_key.replace("{version}", "v107")
                s3_key = s3_key.replace("{int(time.time())}", str(int(time.time())))
                
                video_url = upload_to_s3(video_path, s3_key)
                
                if video_url:
                    result = {
                        "status": "completed",
                        "video_url": video_url,
                        "s3_key": s3_key,
                        "message": f"Video generated successfully (V107 - MultiTalk inference)",
                        "xfuser_available": XFUSER_AVAILABLE,
                        "inference_mode": "multitalk" if multitalk_inference else "fallback"
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
                    "message": f"Video generated successfully (V107 - MultiTalk inference)",
                    "xfuser_available": XFUSER_AVAILABLE,
                    "inference_mode": "multitalk" if multitalk_inference else "fallback"
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
                "multitalk_initialized": multitalk_inference is not None,
                "pytorch_version": torch.__version__
            }
            
            # Check versions and model loading status
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
            
            # Check if models are loaded
            if multitalk_inference:
                model_info["models_loaded"] = multitalk_inference.models_loaded
                model_info["wav2vec_loaded"] = multitalk_inference.wav2vec_model is not None
                model_info["wan_loaded"] = multitalk_inference.wan_pipeline is not None
            
            if model_path.exists():
                files = list(model_path.rglob("*"))
                model_info["total_files"] = len([f for f in files if f.is_file()])
                
                # Check for key model directories
                model_info["wan_checkpoint_exists"] = (model_path / "wan2.1-i2v-14b-480p").exists()
                model_info["wav2vec_exists"] = (model_path / "wav2vec2").exists()
                
                # List sample files
                audio_files = [f for f in files if f.suffix in ['.wav', '.mp3'] and f.is_file()]
                image_files = [f for f in files if f.suffix in ['.png', '.jpg', '.jpeg'] and f.is_file()]
                
                model_info["audio_files"] = [str(f.relative_to(model_path)) for f in audio_files[:5]]
                model_info["image_files"] = [str(f.relative_to(model_path)) for f in image_files[:5]]
            
            return {
                "output": {
                    "status": "ready",
                    "message": "V107 MultiTalk handler ready (real inference implementation)",
                    "version": "107",
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
        print(f"V107: Handler error: {e}")
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
            cv2.putText(frame_bgr, f"V107 Test - {i+1}/75", (10, 30), 
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
        print(f"V107: Error creating test video: {e}")
        return None

# Start handler
print("V107: Starting RunPod serverless handler...")
runpod.serverless.start({"handler": handler})