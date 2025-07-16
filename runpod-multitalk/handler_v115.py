#!/usr/bin/env python3
"""
MultiTalk V115 Handler - Proper MeiGen-MultiTalk Implementation
Uses the correct implementation approach from the working codebase
"""

import runpod
import os
import sys
import json
import time
import torch
import base64
import tempfile
import traceback
import warnings
from pathlib import Path

# Add MultiTalk to path
sys.path.insert(0, '/app/multitalk_official')
sys.path.insert(0, '/app')

print("="*60)
print("V115: MultiTalk Handler - Proper MeiGen-MultiTalk Implementation")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("="*60)

# Import S3 handler
S3_AVAILABLE = False
s3_handler = None
try:
    from s3_handler import s3_handler
    S3_AVAILABLE = True
    print(f"✓ S3 handler imported successfully")
except ImportError as e:
    print(f"✗ S3 handler import failed: {e}")

# Import V115 implementation
MULTITALK_V115_AVAILABLE = False
multitalk_v115 = None

try:
    from multitalk_v115_implementation import MultiTalkV115
    print("✓ MultiTalk V115 implementation imported successfully")
    MULTITALK_V115_AVAILABLE = True
    
    # Initialize MultiTalk V115
    print("\\nInitializing MultiTalk V115...")
    multitalk_v115 = MultiTalkV115()
    
    if multitalk_v115.load_models():
        print("✓ MultiTalk V115 initialized successfully")
        
        # Show model info
        model_info = multitalk_v115.get_model_info()
        print(f"Implementation: {model_info['implementation']}")
        print(f"Device: {model_info['device']}")
        print(f"Models available: {sum(1 for v in model_info['models_available'].values() if v)}/5")
        
        for model_name, available in model_info['models_available'].items():
            status = "✓" if available else "✗"
            print(f"  {status} {model_name}: {'available' if available else 'missing'}")
        
        print(f"Models loaded: {sum(1 for v in model_info['models_loaded'].values() if v)}/3")
        for model_name, loaded in model_info['models_loaded'].items():
            status = "✓" if loaded else "✗"
            print(f"  {status} {model_name}: {'loaded' if loaded else 'not loaded'}")
        
    else:
        print("✗ MultiTalk V115 initialization failed - MeiGen-MultiTalk components required")
        multitalk_v115 = None
        
except ImportError as e:
    print(f"✗ MultiTalk V115 implementation import failed: {e}")
    traceback.print_exc()

print("="*60)

def handler(job):
    """V115 Handler with proper MeiGen-MultiTalk implementation"""
    
    start_time = time.time()
    job_input = job.get('input', {})
    
    try:
        print(f"\\n[{time.strftime('%H:%M:%S')}] Processing job: {job.get('id', 'unknown')}")
        
        # Health check
        if job_input.get('health_check'):
            return {
                "output": {
                    "status": "healthy",
                    "version": "V115",
                    "implementation": "MeiGen-MultiTalk",
                    "multitalk_available": MULTITALK_V115_AVAILABLE,
                    "multitalk_loaded": multitalk_v115 is not None,
                    "s3_available": S3_AVAILABLE,
                    "cuda_available": torch.cuda.is_available(),
                    "model_info": multitalk_v115.get_model_info() if multitalk_v115 else None
                }
            }
        
        # Model information
        if job_input.get('action') == 'model_info':
            if multitalk_v115:
                return {
                    "output": {
                        "status": "success",
                        "model_info": multitalk_v115.get_model_info()
                    }
                }
            else:
                return {
                    "output": {
                        "status": "error",
                        "error": "MultiTalk V115 not available"
                    }
                }
        
        # Video generation
        if job_input.get('action') == 'generate' or any(key in job_input for key in ['audio_1', 'audio', 'condition_image']):
            if not multitalk_v115:
                return {
                    "output": {
                        "status": "error",
                        "error": "MultiTalk V115 not initialized - MeiGen-MultiTalk components required"
                    }
                }
            
            # Get input parameters
            audio_input = job_input.get('audio_1') or job_input.get('audio')
            image_input = job_input.get('condition_image') or job_input.get('image')
            prompt = job_input.get('prompt', "A person talking naturally with expressive facial movements")
            
            # Generation parameters
            num_frames = job_input.get('num_frames', 81)
            sampling_steps = job_input.get('sample_steps', 40)
            seed = job_input.get('seed', 42)
            turbo = job_input.get('turbo', True)
            
            if not audio_input:
                return {
                    "output": {
                        "status": "error",
                        "error": "No audio input provided"
                    }
                }
            
            if not image_input:
                return {
                    "output": {
                        "status": "error",
                        "error": "No image input provided"
                    }
                }
            
            print(f"Generating video with parameters:")
            print(f"  Audio: {audio_input}")
            print(f"  Image: {image_input}")
            print(f"  Prompt: {prompt}")
            print(f"  Frames: {num_frames}, Steps: {sampling_steps}")
            print(f"  Seed: {seed}, Turbo: {turbo}")
            
            # Process inputs (download from S3 if needed)
            try:
                # Handle audio input
                if isinstance(audio_input, str):
                    if S3_AVAILABLE and (audio_input.startswith('s3://') or 'amazonaws.com' in audio_input or len(audio_input) < 100):
                        if audio_input.startswith('s3://') or 'amazonaws.com' in audio_input:
                            audio_data = s3_handler.download_from_s3(audio_input)
                        else:
                            # Treat as S3 key
                            s3_url = f"s3://{s3_handler.default_bucket}/{audio_input}"
                            audio_data = s3_handler.download_from_s3(s3_url)
                        
                        # Save to temporary file
                        audio_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        audio_temp.write(audio_data)
                        audio_temp.close()
                        audio_path = audio_temp.name
                    else:
                        # Assume base64
                        audio_data = base64.b64decode(audio_input)
                        audio_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        audio_temp.write(audio_data)
                        audio_temp.close()
                        audio_path = audio_temp.name
                else:
                    return {
                        "output": {
                            "status": "error",
                            "error": "Invalid audio input format"
                        }
                    }
                
                # Handle image input
                if isinstance(image_input, str):
                    if S3_AVAILABLE and (image_input.startswith('s3://') or 'amazonaws.com' in image_input or len(image_input) < 100):
                        if image_input.startswith('s3://') or 'amazonaws.com' in image_input:
                            image_data = s3_handler.download_from_s3(image_input)
                        else:
                            # Treat as S3 key
                            s3_url = f"s3://{s3_handler.default_bucket}/{image_input}"
                            image_data = s3_handler.download_from_s3(s3_url)
                        
                        # Save to temporary file
                        image_temp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                        image_temp.write(image_data)
                        image_temp.close()
                        image_path = image_temp.name
                    else:
                        # Assume base64
                        image_data = base64.b64decode(image_input)
                        image_temp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                        image_temp.write(image_data)
                        image_temp.close()
                        image_path = image_temp.name
                else:
                    return {
                        "output": {
                            "status": "error",
                            "error": "Invalid image input format"
                        }
                    }
                
                print(f"Inputs processed - Audio: {audio_path}, Image: {image_path}")
                
                # Generate video using V115 implementation
                generation_result = multitalk_v115.generate_video(
                    audio_path=audio_path,
                    image_path=image_path,
                    prompt=prompt,
                    num_frames=num_frames,
                    sampling_steps=sampling_steps,
                    seed=seed,
                    turbo=turbo
                )
                
                # Clean up temporary files
                try:
                    os.unlink(audio_path)
                    os.unlink(image_path)
                except:
                    pass
                
                if not generation_result.get('success'):
                    return {
                        "output": {
                            "status": "error",
                            "error": f"Video generation failed: {generation_result.get('error', 'Unknown error')}",
                            "implementation": "V115"
                        }
                    }
                
                video_path = generation_result['video_path']
                
                if not video_path or not os.path.exists(video_path):
                    return {
                        "output": {
                            "status": "error",
                            "error": "Video file not created",
                            "details": {
                                "video_path": video_path,
                                "file_exists": os.path.exists(video_path) if video_path else False
                            }
                        }
                    }
                
                print(f"Video generated successfully: {video_path}, size: {os.path.getsize(video_path)} bytes")
                
                # Handle output format
                output_format = job_input.get('output_format', 'base64')
                
                if output_format == 's3' and S3_AVAILABLE:
                    # Upload to S3
                    s3_output_key = job_input.get('s3_output_key')
                    if not s3_output_key:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        s3_output_key = f"multitalk-v115/videos/output_{timestamp}.mp4"
                    
                    try:
                        with open(video_path, 'rb') as f:
                            video_data = f.read()
                        
                        video_url = s3_handler.upload_to_s3(
                            video_data,
                            s3_output_key,
                            content_type='video/mp4'
                        )
                        
                        # Clean up local file
                        os.unlink(video_path)
                        
                        processing_time = time.time() - start_time
                        
                        return {
                            "output": {
                                "status": "completed",
                                "video_url": video_url,
                                "s3_key": s3_output_key,
                                "video_size": len(video_data),
                                "processing_time": f"{processing_time:.1f}s",
                                "implementation": "V115",
                                "generation_params": {
                                    "num_frames": num_frames,
                                    "sampling_steps": sampling_steps,
                                    "seed": seed,
                                    "turbo": turbo,
                                    "resolution": "480x480"
                                }
                            }
                        }
                    except Exception as e:
                        print(f"S3 upload failed: {e}, falling back to base64")
                        # Fall through to base64
                
                # Return as base64
                with open(video_path, 'rb') as f:
                    video_data = f.read()
                
                video_base64 = base64.b64encode(video_data).decode('utf-8')
                
                # Clean up local file
                os.unlink(video_path)
                
                processing_time = time.time() - start_time
                
                return {
                    "output": {
                        "status": "completed",
                        "video_base64": video_base64,
                        "video_size": len(video_data),
                        "processing_time": f"{processing_time:.1f}s",
                        "implementation": "V115",
                        "generation_params": {
                            "num_frames": num_frames,
                            "sampling_steps": sampling_steps,
                            "seed": seed,
                            "turbo": turbo,
                            "resolution": "480x480"
                        }
                    }
                }
                
            except Exception as e:
                print(f"Input processing failed: {e}")
                traceback.print_exc()
                return {
                    "output": {
                        "status": "error",
                        "error": f"Input processing failed: {e}"
                    }
                }
        
        # Default response
        return {
            "output": {
                "status": "ready",
                "version": "V115",
                "implementation": "MeiGen-MultiTalk",
                "message": "MultiTalk V115 handler ready for video generation",
                "supported_actions": ["health_check", "model_info", "generate"],
                "example_request": {
                    "action": "generate",
                    "audio_1": "1.wav",
                    "condition_image": "multi1.png",
                    "prompt": "A person talking naturally",
                    "num_frames": 81,
                    "sampling_steps": 40,
                    "seed": 42,
                    "turbo": True,
                    "output_format": "s3"
                }
            }
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Handler error: {e}")
        print(f"Traceback: {error_trace}")
        
        processing_time = time.time() - start_time
        
        return {
            "output": {
                "status": "error",
                "error": f"Handler failed: {str(e)}",
                "processing_time": f"{processing_time:.1f}s",
                "implementation": "V115"
            }
        }

def initialize():
    """Initialize the V115 handler"""
    print("="*60)
    print("MultiTalk V115 Handler Starting...")
    print(f"Implementation: Proper MeiGen-MultiTalk")
    print(f"Volume mounted: {os.path.exists('/runpod-volume')}")
    print(f"Model path: /runpod-volume/models")
    
    if multitalk_v115:
        model_info = multitalk_v115.get_model_info()
        available_models = sum(1 for v in model_info['models_available'].values() if v)
        loaded_models = sum(1 for v in model_info['models_loaded'].values() if v)
        
        print(f"Models available: {available_models}/5")
        print(f"Models loaded: {loaded_models}/3")
        
        if available_models == 5 and loaded_models == 3:
            print("✅ Ready for MeiGen-MultiTalk video generation!")
        else:
            print("❌ ERROR: All MeiGen-MultiTalk components required")
            print("   Missing models will cause video generation to fail")
    else:
        print("❌ MultiTalk V115 not available - MeiGen-MultiTalk components required")
    
    print("="*60)

if __name__ == "__main__":
    initialize()
    runpod.serverless.start({"handler": handler})