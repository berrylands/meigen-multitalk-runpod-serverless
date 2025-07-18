#!/usr/bin/env python3
"""
RunPod serverless handler for MultiTalk V150 with graceful error handling.
Uses cog-MultiTalk reference implementation with proper dependency management.
"""

import os
import sys
import json
import time
import base64
import shutil
import logging
import tempfile
import traceback
from typing import Dict, Any, Optional

import runpod

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("MeiGen MultiTalk Handler V150 with Graceful Error Handling")
logger.info("Using cog-MultiTalk reference with proper dependency management")
logger.info("=" * 80)

# Import S3 handler
class S3Handler:
    def __init__(self):
        self.bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        self.region = os.environ.get('AWS_REGION', 'eu-west-1')
        self.enabled = True
        logger.info(f"S3 handler initialized - bucket: {self.bucket_name}")
    
    def download_from_s3(self, s3_key: str, local_path: str) -> None:
        """Download from S3"""
        logger.info(f"Downloading from S3: {s3_key} -> {local_path}")
        
        try:
            import boto3
            s3_client = boto3.client('s3')
            s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            file_size = os.path.getsize(local_path)
            logger.info(f"✅ Downloaded {s3_key} ({file_size} bytes)")
        except Exception as e:
            logger.error(f"Failed to download from S3: {str(e)}")
            raise
    
    def upload_to_s3(self, local_path: str, s3_key: str) -> str:
        """Upload to S3 and return URL"""
        logger.info(f"Uploading to S3: {local_path} -> {s3_key}")
        
        try:
            import boto3
            s3_client = boto3.client('s3')
            s3_client.upload_file(local_path, self.bucket_name, s3_key)
            
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Uploaded to: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            raise

# Global variables
s3_handler = None
multitalk = None
multitalk_error = None

def initialize_multitalk():
    """Initialize MultiTalk with graceful error handling"""
    global multitalk, multitalk_error
    
    if multitalk is not None:
        return multitalk
    
    if multitalk_error is not None:
        raise RuntimeError(f"MultiTalk initialization failed previously: {multitalk_error}")
    
    try:
        logger.info("🔄 Initializing MultiTalk with graceful error handling...")
        
        # Test basic imports first
        logger.info("Testing basic imports...")
        import torch
        import numpy as np
        logger.info(f"✅ PyTorch {torch.__version__} imported")
        
        # Test model paths
        model_path = os.environ.get('MODEL_PATH', '/runpod-volume/models')
        logger.info(f"Checking model path: {model_path}")
        
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model path not found: {model_path}")
        
        # Test reference implementation path
        ref_path = '/app/cog_multitalk_reference'
        if not os.path.exists(ref_path):
            raise RuntimeError(f"Reference implementation not found: {ref_path}")
        
        logger.info("✅ Basic validation passed")
        
        # Import reference wrapper with error handling
        try:
            from multitalk_reference_wrapper import MultiTalkReferenceWrapper
            logger.info("✅ MultiTalkReferenceWrapper imported successfully")
        except ImportError as e:
            error_msg = f"Failed to import MultiTalkReferenceWrapper: {str(e)}"
            logger.error(error_msg)
            multitalk_error = error_msg
            raise RuntimeError(error_msg)
        
        # Initialize wrapper
        try:
            multitalk = MultiTalkReferenceWrapper()
            logger.info("✅ MultiTalk V150 initialized successfully")
            return multitalk
        except Exception as e:
            error_msg = f"Failed to initialize MultiTalk: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            multitalk_error = error_msg
            raise RuntimeError(error_msg)
            
    except Exception as e:
        error_msg = f"MultiTalk initialization failed: {str(e)}"
        logger.error(error_msg)
        multitalk_error = error_msg
        raise RuntimeError(error_msg)

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function with graceful error handling
    """
    logger.info("=" * 80)
    logger.info("🎬 Handler invoked - MultiTalk V150 Graceful")
    logger.info("=" * 80)
    
    global s3_handler
    
    try:
        # Get job input
        job_input = job.get('input', {})
        logger.info(f"Input data: {job_input}")
        
        # Handle test requests
        if job_input.get('test') == 'health_check':
            logger.info("🏥 Health check requested")
            return {
                "status": "healthy",
                "version": "V150",
                "message": "Handler is running but MultiTalk not initialized yet"
            }
        
        # Parse inputs with fallbacks
        if 'audio_s3_key' in job_input:
            audio_input = job_input.get('audio_s3_key', job_input.get('audio_1'))
            image_input = job_input.get('image_s3_key', job_input.get('condition_image'))
        else:
            audio_input = job_input.get('audio_file', job_input.get('audio_1'))
            image_input = job_input.get('image_file', job_input.get('condition_image'))
        
        prompt = job_input.get('prompt', "A person talking naturally with expressive lip sync")
        output_format = job_input.get('output_format', 's3')
        sample_steps = job_input.get('sampling_steps', job_input.get('sample_steps', 40))
        turbo = job_input.get('turbo', False)  # Default to False for stability
        
        # Validate inputs
        if not audio_input:
            raise ValueError("No audio input provided (audio_file, audio_1, or audio_s3_key)")
        if not image_input:
            raise ValueError("No image input provided (image_file, condition_image, or image_s3_key)")
        
        # Log parsed inputs
        logger.info(f"🎬 Processing request:")
        logger.info(f"  - Audio: {audio_input}")
        logger.info(f"  - Image: {image_input}")
        logger.info(f"  - Prompt: {prompt}")
        logger.info(f"  - Output format: {output_format}")
        logger.info(f"  - Sample steps: {sample_steps}")
        logger.info(f"  - Turbo mode: {turbo}")
        
        # Initialize S3 handler if needed
        if s3_handler is None:
            logger.info("Initializing S3 handler...")
            s3_handler = S3Handler()
        
        # Initialize MultiTalk with graceful error handling
        try:
            multitalk_instance = initialize_multitalk()
        except RuntimeError as e:
            return {
                "error": str(e),
                "status": "failed",
                "error_type": "initialization_error",
                "message": "MultiTalk could not be initialized. Check logs for details."
            }
        
        # Create temporary directory
        with tempfile.TemporaryDirectory(prefix="multitalk_v150_") as temp_dir:
            logger.info(f"Work directory: {temp_dir}")
            
            # Download input files from S3
            audio_path = os.path.join(temp_dir, "input_audio.wav")
            image_path = os.path.join(temp_dir, "input_image.png")
            
            try:
                s3_handler.download_from_s3(audio_input, audio_path)
                s3_handler.download_from_s3(image_input, image_path)
            except Exception as e:
                return {
                    "error": f"Failed to download input files: {str(e)}",
                    "status": "failed",
                    "error_type": "download_error"
                }
            
            # Validate inputs exist
            if not os.path.exists(audio_path):
                return {
                    "error": f"Audio file not found after download: {audio_path}",
                    "status": "failed",
                    "error_type": "file_error"
                }
            if not os.path.exists(image_path):
                return {
                    "error": f"Image file not found after download: {image_path}",
                    "status": "failed",
                    "error_type": "file_error"
                }
            
            # Generate output path
            output_path = os.path.join(temp_dir, "output_video.mp4")
            
            # Calculate number of frames based on audio duration
            try:
                import soundfile as sf
                audio_info = sf.info(audio_path)
                audio_duration = audio_info.duration
                raw_frames = int(audio_duration * 25)  # 25 fps
                num_frames = ((raw_frames + 2) // 4) * 4 + 1
                num_frames = max(25, min(num_frames, 201))
                logger.info(f"Auto-calculated {num_frames} frames for {audio_duration:.2f}s audio")
            except Exception as e:
                logger.error(f"Failed to analyze audio: {str(e)}")
                return {
                    "error": f"Failed to analyze audio file: {str(e)}",
                    "status": "failed",
                    "error_type": "audio_analysis_error"
                }
            
            # Generate video using reference implementation
            try:
                logger.info(f"🎬 Generating video with {sample_steps} steps...")
                generated_path = multitalk_instance.generate(
                    audio_path=audio_path,
                    image_path=image_path,
                    output_path=output_path,
                    prompt=prompt,
                    num_frames=num_frames,
                    sampling_steps=sample_steps,
                    turbo=turbo,
                    seed=42
                )
            except Exception as e:
                error_msg = f"Video generation failed: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return {
                    "error": error_msg,
                    "status": "failed",
                    "error_type": "generation_error",
                    "traceback": traceback.format_exc()
                }
            
            # Verify output exists
            if not os.path.exists(generated_path):
                return {
                    "error": f"Video generation failed - output not found: {generated_path}",
                    "status": "failed",
                    "error_type": "output_error"
                }
            
            # Get file size
            file_size = os.path.getsize(generated_path)
            logger.info(f"✅ Generated video: {file_size / 1024 / 1024:.2f} MB")
            
            # Handle output format
            if output_format == 's3':
                try:
                    # Upload to S3
                    timestamp = int(time.time())
                    s3_key = f"multitalk_outputs/v150_{timestamp}.mp4"
                    video_url = s3_handler.upload_to_s3(generated_path, s3_key)
                    
                    return {
                        "video_url": video_url,
                        "video_s3_key": s3_key,
                        "duration": audio_duration,
                        "frames": num_frames,
                        "size_mb": file_size / 1024 / 1024,
                        "version": "V150",
                        "status": "success"
                    }
                except Exception as e:
                    return {
                        "error": f"Failed to upload video to S3: {str(e)}",
                        "status": "failed",
                        "error_type": "upload_error"
                    }
            else:
                try:
                    # Return base64
                    logger.info("Encoding video to base64...")
                    with open(generated_path, 'rb') as f:
                        video_data = f.read()
                    video_base64 = base64.b64encode(video_data).decode('utf-8')
                    
                    return {
                        "video_base64": video_base64,
                        "duration": audio_duration,
                        "frames": num_frames,
                        "size_mb": file_size / 1024 / 1024,
                        "version": "V150",
                        "status": "success"
                    }
                except Exception as e:
                    return {
                        "error": f"Failed to encode video: {str(e)}",
                        "status": "failed",
                        "error_type": "encoding_error"
                    }
    
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed",
            "error_type": "handler_error",
            "version": "V150"
        }

# RunPod serverless handler
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler V150...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Model path: {os.environ.get('MODEL_PATH', '/runpod-volume/models')}")
    
    # Run startup diagnostics
    try:
        logger.info("Running startup diagnostics...")
        import subprocess
        result = subprocess.run([sys.executable, "/app/startup_diagnostics.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Startup diagnostics passed")
        else:
            logger.warning("⚠️  Startup diagnostics detected issues")
            logger.warning(result.stdout)
            logger.warning(result.stderr)
    except Exception as e:
        logger.error(f"Failed to run startup diagnostics: {e}")
    
    # Log environment variables (safely)
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if any(secret in key.upper() for secret in ['KEY', 'SECRET', 'PASSWORD', 'TOKEN']):
            logger.info(f"  {key}: {'*' * 20}...")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("🚀 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})