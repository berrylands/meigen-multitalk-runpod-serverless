#!/usr/bin/env python3
"""
RunPod serverless handler for MultiTalk V136 Reference Implementation.
Uses cog-MultiTalk reference implementation with direct API calls.
Compatible with V76 S3 handler signature.
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
logger.info("MeiGen MultiTalk Handler V136 Reference Implementation Starting")
logger.info("Using cog-MultiTalk reference with direct API calls")
logger.info("Compatible with V76 S3 handler signature")
logger.info("=" * 80)

# Import S3 handler - V76 style with download_from_s3(key, path)
class S3Handler:
    def __init__(self):
        self.bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        self.region = os.environ.get('AWS_REGION', 'eu-west-1')
        self.enabled = True
        logger.info(f"S3 handler initialized - bucket: {self.bucket_name}")
    
    def download_from_s3(self, s3_key: str, local_path: str) -> None:
        """Download from S3 - V76 style signature"""
        logger.info(f"Downloading from S3: {s3_key} -> {local_path}")
        
        # Import boto3 here to avoid issues if not installed
        try:
            import boto3
            s3_client = boto3.client('s3')
            s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            # Log file size
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
            
            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Uploaded to: {url}")
            return url
        except Exception as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            raise

# Global variables
s3_handler = None
multitalk = None

# Import reference wrapper
logger.info("Importing MultiTalk Reference Wrapper implementation...")
try:
    from multitalk_reference_wrapper import MultiTalkReferenceWrapper
    logger.info("✅ Successfully imported MultiTalkReferenceWrapper")
except Exception as e:
    logger.error(f"Failed to import MultiTalkReferenceWrapper: {str(e)}")
    raise

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function with V76 S3 compatibility
    """
    logger.info("=" * 80)
    logger.info("🎬 Handler invoked - MultiTalk V136 Reference")
    logger.info("=" * 80)
    
    global s3_handler, multitalk
    
    try:
        # Get job input
        job_input = job.get('input', {})
        logger.info(f"Input data type: {type(job_input)}")
        logger.info(f"Input data: {job_input}")
        
        # Handle both direct input and nested 'input' field (V76 compatibility)
        if 'audio_s3_key' in job_input:
            # V76 format with S3 keys
            audio_input = job_input.get('audio_s3_key', job_input.get('audio_1'))
            image_input = job_input.get('image_s3_key', job_input.get('condition_image'))
            prompt = job_input.get('prompt', "A person talking naturally with expressive lip sync")
            output_format = job_input.get('output_format', 's3')
            sample_steps = job_input.get('sampling_steps', job_input.get('sample_steps', 40))
            turbo = job_input.get('turbo', True)
        else:
            # Direct format
            audio_input = job_input.get('audio_1')
            image_input = job_input.get('condition_image')
            prompt = job_input.get('prompt', "A person talking naturally with expressive lip sync")
            output_format = job_input.get('output_format', 's3')
            sample_steps = job_input.get('sampling_steps', job_input.get('sample_steps', 40))
            turbo = job_input.get('turbo', True)
        
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
        
        # Initialize MultiTalk if needed
        if multitalk is None:
            logger.info("Initializing MultiTalk V136...")
            multitalk = MultiTalkReferenceWrapper()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory(prefix="multitalk_") as temp_dir:
            logger.info(f"Work directory: {temp_dir}")
            
            # Download input files from S3
            audio_path = os.path.join(temp_dir, "input_audio.wav")
            image_path = os.path.join(temp_dir, "input_image.png")
            
            s3_handler.download_from_s3(audio_input, audio_path)
            s3_handler.download_from_s3(image_input, image_path)
            
            # Validate inputs exist
            if not os.path.exists(audio_path):
                raise ValueError(f"Audio file not found after download: {audio_path}")
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found after download: {image_path}")
            
            # Generate output path
            output_path = os.path.join(temp_dir, "output_video.mp4")
            
            # Calculate number of frames based on audio duration
            import soundfile as sf
            audio_info = sf.info(audio_path)
            audio_duration = audio_info.duration
            raw_frames = int(audio_duration * 25)  # 25 fps
            num_frames = ((raw_frames + 2) // 4) * 4 + 1
            num_frames = max(25, min(num_frames, 201))
            logger.info(f"Auto-calculated {num_frames} frames for {audio_duration:.2f}s audio")
            
            # Generate video using reference implementation
            logger.info(f"🎬 Generating video with {sample_steps} steps...")
            generated_path = multitalk.generate(
                audio_path=audio_path,
                image_path=image_path,
                output_path=output_path,
                prompt=prompt,
                num_frames=num_frames,
                sampling_steps=sample_steps,
                turbo=turbo,
                seed=42
            )
            
            # Verify output exists
            if not os.path.exists(generated_path):
                raise RuntimeError(f"Video generation failed - output not found: {generated_path}")
            
            # Get file size
            file_size = os.path.getsize(generated_path)
            logger.info(f"✅ Generated video: {file_size / 1024 / 1024:.2f} MB")
            
            # Handle output format
            if output_format == 's3':
                # Upload to S3
                timestamp = int(time.time())
                s3_key = f"multitalk_outputs/v136_{timestamp}.mp4"
                video_url = s3_handler.upload_to_s3(generated_path, s3_key)
                
                return {
                    "video_url": video_url,
                    "video_s3_key": s3_key,
                    "duration": audio_duration,
                    "frames": num_frames,
                    "size_mb": file_size / 1024 / 1024
                }
            else:
                # Return base64
                logger.info("Encoding video to base64...")
                with open(generated_path, 'rb') as f:
                    video_data = f.read()
                video_base64 = base64.b64encode(video_data).decode('utf-8')
                
                return {
                    "video_base64": video_base64,
                    "duration": audio_duration,
                    "frames": num_frames,
                    "size_mb": file_size / 1024 / 1024
                }
    
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Return error in consistent format
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }

# RunPod serverless handler
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Model path: {os.environ.get('MODEL_PATH', '/runpod-volume/models')}")
    
    # Log environment variables (safely)
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if any(secret in key.upper() for secret in ['KEY', 'SECRET', 'PASSWORD', 'TOKEN']):
            logger.info(f"  {key}: {'*' * 20}...")
        else:
            logger.info(f"  {key}: {value}")
    
    runpod.serverless.start({"handler": handler})