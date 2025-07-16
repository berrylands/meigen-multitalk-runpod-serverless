#!/usr/bin/env python3
"""
RunPod serverless handler for MultiTalk V122 with format compatibility.
Handles both V76 format (audio_s3_key/image_s3_key) and new format (audio_1/condition_image).
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
from typing import Dict, Any, Optional, Tuple

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
logger.info("MeiGen MultiTalk Handler V122 Format Fix Starting")
logger.info("Supports both V76 and new input formats")
logger.info("=" * 80)

# Import S3 handler
try:
    from s3_handler import S3Handler
    logger.info("✓ Successfully imported S3Handler")
except Exception as e:
    logger.warning(f"Could not import S3Handler: {str(e)}")
    
    # Minimal S3 handler fallback
    class S3Handler:
        def __init__(self):
            self.bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
            self.region = os.environ.get('AWS_REGION', 'eu-west-1')
            logger.info("Using fallback S3 handler")
        
        def download_from_s3(self, s3_key: str, local_path: str) -> None:
            logger.info(f"[MOCK] Downloading {s3_key} to {local_path}")
            
        def upload_to_s3(self, local_path: str, s3_key: str) -> str:
            return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

# Import MultiTalk V76 JSON Input implementation (the one that's currently working)
try:
    logger.info("Importing MultiTalk V75.0 JSON Input implementation...")
    from multitalk_v75_0_json_input import MultiTalkV75JsonWrapper
    logger.info("✓ Successfully imported MultiTalkV75JsonWrapper")
except Exception as e:
    logger.error(f"❌ Failed to import MultiTalk implementation: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function with format compatibility.
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🎬 Handler invoked - MultiTalk V122 Format Fix")
    logger.info("=" * 80)
    
    try:
        # Extract input
        input_data = event.get('input', {})
        
        # Log input structure
        logger.info(f"Input data type: {type(input_data)}")
        logger.info(f"Input data: {input_data}")
        
        # Convert input format if needed
        if 'audio_s3_key' in input_data or 'image_s3_key' in input_data:
            # V76 format - convert to new format
            logger.info("Detected V76 format, converting to new format...")
            new_input = {
                'action': 'generate',
                'audio_1': input_data.get('audio_s3_key', '1.wav'),
                'condition_image': input_data.get('image_s3_key', 'multi1.png'),
                'prompt': input_data.get('prompt', 'A person talking naturally with expressive lip sync'),
                'num_frames': input_data.get('num_frames'),
                'sampling_steps': input_data.get('sampling_steps', 40),
                'turbo': input_data.get('turbo', False),
                'output_format': input_data.get('output_format', 's3')
            }
            input_data = new_input
            logger.info(f"Converted input: {input_data}")
        
        # Validate required fields
        if 'audio_1' not in input_data or 'condition_image' not in input_data:
            raise ValueError("Both audio_1 and condition_image are required")
        
        # Extract parameters with V76 compatibility
        audio_key = input_data['audio_1']
        image_key = input_data['condition_image']
        prompt = input_data.get('prompt', 'A person talking naturally with expressive lip sync')
        output_format = input_data.get('output_format', 's3')
        sample_steps = input_data.get('sampling_steps', 40)
        turbo_mode = input_data.get('turbo', False)
        
        logger.info(f"🎬 Processing request:")
        logger.info(f"  - Audio: {audio_key}...")
        logger.info(f"  - Image: {image_key}...")
        logger.info(f"  - Prompt: {prompt}")
        logger.info(f"  - Output format: {output_format}")
        logger.info(f"  - Sample steps: {sample_steps}")
        logger.info(f"  - Turbo mode: {turbo_mode}")
        
        # Initialize S3
        logger.info("Initializing S3 handler...")
        s3_handler = S3Handler()
        
        # Initialize MultiTalk
        logger.info("Initializing MultiTalk V122...")
        multitalk = MultiTalkV75JsonWrapper()
        
        # Create work directory
        work_dir = tempfile.mkdtemp(prefix="multitalk_")
        logger.info(f"Work directory: {work_dir}")
        
        # Download files from S3
        audio_path = os.path.join(work_dir, "input_audio.wav")
        image_path = os.path.join(work_dir, "input_image.png")
        
        # Download from S3 and save to files
        try:
            audio_data = s3_handler.download_from_s3(audio_key)
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            logger.info(f"✓ Downloaded audio: {len(audio_data)} bytes")
            
            image_data = s3_handler.download_from_s3(image_key)
            with open(image_path, 'wb') as f:
                f.write(image_data)
            logger.info(f"✓ Downloaded image: {len(image_data)} bytes")
        except Exception as e:
            logger.warning(f"S3 download failed: {e}, using direct key as path")
            # Fallback: assume keys are S3 keys not full URLs
            audio_s3_url = f"s3://{s3_handler.default_bucket}/{audio_key}"
            image_s3_url = f"s3://{s3_handler.default_bucket}/{image_key}"
            
            audio_data = s3_handler.download_from_s3(audio_s3_url)
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            
            image_data = s3_handler.download_from_s3(image_s3_url)
            with open(image_path, 'wb') as f:
                f.write(image_data)
        
        # Auto-calculate frames if not specified
        if 'num_frames' not in input_data or input_data['num_frames'] is None:
            # Get audio duration and calculate frames
            import librosa
            audio_array, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio_array) / sr
            fps = 25
            
            # Calculate frames (must be 4n+1)
            raw_frames = int(duration * fps)
            num_frames = ((raw_frames // 4) * 4) + 1
            if num_frames < 25:
                num_frames = 25
            
            logger.info(f"Auto-calculated {num_frames} frames for {duration:.2f}s audio")
        else:
            num_frames = input_data['num_frames']
        
        # Validate frame count
        if (num_frames - 1) % 4 != 0:
            logger.warning(f"Frame count {num_frames} is not in 4n+1 format, adjusting...")
            num_frames = ((num_frames // 4) * 4) + 1
        
        logger.info(f"🎬 Generating video with {sample_steps} steps...")
        
        # Call V76 implementation with proper parameters
        generated_path = multitalk.generate_with_options(
            audio_path=audio_path,
            image_path=image_path,
            output_path=os.path.join(work_dir, "output.mp4"),
            prompt=prompt,
            sample_steps=sample_steps,
            mode="clip",
            size="multitalk-480",
            use_teacache=turbo_mode,
            text_guidance_scale=7.5,
            audio_guidance_scale=3.5,
            seed=42
        )
        
        if not os.path.exists(generated_path):
            raise RuntimeError("Video file not created")
        
        logger.info(f"✓ Video generated: {generated_path} ({os.path.getsize(generated_path)} bytes)")
        
        # Return result based on format
        if output_format == 'base64':
            with open(generated_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            response = {
                'video_data': video_data,
                'format': 'mp4',
                'success': True,
                'implementation': 'V122_FORMAT_FIX'
            }
        else:
            # Upload to S3
            timestamp = int(time.time() * 1000)
            s3_key = f"multitalk_v122_output_{timestamp}.mp4"
            video_url = s3_handler.upload_to_s3(generated_path, s3_key)
            response = {
                'video_url': video_url,
                'format': 'mp4',
                'success': True,
                's3_key': s3_key,
                'implementation': 'V122_FORMAT_FIX'
            }
        
        # Clean up
        shutil.rmtree(work_dir, ignore_errors=True)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Request completed in {elapsed:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up on error
        if 'work_dir' in locals():
            shutil.rmtree(work_dir, ignore_errors=True)
        
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False,
            'implementation': 'V122_FORMAT_FIX'
        }

# RunPod handler
logger.info("Starting RunPod serverless handler...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Model path: {os.environ.get('MODEL_PATH', '/runpod-volume/models')}")

# Log environment
logger.info("Environment variables:")
for key, value in sorted(os.environ.items()):
    if 'KEY' in key or 'SECRET' in key or 'PASSWORD' in key:
        logger.info(f"  {key}: {value[:20]}...")
    else:
        logger.info(f"  {key}: {value}")

runpod.serverless.start({"handler": handler})