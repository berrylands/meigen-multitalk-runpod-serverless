#!/usr/bin/env python3
"""
RunPod Serverless Handler for MultiTalk V71 - Pre-installed Dependencies
"""
import os
import sys
import json
import time
import boto3
import base64
import logging
import tempfile
import runpod
from typing import Dict, Any
import torch  # For version check

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import V71 pipeline
from multitalk_v71_official_wrapper import MultiTalkV71Pipeline, S3Handler

# Global variables
multitalk_pipeline = None
s3_handler = None

# Constants
VERSION = "71.0.0"
BUILD_ID = os.environ.get("BUILD_ID", "multitalk-v71-preinstalled")
MODEL_PATH = os.environ.get("MODEL_PATH", "/runpod-volume/models")
S3_BUCKET = os.environ.get("S3_BUCKET", "760572149-framepack")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")


def init_multitalk():
    """Initialize MultiTalk pipeline"""
    global multitalk_pipeline
    
    if multitalk_pipeline is None:
        logger.info("Initializing MultiTalk V71 with model path: %s", MODEL_PATH)
        try:
            multitalk_pipeline = MultiTalkV71Pipeline(model_path=MODEL_PATH)
            logger.info("MultiTalk V71 pipeline initialized")
        except Exception as e:
            logger.error("Failed to initialize MultiTalk: %s", str(e))
            import traceback
            logger.error(traceback.format_exc())
            raise


def init_s3():
    """Initialize S3 client"""
    global s3_handler
    
    if s3_handler is None:
        logger.info("Initializing S3...")
        try:
            s3_handler = S3Handler(bucket_name=S3_BUCKET, region=AWS_REGION)
            logger.info("S3 initialized for bucket: %s", S3_BUCKET)
        except Exception as e:
            logger.error("Failed to initialize S3: %s", str(e))
            raise


def download_from_s3(s3_key: str) -> bytes:
    """Download file from S3 and return bytes"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        temp_path = tmp_file.name
        
    try:
        if s3_handler.download_file(s3_key, temp_path):
            with open(temp_path, 'rb') as f:
                data = f.read()
            logger.info("Downloaded %s: %d bytes", s3_key, len(data))
            return data
        else:
            raise RuntimeError(f"Failed to download {s3_key}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def upload_to_s3(data: bytes, s3_key: str) -> str:
    """Upload data to S3 and return URL"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(data)
        temp_path = tmp_file.name
        
    try:
        s3_url = s3_handler.upload_file(temp_path, s3_key)
        if s3_url:
            logger.info("Uploaded to: %s", s3_url)
            return s3_url
        else:
            raise RuntimeError(f"Failed to upload to {s3_key}")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function
    
    Expected input format:
    {
        "input": {
            "audio": "path/to/audio.wav",  # S3 key
            "image": "path/to/image.png",  # S3 key
            "prompt": "Optional prompt text",
            "sample_steps": 40,  # Optional
            "use_teacache": true,  # Optional
            "mode": "streaming",  # Optional
            "low_vram": false  # Optional
        }
    }
    """
    try:
        job_input = job.get("input", {})
        
        # Extract parameters
        audio_key = job_input.get("audio")
        image_key = job_input.get("image")
        prompt = job_input.get("prompt", "A person talking naturally")
        sample_steps = job_input.get("sample_steps", 40)
        use_teacache = job_input.get("use_teacache", True)
        mode = job_input.get("mode", "streaming")
        low_vram = job_input.get("low_vram", False)
        
        # Check if we have S3 keys or direct data
        audio_data = None
        image_data = None
        
        # Option 1: Check for S3 keys
        if audio_key and image_key:
            logger.info("Processing from S3: audio=%s, image=%s", audio_key, image_key)
            audio_data = download_from_s3(audio_key)
            image_data = download_from_s3(image_key)
        else:
            # Option 2: Check for direct data (base64 or bytes)
            audio_data_raw = job_input.get("audio_data")
            image_data_raw = job_input.get("image_data")
            
            if audio_data_raw and image_data_raw:
                # Decode base64 if needed
                import base64
                if isinstance(audio_data_raw, str):
                    audio_data = base64.b64decode(audio_data_raw)
                else:
                    audio_data = audio_data_raw
                    
                if isinstance(image_data_raw, str):
                    image_data = base64.b64decode(image_data_raw)
                else:
                    image_data = image_data_raw
                    
                logger.info("Processing from direct data: audio=%d bytes, image=%d bytes", 
                           len(audio_data), len(image_data))
            else:
                # Option 3: Default to test files if no input provided
                logger.info("No input provided, using default test files")
                audio_key = "1.wav"
                image_key = "multi1.png"
                logger.info("Using default test files: %s, %s", audio_key, image_key)
                audio_data = download_from_s3(audio_key)
                image_data = download_from_s3(image_key)
        
        # Process with MultiTalk
        result = multitalk_pipeline.process_audio_to_video(
            audio_data=audio_data,
            reference_image=image_data,
            prompt=prompt,
            sample_steps=sample_steps,
            use_teacache=use_teacache,
            mode=mode,
            low_vram=low_vram
        )
        
        if result["success"]:
            # Upload output video to S3
            timestamp = int(time.time())
            output_key = f"multitalk-out/v71_output_{timestamp}.mp4"
            s3_url = upload_to_s3(result["video_data"], output_key)
            
            return {
                "success": True,
                "s3_url": s3_url,
                "model": result["model"],
                "num_frames": result["num_frames"],
                "fps": result["fps"],
                "processing_time": result["processing_time"],
                "message": f"Generated with MultiTalk V71 using official implementation (pre-installed deps)"
            }
        else:
            raise RuntimeError("Video generation failed")
            
    except Exception as e:
        logger.error("Handler error: %s", str(e))
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "error": str(e),
            "success": False
        }


# RunPod serverless entrypoint
if __name__ == "__main__":
    logger.info("PyTorch available: %s", torch.__version__ if 'torch' in sys.modules else "Not imported")
    logger.info("=== MULTITALK V71 HANDLER STARTING ===")
    logger.info("Version: %s", VERSION)
    logger.info("Build ID: %s", BUILD_ID)
    logger.info("Model base path: %s", MODEL_PATH)
    logger.info("Expected paths:")
    logger.info("  - wan2.1-i2v-14b-480p: %s", os.path.join(MODEL_PATH, "wan2.1-i2v-14b-480p"))
    logger.info("  - meigen-multitalk: %s", os.path.join(MODEL_PATH, "meigen-multitalk"))
    logger.info("  - wav2vec2-base-960h: %s", os.path.join(MODEL_PATH, "wav2vec2-base-960h"))
    
    # Check if official script exists
    official_script = "/app/multitalk_official/generate_multitalk.py"
    if os.path.exists(official_script):
        logger.info("✓ Official MultiTalk script found at: %s", official_script)
    else:
        logger.warning("✗ Official MultiTalk script not found at: %s", official_script)
    
    # Initialize S3
    init_s3()
    
    # Initialize MultiTalk
    init_multitalk()
    
    logger.info("Starting RunPod serverless handler...")
    
    # Start the serverless handler
    runpod.serverless.start({
        "handler": handler
    })