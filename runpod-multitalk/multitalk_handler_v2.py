#!/usr/bin/env python3
"""
MultiTalk Handler with Real Implementation
"""
import runpod
import os
import sys
import json
import time
import base64
import tempfile
import logging
import traceback
import subprocess
import boto3
from pathlib import Path
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError

# Import the real MultiTalk implementation
sys.path.insert(0, '/app')
from real_multitalk_inference import RealMultiTalkInference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Model paths
MODEL_BASE = Path(os.environ.get("MODEL_PATH", "/runpod-volume/models"))

# Global MultiTalk instance
multitalk_inference = None

# S3 Configuration
S3_ENABLED = False
s3_client = None
s3_bucket = None

def init_multitalk():
    """Initialize MultiTalk inference engine"""
    global multitalk_inference
    
    try:
        logger.info("Initializing MultiTalk inference engine...")
        multitalk_inference = RealMultiTalkInference(MODEL_BASE)
        logger.info("✓ MultiTalk inference engine ready")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MultiTalk: {e}")
        return False

def init_s3():
    """Initialize S3 client"""
    global S3_ENABLED, s3_client, s3_bucket
    
    try:
        if 'AWS_ACCESS_KEY_ID' in os.environ and 'AWS_SECRET_ACCESS_KEY' in os.environ:
            region = os.environ.get('AWS_REGION', 'eu-west-1')
            s3_bucket = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
            
            logger.info(f"Initializing S3 client for region: {region}, bucket: {s3_bucket}")
            
            s3_client = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
            )
            
            # Test access
            try:
                s3_client.head_bucket(Bucket=s3_bucket)
                S3_ENABLED = True
                logger.info("✓ S3 integration enabled")
            except Exception as e:
                logger.error(f"S3 bucket access test failed: {e}")
                S3_ENABLED = False
        else:
            logger.info("AWS credentials not found, S3 integration disabled")
            
    except Exception as e:
        logger.error(f"Failed to initialize S3: {e}")
        S3_ENABLED = False

def download_from_s3(filename: str) -> bytes:
    """Download file from S3 bucket"""
    if not S3_ENABLED or not s3_client:
        raise RuntimeError("S3 not available")
        
    try:
        logger.info(f"Downloading from S3: {s3_bucket}/{filename}")
        response = s3_client.get_object(Bucket=s3_bucket, Key=filename)
        data = response['Body'].read()
        logger.info(f"Downloaded {len(data)} bytes from S3")
        return data
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 download failed - Error code: {error_code}")
        raise

def upload_to_s3(data: bytes, filename: str) -> str:
    """Upload data to S3 and return URL"""
    if not S3_ENABLED or not s3_client:
        raise RuntimeError("S3 not available")
        
    try:
        logger.info(f"Uploading to S3: {s3_bucket}/{filename}")
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=filename,
            Body=data,
            ContentType='video/mp4'
        )
        
        # Generate URL
        url = f"s3://{s3_bucket}/{filename}"
        logger.info(f"Uploaded to S3: {url}")
        return url
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        raise

def process_input(input_data: Any, input_type: str = "audio") -> bytes:
    """Process input from various sources"""
    if isinstance(input_data, bytes):
        return input_data
        
    if isinstance(input_data, str):
        # Check if it's a filename (short string without base64 characteristics)
        if len(input_data) < 100 and not input_data.startswith('/') and '=' not in input_data:
            # Try to download from S3
            if S3_ENABLED:
                try:
                    return download_from_s3(input_data)
                except Exception as e:
                    logger.error(f"Failed to download {input_data} from S3: {e}")
                    
        # Try base64 decode
        try:
            return base64.b64decode(input_data)
        except Exception as e:
            logger.error(f"Failed to decode as base64: {e}")
            raise ValueError(f"Could not process {input_type} input")
            
    raise ValueError(f"Unsupported {input_type} input type: {type(input_data)}")

def handler(job):
    """Main handler function"""
    try:
        job_input = job.get('input', {})
        job_id = job.get('id', 'unknown')
        
        logger.info(f"Processing job: {job_id}")
        
        # Health check
        if job_input.get('health_check'):
            return {
                "status": "healthy",
                "handler": "multitalk_handler.py",
                "version": "2.0.0",
                "multitalk_ready": multitalk_inference is not None,
                "s3_enabled": S3_ENABLED,
                "gpu_available": multitalk_inference.device.type == 'cuda' if multitalk_inference else False
            }
            
        # Ensure MultiTalk is initialized
        if not multitalk_inference:
            return {"error": "MultiTalk not initialized"}
            
        # Process audio
        audio_input = job_input.get('audio')
        if not audio_input:
            return {"error": "No audio input provided"}
            
        logger.info(f"Processing audio input...")
        audio_data = process_input(audio_input, "audio")
        logger.info(f"Audio processed: {len(audio_data)} bytes")
        
        # Process reference image (optional)
        reference_image = None
        if 'reference_image' in job_input:
            logger.info("Processing reference image...")
            reference_image = process_input(job_input['reference_image'], "image")
            logger.info(f"Reference image processed: {len(reference_image)} bytes")
            
        # Get generation parameters
        params = {
            "duration": job_input.get('duration', 5.0),
            "fps": job_input.get('fps', 30),
            "width": job_input.get('width', 480),
            "height": job_input.get('height', 480),
            "prompt": job_input.get('prompt', "A person talking naturally")
        }
        
        logger.info(f"Generation params: {params}")
        
        # Generate video using MultiTalk
        result = multitalk_inference.process_audio_to_video(
            audio_data=audio_data,
            reference_image=reference_image,
            **params
        )
        
        if result.get("success"):
            video_data = result["video_data"]
            
            # Handle output format
            output_format = job_input.get('output_format', 'base64')
            
            if output_format == 's3' and S3_ENABLED:
                # Upload to S3
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                filename = f"multitalk_output_{timestamp}.mp4"
                s3_url = upload_to_s3(video_data, filename)
                
                return {
                    "video_url": s3_url,
                    "format": "mp4",
                    "model": result.get("model", "multitalk"),
                    "resolution": result.get("resolution"),
                    "duration": result.get("duration"),
                    "fps": result.get("fps")
                }
            else:
                # Return as base64
                video_base64 = base64.b64encode(video_data).decode('utf-8')
                
                return {
                    "video": video_base64,
                    "format": "mp4",
                    "model": result.get("model", "multitalk"),
                    "resolution": result.get("resolution"),
                    "duration": result.get("duration"),
                    "fps": result.get("fps")
                }
        else:
            return {
                "error": "Video generation failed",
                "details": result.get("error")
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}\n{traceback.format_exc()}")
        return {"error": f"Handler failed: {e}"}

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MultiTalk Handler Starting...")
    logger.info(f"Model path: {MODEL_BASE}")
    logger.info(f"Volume mounted: {os.path.exists('/runpod-volume')}")
    
    # Initialize S3
    init_s3()
    
    # Initialize MultiTalk
    if init_multitalk():
        logger.info("✅ Ready for MultiTalk video generation!")
    else:
        logger.error("⚠️  MultiTalk initialization failed - limited functionality")
        
    logger.info("Starting RunPod serverless handler...")
    logger.info("=" * 60)
    
    runpod.serverless.start({"handler": handler})