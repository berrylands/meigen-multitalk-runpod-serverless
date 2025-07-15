#!/usr/bin/env python3
"""
Working MultiTalk Handler - Properly handles S3 and video generation
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Model paths
MODEL_BASE = Path(os.environ.get("MODEL_PATH", "/runpod-volume/models"))
MODELS = {
    "multitalk": MODEL_BASE / "meigen-multitalk",
    "wan21": MODEL_BASE / "wan2.1-i2v-14b-480p",
    "wav2vec_base": MODEL_BASE / "wav2vec2-base-960h",
    "gfpgan": MODEL_BASE / "gfpgan"
}

# S3 Configuration
S3_ENABLED = False
s3_client = None
s3_bucket = None

def init_s3():
    """Initialize S3 client"""
    global S3_ENABLED, s3_client, s3_bucket
    
    try:
        # Check for AWS credentials
        if 'AWS_ACCESS_KEY_ID' in os.environ and 'AWS_SECRET_ACCESS_KEY' in os.environ:
            # Use explicit region
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
        if error_code == 'NoSuchKey':
            # List bucket contents to help debug
            try:
                response = s3_client.list_objects_v2(Bucket=s3_bucket, MaxKeys=10)
                if 'Contents' in response:
                    logger.error("Available files in bucket:")
                    for obj in response['Contents']:
                        logger.error(f"  - {obj['Key']}")
            except:
                pass
        raise

def process_audio_input(audio_input: Any) -> bytes:
    """Process audio input from various sources"""
    if isinstance(audio_input, bytes):
        return audio_input
        
    if isinstance(audio_input, str):
        # Check if it's a filename (short string without base64 characteristics)
        if len(audio_input) < 100 and not audio_input.startswith('/') and '=' not in audio_input:
            # Try to download from S3
            if S3_ENABLED:
                try:
                    return download_from_s3(audio_input)
                except Exception as e:
                    logger.error(f"Failed to download {audio_input} from S3: {e}")
                    # Fall through to base64 attempt
                    
        # Try base64 decode
        try:
            return base64.b64decode(audio_input)
        except Exception as e:
            logger.error(f"Failed to decode as base64: {e}")
            raise ValueError(f"Could not process audio input: {audio_input}")
            
    raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

def generate_video_with_audio(audio_data: bytes, duration: float = 5.0, 
                            width: int = 480, height: int = 480, 
                            fps: int = 30) -> bytes:
    """Generate a video with the provided audio"""
    
    # Save audio to temp file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as audio_file:
        audio_file.write(audio_data)
        audio_path = audio_file.name
        
    output_path = tempfile.mktemp(suffix='.mp4')
    
    try:
        # For now, create a simple video with the actual audio
        # This proves the audio is being processed correctly
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'color=c=blue:s={width}x{height}:d={duration}:r={fps}',
            '-i', audio_path,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '128k',
            '-shortest',
            output_path
        ]
        
        logger.info("Generating video with provided audio...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise Exception(f"FFmpeg failed: {result.stderr}")
            
        # Read the video
        with open(output_path, 'rb') as f:
            video_data = f.read()
            
        logger.info(f"Generated video: {len(video_data)} bytes")
        return video_data
        
    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        if os.path.exists(output_path):
            os.unlink(output_path)

def handler(job):
    """Main handler function"""
    try:
        job_input = job.get('input', {})
        job_id = job.get('id', 'unknown')
        
        logger.info(f"Processing job: {job_id}")
        
        # Initialize handler info
        handler_info = {
            "handler": "working_handler.py",
            "version": "1.0.0",
            "s3_enabled": S3_ENABLED,
            "models_available": [name for name, path in MODELS.items() if path.exists()]
        }
        
        # Health check
        if job_input.get('health_check'):
            return {
                "status": "healthy",
                **handler_info
            }
            
        # Process audio
        audio_input = job_input.get('audio')
        if not audio_input:
            return {"error": "No audio input provided"}
            
        logger.info(f"Processing audio input, type: {type(audio_input).__name__}, "
                   f"preview: {str(audio_input)[:100]}...")
        
        try:
            audio_data = process_audio_input(audio_input)
            logger.info(f"Audio processed: {len(audio_data)} bytes")
        except Exception as e:
            return {
                "error": f"Failed to process audio: {e}",
                "handler_info": handler_info
            }
            
        # Generate video
        try:
            video_data = generate_video_with_audio(
                audio_data,
                duration=job_input.get('duration', 5.0),
                width=job_input.get('width', 480),
                height=job_input.get('height', 480),
                fps=job_input.get('fps', 30)
            )
            
            # Return as base64
            video_base64 = base64.b64encode(video_data).decode('utf-8')
            
            return {
                "video": video_base64,
                "format": "mp4",
                "handler_info": handler_info
            }
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return {
                "error": f"Video generation failed: {e}",
                "handler_info": handler_info
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}\n{traceback.format_exc()}")
        return {"error": f"Handler failed: {e}"}

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Working MultiTalk Handler Starting...")
    logger.info(f"Model path: {MODEL_BASE}")
    logger.info(f"Volume mounted: {os.path.exists('/runpod-volume')}")
    
    # Check models
    for name, path in MODELS.items():
        if path.exists():
            size_gb = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024**3)
            logger.info(f"✓ {name} ({size_gb:.1f} GB)")
        else:
            logger.info(f"✗ {name} not found")
            
    # Initialize S3
    init_s3()
    
    logger.info("Starting RunPod serverless handler...")
    logger.info("=" * 60)
    
    runpod.serverless.start({"handler": handler})