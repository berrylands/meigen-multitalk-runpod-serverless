#!/usr/bin/env python3
"""
RunPod Serverless Handler for MeiGen MultiTalk
Version: 74.1.0 - Fixed f-string syntax error

Key fixes:
- Fixed f-string backslash syntax error
- Installs build-essential for gcc/g++/make
- Rebuilds xformers to match PyTorch version
- Maintains all V72 functionality
"""

import runpod
import os
import sys
import json
import base64
import logging
import traceback
import time
from typing import Dict, Any, Optional, Union
import boto3
from botocore.exceptions import ClientError
import requests
from pathlib import Path
import shutil
import subprocess

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Version info
logger.info("=" * 80)
logger.info("MeiGen MultiTalk Handler V74.1.0 Starting")
logger.info("GCC Installation Fix - Fixed Syntax Error")
logger.info("=" * 80)

# Import after logging setup
try:
    logger.info("Importing MultiTalk V74.1 implementation...")
    from multitalk_v74_1_official_wrapper import MultiTalkV74OfficialWrapper
    logger.info("✅ Successfully imported MultiTalkV74OfficialWrapper")
except Exception as e:
    logger.error(f"❌ Failed to import MultiTalk implementation: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

class S3Handler:
    """Handle S3 operations with proper error handling."""
    
    def __init__(self):
        """Initialize S3 client with credentials from environment."""
        self.bucket_name = os.environ.get('S3_BUCKET', '760572149-framepack')
        self.region = os.environ.get('AWS_DEFAULT_REGION', 'eu-west-1')
        
        logger.info(f"Initializing S3 handler for bucket: {self.bucket_name}")
        logger.info(f"AWS Region: {self.region}")
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                endpoint_url=f'https://s3.{self.region}.amazonaws.com'
            )
            logger.info("✅ S3 client initialized successfully")
            
            # Test S3 access
            self._test_s3_access()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {str(e)}")
            raise

    def _test_s3_access(self):
        """Test S3 access by listing objects."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=1
            )
            logger.info(f"✅ S3 access verified. Bucket contains {response.get('KeyCount', 0)} objects")
        except Exception as e:
            logger.error(f"❌ S3 access test failed: {str(e)}")
            raise

    def download_from_s3(self, s3_key: str, local_path: str) -> str:
        """Download a file from S3."""
        try:
            logger.info(f"Downloading from S3: {s3_key} -> {local_path}")
            
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=s3_key,
                Filename=local_path
            )
            
            logger.info(f"✅ Downloaded {s3_key} ({os.path.getsize(local_path)} bytes)")
            return local_path
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.error(f"❌ File not found in S3: {s3_key}")
            else:
                logger.error(f"❌ S3 download error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Download error: {str(e)}")
            raise

    def upload_to_s3(self, local_path: str, s3_key: str) -> str:
        """Upload a file to S3 and return the URL."""
        try:
            logger.info(f"Uploading to S3: {local_path} -> {s3_key}")
            
            # Determine content type
            content_type = 'video/mp4' if s3_key.endswith('.mp4') else 'application/octet-stream'
            
            # Upload file
            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=self.bucket_name,
                Key=s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'ACL': 'public-read'
                }
            )
            
            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Uploaded to S3: {url}")
            return url
            
        except Exception as e:
            logger.error(f"❌ Upload error: {str(e)}")
            raise

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.
    
    Args:
        event: RunPod event containing:
            - input: User input data
                - audio_s3_key or audio_data
                - image_s3_key or image_data
                - output_format: "url" or "base64"
    
    Returns:
        Dict containing video URL or base64 data
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("Handler invoked with event")
    logger.info("=" * 80)
    
    try:
        # Extract input
        input_data = event.get('input', {})
        
        # Log input structure
        logger.info(f"Input data type: {type(input_data)}")
        logger.info(f"Input data: {input_data}")
        
        # Handle different input formats flexibly
        audio_key = audio_data = image_key = image_data = None
        output_format = 'url'  # Default output format
        
        # Check for S3 keys - support multiple input formats
        if isinstance(input_data, dict):
            # Support your specific format: audio_1, condition_image
            audio_key = (input_data.get('audio_s3_key') or 
                        input_data.get('audio_key') or 
                        input_data.get('audio_1'))
            image_key = (input_data.get('image_s3_key') or 
                        input_data.get('image_key') or
                        input_data.get('condition_image'))
            audio_data = input_data.get('audio_data')
            image_data = input_data.get('image_data')
            
            # Handle output format
            output_format = input_data.get('output_format', 'url')
            if output_format == 's3':
                output_format = 'url'  # S3 upload returns URL
        
        # Default to test files if no input provided
        if not any([audio_key, audio_data, image_key, image_data]):
            logger.info("No input provided, using default test files")
            audio_key = "1.wav"
            image_key = "multi1.png"
        
        
        logger.info(f"Processing request:")
        logger.info(f"  - Audio: {'S3 key: ' + audio_key if audio_key else 'base64 data' if audio_data else 'none'}")
        logger.info(f"  - Image: {'S3 key: ' + image_key if image_key else 'base64 data' if image_data else 'none'}")
        logger.info(f"  - Output format: {output_format}")
        
        # Initialize components
        logger.info("Initializing S3 handler...")
        s3_handler = S3Handler()
        
        logger.info("Initializing MultiTalk V74.1...")
        multitalk = MultiTalkV74OfficialWrapper()
        
        # Prepare paths
        work_dir = Path("/tmp/multitalk_work")
        work_dir.mkdir(exist_ok=True)
        
        # Get audio file
        if audio_key:
            audio_path = str(work_dir / "input_audio.wav")
            s3_handler.download_from_s3(audio_key, audio_path)
        elif audio_data:
            audio_path = str(work_dir / "input_audio.wav")
            audio_bytes = base64.b64decode(audio_data)
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"✅ Decoded audio data ({len(audio_bytes)} bytes)")
        else:
            raise ValueError("No audio input provided")
        
        # Get image file
        if image_key:
            image_path = str(work_dir / "input_image.png")
            s3_handler.download_from_s3(image_key, image_path)
        elif image_data:
            image_path = str(work_dir / "input_image.png")
            image_bytes = base64.b64decode(image_data)
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            logger.info(f"✅ Decoded image data ({len(image_bytes)} bytes)")
        else:
            raise ValueError("No image input provided")
        
        # Generate video
        logger.info("Generating video with MultiTalk V74.1...")
        output_path = str(work_dir / "output_video.mp4")
        
        result = multitalk.generate(
            audio_path=audio_path,
            image_path=image_path,
            output_path=output_path
        )
        
        if not result or not os.path.exists(output_path):
            raise RuntimeError("Video generation failed")
        
        logger.info(f"✅ Video generated: {output_path} ({os.path.getsize(output_path)} bytes)")
        
        # Return result based on format
        if output_format == 'base64':
            with open(output_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            result = {
                'video_data': video_data,
                'format': 'mp4',
                'duration': result.get('duration', 0)
            }
        else:
            # Upload to S3
            timestamp = int(time.time() * 1000)
            s3_key = f"multitalk_output_{timestamp}.mp4"
            video_url = s3_handler.upload_to_s3(output_path, s3_key)
            result = {
                'video_url': video_url,
                'format': 'mp4',
                'duration': result.get('duration', 0),
                's3_key': s3_key
            }
        
        # Clean up
        shutil.rmtree(work_dir, ignore_errors=True)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Request completed in {elapsed:.2f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up on error
        if 'work_dir' in locals():
            shutil.rmtree(work_dir, ignore_errors=True)
        
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }

# RunPod serverless entrypoint
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Available models: {os.environ.get('MODEL_PATH', '/runpod-volume/models')}")
    
    runpod.serverless.start({
        "handler": handler
    })