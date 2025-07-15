#!/usr/bin/env python3
"""
RunPod Serverless Handler for MeiGen MultiTalk
Version: 74.3.0 - Comprehensive Validation Framework

Proactive validation of:
- System dependencies and versions
- Resource availability (GPU, RAM, disk)
- Model structure and completeness
- Input file validation
- Environment configuration
- Runtime monitoring with timeouts
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
from pathlib import Path
import shutil

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
logger.info("MeiGen MultiTalk Handler V74.3.0 Starting")
logger.info("Comprehensive Validation Framework - Proactive Issue Prevention")
logger.info("=" * 80)

# Import after logging setup
try:
    logger.info("Importing MultiTalk V74.3 comprehensive implementation...")
    from multitalk_v74_3_comprehensive import MultiTalkV74ComprehensiveWrapper
    logger.info("✅ Successfully imported MultiTalkV74ComprehensiveWrapper")
except Exception as e:
    logger.error(f"❌ Failed to import MultiTalk implementation: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

class S3Handler:
    """Handle S3 operations with validation."""
    
    def __init__(self):
        """Initialize S3 client with comprehensive validation."""
        self.bucket_name = os.environ.get('S3_BUCKET', '760572149-framepack')
        self.region = os.environ.get('AWS_DEFAULT_REGION', 'eu-west-1')
        
        logger.info(f"Initializing S3 handler for bucket: {self.bucket_name}")
        logger.info(f"AWS Region: {self.region}")
        
        # Validate S3 configuration
        self._validate_s3_config()
        
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

    def _validate_s3_config(self):
        """Validate S3 configuration."""
        required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.warning(f"⚠️ Missing S3 environment variables: {missing_vars}")
            logger.warning("S3 operations may fail")

    def _test_s3_access(self):
        """Test S3 access with comprehensive validation."""
        try:
            # Test bucket access
            response = self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info("✅ S3 bucket access verified")
            
            # Test list operation
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=1
            )
            object_count = response.get('KeyCount', 0)
            logger.info(f"✅ S3 list operation verified. Bucket contains {object_count} objects")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise RuntimeError(f"S3 bucket does not exist: {self.bucket_name}")
            elif error_code == 'AccessDenied':
                raise RuntimeError(f"S3 access denied for bucket: {self.bucket_name}")
            else:
                raise RuntimeError(f"S3 access test failed: {e}")
        except Exception as e:
            logger.error(f"❌ S3 access test failed: {str(e)}")
            raise

    def download_from_s3(self, s3_key: str, local_path: str) -> str:
        """Download a file from S3 with validation."""
        try:
            logger.info(f"Downloading from S3: {s3_key} -> {local_path}")
            
            # Validate key exists
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    raise RuntimeError(f"File not found in S3: {s3_key}")
                else:
                    raise RuntimeError(f"S3 head object failed: {e}")
            
            # Ensure local directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=s3_key,
                Filename=local_path
            )
            
            # Validate download
            if not os.path.exists(local_path):
                raise RuntimeError(f"Download failed - file not created: {local_path}")
            
            file_size = os.path.getsize(local_path)
            if file_size == 0:
                raise RuntimeError(f"Downloaded file is empty: {local_path}")
            
            logger.info(f"✅ Downloaded {s3_key} ({file_size} bytes)")
            return local_path
            
        except Exception as e:
            logger.error(f"❌ Download error: {str(e)}")
            raise

    def upload_to_s3(self, local_path: str, s3_key: str) -> str:
        """Upload a file to S3 with validation."""
        try:
            logger.info(f"Uploading to S3: {local_path} -> {s3_key}")
            
            # Validate local file
            if not os.path.exists(local_path):
                raise RuntimeError(f"Local file not found: {local_path}")
            
            file_size = os.path.getsize(local_path)
            if file_size == 0:
                raise RuntimeError(f"Local file is empty: {local_path}")
            
            # Determine content type
            content_type = 'video/mp4' if s3_key.endswith('.mp4') else 'application/octet-stream'
            
            # Upload file
            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=self.bucket_name,
                Key=s3_key,
                ExtraArgs={
                    'ContentType': content_type
                }
            )
            
            # Verify upload
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            except ClientError:
                raise RuntimeError(f"Upload verification failed: {s3_key}")
            
            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Uploaded to S3: {url} ({file_size} bytes)")
            return url
            
        except Exception as e:
            logger.error(f"❌ Upload error: {str(e)}")
            raise

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler with comprehensive validation.
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("Handler invoked - Comprehensive Validation Mode")
    logger.info("=" * 80)
    
    try:
        # Extract and validate input
        input_data = event.get('input', {})
        
        logger.info(f"Input data type: {type(input_data)}")
        logger.info(f"Input data: {input_data}")
        
        # Handle different input formats flexibly
        audio_key = audio_data = image_key = image_data = None
        output_format = 'url'
        
        if isinstance(input_data, dict):
            # Support multiple input formats
            audio_key = (input_data.get('audio_s3_key') or 
                        input_data.get('audio_key') or 
                        input_data.get('audio_1'))
            image_key = (input_data.get('image_s3_key') or 
                        input_data.get('image_key') or
                        input_data.get('condition_image'))
            audio_data = input_data.get('audio_data')
            image_data = input_data.get('image_data')
            
            output_format = input_data.get('output_format', 'url')
            if output_format == 's3':
                output_format = 'url'
            
            custom_s3_key = input_data.get('s3_output_key')
        
        # Default to test files if no input provided
        if not any([audio_key, audio_data, image_key, image_data]):
            logger.info("No input provided, using default test files")
            audio_key = "1.wav"
            image_key = "multi1.png"
        
        logger.info(f"Processing request:")
        logger.info(f"  - Audio: {'S3 key: ' + audio_key if audio_key else 'base64 data' if audio_data else 'none'}")
        logger.info(f"  - Image: {'S3 key: ' + image_key if image_key else 'base64 data' if image_data else 'none'}")
        logger.info(f"  - Output format: {output_format}")
        
        # Initialize components with validation
        logger.info("Initializing S3 handler with validation...")
        s3_handler = S3Handler()
        
        logger.info("Initializing MultiTalk V74.3 with comprehensive validation...")
        multitalk = MultiTalkV74ComprehensiveWrapper()
        
        # Prepare work directory
        work_dir = Path("/tmp/multitalk_work")
        work_dir.mkdir(exist_ok=True)
        
        # Get and validate audio file
        if audio_key:
            audio_path = str(work_dir / "input_audio.wav")
            s3_handler.download_from_s3(audio_key, audio_path)
        elif audio_data:
            audio_path = str(work_dir / "input_audio.wav")
            try:
                audio_bytes = base64.b64decode(audio_data)
                if len(audio_bytes) == 0:
                    raise ValueError("Base64 audio data is empty")
                
                with open(audio_path, 'wb') as f:
                    f.write(audio_bytes)
                logger.info(f"✅ Decoded audio data ({len(audio_bytes)} bytes)")
            except Exception as e:
                raise ValueError(f"Invalid base64 audio data: {e}")
        else:
            raise ValueError("No audio input provided")
        
        # Get and validate image file
        if image_key:
            image_path = str(work_dir / "input_image.png")
            s3_handler.download_from_s3(image_key, image_path)
        elif image_data:
            image_path = str(work_dir / "input_image.png")
            try:
                image_bytes = base64.b64decode(image_data)
                if len(image_bytes) == 0:
                    raise ValueError("Base64 image data is empty")
                
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                logger.info(f"✅ Decoded image data ({len(image_bytes)} bytes)")
            except Exception as e:
                raise ValueError(f"Invalid base64 image data: {e}")
        else:
            raise ValueError("No image input provided")
        
        # Generate video with comprehensive monitoring
        logger.info("Generating video with comprehensive validation...")
        output_path = str(work_dir / "output_video.mp4")
        
        result = multitalk.generate(
            audio_path=audio_path,
            image_path=image_path,
            output_path=output_path
        )
        
        if not result.get('success') or not os.path.exists(output_path):
            raise RuntimeError("Video generation failed")
        
        logger.info(f"✅ Video generated successfully: {output_path} ({result.get('size', 0)} bytes)")
        
        # Return result based on format
        if output_format == 'base64':
            with open(output_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            result_data = {
                'video_data': video_data,
                'format': 'mp4',
                'duration': result.get('duration', 0),
                'generation_time': result.get('generation_time', 0)
            }
        else:
            # Upload to S3
            if 'custom_s3_key' in locals() and custom_s3_key:
                s3_key = custom_s3_key
            else:
                timestamp = int(time.time() * 1000)
                s3_key = f"multitalk_output_{timestamp}.mp4"
            
            video_url = s3_handler.upload_to_s3(output_path, s3_key)
            result_data = {
                'video_url': video_url,
                'format': 'mp4',
                'duration': result.get('duration', 0),
                'generation_time': result.get('generation_time', 0),
                's3_key': s3_key
            }
        
        # Clean up
        shutil.rmtree(work_dir, ignore_errors=True)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Request completed successfully in {elapsed:.2f}s")
        
        return result_data
        
    except Exception as e:
        logger.error(f"❌ Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up on error
        if 'work_dir' in locals():
            shutil.rmtree(work_dir, ignore_errors=True)
        
        return {
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        }

# RunPod serverless entrypoint
if __name__ == "__main__":
    logger.info("Starting RunPod serverless handler - Comprehensive Validation Mode")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Model path: {os.environ.get('MODEL_PATH', '/runpod-volume/models')}")
    
    # Log environment (excluding sensitive data)
    logger.info("Environment variables:")
    for key, value in os.environ.items():
        if not any(sensitive in key.upper() for sensitive in ['SECRET', 'KEY', 'PASSWORD', 'TOKEN']):
            logger.info(f"  {key}: {value}")
    
    runpod.serverless.start({
        "handler": handler
    })