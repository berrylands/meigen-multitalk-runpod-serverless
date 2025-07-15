"""
RunPod Handler for Working MultiTalk Implementation
Based on official patterns but deployable now
"""
import os
import sys
import traceback
import logging
import base64
from datetime import datetime
from pathlib import Path

import runpod
import boto3
from botocore.exceptions import ClientError

# Import our working implementation
from multitalk_working_implementation import WorkingMultiTalkPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(filename)-20s:%(lineno)-4d %(asctime)s,%(msecs)d %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global variables
multitalk_pipeline = None
s3_client = None
s3_bucket = None

def init_s3():
    """Initialize S3 client"""
    global s3_client, s3_bucket
    
    try:
        region = os.environ.get('AWS_REGION', 'eu-west-1')
        s3_bucket = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        
        logger.info(f"Initializing S3 client for region: {region}, bucket: {s3_bucket}")
        
        s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        
        # Test connection
        s3_client.head_bucket(Bucket=s3_bucket)
        logger.info("✓ S3 integration enabled")
        
    except Exception as e:
        logger.warning(f"S3 initialization failed: {e}")
        s3_client = None

def init_multitalk():
    """Initialize Working MultiTalk pipeline"""
    global multitalk_pipeline
    
    try:
        model_path = os.environ.get('MODEL_PATH', '/runpod-volume/models')
        logger.info("Initializing Working MultiTalk pipeline...")
        
        multitalk_pipeline = WorkingMultiTalkPipeline(model_path=model_path)
        logger.info("✓ Working MultiTalk pipeline ready")
        
    except Exception as e:
        logger.error(f"Failed to initialize MultiTalk: {e}")
        raise

def download_from_s3(filename: str) -> bytes:
    """Download file from S3 bucket"""
    if s3_client is None:
        raise RuntimeError("S3 not configured")
    
    try:
        logger.info(f"Downloading from S3: {s3_bucket}/{filename}")
        response = s3_client.get_object(Bucket=s3_bucket, Key=filename)
        data = response['Body'].read()
        logger.info(f"Downloaded {len(data)} bytes from S3")
        return data
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            raise FileNotFoundError(f"File not found in S3: {filename}")
        else:
            raise RuntimeError(f"S3 error ({error_code}): {e}")

def upload_to_s3(data: bytes, filename: str) -> str:
    """Upload data to S3 and return URL"""
    if s3_client is None:
        raise RuntimeError("S3 not configured")
    
    try:
        logger.info(f"Uploading to S3: {s3_bucket}/{filename}")
        
        s3_client.put_object(
            Bucket=s3_bucket,
            Key=filename,
            Body=data,
            ContentType='video/mp4'
        )
        
        url = f"s3://{s3_bucket}/{filename}"
        logger.info(f"Uploaded to S3: {url}")
        return url
        
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise

def process_input_data(input_data: str, input_type: str) -> bytes:
    """Process input data (base64 or S3 filename)"""
    try:
        # Check if it looks like base64
        if input_data.startswith('data:') or len(input_data) > 1000:
            # Likely base64 encoded data
            if input_data.startswith('data:'):
                # Remove data URL prefix
                input_data = input_data.split(',', 1)[1]
            
            return base64.b64decode(input_data)
        else:
            # Likely S3 filename
            return download_from_s3(input_data)
            
    except Exception as e:
        logger.error(f"Error processing {input_type} input: {e}")
        raise

def handler(event):
    """Main handler function"""
    try:
        input_data = event.get('input', {})
        request_id = event.get('id', 'unknown')
        
        logger.info(f"Processing job: {request_id}")
        
        # Extract parameters
        audio_input = input_data.get('audio_1') or input_data.get('audio')
        image_input = input_data.get('reference_image')
        prompt = input_data.get('prompt', 'A person talking naturally')
        
        # Validate inputs
        if not audio_input:
            return {"error": "Missing required parameter: audio_1 or audio"}
        if not image_input:
            return {"error": "Missing required parameter: reference_image"}
        
        logger.info("Processing audio input...")
        audio_data = process_input_data(audio_input, "audio")
        logger.info(f"Audio processed: {len(audio_data)} bytes")
        
        logger.info("Processing reference image...")
        image_data = process_input_data(image_input, "image")
        logger.info(f"Reference image processed: {len(image_data)} bytes")
        
        logger.info(f"Prompt: {prompt}")
        
        # Generate video using Working MultiTalk
        result = multitalk_pipeline.process_audio_to_video(
            audio_data=audio_data,
            reference_image=image_data,
            prompt=prompt
        )
        
        if not result.get('success'):
            return {"error": result.get('error', 'Unknown error during video generation')}
        
        video_data = result['video_data']
        logger.info(f"Video generated: {len(video_data)} bytes")
        
        # Upload to S3 or return base64
        if s3_client:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multitalk_working_output_{timestamp}.mp4"
            video_url = upload_to_s3(video_data, filename)
            
            return {
                "video_url": video_url,
                "model": result['model'],
                "num_frames": result.get('num_frames'),
                "fps": result.get('fps'),
                "format": "mp4"
            }
        else:
            # Return base64 encoded video
            video_base64 = base64.b64encode(video_data).decode('utf-8')
            
            return {
                "video_base64": video_base64,
                "model": result['model'],
                "num_frames": result.get('num_frames'),
                "fps": result.get('fps'),
                "format": "mp4"
            }
    
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("Working MultiTalk Handler Starting...")
    
    # Check model path
    model_path = os.environ.get('MODEL_PATH', '/runpod-volume/models')
    logger.info(f"Model path: {model_path}")
    logger.info(f"Volume mounted: {Path('/runpod-volume').exists()}")
    
    # Initialize S3 (optional)
    init_s3()
    
    # Initialize MultiTalk (required)
    init_multitalk()
    
    logger.info("✅ Ready for Working MultiTalk video generation!")
    logger.info("=" * 60)
    logger.info("Starting RunPod serverless handler...")
    logger.info("=" * 60)
    
    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler})

if __name__ == "__main__":
    main()