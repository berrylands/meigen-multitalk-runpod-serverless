"""
RunPod Serverless Handler - V51 Using Correct Model Paths
Based on v50 debug findings
"""
import os
import sys
import json
import logging
import traceback
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import required modules
try:
    import runpod
    HAS_RUNPOD = True
except ImportError:
    logger.error("RunPod not available")
    HAS_RUNPOD = False

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    HAS_BOTO3 = True
except ImportError:
    logger.error("Boto3 not available")
    HAS_BOTO3 = False

try:
    import torch
    HAS_TORCH = True
    logger.info(f"PyTorch available: {torch.__version__}")
except ImportError:
    logger.error("PyTorch not available")
    HAS_TORCH = False

# Global variables
s3_client = None
multitalk_pipeline = None

def init_s3():
    """Initialize S3 client"""
    global s3_client
    
    if not HAS_BOTO3:
        logger.warning("Boto3 not available")
        return None
    
    try:
        region = os.environ.get('AWS_REGION', 'eu-west-1')
        s3_client = boto3.client('s3', region_name=region)
        
        bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"S3 initialized for bucket: {bucket_name}")
        return s3_client
        
    except Exception as e:
        logger.error(f"S3 initialization error: {e}")
        return None

def download_from_s3(filename: str) -> Optional[bytes]:
    """Download file from S3"""
    if not s3_client:
        return None
    
    try:
        bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        response = s3_client.get_object(Bucket=bucket_name, Key=filename)
        data = response['Body'].read()
        logger.info(f"Downloaded {filename}: {len(data)} bytes")
        return data
        
    except Exception as e:
        logger.error(f"Download error for {filename}: {e}")
        return None

def upload_to_s3(data: bytes, filename: str) -> Optional[str]:
    """Upload data to S3"""
    if not s3_client:
        return None
    
    try:
        bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        s3_key = f"multitalk-out/{filename}"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=data,
            ContentType='video/mp4'
        )
        
        s3_url = f"s3://{bucket_name}/{s3_key}"
        logger.info(f"Uploaded to: {s3_url}")
        return s3_url
        
    except Exception as e:
        logger.error(f"S3 upload error: {e}")
        return None

def init_multitalk():
    """Initialize MultiTalk pipeline"""
    global multitalk_pipeline
    
    if multitalk_pipeline:
        return multitalk_pipeline
    
    try:
        from multitalk_v51_implementation import MultiTalkV51Pipeline
        
        model_path = os.environ.get('MODEL_PATH', '/runpod-volume/models')
        logger.info(f"Initializing MultiTalk V51 with model path: {model_path}")
        
        multitalk_pipeline = MultiTalkV51Pipeline(model_path=model_path)
        logger.info("MultiTalk V51 pipeline initialized")
        return multitalk_pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize MultiTalk: {e}")
        logger.error(traceback.format_exc())
        return None

def handler(job):
    """RunPod handler function"""
    try:
        # Extract input
        job_input = job.get('input', {})
        
        # Get parameters
        audio_filename = job_input.get('audio_1', job_input.get('audio'))
        image_filename = job_input.get('condition_image', job_input.get('reference_image'))
        prompt = job_input.get('prompt', 'A person talking naturally')
        
        # Model parameters
        num_frames = job_input.get('num_frames', 81)
        fps = job_input.get('fps', 25)
        
        # Validate inputs
        if not audio_filename or not image_filename:
            return {
                "error": "Missing required inputs: audio_1 and condition_image",
                "success": False
            }
        
        logger.info(f"Processing: audio={audio_filename}, image={image_filename}")
        
        # Download files
        audio_data = download_from_s3(audio_filename)
        if not audio_data:
            return {
                "error": f"Failed to download audio: {audio_filename}",
                "success": False
            }
        
        image_data = download_from_s3(image_filename)
        if not image_data:
            return {
                "error": f"Failed to download image: {image_filename}",
                "success": False
            }
        
        # Initialize pipeline
        pipeline = init_multitalk()
        if not pipeline:
            return {
                "error": "MultiTalk pipeline initialization failed",
                "success": False
            }
        
        # Process
        start_time = time.time()
        result = pipeline.process_audio_to_video(
            audio_data=audio_data,
            reference_image=image_data,
            prompt=prompt,
            num_frames=num_frames,
            fps=fps
        )
        
        if not result.get("success"):
            return {
                "error": f"Processing failed: {result.get('error', 'Unknown error')}",
                "success": False
            }
        
        # Upload result
        video_data = result["video_data"]
        if video_data:
            timestamp = int(time.time())
            output_filename = f"v51_output_{timestamp}.mp4"
            s3_url = upload_to_s3(video_data, output_filename)
            
            if s3_url:
                processing_time = time.time() - start_time
                return {
                    "success": True,
                    "s3_url": s3_url,
                    "model": result.get("model", "multitalk-v51"),
                    "num_frames": num_frames,
                    "fps": fps,
                    "processing_time": processing_time,
                    "message": "Generated with MultiTalk V51 using correct model paths"
                }
        
        return {
            "error": "Failed to upload result",
            "success": False
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": f"Handler error: {str(e)}",
            "success": False
        }

def main():
    """Main function"""
    try:
        logger.info("=== MULTITALK V51 HANDLER STARTING ===")
        logger.info(f"Version: {os.environ.get('VERSION', 'unknown')}")
        logger.info(f"Build ID: {os.environ.get('BUILD_ID', 'unknown')}")
        
        # Log model paths we'll be using
        model_path = Path(os.environ.get('MODEL_PATH', '/runpod-volume/models'))
        logger.info(f"Model base path: {model_path}")
        logger.info(f"Expected paths:")
        logger.info(f"  - wan2.1-i2v-14b-480p: {model_path / 'wan2.1-i2v-14b-480p'}")
        logger.info(f"  - meigen-multitalk: {model_path / 'meigen-multitalk'}")
        logger.info(f"  - wav2vec2-base-960h: {model_path / 'wav2vec2-base-960h'}")
        
        # Initialize S3
        logger.info("Initializing S3...")
        init_s3()
        
        # Start handler
        if HAS_RUNPOD:
            logger.info("Starting RunPod serverless handler...")
            runpod.serverless.start({
                "handler": handler
            })
        else:
            logger.error("RunPod not available")
            sys.exit(1)
                
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()