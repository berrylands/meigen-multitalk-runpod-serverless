"""
RunPod Serverless Handler - V46 Robust Implementation
Defensive programming to prevent container health issues
"""
import os
import sys
import json
import logging
import traceback
import tempfile
import time
import gc
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging immediately
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required modules with fallbacks
try:
    import runpod
    HAS_RUNPOD = True
    logger.info("✓ RunPod available")
except ImportError:
    logger.error("✗ RunPod not available")
    HAS_RUNPOD = False

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    HAS_BOTO3 = True
    logger.info("✓ Boto3 available")
except ImportError:
    logger.error("✗ Boto3 not available")
    HAS_BOTO3 = False

try:
    import torch
    HAS_TORCH = True
    logger.info(f"✓ PyTorch available: {torch.__version__}")
except ImportError:
    logger.error("✗ PyTorch not available")
    HAS_TORCH = False

# Global variables for initialization
s3_client = None
multitalk_pipeline = None
model_loading_attempted = False

def log_environment():
    """Log environment information for debugging"""
    logger.info("=== ENVIRONMENT INFORMATION ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Available dependencies: RunPod={HAS_RUNPOD}, Boto3={HAS_BOTO3}, PyTorch={HAS_TORCH}")
    
    # Log environment variables (obscure secrets)
    env_vars = [
        'AWS_REGION', 'AWS_S3_BUCKET_NAME', 'MODEL_PATH', 
        'RUNPOD_DEBUG_LEVEL', 'VERSION', 'BUILD_ID'
    ]
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        logger.info(f"{var}: {value}")
    
    # Log secret keys existence (not values)
    secret_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    for var in secret_vars:
        exists = var in os.environ
        logger.info(f"{var}: {'Set' if exists else 'Not set'}")

def init_s3_safe():
    """Initialize S3 client with comprehensive error handling"""
    global s3_client
    
    if not HAS_BOTO3:
        logger.warning("Boto3 not available, S3 operations will fail")
        return None
    
    try:
        region = os.environ.get('AWS_REGION', 'eu-west-1')
        s3_client = boto3.client('s3', region_name=region)
        
        # Test S3 connection
        bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"✓ S3 connection successful to bucket: {bucket_name}")
        return s3_client
        
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        return None
    except ClientError as e:
        logger.error(f"S3 client error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected S3 initialization error: {e}")
        return None

def download_from_s3_safe(filename: str) -> Optional[bytes]:
    """Download file from S3 with comprehensive error handling"""
    if not s3_client:
        logger.error("S3 client not initialized")
        return None
    
    try:
        bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        logger.info(f"Downloading {filename} from S3 bucket {bucket_name}")
        
        response = s3_client.get_object(Bucket=bucket_name, Key=filename)
        data = response['Body'].read()
        
        logger.info(f"✓ Downloaded {filename}: {len(data)} bytes")
        return data
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"S3 download error for {filename}: {error_code} - {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected download error for {filename}: {e}")
        return None

def upload_to_s3_safe(data: bytes, filename: str) -> Optional[str]:
    """Upload data to S3 with error handling"""
    if not s3_client:
        logger.error("S3 client not initialized")
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
        logger.info(f"✓ Uploaded to S3: {s3_url}")
        return s3_url
        
    except Exception as e:
        logger.error(f"S3 upload error: {e}")
        return None

def init_multitalk_safe():
    """Initialize MultiTalk pipeline with defensive programming"""
    global multitalk_pipeline, model_loading_attempted
    
    if model_loading_attempted:
        return multitalk_pipeline
    
    model_loading_attempted = True
    
    try:
        # Try to import the robust implementation
        from multitalk_robust_implementation import RobustMultiTalkPipeline
        
        model_path = os.environ.get('MODEL_PATH', '/runpod-volume/models')
        logger.info(f"Initializing MultiTalk with model path: {model_path}")
        
        multitalk_pipeline = RobustMultiTalkPipeline(model_path=model_path)
        logger.info("✓ MultiTalk pipeline initialized successfully")
        return multitalk_pipeline
        
    except ImportError as e:
        logger.error(f"Failed to import RobustMultiTalkPipeline: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize MultiTalk: {e}")
        logger.error(traceback.format_exc())
        return None

def generate_fallback_response(error_msg: str) -> Dict[str, Any]:
    """Generate fallback response when everything fails"""
    return {
        "success": False,
        "error": f"MultiTalk processing failed: {error_msg}",
        "fallback": True,
        "timestamp": time.time()
    }

def process_request_safe(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Process request with comprehensive error handling"""
    try:
        logger.info("Processing MultiTalk request...")
        logger.info(f"Input keys: {list(job_input.keys())}")
        
        # Extract inputs with defaults
        audio_filename = job_input.get('audio_1', job_input.get('audio'))
        image_filename = job_input.get('condition_image', job_input.get('reference_image'))
        prompt = job_input.get('prompt', 'A person talking naturally')
        
        if not audio_filename or not image_filename:
            return generate_fallback_response("Missing required inputs: audio_1 and condition_image")
        
        logger.info(f"Processing: audio={audio_filename}, image={image_filename}")
        
        # Download files from S3
        audio_data = download_from_s3_safe(audio_filename)
        if not audio_data:
            return generate_fallback_response(f"Failed to download audio: {audio_filename}")
        
        image_data = download_from_s3_safe(image_filename)
        if not image_data:
            return generate_fallback_response(f"Failed to download image: {image_filename}")
        
        # Initialize MultiTalk if needed
        pipeline = init_multitalk_safe()
        if not pipeline:
            return generate_fallback_response("MultiTalk pipeline initialization failed")
        
        # Process with MultiTalk
        logger.info("Starting MultiTalk processing...")
        result = pipeline.process_audio_to_video(
            audio_data=audio_data,
            reference_image=image_data,
            prompt=prompt
        )
        
        if not result.get("success"):
            return generate_fallback_response(f"MultiTalk processing failed: {result.get('error', 'Unknown error')}")
        
        # Upload result to S3
        video_data = result["video_data"]
        if video_data:
            timestamp = int(time.time())
            output_filename = f"output_{timestamp}.mp4"
            s3_url = upload_to_s3_safe(video_data, output_filename)
            
            if s3_url:
                return {
                    "success": True,
                    "s3_url": s3_url,
                    "model": result.get("model", "robust-multitalk"),
                    "num_frames": result.get("num_frames", 81),
                    "fps": result.get("fps", 8),
                    "processing_time": time.time() - result.get("start_time", time.time())
                }
        
        return generate_fallback_response("Failed to upload result to S3")
        
    except Exception as e:
        logger.error(f"Request processing error: {e}")
        logger.error(traceback.format_exc())
        return generate_fallback_response(f"Unexpected error: {str(e)}")

def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "dependencies": {
                "runpod": HAS_RUNPOD,
                "boto3": HAS_BOTO3,
                "torch": HAS_TORCH,
                "s3_client": s3_client is not None,
                "multitalk": multitalk_pipeline is not None
            },
            "environment": {
                "aws_region": os.environ.get('AWS_REGION', 'not_set'),
                "bucket": os.environ.get('AWS_S3_BUCKET_NAME', 'not_set'),
                "model_path": os.environ.get('MODEL_PATH', 'not_set')
            }
        }
        
        # Check if critical components are working
        if not HAS_RUNPOD:
            status["status"] = "degraded"
            status["warnings"] = ["RunPod not available"]
        
        return status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

def main():
    """Main function with comprehensive startup handling"""
    try:
        logger.info("=== MULTITALK V46 ROBUST HANDLER STARTING ===")
        log_environment()
        
        # Initialize S3 early
        logger.info("Initializing S3 client...")
        init_s3_safe()
        
        # Don't initialize MultiTalk during startup to avoid memory issues
        logger.info("Deferring MultiTalk initialization until first request...")
        
        if HAS_RUNPOD:
            logger.info("Starting RunPod serverless handler...")
            runpod.serverless.start({
                "handler": process_request_safe,
                "return_aggregate_stream": True
            })
        else:
            logger.error("RunPod not available - running in standalone mode")
            # Keep container alive for debugging
            while True:
                time.sleep(60)
                logger.info("Standalone mode - container alive")
                
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal startup error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()