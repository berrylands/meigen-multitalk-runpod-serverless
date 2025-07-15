"""
RunPod Serverless Handler - V50 Debug
Comprehensive debugging to test all assumptions
"""
import os
import sys
import json
import logging
import traceback
import time
from pathlib import Path
from typing import Dict, Any

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
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
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError:
    logger.error("PyTorch not available")
    HAS_TORCH = False

# Global variables
s3_client = None
debug_pipeline = None

def init_s3():
    """Initialize S3 client"""
    global s3_client
    
    if not HAS_BOTO3:
        logger.warning("Boto3 not available, S3 operations will fail")
        return None
    
    try:
        region = os.environ.get('AWS_REGION', 'eu-west-1')
        s3_client = boto3.client('s3', region_name=region)
        
        bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"S3 initialized successfully for bucket: {bucket_name}")
        return s3_client
        
    except Exception as e:
        logger.error(f"S3 initialization error: {e}")
        return None

def download_from_s3(filename: str):
    """Download file from S3"""
    if not s3_client:
        logger.error("S3 client not initialized")
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

def upload_debug_report(report: Dict[str, Any]):
    """Upload debug report to S3"""
    if not s3_client:
        return None
    
    try:
        bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        timestamp = int(time.time())
        report_key = f"multitalk-debug/debug_report_{timestamp}.json"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=report_key,
            Body=json.dumps(report, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"Debug report uploaded to: s3://{bucket_name}/{report_key}")
        return f"s3://{bucket_name}/{report_key}"
        
    except Exception as e:
        logger.error(f"Failed to upload debug report: {e}")
        return None

def init_debug_pipeline():
    """Initialize debug pipeline"""
    global debug_pipeline
    
    try:
        from multitalk_debug_implementation import MultiTalkDebugPipeline
        
        model_path = os.environ.get('MODEL_PATH', '/runpod-volume/models')
        logger.info(f"Initializing debug pipeline with model path: {model_path}")
        
        debug_pipeline = MultiTalkDebugPipeline(model_path=model_path)
        logger.info("Debug pipeline initialized")
        
        return debug_pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize debug pipeline: {e}")
        logger.error(traceback.format_exc())
        return None

def handler(job):
    """RunPod handler function"""
    try:
        logger.info("=== DEBUG HANDLER CALLED ===")
        
        # Extract input
        job_input = job.get('input', {})
        
        # Log input structure
        logger.info(f"Job input keys: {list(job_input.keys())}")
        logger.info(f"Job input: {json.dumps(job_input, indent=2)}")
        
        # Get parameters
        audio_filename = job_input.get('audio_1', job_input.get('audio'))
        image_filename = job_input.get('condition_image', job_input.get('reference_image'))
        run_debug = job_input.get('debug', True)
        
        # Initialize debug pipeline
        pipeline = init_debug_pipeline()
        if not pipeline:
            return {
                "error": "Failed to initialize debug pipeline",
                "success": False
            }
        
        # If we have input files, try to process them
        if audio_filename and image_filename:
            logger.info(f"Processing files: audio={audio_filename}, image={image_filename}")
            
            audio_data = download_from_s3(audio_filename)
            image_data = download_from_s3(image_filename)
            
            if audio_data and image_data:
                result = pipeline.process_audio_to_video(
                    audio_data=audio_data,
                    reference_image=image_data,
                    prompt=job_input.get('prompt', 'A person talking naturally')
                )
            else:
                result = {
                    "success": False,
                    "error": "Failed to download input files",
                    "debug_info": pipeline.debug_info
                }
        else:
            # Just return debug info without processing
            result = {
                "success": True,
                "message": "Debug run without input files",
                "debug_info": pipeline.debug_info
            }
        
        # Upload debug report
        debug_report = {
            "timestamp": time.time(),
            "job_input": job_input,
            "environment": {
                "model_path": os.environ.get('MODEL_PATH', '/runpod-volume/models'),
                "pythonpath": os.environ.get('PYTHONPATH', ''),
                "cuda_available": torch.cuda.is_available() if HAS_TORCH else False,
                "working_directory": os.getcwd(),
                "sys_path": sys.path[:10]  # First 10 entries
            },
            "result": result
        }
        
        report_url = upload_debug_report(debug_report)
        
        return {
            "success": True,
            "debug_report_url": report_url,
            "debug_summary": {
                "models_found": result.get("debug_info", {}).get("models_found", {}),
                "dependencies": result.get("debug_info", {}).get("dependencies", {}),
                "imports": result.get("debug_info", {}).get("imports", {}),
                "errors": result.get("debug_info", {}).get("errors", [])
            },
            "message": "Debug run completed - check logs and debug report for details"
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": f"Debug handler error: {str(e)}",
            "success": False
        }

def main():
    """Main function"""
    try:
        logger.info("=== MULTITALK V50 DEBUG HANDLER STARTING ===")
        logger.info(f"Version: {os.environ.get('VERSION', 'unknown')}")
        logger.info(f"Build ID: {os.environ.get('BUILD_ID', 'unknown')}")
        
        # Log environment
        logger.info("\nENVIRONMENT VARIABLES:")
        for key in ['MODEL_PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'AWS_REGION']:
            logger.info(f"  {key}: {os.environ.get(key, 'not set')}")
        
        # Initialize S3
        logger.info("\nInitializing S3 client...")
        init_s3()
        
        # Start handler
        if HAS_RUNPOD:
            logger.info("Starting RunPod serverless handler...")
            runpod.serverless.start({
                "handler": handler
            })
        else:
            logger.error("RunPod not available - running test mode")
            # Run a test
            test_result = handler({"input": {}})
            logger.info(f"Test result: {json.dumps(test_result, indent=2)}")
                
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()