"""
RunPod Serverless Handler for MeiGen MultiTalk
"""

import os
import sys
import json
import base64
import tempfile
import traceback
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import runpod
import torch
import numpy as np
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add local imports
sys.path.append('/app/src')
from multitalk_inference import MultiTalkInference

# Global model instance
model = None

def health_check(job: Dict[str, Any]) -> Dict[str, Any]:
    """Health check endpoint."""
    logger.info("Health check requested")
    
    # Check if models directory exists
    model_path = os.environ.get('MODEL_PATH', '/runpod-volume/models')
    models_exist = os.path.exists(model_path) and os.path.isdir(model_path)
    
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    
    # Check model loading status
    model_loaded = model is not None
    
    return {
        "status": "healthy",
        "models_directory_exists": models_exist,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "model_loaded": model_loaded,
        "model_path": model_path
    }

def download_input_file(url_or_base64: str, file_type: str) -> str:
    """Download or decode input file and save to temp location."""
    temp_dir = tempfile.gettempdir()
    
    if url_or_base64.startswith('http'):
        # Download from URL
        import requests
        response = requests.get(url_or_base64)
        response.raise_for_status()
        
        ext = '.jpg' if file_type == 'image' else '.wav'
        temp_path = os.path.join(temp_dir, f"input_{file_type}{ext}")
        
        with open(temp_path, 'wb') as f:
            f.write(response.content)
    else:
        # Decode base64
        try:
            decoded = base64.b64decode(url_or_base64)
            ext = '.jpg' if file_type == 'image' else '.wav'
            temp_path = os.path.join(temp_dir, f"input_{file_type}{ext}")
            
            with open(temp_path, 'wb') as f:
                f.write(decoded)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 {file_type}: {str(e)}")
    
    return temp_path

def inference_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for MultiTalk inference.
    
    Expected input format:
    {
        "input": {
            "reference_image": "base64_or_url",
            "audio_1": "base64_or_url",
            "audio_2": "base64_or_url" (optional),
            "prompt": "conversation description",
            "num_frames": 100,
            "seed": 42,
            "turbo": false,
            "sampling_steps": 20
        }
    }
    """
    global model
    
    try:
        job_input = job.get('input', {})
        
        # Validate required inputs
        if not job_input.get('reference_image'):
            return {"error": "reference_image is required"}
        if not job_input.get('audio_1'):
            return {"error": "audio_1 is required"}
        if not job_input.get('prompt'):
            return {"error": "prompt is required"}
        
        # Initialize model if not already loaded
        if model is None:
            logger.info("Loading MultiTalk model...")
            try:
                model = MultiTalkInference(
                    model_path=os.environ.get('MODEL_PATH', '/runpod-volume/models')
                )
                logger.info("Model loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                return {"error": f"Model initialization failed: {str(e)}"}
        
        # Download/decode input files
        logger.info("Processing input files...")
        image_path = download_input_file(job_input['reference_image'], 'image')
        audio1_path = download_input_file(job_input['audio_1'], 'audio')
        audio2_path = None
        
        if job_input.get('audio_2'):
            audio2_path = download_input_file(job_input['audio_2'], 'audio')
        
        # Set generation parameters
        params = {
            'prompt': job_input['prompt'],
            'num_frames': job_input.get('num_frames', 100),
            'seed': job_input.get('seed', 42),
            'turbo': job_input.get('turbo', False),
            'sampling_steps': job_input.get('sampling_steps', 20),
            'guidance_scale': job_input.get('guidance_scale', 7.5),
            'fps': job_input.get('fps', 8)
        }
        
        logger.info(f"Generating video with params: {params}")
        
        # Generate video
        output_path = model.generate(
            reference_image_path=image_path,
            audio1_path=audio1_path,
            audio2_path=audio2_path,
            **params
        )
        
        # Upload result to S3 or return as base64
        if os.environ.get('S3_BUCKET'):
            # Upload to S3 (implementation needed)
            video_url = upload_to_s3(output_path)
            result = {"video_url": video_url}
        else:
            # Return as base64
            with open(output_path, 'rb') as f:
                video_base64 = base64.b64encode(f.read()).decode('utf-8')
            result = {"video_base64": video_base64}
        
        # Cleanup temp files
        for path in [image_path, audio1_path, audio2_path, output_path]:
            if path and os.path.exists(path):
                os.remove(path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in handler: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

def upload_to_s3(file_path: str) -> str:
    """Upload file to S3 and return URL."""
    # TODO: Implement S3 upload
    # This is a placeholder - you'll need to implement actual S3 upload
    import boto3
    from datetime import datetime
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_REGION', 'us-east-1')
    )
    
    bucket = os.environ['S3_BUCKET']
    key = f"multitalk-output/{datetime.now().isoformat()}.mp4"
    
    s3_client.upload_file(file_path, bucket, key)
    
    # Generate presigned URL (valid for 24 hours)
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=86400
    )
    
    return url

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler that routes requests."""
    job_input = job.get('input', {})
    
    # Check if this is a health check request
    if job_input.get('health_check') or job_input.get('type') == 'health':
        return health_check(job)
    
    # Otherwise, handle as inference request
    return inference_handler(job)

# RunPod serverless worker
runpod.serverless.start({"handler": handler})