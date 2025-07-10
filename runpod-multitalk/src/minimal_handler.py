#!/usr/bin/env python3
"""
Minimal RunPod handler for testing basic functionality
"""

import os
import sys
import json
import base64
import tempfile
import logging
from typing import Dict, Any

import runpod
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def health_check(job: Dict[str, Any]) -> Dict[str, Any]:
    """Health check endpoint."""
    logger.info("Health check requested")
    
    return {
        "status": "healthy",
        "message": "Minimal MultiTalk handler is running",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "python_version": sys.version,
        "environment": {
            "MODEL_PATH": os.environ.get('MODEL_PATH', 'Not set'),
            "RUNPOD_DEBUG_LEVEL": os.environ.get('RUNPOD_DEBUG_LEVEL', 'Not set')
        }
    }

def echo_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Simple echo handler for testing."""
    job_input = job.get('input', {})
    
    logger.info(f"Echo handler received: {job_input}")
    
    return {
        "echo": job_input,
        "message": "Echo successful",
        "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown'),
        "timestamp": str(torch.cuda.current_device()) if torch.cuda.is_available() else "no-gpu"
    }

def simple_image_handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Simple image processing handler."""
    job_input = job.get('input', {})
    
    if not job_input.get('image_base64'):
        return {"error": "image_base64 is required"}
    
    try:
        # Decode image
        image_data = base64.b64decode(job_input['image_base64'])
        logger.info(f"Decoded image size: {len(image_data)} bytes")
        
        # Simple processing - just return image info
        return {
            "message": "Image processed successfully",
            "image_size_bytes": len(image_data),
            "processing_device": "gpu" if torch.cuda.is_available() else "cpu",
            "echo_back": job_input.get('echo_text', 'No echo text provided')
        }
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return {"error": f"Image processing failed: {str(e)}"}

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler that routes requests."""
    try:
        job_input = job.get('input', {})
        
        # Route based on input type
        if job_input.get('health_check') or job_input.get('type') == 'health':
            return health_check(job)
        elif job_input.get('type') == 'echo':
            return echo_handler(job)
        elif job_input.get('image_base64'):
            return simple_image_handler(job)
        else:
            # Default behavior
            return {
                "message": "Minimal MultiTalk handler",
                "available_endpoints": [
                    "health_check: {\"input\": {\"health_check\": true}}",
                    "echo: {\"input\": {\"type\": \"echo\", \"message\": \"test\"}}",
                    "image: {\"input\": {\"image_base64\": \"...\", \"echo_text\": \"...\"}}"
                ],
                "input_received": job_input
            }
            
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": f"Handler failed: {str(e)}"}

# Start the RunPod serverless worker
if __name__ == "__main__":
    logger.info("Starting minimal MultiTalk handler...")
    runpod.serverless.start({"handler": handler})