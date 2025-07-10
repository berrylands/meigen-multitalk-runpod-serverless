#!/usr/bin/env python3
"""
Ultra-minimal test handler for RunPod
"""

import os
import sys
import json
import base64
import logging
from typing import Dict, Any

import runpod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """Ultra-simple test handler."""
    try:
        job_input = job.get('input', {})
        
        logger.info(f"Received job: {job_input}")
        
        # Health check
        if job_input.get('health_check'):
            return {
                "status": "healthy",
                "message": "Ultra-minimal test handler is working!",
                "python_version": sys.version,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown'),
                "environment": {
                    "MODEL_PATH": os.environ.get('MODEL_PATH', 'Not set'),
                    "RUNPOD_DEBUG_LEVEL": os.environ.get('RUNPOD_DEBUG_LEVEL', 'Not set')
                }
            }
        
        # Echo test
        return {
            "message": "Test handler successful!",
            "echo": job_input,
            "server_info": {
                "python_version": sys.version,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown')
            }
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {"error": f"Handler failed: {str(e)}"}

# Start the RunPod serverless worker
if __name__ == "__main__":
    logger.info("Starting ultra-minimal test handler...")
    runpod.serverless.start({"handler": handler})