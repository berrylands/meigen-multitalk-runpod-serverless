import runpod
import os
import sys
import json
import time
import base64
import tempfile
import traceback
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

# Enhanced S3 debugging
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] Starting S3 debug handler", flush=True)

# Check environment variables before any imports
env_debug = {
    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", "NOT_SET"),
    "AWS_SECRET_ACCESS_KEY": "SET" if os.environ.get("AWS_SECRET_ACCESS_KEY") else "NOT_SET",
    "AWS_REGION": os.environ.get("AWS_REGION", "NOT_SET"),
    "AWS_S3_BUCKET_NAME": os.environ.get("AWS_S3_BUCKET_NAME", "NOT_SET"),
    "BUCKET_ENDPOINT_URL": os.environ.get("BUCKET_ENDPOINT_URL", "NOT_SET"),
}

# Check for empty strings
for key, value in env_debug.items():
    if key != "AWS_SECRET_ACCESS_KEY" and value == "":
        env_debug[f"{key}_IS_EMPTY"] = True

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] Environment check: {json.dumps(env_debug, indent=2)}", flush=True)

# Try importing S3 handler with detailed error catching
S3_AVAILABLE = False
s3_handler = None
s3_import_error = None

try:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] Attempting to import s3_handler...", flush=True)
    from s3_handler import S3Handler
    
    # Create instance with debug logging
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] Creating S3Handler instance...", flush=True)
    s3_handler = S3Handler()
    
    # Import the convenience functions
    from s3_handler import download_input, prepare_output
    
    S3_AVAILABLE = True
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO] S3 handler imported successfully", flush=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] S3 enabled: {s3_handler.enabled}", flush=True)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] S3 bucket: {s3_handler.default_bucket}", flush=True)
    
except ImportError as e:
    s3_import_error = f"ImportError: {e}"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] S3 handler import failed: {e}", flush=True)
except Exception as e:
    s3_import_error = f"Exception: {e}"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR] Unexpected error importing S3 handler: {e}", flush=True)
    traceback.print_exc()

# Model paths
MODEL_BASE = Path(os.environ.get("MODEL_PATH", "/runpod-volume/models"))
MODELS = {
    "multitalk": MODEL_BASE / "meigen-multitalk",
    "wan21": MODEL_BASE / "wan2.1-i2v-14b-480p", 
    "chinese_wav2vec": MODEL_BASE / "chinese-wav2vec2-base",
    "kokoro": MODEL_BASE / "kokoro-82m",
    "wav2vec_base": MODEL_BASE / "wav2vec2-base-960h",
    "wav2vec_large": MODEL_BASE / "wav2vec2-large-960h",
    "gfpgan": MODEL_BASE / "gfpgan"
}

def log_message(message: str, level: str = "INFO"):
    """Enhanced logging with timestamps."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)

def handler(job):
    """Debug handler that provides detailed S3 diagnostics"""
    log_message(f"Debug handler received job: {job.get('id', 'unknown')}", "DEBUG")
    
    job_input = job.get("input", {})
    action = job_input.get("action", "health")
    
    if action == "health":
        # Comprehensive health check
        health_status = {
            "handler": "s3_debug",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "s3_available": S3_AVAILABLE,
            "s3_import_error": s3_import_error,
            "environment": {
                "AWS_ACCESS_KEY_ID": "SET" if os.environ.get("AWS_ACCESS_KEY_ID") else "NOT_SET",
                "AWS_SECRET_ACCESS_KEY": "SET" if os.environ.get("AWS_SECRET_ACCESS_KEY") else "NOT_SET",
                "AWS_REGION": os.environ.get("AWS_REGION", "NOT_SET"),
                "AWS_S3_BUCKET_NAME": os.environ.get("AWS_S3_BUCKET_NAME", "NOT_SET"),
                "BUCKET_ENDPOINT_URL": os.environ.get("BUCKET_ENDPOINT_URL", "NOT_SET"),
            }
        }
        
        # Check for empty values
        for key in ["AWS_ACCESS_KEY_ID", "AWS_REGION", "AWS_S3_BUCKET_NAME", "BUCKET_ENDPOINT_URL"]:
            val = os.environ.get(key, "")
            if val == "":
                health_status["environment"][f"{key}_IS_EMPTY_STRING"] = True
        
        # Check S3 handler status
        if S3_AVAILABLE and s3_handler:
            health_status["s3_handler"] = {
                "enabled": s3_handler.enabled,
                "default_bucket": s3_handler.default_bucket,
                "has_client": s3_handler.s3_client is not None
            }
            
            # Try to check S3 access
            if s3_handler.enabled:
                try:
                    s3_status = s3_handler.check_s3_access()
                    health_status["s3_access"] = s3_status
                except Exception as e:
                    health_status["s3_access_error"] = str(e)
        
        return health_status
    
    elif action == "test_s3":
        # Test S3 functionality
        if not S3_AVAILABLE:
            return {"error": "S3 handler not available", "import_error": s3_import_error}
        
        if not s3_handler or not s3_handler.enabled:
            return {
                "error": "S3 not enabled",
                "s3_available": S3_AVAILABLE,
                "s3_enabled": s3_handler.enabled if s3_handler else False,
                "environment": {
                    "AWS_ACCESS_KEY_ID": "SET" if os.environ.get("AWS_ACCESS_KEY_ID") else "NOT_SET",
                    "AWS_SECRET_ACCESS_KEY": "SET" if os.environ.get("AWS_SECRET_ACCESS_KEY") else "NOT_SET",
                }
            }
        
        # Try to test S3
        test_url = job_input.get("test_url", "s3://test-bucket/test.txt")
        
        result = {
            "test_url": test_url,
            "is_s3_url": s3_handler.is_s3_url(test_url),
            "s3_enabled": s3_handler.enabled,
            "default_bucket": s3_handler.default_bucket
        }
        
        if s3_handler.is_s3_url(test_url):
            try:
                # Don't actually download, just parse
                bucket, key = s3_handler.parse_s3_url(test_url)
                result["parsed"] = {"bucket": bucket, "key": key}
            except Exception as e:
                result["parse_error"] = str(e)
        
        return result
    
    elif action == "debug_env":
        # Return raw environment info for debugging
        return {
            "raw_env": {
                k: v if "SECRET" not in k else "REDACTED"
                for k, v in os.environ.items()
                if k.startswith("AWS") or k.startswith("BUCKET")
            },
            "s3_import_status": {
                "available": S3_AVAILABLE,
                "error": s3_import_error,
                "handler_exists": s3_handler is not None,
                "handler_enabled": s3_handler.enabled if s3_handler else False
            }
        }
    
    else:
        # Fallback
        return {
            "message": "Debug handler - use action=health, test_s3, or debug_env",
            "s3_available": S3_AVAILABLE
        }

if __name__ == "__main__":
    log_message("S3 Debug Handler Starting...", "INFO")
    log_message("=" * 60, "INFO")
    
    # Run initialization debug
    log_message(f"Python: {sys.version}", "DEBUG")
    log_message(f"Working directory: {os.getcwd()}", "DEBUG")
    log_message(f"S3 Available: {S3_AVAILABLE}", "INFO")
    log_message(f"S3 Import Error: {s3_import_error}", "INFO" if not s3_import_error else "ERROR")
    
    if S3_AVAILABLE and s3_handler:
        log_message(f"S3 Enabled: {s3_handler.enabled}", "INFO")
        log_message(f"S3 Bucket: {s3_handler.default_bucket}", "INFO")
    
    log_message("=" * 60, "INFO")
    
    # Start RunPod handler
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        log_message(f"Failed to start RunPod handler: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1)