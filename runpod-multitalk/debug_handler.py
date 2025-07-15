#!/usr/bin/env python3
"""
Debug handler to identify startup issues
"""
import sys
import os
import traceback

def debug_startup():
    """Debug the startup process"""
    print("=" * 60)
    print("MultiTalk Debug Handler Starting...")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check environment variables
    print("\nEnvironment Variables:")
    for key in ['MODEL_PATH', 'AWS_ACCESS_KEY_ID', 'AWS_REGION', 'AWS_S3_BUCKET_NAME', 
                'RUNPOD_WEBHOOK_GET_JOB', 'RUNPOD_AI_API_KEY']:
        value = os.environ.get(key, 'NOT_SET')
        if 'KEY' in key or 'SECRET' in key:
            value = value[:4] + '...' if value != 'NOT_SET' else 'NOT_SET'
        print(f"  {key}: {value}")
    
    # Check if running on RunPod
    print(f"\nRunPod environment: {'Yes' if 'RUNPOD_POD_ID' in os.environ else 'No'}")
    
    # Try importing critical modules
    print("\nChecking imports:")
    
    # Check runpod
    try:
        import runpod
        print("✓ runpod imported successfully")
    except ImportError as e:
        print(f"✗ runpod import failed: {e}")
        return False
    
    # Check boto3
    try:
        import boto3
        print("✓ boto3 imported successfully")
    except ImportError as e:
        print(f"✗ boto3 import failed: {e}")
    
    # Check torch
    try:
        import torch
        print(f"✓ torch imported successfully (version: {torch.__version__})")
        if torch.cuda.is_available():
            print(f"  GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("  No GPU available")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
    
    # Check s3_handler
    try:
        from s3_handler import s3_handler, download_input, prepare_output
        print("✓ s3_handler imported successfully")
        print(f"  S3 enabled: {s3_handler.enabled}")
        if s3_handler.enabled:
            print(f"  Default bucket: {s3_handler.default_bucket}")
    except ImportError as e:
        print(f"✗ s3_handler import failed: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"✗ s3_handler unexpected error: {e}")
        traceback.print_exc()
    
    # Check file system
    print("\nFile system check:")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Check model path
    model_path = os.environ.get("MODEL_PATH", "/runpod-volume/models")
    print(f"\nModel path: {model_path}")
    if os.path.exists(model_path):
        print(f"Model path exists: Yes")
        try:
            models = os.listdir(model_path)
            print(f"Models found: {len(models)}")
            for model in models[:5]:  # Show first 5
                print(f"  - {model}")
        except Exception as e:
            print(f"Error listing models: {e}")
    else:
        print(f"Model path exists: No")
    
    print("\n" + "=" * 60)
    return True

def handler(job):
    """Debug handler for RunPod"""
    print("Debug handler called with job:", job)
    
    # Run debug
    success = debug_startup()
    
    return {
        "status": "completed" if success else "failed",
        "debug_complete": True,
        "message": "See logs for debug information"
    }

if __name__ == "__main__":
    print("Starting debug handler...")
    
    # First run debug
    debug_startup()
    
    # Then try to start runpod
    try:
        import runpod
        print("\nStarting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"\nFailed to start RunPod handler: {e}")
        traceback.print_exc()
        sys.exit(1)