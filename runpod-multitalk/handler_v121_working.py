#!/usr/bin/env python3
"""
RunPod Serverless Handler for MeiGen MultiTalk V121
Working Implementation based on proven cog-MultiTalk code
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
logger.info("MeiGen MultiTalk Handler V121 Working Implementation")
logger.info("Based on proven cog-MultiTalk code")
logger.info("=" * 80)

# Import S3 handler
try:
    from s3_handler import S3Handler
    logger.info("✓ S3 handler imported successfully")
except ImportError as e:
    logger.error(f"❌ S3 handler import failed: {e}")
    
    # Simple S3 handler fallback
    class S3Handler:
        def __init__(self):
            self.bucket_name = "760572149-framepack"
            self.region = "eu-west-1"
            logger.info("Using fallback S3 handler")
        
        def upload_to_s3(self, local_path: str, s3_key: str) -> str:
            return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"

# Import MultiTalk implementation
try:
    logger.info("Importing Working MultiTalk V121 implementation...")
    from multitalk_v121_working_implementation import MultiTalkV121Working
    logger.info("✓ Successfully imported MultiTalkV121Working")
except Exception as e:
    logger.error(f"❌ Failed to import MultiTalk implementation: {str(e)}")
    logger.error(traceback.format_exc())
    
    # Fallback to V115 implementation
    try:
        from multitalk_v115_implementation import MultiTalkV115 as MultiTalkV121Working
        logger.info("✓ Using V115 implementation as fallback")
    except Exception as e2:
        logger.error(f"❌ Fallback import also failed: {str(e2)}")
        sys.exit(1)

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler function.
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("Handler invoked - Working MultiTalk V121")
    logger.info("=" * 80)
    
    try:
        # Extract input
        input_data = event.get('input', {})
        
        # Log input structure
        logger.info(f"Input data type: {type(input_data)}")
        logger.info(f"Input data keys: {list(input_data.keys()) if isinstance(input_data, dict) else 'Not a dict'}")
        
        # Handle different input formats
        audio_key = audio_data = image_key = image_data = None
        
        # Check for S3 keys or direct data
        if isinstance(input_data, dict):
            audio_key = input_data.get('audio_s3_key') or input_data.get('audio_key')
            image_key = input_data.get('image_s3_key') or input_data.get('image_key')
            audio_data = input_data.get('audio_data')
            image_data = input_data.get('image_data')
        
        # Default to test files if no input provided
        if not any([audio_key, audio_data, image_key, image_data]):
            logger.info("No input provided, using default test files")
            audio_key = "1.wav"
            image_key = "multi1.png"
        
        output_format = input_data.get('output_format', 'url') if isinstance(input_data, dict) else 'url'
        
        # Generation parameters
        prompt = input_data.get('prompt', 'A person talking naturally with expressive facial movements') if isinstance(input_data, dict) else 'A person talking naturally with expressive facial movements'
        num_frames = input_data.get('num_frames', 81) if isinstance(input_data, dict) else 81
        sampling_steps = input_data.get('sampling_steps', 30) if isinstance(input_data, dict) else 30
        seed = input_data.get('seed') if isinstance(input_data, dict) else None
        turbo = input_data.get('turbo', True) if isinstance(input_data, dict) else True
        
        logger.info(f"Generation parameters:")
        logger.info(f"  - Prompt: {prompt}")
        logger.info(f"  - Frames: {num_frames}")
        logger.info(f"  - Steps: {sampling_steps}")
        logger.info(f"  - Seed: {seed}")
        logger.info(f"  - Turbo: {turbo}")
        logger.info(f"  - Output format: {output_format}")
        
        # Initialize components
        logger.info("Initializing S3 handler...")
        s3_handler = S3Handler()
        
        logger.info("Initializing Working MultiTalk V121...")
        multitalk = MultiTalkV121Working()
        
        # Load models
        logger.info("Loading models...")
        if not multitalk.load_models():
            raise RuntimeError("Failed to load MultiTalk models")
        
        # Show model info
        model_info = multitalk.get_model_info()
        logger.info(f"Model info: {json.dumps(model_info, indent=2)}")
        
        # Prepare working directory
        work_dir = Path("/tmp/multitalk_work")
        work_dir.mkdir(exist_ok=True)
        
        # Get audio file
        if audio_key:
            audio_path = str(work_dir / "input_audio.wav")
            s3_handler.download_from_s3(audio_key, audio_path)
        elif audio_data:
            audio_path = str(work_dir / "input_audio.wav")
            audio_bytes = base64.b64decode(audio_data)
            with open(audio_path, 'wb') as f:
                f.write(audio_bytes)
            logger.info(f"✓ Decoded audio data ({len(audio_bytes)} bytes)")
        else:
            raise ValueError("No audio input provided")
        
        # Get image file
        if image_key:
            image_path = str(work_dir / "input_image.png")
            s3_handler.download_from_s3(image_key, image_path)
        elif image_data:
            image_path = str(work_dir / "input_image.png")
            image_bytes = base64.b64decode(image_data)
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            logger.info(f"✓ Decoded image data ({len(image_bytes)} bytes)")
        else:
            raise ValueError("No image input provided")
        
        # Generate video using working implementation
        logger.info("Generating video with Working MultiTalk V121...")
        
        result = multitalk.generate_video(
            audio_path=audio_path,
            image_path=image_path,
            prompt=prompt,
            num_frames=num_frames,
            sampling_steps=sampling_steps,
            seed=seed,
            turbo=turbo
        )
        
        if not result.get('success'):
            raise RuntimeError(f"Video generation failed: {result.get('error', 'Unknown error')}")
        
        output_path = result['video_path']
        
        if not os.path.exists(output_path):
            raise RuntimeError("Video file not created")
        
        logger.info(f"✓ Video generated: {output_path} ({os.path.getsize(output_path)} bytes)")
        
        # Return result based on format
        if output_format == 'base64':
            with open(output_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode('utf-8')
            response = {
                'video_data': video_data,
                'format': 'mp4',
                'success': True,
                'generation_info': result
            }
        else:
            # Upload to S3
            timestamp = int(time.time() * 1000)
            s3_key = f"multitalk_v121_output_{timestamp}.mp4"
            video_url = s3_handler.upload_to_s3(output_path, s3_key)
            response = {
                'video_url': video_url,
                'format': 'mp4',
                'success': True,
                's3_key': s3_key,
                'generation_info': result
            }
        
        # Clean up
        shutil.rmtree(work_dir, ignore_errors=True)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Request completed in {elapsed:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Handler error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up on error
        if 'work_dir' in locals():
            shutil.rmtree(work_dir, ignore_errors=True)
        
        return {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }

# RunPod serverless entrypoint
if __name__ == "__main__":
    # Display startup info
    logger.info("Starting Working MultiTalk V121 Handler...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Model path: {os.environ.get('MODEL_PATH', '/runpod-volume/models')}")
    
    # Volume info
    volume_path = Path("/runpod-volume")
    if volume_path.exists():
        logger.info(f"Volume mounted: True")
        models_path = volume_path / "models"
        if models_path.exists():
            logger.info(f"Models directory exists: {models_path}")
            # List model directories
            for item in models_path.iterdir():
                if item.is_dir():
                    logger.info(f"  - {item.name}")
        else:
            logger.info("Models directory not found")
    else:
        logger.info("Volume not mounted")
    
    logger.info("=" * 80)
    
    runpod.serverless.start({
        "handler": handler
    })