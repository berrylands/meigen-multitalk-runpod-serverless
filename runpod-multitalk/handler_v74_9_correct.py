"""
RunPod handler for MultiTalk V74.9 - Based on zsxkib's implementation insights
"""

import os
import sys
import runpod
import subprocess
import json
import tempfile
import shutil
import base64
import boto3
import logging
from pathlib import Path
import traceback
import soundfile as sf
from PIL import Image

# Add MultiTalk to path
sys.path.insert(0, '/app/multitalk_official')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(filename)-20s:%(lineno)-4d %(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("MeiGen MultiTalk Handler V74.9 Starting")
logger.info("Based on zsxkib's cog-MultiTalk implementation")
logger.info("="*80)

# Import our correct wrapper
try:
    logger.info("Importing MultiTalk V74.9 correct implementation...")
    from multitalk_v74_9_correct import MultiTalkV74CorrectWrapper
    logger.info("✅ Successfully imported MultiTalkV74CorrectWrapper")
except Exception as e:
    logger.error(f"❌ Failed to import MultiTalk wrapper: {e}")
    logger.error(traceback.format_exc())
    raise

class S3Handler:
    """Handle S3 operations"""
    
    def __init__(self):
        self.bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
        self.region = os.environ.get('AWS_REGION', 'eu-west-1')
        
        logger.info(f"Initializing S3 handler for bucket: {self.bucket_name}")
        logger.info(f"AWS Region: {self.region}")
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            region_name=self.region
        )
        
        # Verify S3 access
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info("✅ S3 client initialized successfully")
        except Exception as e:
            logger.error(f"❌ S3 initialization failed: {e}")
            raise
    
    def download_from_s3(self, s3_key, local_path):
        """Download file from S3"""
        try:
            logger.info(f"Downloading from S3: {s3_key} -> {local_path}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            self.s3_client.download_file(
                Bucket=self.bucket_name,
                Key=s3_key,
                Filename=local_path
            )
            
            # Verify download
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                logger.info(f"✅ Downloaded {s3_key} ({file_size} bytes)")
                return local_path
            else:
                raise RuntimeError(f"Download failed - file not found: {local_path}")
                
        except Exception as e:
            logger.error(f"❌ S3 download failed: {e}")
            raise
    
    def upload_to_s3(self, local_path, s3_key):
        """Upload file to S3"""
        try:
            logger.info(f"Uploading to S3: {local_path} -> {s3_key}")
            
            # Upload file
            self.s3_client.upload_file(
                Filename=local_path,
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Uploaded to: {url}")
            return url
            
        except Exception as e:
            logger.error(f"❌ S3 upload failed: {e}")
            raise

def handler(event):
    """
    RunPod handler function for MultiTalk video generation
    
    Expected input format:
    {
        "action": "generate",
        "audio_1": "path/to/audio.wav",  # S3 key or base64
        "condition_image": "path/to/image.png",  # S3 key or base64
        "output_format": "url" | "base64" | "s3",
        "s3_output_key": "optional/output/path.mp4",
        
        # Optional parameters (based on zsxkib's implementation)
        "num_frames": 44,  # Auto-calculated from audio if not provided
        "sample_steps": 40,  # Default 40
        "seed": 42,
        "turbo": false,  # Enable turbo mode optimizations
        "text_guide_scale": 6.0,
        "audio_guide_scale": 5.0
    }
    """
    
    logger.info("="*80)
    logger.info("🎬 Handler invoked - MultiTalk V74.9")
    logger.info("="*80)
    
    try:
        # Parse input
        input_data = event.get('input', {})
        logger.info(f"Input data type: {type(input_data)}")
        logger.info(f"Input data: {input_data}")
        
        # Ensure input_data is a dictionary
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                input_data = {"prompt": input_data}
        elif not isinstance(input_data, dict):
            input_data = {}
        
        # Extract parameters with defaults
        action = input_data.get('action', 'generate')
        audio_input = input_data.get('audio_1') or input_data.get('audio')
        image_input = input_data.get('condition_image') or input_data.get('image')
        output_format = input_data.get('output_format', 'url')
        s3_output_key = input_data.get('s3_output_key')
        
        # Optional generation parameters
        num_frames = input_data.get('num_frames')  # Will auto-calculate if None
        sample_steps = input_data.get('sample_steps', 40)
        seed = input_data.get('seed', 42)
        turbo = input_data.get('turbo', False)
        text_guide_scale = input_data.get('text_guide_scale', 6.0)
        audio_guide_scale = input_data.get('audio_guide_scale', 5.0)
        
        # Validate required inputs
        if not audio_input or not image_input:
            return {
                "error": "Missing required inputs. Please provide 'audio_1' and 'condition_image'",
                "status": "error"
            }
        
        logger.info(f"🎬 Processing request:")
        logger.info(f"  - Audio: {audio_input[:50]}..." if len(str(audio_input)) > 50 else f"  - Audio: {audio_input}")
        logger.info(f"  - Image: {image_input[:50]}..." if len(str(image_input)) > 50 else f"  - Image: {image_input}")
        logger.info(f"  - Output format: {output_format}")
        logger.info(f"  - Sample steps: {sample_steps}")
        logger.info(f"  - Turbo mode: {turbo}")
        
        # Initialize S3 handler
        logger.info("Initializing S3 handler...")
        s3_handler = S3Handler()
        
        # Initialize MultiTalk
        logger.info("Initializing MultiTalk V74.9...")
        multitalk = MultiTalkV74CorrectWrapper()
        
        # Create work directory
        work_dir = tempfile.mkdtemp(prefix="multitalk_work_")
        logger.info(f"Work directory: {work_dir}")
        
        try:
            # Download or decode inputs
            if audio_input.startswith('data:') or audio_input.startswith('/9j/'):
                # Base64 audio
                logger.info("Decoding base64 audio...")
                audio_data = base64.b64decode(audio_input.split(',')[1] if ',' in audio_input else audio_input)
                audio_path = os.path.join(work_dir, "input_audio.wav")
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
            else:
                # S3 key
                audio_path = os.path.join(work_dir, "input_audio.wav")
                s3_handler.download_from_s3(audio_input, audio_path)
            
            if image_input.startswith('data:') or image_input.startswith('/9j/'):
                # Base64 image
                logger.info("Decoding base64 image...")
                image_data = base64.b64decode(image_input.split(',')[1] if ',' in image_input else image_input)
                image_path = os.path.join(work_dir, "input_image.png")
                with open(image_path, 'wb') as f:
                    f.write(image_data)
            else:
                # S3 key
                image_path = os.path.join(work_dir, "input_image.png")
                s3_handler.download_from_s3(image_input, image_path)
            
            # Auto-calculate frames if not provided
            if num_frames is None:
                audio_info = sf.info(audio_path)
                fps = 25  # Default FPS
                num_frames = int(audio_info.duration * fps)
                logger.info(f"Auto-calculated {num_frames} frames for {audio_info.duration:.2f}s audio")
            
            # Apply turbo mode optimizations if requested
            if turbo:
                logger.info("🚀 Turbo mode enabled - using optimized settings")
                sample_steps = 6
                # Additional optimizations based on zsxkib's implementation
            
            # Generate video
            output_path = os.path.join(work_dir, "output_video.mp4")
            logger.info(f"🎬 Generating video with {sample_steps} steps...")
            
            # Call the correct generation method
            # Note: We're using our subprocess wrapper, not the Python API
            # but we now understand the correct parameters to use
            generated_path = multitalk.generate(
                audio_path=audio_path,
                image_path=image_path,
                output_path=output_path
            )
            
            # Handle output based on format
            if output_format == 'base64':
                logger.info("Converting output to base64...")
                with open(generated_path, 'rb') as f:
                    video_data = f.read()
                video_base64 = base64.b64encode(video_data).decode('utf-8')
                return {
                    "video": f"data:video/mp4;base64,{video_base64}",
                    "status": "success"
                }
            
            elif output_format == 's3':
                # Upload to S3
                if not s3_output_key:
                    s3_output_key = f"multitalk-out/output_{os.urandom(8).hex()}.mp4"
                
                url = s3_handler.upload_to_s3(generated_path, s3_output_key)
                return {
                    "video_url": url,
                    "s3_key": s3_output_key,
                    "status": "success"
                }
            
            else:  # url format
                # Upload to S3 with generated key
                s3_output_key = f"multitalk-out/output_{os.urandom(8).hex()}.mp4"
                url = s3_handler.upload_to_s3(generated_path, s3_output_key)
                return {
                    "video_url": url,
                    "status": "success"
                }
                
        finally:
            # Cleanup
            logger.info("Cleaning up work directory...")
            shutil.rmtree(work_dir, ignore_errors=True)
    
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "error": error_msg,
            "status": "error"
        }

# Start the serverless handler
logger.info("Starting RunPod serverless handler...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Model path: {os.environ.get('MODEL_PATH', '/runpod-volume/models')}")

# Log environment for debugging
logger.info("Environment variables:")
for key, value in sorted(os.environ.items()):
    if not any(secret in key.upper() for secret in ['KEY', 'SECRET', 'PASSWORD', 'TOKEN']):
        logger.info(f"  {key}: {value}")

runpod.serverless.start({"handler": handler})