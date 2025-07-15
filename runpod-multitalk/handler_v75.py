"""
RunPod Handler for MeiGen MultiTalk V75.0
Using correct JSON input format for official MultiTalk
"""
import runpod
import os
import sys
import json
import base64
import time
import logging
from pathlib import Path
import traceback
import tempfile
import requests
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)-20s:%(lineno)-4d %(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Version info
VERSION = "75.0.0"
BUILD_ID = os.environ.get("BUILD_ID", "multitalk-v75-json-input")
IMPLEMENTATION = "JSON_INPUT_FORMAT"

logger.info("="*80)
logger.info("MeiGen MultiTalk Handler V75.0 Starting")
logger.info("Using correct JSON input format for official MultiTalk")
logger.info("="*80)

# Import MultiTalk implementation
try:
    from multitalk_v75_0_json_input import MultiTalkV75JsonWrapper
    logger.info("✅ Successfully imported MultiTalkV75JsonWrapper")
except ImportError as e:
    logger.error(f"Failed to import MultiTalk V75.0: {e}")
    MultiTalkV75JsonWrapper = None

# S3 Handler
class S3Handler:
    """Handle S3 operations for input/output files"""
    
    def __init__(self, bucket_name: str, region: str = "eu-west-1"):
        self.bucket_name = bucket_name
        self.region = region
        
        logger.info(f"Initializing S3 handler for bucket: {bucket_name}")
        logger.info(f"AWS Region: {region}")
        
        # Initialize boto3 client
        try:
            import boto3
            self.s3_client = boto3.client('s3', region_name=region)
            
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info("✅ S3 client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download file from S3"""
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return False
            
        try:
            logger.info(f"Downloading from S3: {s3_key} -> {local_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            # Verify download
            if os.path.exists(local_path):
                size = os.path.getsize(local_path)
                logger.info(f"✅ Downloaded {s3_key} ({size} bytes)")
                return True
            else:
                logger.error(f"Download completed but file not found: {local_path}")
                return False
                
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False
    
    def upload_file(self, local_path: str, s3_key: str) -> Optional[str]:
        """Upload file to S3"""
        if not self.s3_client:
            logger.error("S3 client not initialized")
            return None
            
        try:
            logger.info(f"Uploading to S3: {local_path} -> {s3_key}")
            
            # Get file size
            file_size = os.path.getsize(local_path)
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            # Upload file
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            
            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Uploaded to: {url}")
            return url
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return None


def download_file_from_url(url: str, local_path: str) -> bool:
    """Download file from URL"""
    try:
        logger.info(f"Downloading from URL: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size = os.path.getsize(local_path)
        logger.info(f"✅ Downloaded {size} bytes")
        return True
        
    except Exception as e:
        logger.error(f"URL download failed: {e}")
        return False


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler function for MultiTalk V75.0
    
    Expected input format:
    {
        "action": "generate",
        "audio_1": "path/to/audio.wav" or base64 data,
        "condition_image": "path/to/image.png" or base64 data,
        "prompt": "A person talking naturally",
        "sample_steps": 40,
        "turbo": false,
        "output_format": "s3" or "base64",
        "s3_output_key": "optional/path/for/s3/output.mp4"
    }
    """
    try:
        logger.info("="*80)
        logger.info("🎬 Handler invoked - MultiTalk V75.0 JSON Input")
        logger.info("="*80)
        
        # Extract input
        job_input = job.get("input", {})
        logger.info(f"Input data type: {type(job_input)}")
        logger.info(f"Input data: {job_input}")
        
        # Validate action
        action = job_input.get("action", "generate")
        if action == "status":
            return {
                "status": "ready",
                "version": VERSION,
                "build_id": BUILD_ID,
                "implementation": IMPLEMENTATION,
                "models_loaded": MultiTalkV75JsonWrapper is not None
            }
        
        if action != "generate":
            raise ValueError(f"Unknown action: {action}")
        
        # Validate MultiTalk is available
        if MultiTalkV75JsonWrapper is None:
            raise RuntimeError("MultiTalk V75.0 not properly initialized")
        
        # Extract parameters
        audio_input = job_input.get("audio_1")
        image_input = job_input.get("condition_image")
        prompt = job_input.get("prompt", "A person talking naturally with expressive lip sync")
        sample_steps = job_input.get("sample_steps", 40)
        turbo = job_input.get("turbo", False)
        output_format = job_input.get("output_format", "s3")
        s3_output_key = job_input.get("s3_output_key", None)
        
        if not audio_input or not image_input:
            raise ValueError("Both audio_1 and condition_image are required")
        
        logger.info(f"🎬 Processing request:")
        logger.info(f"  - Audio: {audio_input[:50]}..." if isinstance(audio_input, str) else "  - Audio: binary data")
        logger.info(f"  - Image: {image_input[:50]}..." if isinstance(image_input, str) else "  - Image: binary data")
        logger.info(f"  - Prompt: {prompt}")
        logger.info(f"  - Output format: {output_format}")
        logger.info(f"  - Sample steps: {sample_steps}")
        logger.info(f"  - Turbo mode: {turbo}")
        
        # Initialize S3 handler
        logger.info("Initializing S3 handler...")
        s3_handler = S3Handler(
            bucket_name=os.environ.get("AWS_S3_BUCKET_NAME", "760572149-framepack"),
            region=os.environ.get("AWS_REGION", "eu-west-1")
        )
        
        # Initialize MultiTalk
        logger.info("Initializing MultiTalk V75.0...")
        multitalk = MultiTalkV75JsonWrapper()
        
        # Create work directory
        with tempfile.TemporaryDirectory() as work_dir:
            work_path = Path(work_dir)
            logger.info(f"Work directory: {work_path}")
            
            # Process audio input
            audio_path = work_path / "input_audio.wav"
            if audio_input.startswith("data:") or len(audio_input) > 1000:
                # Base64 data
                logger.info("Decoding base64 audio...")
                if "," in audio_input:
                    audio_data = base64.b64decode(audio_input.split(",")[1])
                else:
                    audio_data = base64.b64decode(audio_input)
                with open(audio_path, "wb") as f:
                    f.write(audio_data)
            elif audio_input.startswith("http"):
                # URL
                download_file_from_url(audio_input, str(audio_path))
            else:
                # S3 path
                s3_handler.download_file(audio_input, str(audio_path))
            
            # Process image input
            image_path = work_path / "input_image.png"
            if image_input.startswith("data:") or len(image_input) > 1000:
                # Base64 data
                logger.info("Decoding base64 image...")
                if "," in image_input:
                    image_data = base64.b64decode(image_input.split(",")[1])
                else:
                    image_data = base64.b64decode(image_input)
                with open(image_path, "wb") as f:
                    f.write(image_data)
            elif image_input.startswith("http"):
                # URL
                download_file_from_url(image_input, str(image_path))
            else:
                # S3 path
                s3_handler.download_file(image_input, str(image_path))
            
            # Verify inputs exist
            if not audio_path.exists():
                raise RuntimeError(f"Audio file not found: {audio_path}")
            if not image_path.exists():
                raise RuntimeError(f"Image file not found: {image_path}")
            
            # Calculate number of frames based on audio duration
            import soundfile as sf
            audio_info = sf.info(str(audio_path))
            fps = 25
            num_frames = int(audio_info.duration * fps)
            logger.info(f"Auto-calculated {num_frames} frames for {audio_info.duration:.2f}s audio")
            
            # Generate video
            output_path = work_path / "output_video.mp4"
            
            # Adjust sample steps for turbo mode
            if turbo:
                sample_steps = min(sample_steps, 20)
                logger.info(f"🚀 Turbo mode: reduced to {sample_steps} steps")
            
            logger.info(f"🎬 Generating video with {sample_steps} steps...")
            start_time = time.time()
            
            # Use the new generate_with_options method for more control
            generated_path = multitalk.generate_with_options(
                audio_path=str(audio_path),
                image_path=str(image_path),
                output_path=str(output_path),
                prompt=prompt,
                sample_steps=sample_steps,
                mode="clip",
                size="multitalk-480",
                use_teacache=True,
                text_guide_scale=7.5,
                audio_guide_scale=3.5,
                seed=42
            )
            
            generation_time = time.time() - start_time
            logger.info(f"✅ Video generated in {generation_time:.2f}s")
            
            # Verify output exists
            if not os.path.exists(generated_path):
                raise RuntimeError(f"Generated video not found: {generated_path}")
            
            # Get video info
            video_size = os.path.getsize(generated_path) / (1024 * 1024)
            logger.info(f"📹 Generated video: {video_size:.2f} MB")
            
            # Handle output format
            if output_format == "base64":
                # Return base64 encoded video
                logger.info("Encoding video to base64...")
                with open(generated_path, "rb") as f:
                    video_data = f.read()
                video_base64 = base64.b64encode(video_data).decode('utf-8')
                
                return {
                    "video_base64": f"data:video/mp4;base64,{video_base64}",
                    "duration": audio_info.duration,
                    "frames": num_frames,
                    "fps": fps,
                    "size_mb": video_size,
                    "generation_time": generation_time,
                    "model": "multitalk-v75-json",
                    "version": VERSION
                }
            else:
                # Upload to S3
                if not s3_output_key:
                    timestamp = int(time.time())
                    s3_output_key = f"multitalk-output/video_{timestamp}.mp4"
                
                video_url = s3_handler.upload_file(generated_path, s3_output_key)
                
                if not video_url:
                    raise RuntimeError("Failed to upload video to S3")
                
                return {
                    "video_url": video_url,
                    "s3_key": s3_output_key,
                    "duration": audio_info.duration,
                    "frames": num_frames,
                    "fps": fps,
                    "size_mb": video_size,
                    "generation_time": generation_time,
                    "model": "multitalk-v75-json",
                    "version": VERSION
                }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "version": VERSION,
            "build_id": BUILD_ID
        }


# RunPod serverless handler
logger.info("Starting RunPod serverless handler...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Model path: {os.environ.get('MODEL_PATH', '/runpod-volume/models')}")

# Log environment info
logger.info("Environment variables:")
for key in sorted(os.environ.keys()):
    if any(k in key for k in ['AWS', 'RUNPOD', 'MODEL', 'CUDA', 'PYTHON', 'VERSION', 'BUILD', 'BUCKET', 'IMPLEMENTATION']):
        logger.info(f"  {key}: {os.environ[key]}")

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})