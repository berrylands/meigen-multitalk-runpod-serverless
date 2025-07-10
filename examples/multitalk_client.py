#!/usr/bin/env python3
"""
Client example for MultiTalk RunPod serverless endpoint
"""

import os
import base64
import time
import runpod
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "kkx3cfy484jszl"  # Your endpoint ID

def encode_file(file_path):
    """Encode a file to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def save_video(video_b64, output_path):
    """Save base64 video to file."""
    video_data = base64.b64decode(video_b64)
    with open(output_path, "wb") as f:
        f.write(video_data)

def generate_talking_video(
    audio_path: str,
    reference_image_path: str = None,
    output_path: str = "output.mp4",
    duration: float = None,
    fps: int = 30
):
    """Generate a talking head video from audio."""
    
    print("MultiTalk Video Generation")
    print("=" * 60)
    
    # Initialize RunPod client
    runpod.api_key = RUNPOD_API_KEY
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Check endpoint health
    print("Checking endpoint health...")
    health = endpoint.health()
    print(f"Workers ready: {health.get('workers', {}).get('ready', 0)}")
    
    # Prepare input
    job_input = {
        "action": "generate",
        "audio": encode_file(audio_path),
        "fps": fps
    }
    
    if reference_image_path:
        job_input["reference_image"] = encode_file(reference_image_path)
    
    if duration:
        job_input["duration"] = duration
    
    # Submit job
    print(f"\nSubmitting video generation job...")
    print(f"  Audio: {audio_path}")
    if reference_image_path:
        print(f"  Reference image: {reference_image_path}")
    print(f"  FPS: {fps}")
    
    start_time = time.time()
    job = endpoint.run(job_input)
    print(f"Job ID: {job.job_id}")
    
    # Monitor progress
    print("\nProcessing...")
    last_status = None
    
    while True:
        status = job.status()
        if status != last_status:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Status: {status}")
            last_status = status
        
        if status not in ["IN_QUEUE", "IN_PROGRESS"]:
            break
            
        if time.time() - start_time > 600:  # 10 minute timeout
            print("Timeout waiting for completion")
            return None
            
        time.sleep(5)
    
    # Get result
    if job.status() == "COMPLETED":
        output = job.output()
        
        if output.get("success"):
            print(f"\n✓ Video generated successfully!")
            print(f"  Duration: {output.get('duration')}s")
            print(f"  Frames: {output.get('frames')}")
            print(f"  Processing time: {output.get('processing_time')}")
            
            # Save video
            video_b64 = output.get("video")
            if video_b64:
                save_video(video_b64, output_path)
                print(f"  Saved to: {output_path}")
                return output_path
            else:
                print("  Error: No video data in response")
                return None
        else:
            print(f"\n✗ Generation failed: {output.get('error')}")
            return None
    else:
        print(f"\n✗ Job failed with status: {job.status()}")
        try:
            print(f"Error: {job.output()}")
        except:
            pass
        return None

def create_test_audio(output_path="test_audio.wav", duration=5):
    """Create a test audio file."""
    import subprocess
    
    # Generate test audio with text-to-speech or sine wave
    cmd = [
        "ffmpeg", "-f", "lavfi", 
        "-i", f"sine=frequency=440:duration={duration}",
        "-ar", "16000", "-ac", "1", "-y", output_path
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Created test audio: {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage
    print("MultiTalk Client Example")
    print("=" * 60)
    
    # Create test audio if needed
    audio_file = "test_audio.wav"
    if not os.path.exists(audio_file):
        print("Creating test audio...")
        audio_file = create_test_audio()
    
    # Optional: Use a reference image
    reference_image = None  # Set to path of an image file if available
    
    # Generate video
    output_video = generate_talking_video(
        audio_path=audio_file,
        reference_image_path=reference_image,
        output_path="multitalk_output.mp4",
        duration=5.0,
        fps=30
    )
    
    if output_video:
        print(f"\n✓ Success! Video saved to: {output_video}")
        print(f"File size: {os.path.getsize(output_video) / (1024*1024):.1f} MB")
    else:
        print("\n✗ Failed to generate video")