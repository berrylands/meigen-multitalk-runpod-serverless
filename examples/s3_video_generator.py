#!/usr/bin/env python3
"""
S3-Enabled MultiTalk Video Generator
Generate videos using S3 for input/output with the MultiTalk serverless endpoint.
"""

import os
import sys
import time
import runpod
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

# Configuration
ENDPOINT_ID = "kkx3cfy484jszl"

# S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME")


def upload_to_s3(file_path, s3_key):
    """Upload a local file to S3."""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    print(f"Uploading {file_path} to s3://{S3_BUCKET}/{s3_key}")
    s3_client.upload_file(file_path, S3_BUCKET, s3_key)
    return f"s3://{S3_BUCKET}/{s3_key}"


def download_from_s3(s3_url, local_path):
    """Download a file from S3."""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION
    )
    
    # Parse S3 URL
    if s3_url.startswith("s3://"):
        parts = s3_url[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1]
    else:
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    
    print(f"Downloading s3://{bucket}/{key} to {local_path}")
    s3_client.download_file(bucket, key, local_path)


def generate_video_s3(
    audio_s3_url=None,
    audio_file=None,
    reference_image_s3_url=None,
    output_s3_key=None,
    use_s3_output=True,
    duration=5.0,
    fps=30,
    width=512,
    height=512
):
    """Generate video using S3 for inputs and optionally outputs."""
    
    print("MultiTalk S3 Video Generator")
    print("=" * 50)
    
    # Prepare audio input
    if audio_file and not audio_s3_url:
        # Upload local audio to S3
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        audio_s3_key = f"multitalk/input/audio_{timestamp}.wav"
        audio_s3_url = upload_to_s3(audio_file, audio_s3_key)
    
    if not audio_s3_url:
        print("Error: No audio input provided")
        return None
    
    print(f"\nAudio input: {audio_s3_url}")
    if reference_image_s3_url:
        print(f"Reference image: {reference_image_s3_url}")
    
    # Prepare job input
    job_input = {
        "action": "generate",
        "audio": audio_s3_url,
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height
    }
    
    if reference_image_s3_url:
        job_input["reference_image"] = reference_image_s3_url
    
    if use_s3_output:
        job_input["output_format"] = "s3"
        if output_s3_key:
            job_input["s3_output_key"] = output_s3_key
    
    # Connect to endpoint
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    print(f"\nSubmitting job to MultiTalk endpoint...")
    print(f"Output format: {'S3' if use_s3_output else 'Base64'}")
    
    try:
        # Submit job
        job = endpoint.run(job_input)
        print(f"Job ID: {job.job_id}")
        
        # Monitor progress
        start_time = time.time()
        last_status = None
        
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            elapsed = time.time() - start_time
            status = job.status()
            
            if status != last_status:
                print(f"[{elapsed:.1f}s] Status: {status}")
                last_status = status
            
            time.sleep(2)
            
            if elapsed > 300:  # 5 minute timeout
                print("Timeout! Job taking too long.")
                return None
        
        # Get result
        final_time = time.time() - start_time
        print(f"\nJob completed in {final_time:.1f} seconds")
        print(f"Final status: {job.status()}")
        
        if job.status() == "COMPLETED":
            result = job.output()
            
            if result and result.get("success"):
                print("\n✅ Video generated successfully!")
                
                output_type = result.get("output_type", "base64")
                
                if output_type == "s3_url":
                    video_s3_url = result.get("video")
                    s3_key = result.get("s3_key")
                    s3_bucket = result.get("s3_bucket")
                    
                    print(f"\n📦 S3 Output:")
                    print(f"   URL: {video_s3_url}")
                    print(f"   Bucket: {s3_bucket}")
                    print(f"   Key: {s3_key}")
                    
                    # Optionally download the video
                    download_choice = input("\nDownload video locally? (y/n): ")
                    if download_choice.lower() == 'y':
                        local_path = f"output_{int(time.time())}.mp4"
                        download_from_s3(video_s3_url, local_path)
                        print(f"Video downloaded: {local_path}")
                    
                    return video_s3_url
                    
                else:
                    # Base64 output
                    import base64
                    video_b64 = result.get("video")
                    if video_b64:
                        video_data = base64.b64decode(video_b64)
                        output_file = f"output_{int(time.time())}.mp4"
                        
                        with open(output_file, "wb") as f:
                            f.write(video_data)
                        
                        print(f"✅ Video saved: {output_file}")
                        print(f"   Size: {len(video_data):,} bytes")
                        return output_file
                
                # Show additional info
                if "processing_time" in result:
                    print(f"\n⏱️  Processing time: {result['processing_time']}")
                if "models_used" in result:
                    print(f"🤖 Models used: {', '.join(result['models_used'])}")
                if "parameters" in result:
                    params = result["parameters"]
                    print(f"📊 Parameters:")
                    print(f"   Resolution: {params.get('resolution')}")
                    print(f"   FPS: {params.get('fps')}")
                    print(f"   Duration: {params.get('duration')}s")
                    
            else:
                print(f"❌ Video generation failed: {result.get('error', 'Unknown error')}")
                return None
        else:
            print(f"❌ Job failed: {job.output()}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function with examples."""
    print("MultiTalk S3 Video Generator")
    print("=" * 50)
    
    if not AWS_ACCESS_KEY_ID or not S3_BUCKET:
        print("❌ S3 credentials not configured!")
        print("Please set these environment variables:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY") 
        print("  - AWS_S3_BUCKET_NAME")
        return
    
    print(f"✅ S3 configured (bucket: {S3_BUCKET})")
    
    # Example usage modes
    if len(sys.argv) > 1:
        # Mode 1: Local audio file -> S3 video
        if sys.argv[1].endswith('.wav'):
            audio_file = sys.argv[1]
            print(f"\nMode: Local audio -> S3 video")
            print(f"Audio file: {audio_file}")
            
            generate_video_s3(
                audio_file=audio_file,
                use_s3_output=True
            )
        
        # Mode 2: S3 URL provided
        elif sys.argv[1].startswith('s3://'):
            audio_s3_url = sys.argv[1]
            print(f"\nMode: S3 audio -> S3 video")
            print(f"Audio S3 URL: {audio_s3_url}")
            
            # Optional reference image
            reference_image = sys.argv[2] if len(sys.argv) > 2 else None
            
            generate_video_s3(
                audio_s3_url=audio_s3_url,
                reference_image_s3_url=reference_image,
                use_s3_output=True
            )
    
    else:
        # Interactive mode
        print("\nUsage examples:")
        print("1. Local audio to S3:")
        print("   python s3_video_generator.py audio.wav")
        print("\n2. S3 audio to S3:")
        print("   python s3_video_generator.py s3://bucket/audio.wav")
        print("\n3. S3 audio + reference image:")
        print("   python s3_video_generator.py s3://bucket/audio.wav s3://bucket/face.jpg")
        
        # Demo with test audio
        print("\n" + "-" * 50)
        demo_choice = input("Run demo with test audio? (y/n): ")
        
        if demo_choice.lower() == 'y':
            # Create test audio
            import numpy as np
            import wave
            
            print("\nCreating test audio...")
            sample_rate = 16000
            duration = 3.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = np.sin(2 * np.pi * 440 * t) * 0.5
            audio_int16 = (audio * 32767).astype(np.int16)
            
            test_file = "test_audio_s3.wav"
            with wave.open(test_file, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(audio_int16.tobytes())
            
            print(f"Created test audio: {test_file}")
            
            # Generate video
            generate_video_s3(
                audio_file=test_file,
                use_s3_output=True,
                duration=duration
            )
            
            # Cleanup
            os.remove(test_file)


if __name__ == "__main__":
    main()