#!/usr/bin/env python3
"""
Generate a demonstration video with MultiTalk to prove end-to-end functionality
"""

import os
import time
import runpod
import base64
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

def generate_speech_audio():
    """Generate a more realistic audio signal that simulates speech patterns."""
    
    print("Creating speech-like audio...")
    
    # Parameters
    sample_rate = 16000
    duration = 5.0  # 5 seconds for better demonstration
    
    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a more complex audio signal that simulates speech
    audio_signal = np.zeros_like(t)
    
    # Add multiple frequency components to simulate formants
    # F1: 700 Hz (first formant)
    audio_signal += 0.3 * np.sin(2 * np.pi * 700 * t)
    
    # F2: 1220 Hz (second formant)  
    audio_signal += 0.2 * np.sin(2 * np.pi * 1220 * t)
    
    # F3: 2600 Hz (third formant)
    audio_signal += 0.1 * np.sin(2 * np.pi * 2600 * t)
    
    # Add fundamental frequency with vibrato (simulating vocal pitch)
    vibrato = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz vibrato
    f0 = 150 + 20 * vibrato  # Fundamental frequency around 150 Hz with vibrato
    audio_signal += 0.4 * np.sin(2 * np.pi * f0 * t)
    
    # Add amplitude modulation to simulate syllables
    syllable_rate = 4  # 4 syllables per second
    amplitude_mod = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t)
    audio_signal *= amplitude_mod
    
    # Add some noise for realism
    audio_signal += 0.02 * np.random.normal(0, 1, len(t))
    
    # Apply envelope (fade in/out)
    fade_samples = int(0.1 * sample_rate)  # 0.1 second fade
    audio_signal[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio_signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # Normalize
    audio_signal = audio_signal / np.max(np.abs(audio_signal)) * 0.8
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_signal * 32767).astype(np.int16)
    
    print(f"   Duration: {duration} seconds")
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Audio shape: {audio_int16.shape}")
    print(f"   Audio range: [{audio_int16.min()}, {audio_int16.max()}]")
    
    return audio_int16, sample_rate, duration

def generate_demo_video():
    """Generate a demonstration video with detailed logging."""
    
    print("\n" + "="*60)
    print("MULTITALK DEMO VIDEO GENERATION")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Endpoint: {ENDPOINT_ID}")
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Generate audio
    audio_data, sample_rate, duration = generate_speech_audio()
    audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
    
    print(f"\n✅ Audio prepared:")
    print(f"   Base64 size: {len(audio_b64)} characters")
    print(f"   Raw size: {len(audio_data) * 2} bytes")
    
    # Prepare job input
    job_input = {
        "action": "generate",
        "audio": audio_b64,
        "duration": duration,
        "fps": 30,  # Higher FPS for smoother video
        "width": 512,  # Slightly higher resolution
        "height": 512
    }
    
    print(f"\n📤 Submitting video generation job:")
    print(f"   Resolution: {job_input['width']}x{job_input['height']}")
    print(f"   FPS: {job_input['fps']}")
    print(f"   Duration: {job_input['duration']}s")
    print(f"   Expected frames: {int(job_input['fps'] * job_input['duration'])}")
    
    try:
        # Submit job
        start_time = time.time()
        job = endpoint.run(job_input)
        print(f"\n🚀 Job submitted: {job.job_id}")
        
        # Monitor progress
        print("\n⏳ Processing:")
        last_status = None
        status_counts = {"IN_QUEUE": 0, "IN_PROGRESS": 0}
        
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            elapsed = time.time() - start_time
            status = job.status()
            
            if status in status_counts:
                status_counts[status] += 1
            
            if status != last_status:
                print(f"   [{elapsed:.1f}s] Status changed: {last_status} → {status}")
                last_status = status
            elif int(elapsed) % 10 == 0:  # Update every 10 seconds
                print(f"   [{elapsed:.1f}s] Still {status}...")
            
            time.sleep(2)
            
            if elapsed > 300:  # 5 minute timeout
                print(f"\n⚠️  Timeout after {elapsed:.1f} seconds")
                break
        
        # Get final result
        final_time = time.time() - start_time
        final_status = job.status()
        
        print(f"\n📊 Job completed in {final_time:.1f} seconds")
        print(f"   Final status: {final_status}")
        print(f"   Time in queue: {status_counts.get('IN_QUEUE', 0) * 2}s")
        print(f"   Time processing: {status_counts.get('IN_PROGRESS', 0) * 2}s")
        
        if final_status == "COMPLETED":
            result = job.output()
            
            if result and result.get("success"):
                print(f"\n✅ VIDEO GENERATION SUCCESSFUL!")
                
                # Extract metadata
                print(f"\n📋 Generation Details:")
                print(f"   Processing time: {result.get('processing_time', 'N/A')}")
                print(f"   Models used: {', '.join(result.get('models_used', []))}")
                
                if 'parameters' in result:
                    params = result['parameters']
                    print(f"\n📐 Video Parameters:")
                    print(f"   Resolution: {params.get('resolution', 'N/A')}")
                    print(f"   Duration: {params.get('duration', 'N/A')}")
                    print(f"   Audio size: {params.get('audio_size', 0):,} bytes")
                    print(f"   Video size: {params.get('video_size', 0):,} bytes")
                
                if 'video_info' in result:
                    info = result['video_info']
                    print(f"\n🎬 Video Information:")
                    print(f"   Resolution: {info.get('resolution', 'N/A')}")
                    print(f"   FPS: {info.get('fps', 'N/A')}")
                    print(f"   Frames: {info.get('frames', 'N/A')}")
                    print(f"   Duration: {info.get('duration', 'N/A')}s")
                    print(f"   Processing note: {info.get('processing_note', 'N/A')}")
                
                # Save the video
                if 'video' in result:
                    video_b64 = result['video']
                    video_data = base64.b64decode(video_b64)
                    
                    # Create filename with timestamp
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_file = f"multitalk_demo_{timestamp}.mp4"
                    
                    with open(output_file, "wb") as f:
                        f.write(video_data)
                    
                    print(f"\n💾 Video saved successfully:")
                    print(f"   Filename: {output_file}")
                    print(f"   Size: {len(video_data):,} bytes ({len(video_data)/1024:.1f} KB)")
                    print(f"   Location: {os.path.abspath(output_file)}")
                    
                    # Verify the video file
                    if os.path.exists(output_file):
                        file_size = os.path.getsize(output_file)
                        print(f"\n✅ Video file verified:")
                        print(f"   File exists: Yes")
                        print(f"   File size matches: {file_size == len(video_data)}")
                        
                        # Get file info
                        import subprocess
                        try:
                            result = subprocess.run(['file', output_file], 
                                                  capture_output=True, text=True)
                            print(f"   File type: {result.stdout.strip()}")
                        except:
                            pass
                    
                    print(f"\n🎉 SUCCESS! MultiTalk has generated a video from audio!")
                    print(f"   The video demonstrates the complete end-to-end pipeline")
                    print(f"   Audio → MultiTalk Processing → Video Output")
                    
                    return True, output_file
                else:
                    print(f"\n❌ No video data in response")
            else:
                print(f"\n❌ Generation failed:")
                print(f"   Success: {result.get('success', False)}")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                if 'details' in result:
                    print(f"   Details: {result['details']}")
        else:
            print(f"\n❌ Job failed with status: {final_status}")
            error = job.output()
            if error:
                print(f"   Error output: {error}")
                
    except Exception as e:
        print(f"\n❌ Exception occurred: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return False, None

def main():
    print("MultiTalk Serverless - Video Generation Demonstration")
    print("This will generate a video from synthetic speech-like audio")
    
    success, video_file = generate_demo_video()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    
    if success and video_file:
        print(f"\n✅ Video generation proven successful!")
        print(f"✅ Output file: {video_file}")
        print(f"✅ You can now play this video file to see the result")
        print(f"\n📹 To view the video:")
        print(f"   - Open Finder and navigate to: {os.path.dirname(os.path.abspath(video_file))}")
        print(f"   - Double-click: {os.path.basename(video_file)}")
        print(f"   - Or use QuickTime Player, VLC, or any video player")
        print(f"\n🎯 This demonstrates that the serverless MultiTalk system is fully operational!")
    else:
        print(f"\n⚠️  Video generation did not complete successfully")
        print(f"Please check the error messages above")

if __name__ == "__main__":
    main()