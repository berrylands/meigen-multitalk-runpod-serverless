#!/usr/bin/env python3
"""
Simple MultiTalk Video Generator
Generate a video from an audio file using the deployed MultiTalk serverless endpoint.
"""

import os
import sys
import time
import runpod
import base64
import wave
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

# Configuration
ENDPOINT_ID = "kkx3cfy484jszl"

def load_wav_file(filename):
    """Load audio from a WAV file."""
    try:
        with wave.open(filename, 'rb') as wav:
            # Get audio parameters
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            
            print(f"Audio file info:")
            print(f"  Channels: {channels}")
            print(f"  Sample width: {sample_width} bytes")
            print(f"  Sample rate: {framerate} Hz")
            print(f"  Duration: {n_frames/framerate:.2f} seconds")
            
            # Read audio data
            frames = wav.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:
                audio = np.frombuffer(frames, dtype=np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert stereo to mono if needed
            if channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
                print("  Converted stereo to mono")
            
            # Resample to 16kHz if needed
            if framerate != 16000:
                print(f"  Resampling from {framerate}Hz to 16000Hz...")
                # Simple resampling (for better quality, use scipy.signal.resample)
                ratio = 16000 / framerate
                new_length = int(len(audio) * ratio)
                x_old = np.linspace(0, len(audio)-1, len(audio))
                x_new = np.linspace(0, len(audio)-1, new_length)
                audio = np.interp(x_new, x_old, audio).astype(np.int16)
                framerate = 16000
            
            duration = len(audio) / framerate
            return audio, framerate, duration
            
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None, None

def create_test_audio(duration=3.0):
    """Create a test audio signal."""
    print("Creating test audio signal...")
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a more interesting sound
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some harmonics
    audio += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
    audio += 0.25 * np.sin(2 * np.pi * frequency * 3 * t)
    
    # Add envelope
    envelope = np.exp(-t / duration)
    audio = audio * envelope
    
    # Normalize and convert to int16
    audio = audio / np.max(np.abs(audio)) * 0.8
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16, sample_rate, duration

def generate_video(audio_data, sample_rate, duration, output_filename="output.mp4"):
    """Generate video from audio data using MultiTalk."""
    
    print(f"\nGenerating video...")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Output: {output_filename}")
    
    # Encode audio to base64
    audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
    print(f"  Audio size: {len(audio_data) * 2:,} bytes")
    
    # Connect to endpoint
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Submit job
    job_input = {
        "action": "generate",
        "audio": audio_b64,
        "duration": duration,
        "fps": 30,
        "width": 512,
        "height": 512
    }
    
    print(f"\nSubmitting job to MultiTalk endpoint...")
    try:
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
                return False
        
        # Get result
        final_time = time.time() - start_time
        print(f"\nJob completed in {final_time:.1f} seconds")
        print(f"Final status: {job.status()}")
        
        if job.status() == "COMPLETED":
            result = job.output()
            
            if result and result.get("success"):
                print("\n✅ Video generated successfully!")
                
                # Save video
                video_b64 = result.get("video")
                if video_b64:
                    video_data = base64.b64decode(video_b64)
                    
                    with open(output_filename, "wb") as f:
                        f.write(video_data)
                    
                    print(f"✅ Video saved: {output_filename}")
                    print(f"   Size: {len(video_data):,} bytes ({len(video_data)/1024:.1f} KB)")
                    
                    # Show additional info
                    if "processing_time" in result:
                        print(f"   Processing time: {result['processing_time']}")
                    if "models_used" in result:
                        print(f"   Models used: {', '.join(result['models_used'])}")
                    
                    return True
                else:
                    print("❌ No video data in response")
            else:
                print(f"❌ Video generation failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ Job failed: {job.output()}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    return False

def main():
    """Main function."""
    print("MultiTalk Video Generator")
    print("=" * 50)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
        
        # Load audio from file
        audio_data, sample_rate, duration = load_wav_file(audio_file)
        
        if audio_data is None:
            print("Failed to load audio file. Using test audio instead.")
            audio_data, sample_rate, duration = create_test_audio()
    else:
        print("No audio file provided. Creating test audio...")
        print("\nUsage: python simple_video_generator.py <audio_file.wav> [output.mp4]")
        audio_data, sample_rate, duration = create_test_audio()
        output_file = "test_output.mp4"
    
    # Generate video
    success = generate_video(audio_data, sample_rate, duration, output_file)
    
    if success:
        print(f"\n🎉 Success! Video saved as: {output_file}")
        print(f"You can now play this video in any media player.")
    else:
        print(f"\n❌ Video generation failed.")

if __name__ == "__main__":
    main()