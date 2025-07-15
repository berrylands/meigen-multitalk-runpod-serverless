#!/usr/bin/env python3
"""
Test MultiTalk with base64 audio (no S3 required)
"""

import runpod
import os
import base64
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_test_audio():
    """Create a simple test audio file"""
    import numpy as np
    import wave
    import io
    
    # Generate a simple sine wave (1 second, 440 Hz)
    sample_rate = 16000
    duration = 3.0
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    # Get bytes and encode to base64
    buffer.seek(0)
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

def test_base64():
    """Test with base64 audio"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🎵 Testing MultiTalk with Base64 Audio")
    print("=" * 60)
    
    # Create test audio
    print("1. Creating test audio...")
    try:
        audio_base64 = create_test_audio()
        print(f"✅ Created test audio ({len(audio_base64)} base64 chars)")
    except Exception as e:
        print(f"❌ Failed to create test audio: {e}")
        print("\nUsing a minimal base64 audio instead...")
        # Minimal valid WAV header + silence
        audio_base64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA="
    
    # Test with base64 audio
    print("\n2. Submitting job with base64 audio...")
    
    job_input = {
        "action": "generate",
        "audio": audio_base64,
        "duration": 3.0,
        "width": 480,
        "height": 480,
        "fps": 30
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"✅ Job submitted: {job.job_id}")
        
        # Wait for result
        print("\n3. Waiting for result...")
        start_time = time.time()
        
        while True:
            status = job.status()
            elapsed = time.time() - start_time
            
            print(f"[{elapsed:.1f}s] Status: {status}")
            
            if status in ["COMPLETED", "FAILED"]:
                break
                
            if elapsed > 120:
                print("❌ Timeout after 2 minutes")
                break
                
            time.sleep(2)
        
        # Get result
        result = job.output()
        
        if status == "COMPLETED":
            print("\n✅ Job completed successfully!")
            if isinstance(result, dict):
                if 'video' in result:
                    video_data = result['video']
                    if video_data.startswith('data:'):
                        print(f"📹 Video generated (data URL): {len(video_data)} chars")
                    else:
                        print(f"📹 Video generated (base64): {len(video_data)} chars")
                else:
                    print("Result:", result)
        else:
            print("\n❌ Job failed")
            print("Error:", result)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        
    print("\n📝 Next steps:")
    print("1. If this works, the issue is just the missing S3 file")
    print("2. Upload your audio file to S3: s3://760572149-framepack/1.wav")
    print("3. Then test with S3 URL again")

if __name__ == "__main__":
    test_base64()