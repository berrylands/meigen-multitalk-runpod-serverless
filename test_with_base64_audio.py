#!/usr/bin/env python3
"""
Test MultiTalk with proper audio data (base64 encoded)
"""

import runpod
import os
import base64
import numpy as np
import wave
import io
from dotenv import load_dotenv

load_dotenv()

def create_test_audio(duration=5.0, sample_rate=16000):
    """Create a simple test audio (sine wave)"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Scale to 16-bit integer range
    audio = (audio * 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    buffer.seek(0)
    return buffer.read()

def test_multitalk_with_audio():
    """Test MultiTalk with properly formatted audio"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🎵 Testing MultiTalk with Real Audio Data")
    print("=" * 60)
    
    # Create test audio
    print("1. Creating test audio...")
    audio_data = create_test_audio(duration=3.0)
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    print(f"   Audio size: {len(audio_data):,} bytes")
    print(f"   Base64 size: {len(audio_base64):,} chars")
    
    # Test generation
    print("\n2. Sending to MultiTalk...")
    job = endpoint.run({
        "action": "generate",
        "audio": audio_base64,  # Base64 encoded audio
        "duration": 3.0,
        "width": 256,
        "height": 256,
        "fps": 25,
        "output_format": "s3",
        "s3_output_key": "test_outputs/multitalk_real_test.mp4"
    })
    
    print("   Waiting for result...")
    result = job.output(timeout=180)
    
    if result and result.get('success'):
        video_info = result.get('video_info', {})
        note = video_info.get('processing_note', '')
        models = video_info.get('models_used', [])
        
        print(f"\n✅ Video generated successfully!")
        print(f"   Processing note: {note}")
        print(f"   Models used: {models}")
        print(f"   Processing time: {result.get('processing_time', 'N/A')}")
        
        if 'audio_features_shape' in video_info:
            print(f"   Audio features shape: {video_info['audio_features_shape']}")
        
        if result.get('output_type') == 's3_url':
            print(f"   Video saved to: {result.get('video', 'N/A')}")
        
        if "Real MultiTalk inference" in note:
            print("\n🎉 SUCCESS! Real MultiTalk inference is working!")
            print("   The audio was processed correctly and video was generated.")
        elif "Fallback" in note:
            print("\n⚠️  Still using fallback - check worker logs for errors")
    else:
        print(f"\n❌ Generation failed!")
        if result:
            print(f"   Error: {result.get('error', 'Unknown')}")
            if 'details' in result:
                print(f"   Details: {result['details']}")
    
    print("\n" + "=" * 60)
    print("\n💡 The issue with your S3 file '1.wav' is that it only contains 3 bytes.")
    print("   It's not a valid audio file. Try uploading a real WAV file to S3.")

if __name__ == "__main__":
    test_multitalk_with_audio()