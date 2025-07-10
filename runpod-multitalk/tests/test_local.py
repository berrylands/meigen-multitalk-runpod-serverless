#!/usr/bin/env python3
"""
Local testing script for MultiTalk inference
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from multitalk_inference import MultiTalkInference
import tempfile
import numpy as np
from PIL import Image
import wave

def create_test_image(path: str, size=(640, 480)):
    """Create a test image."""
    # Create a simple gradient image
    img = Image.new('RGB', size)
    pixels = img.load()
    
    for i in range(size[0]):
        for j in range(size[1]):
            r = int(255 * i / size[0])
            g = int(255 * j / size[1])
            b = 128
            pixels[i, j] = (r, g, b)
    
    img.save(path)
    print(f"Created test image: {path}")

def create_test_audio(path: str, duration=3.0, frequency=440):
    """Create a test audio file."""
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Generate sine wave
    t = np.linspace(0, duration, num_samples)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Scale to 16-bit integer
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"Created test audio: {path} ({duration}s, {frequency}Hz)")

def test_multitalk_inference():
    """Test the MultiTalk inference pipeline."""
    print("Testing MultiTalk Inference...")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test inputs
        image_path = os.path.join(temp_dir, "test_image.jpg")
        audio1_path = os.path.join(temp_dir, "test_audio1.wav")
        audio2_path = os.path.join(temp_dir, "test_audio2.wav")
        
        create_test_image(image_path)
        create_test_audio(audio1_path, duration=3.0, frequency=440)
        create_test_audio(audio2_path, duration=3.0, frequency=880)
        
        # Test model initialization (will use local models path for testing)
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        if not os.path.exists(models_path):
            print(f"WARNING: Models directory not found at {models_path}")
            print("Creating mock model directory for testing...")
            os.makedirs(models_path, exist_ok=True)
            
            # Create mock model directories
            for subdir in ['wan2.1-i2v-14b-480p', 'meigen-multitalk', 
                          'chinese-wav2vec2-base', 'kokoro-82m', 'wan2.1-vae']:
                os.makedirs(os.path.join(models_path, subdir), exist_ok=True)
        
        try:
            # Initialize model
            print("\nInitializing MultiTalk model...")
            model = MultiTalkInference(model_path=models_path)
            
            # Test single person generation
            print("\nTesting single person video generation...")
            output_path = model.generate(
                reference_image_path=image_path,
                audio1_path=audio1_path,
                prompt="A person speaking",
                num_frames=50,
                seed=42,
                turbo=True,
                sampling_steps=10
            )
            
            assert os.path.exists(output_path), "Output video not created"
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"✓ Single person video created: {output_path} ({file_size:.2f} KB)")
            
            # Test multi-person generation
            print("\nTesting multi-person video generation...")
            output_path2 = model.generate(
                reference_image_path=image_path,
                audio1_path=audio1_path,
                audio2_path=audio2_path,
                prompt="Two people having a conversation",
                num_frames=75,
                seed=123,
                turbo=False,
                sampling_steps=20
            )
            
            assert os.path.exists(output_path2), "Output video not created"
            file_size2 = os.path.getsize(output_path2) / 1024  # KB
            print(f"✓ Multi-person video created: {output_path2} ({file_size2:.2f} KB)")
            
            # Cleanup
            model.cleanup()
            print("\n✓ All tests passed!")
            
        except Exception as e:
            print(f"\n✗ Test failed: {str(e)}")
            raise

def test_handler():
    """Test the RunPod handler."""
    print("\nTesting RunPod Handler...")
    
    # Mock RunPod job
    job = {
        "input": {
            "reference_image": "data:image/jpeg;base64,/9j/4AAQ...",  # Mock base64
            "audio_1": "data:audio/wav;base64,UklGR...",  # Mock base64
            "prompt": "Test conversation",
            "num_frames": 50,
            "seed": 42
        }
    }
    
    print("✓ Handler test structure created")
    print("Note: Full handler test requires RunPod environment")

if __name__ == "__main__":
    print("MultiTalk Local Testing Suite")
    print("="*50)
    
    # Run tests
    test_multitalk_inference()
    test_handler()
    
    print("\n" + "="*50)
    print("Testing complete!")