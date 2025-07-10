#!/usr/bin/env python3
"""
Deploy full MultiTalk implementation to RunPod
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "kkx3cfy484jszl"

def submit_large_model_download():
    """Submit a job to download the large Wan2.1 model."""
    
    runpod.api_key = RUNPOD_API_KEY
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    print("Submitting large model download job...")
    print("This will download ~11GB Wan2.1 GGUF model")
    
    # Job to download large models
    job_input = {
        "action": "download_large_models",
        "models": [
            {
                "name": "Wan2.1 GGUF Q4",
                "url": "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/Wan2.1-I2V-14B-480P_Q4_K_M.gguf",
                "path": "wan2.1-i2v-14b-480p/Wan2.1-I2V-14B-480P_Q4_K_M.gguf",
                "size_gb": 11.2
            }
        ]
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"Job submitted: {job.job_id}")
        
        # Monitor for completion
        start_time = time.time()
        while True:
            status = job.status()
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Status: {status}")
            
            if status not in ["IN_QUEUE", "IN_PROGRESS"]:
                break
                
            if elapsed > 1800:  # 30 minute timeout
                print("Timeout - download may still be in progress")
                break
                
            time.sleep(30)  # Check every 30 seconds
        
        if job.status() == "COMPLETED":
            result = job.output()
            print(f"\n✓ Download completed: {result}")
            return True
        else:
            print(f"\n✗ Download failed: {job.output()}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_multitalk_generation():
    """Test the full MultiTalk video generation."""
    
    runpod.api_key = RUNPOD_API_KEY  
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    print("\nTesting MultiTalk video generation...")
    
    # Create test audio data (sine wave)
    import base64
    import numpy as np
    
    # Generate 5 seconds of sine wave audio at 16kHz
    sample_rate = 16000
    duration = 5.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit PCM and encode
    audio_int16 = (audio_data * 32767).astype(np.int16)
    audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
    
    job_input = {
        "action": "generate",
        "audio": audio_b64,
        "duration": 5.0,
        "fps": 30
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"Generation job: {job.job_id}")
        
        start_time = time.time()
        while True:
            status = job.status()
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Status: {status}")
            
            if status not in ["IN_QUEUE", "IN_PROGRESS"]:
                break
                
            if elapsed > 300:  # 5 minute timeout
                print("Generation timeout")
                break
                
            time.sleep(10)
        
        if job.status() == "COMPLETED":
            result = job.output()
            if result.get("success"):
                print(f"\n✓ Video generated!")
                print(f"  Duration: {result.get('duration')}s")
                print(f"  Processing time: {result.get('processing_time')}")
                
                # Save video
                video_b64 = result.get("video")
                if video_b64:
                    video_data = base64.b64decode(video_b64)
                    with open("test_output.mp4", "wb") as f:
                        f.write(video_data)
                    print(f"  Saved test video: test_output.mp4 ({len(video_data)} bytes)")
                
                return True
            else:
                print(f"\n✗ Generation failed: {result}")
                return False
        else:
            print(f"\n✗ Job failed: {job.output()}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main deployment workflow."""
    
    print("MultiTalk Full Deployment")
    print("=" * 60)
    
    # Check current endpoint status
    runpod.api_key = RUNPOD_API_KEY
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    print("1. Checking endpoint health...")
    try:
        health = endpoint.health()
        workers_ready = health.get('workers', {}).get('ready', 0)
        print(f"   Workers ready: {workers_ready}")
        
        if workers_ready == 0:
            print("   ⚠️  No ready workers - endpoint may be starting up")
        
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Check current models
    print("\n2. Checking current models...")
    try:
        job = endpoint.run({"action": "list_models"})
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(2)
        
        if job.status() == "COMPLETED":
            result = job.output()
            models = result.get('models', [])
            print(f"   Models available: {len(models)}")
            
            wan21_exists = any('wan2.1' in m.get('name', '').lower() for m in models)
            if wan21_exists:
                print("   ✓ Wan2.1 model found")
            else:
                print("   ✗ Wan2.1 model missing - need to download")
                
                # Download large model
                print("\n3. Downloading large Wan2.1 model...")
                if not submit_large_model_download():
                    print("   Failed to download model - continuing with test")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test video generation
    print("\n4. Testing video generation...")
    if test_multitalk_generation():
        print("\n🎉 SUCCESS! MultiTalk is working!")
        print("\nNext steps:")
        print("- Use the client example to generate videos with your own audio")
        print("- Deploy to production with proper monitoring")
        print("- Optimize cold start times and model loading")
    else:
        print("\n❌ Generation test failed")
        print("Check the handler logs and model availability")

if __name__ == "__main__":
    main()