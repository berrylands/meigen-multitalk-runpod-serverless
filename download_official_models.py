#!/usr/bin/env python3
"""
Download the official MeiGen MultiTalk models from HuggingFace
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

def download_official_models():
    """Download the official MeiGen MultiTalk models."""
    
    print("Downloading Official MeiGen MultiTalk Models")
    print("=" * 60)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Official models from the correct HuggingFace repositories
    official_models = [
        {
            "name": "MeiGen-MultiTalk",
            "repo_id": "MeiGen-AI/MeiGen-MultiTalk",
            "path": "meigen-multitalk",
            "files": ["multitalk.safetensors"]  # Main model file ~10GB
        },
        {
            "name": "Wan2.1-I2V-14B-480P",
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-480P",
            "path": "wan2.1-i2v-14b-480p"
        },
        {
            "name": "Chinese-wav2vec2-base",
            "repo_id": "TencentGameMate/chinese-wav2vec2-base",
            "path": "chinese-wav2vec2-base"
        },
        {
            "name": "Kokoro-82M",
            "repo_id": "hexgrad/Kokoro-82M",
            "path": "kokoro-82m"
        }
    ]
    
    print(f"Official models to download: {len(official_models)}")
    for model in official_models:
        print(f"  - {model['name']} ({model['repo_id']})")
    
    print(f"\nEstimated download size: ~25-30GB")
    print("This may take 30-60 minutes depending on network speed...")
    
    # Submit download job
    job_input = {
        "action": "download_models",
        "models": official_models
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"\nDownload job submitted: {job.job_id}")
        
        # Monitor progress with extended timeout for very large downloads
        start_time = time.time()
        last_status = None
        last_update = time.time()
        
        while True:
            status = job.status()
            current_time = time.time()
            
            if status != last_status or (current_time - last_update) > 120:  # Update every 2 minutes
                elapsed = current_time - start_time
                print(f"[{elapsed/60:.1f}min] Status: {status}")
                last_status = status
                last_update = current_time
                
            if status not in ["IN_QUEUE", "IN_PROGRESS"]:
                break
                
            if elapsed > 3600:  # 60 minute timeout for very large downloads
                print("Download taking longer than expected...")
                print("Check RunPod dashboard for detailed progress.")
                print("Large models may take significant time to download.")
                break
                
            time.sleep(30)  # Check every 30 seconds
        
        # Get final result
        if job.status() == "COMPLETED":
            output = job.output()
            
            if isinstance(output, dict) and output.get("success"):
                print(f"\n✅ Official models downloaded successfully!")
                
                if "downloaded" in output:
                    total_size = 0
                    print("\nDownloaded models:")
                    for model in output["downloaded"]:
                        size_mb = model.get('size', 0) / (1024 * 1024)
                        total_size += model.get('size', 0)
                        print(f"  ✅ {model.get('name')} ({size_mb:.1f} MB)")
                        print(f"      Path: {model.get('path')}")
                    
                    total_gb = total_size / (1024 * 1024 * 1024)
                    print(f"\nTotal downloaded: {total_gb:.2f} GB")
                
                if "errors" in output and output["errors"]:
                    print("\nPartial download - some models failed:")
                    for error in output["errors"]:
                        print(f"  ❌ {error}")
                
                return True
            else:
                print(f"\n❌ Download failed")
                if isinstance(output, dict):
                    errors = output.get('errors', [])
                    for error in errors:
                        print(f"  ❌ {error}")
                    
                    # Check if it's a model size/availability issue
                    if any("404" in str(error) for error in errors):
                        print("\n💡 Some models may not be publicly available.")
                        print("Checking what we can download...")
                        return "partial"
                        
                return False
        else:
            print(f"\n❌ Job failed with status: {job.status()}")
            try:
                error_output = job.output()
                print(f"Error details: {error_output}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"\nError: {e}")
        return False

def try_alternative_downloads():
    """Try downloading alternative/smaller models if official ones fail."""
    
    print(f"\n" + "=" * 60)
    print("Trying Alternative Model Sources...")
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Alternative models that might be more readily available
    alternative_models = [
        {
            "name": "Stable-Video-Diffusion",
            "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "path": "stable-video-diffusion",
            "files": ["svd_xt.safetensors"]
        },
        {
            "name": "AnimateDiff",
            "repo_id": "guoyww/animatediff",
            "path": "animatediff"
        }
    ]
    
    job_input = {
        "action": "download_models", 
        "models": alternative_models
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"Alternative download job: {job.job_id}")
        
        # Shorter timeout for alternatives
        start_time = time.time()
        while True:
            status = job.status()
            elapsed = time.time() - start_time
            
            if elapsed % 60 == 0:  # Log every minute
                print(f"[{elapsed/60:.0f}min] Status: {status}")
                
            if status not in ["IN_QUEUE", "IN_PROGRESS"]:
                break
                
            if elapsed > 1800:  # 30 minute timeout
                break
                
            time.sleep(30)
        
        if job.status() == "COMPLETED":
            output = job.output()
            print(f"✅ Alternative models result: {output}")
            return True
        else:
            print(f"❌ Alternative download failed: {job.output()}")
            return False
            
    except Exception as e:
        print(f"Error with alternatives: {e}")
        return False

if __name__ == "__main__":
    print("Starting comprehensive model download for full MultiTalk functionality...")
    
    # Try official models first
    result = download_official_models()
    
    if result == True:
        print(f"\n🎉 SUCCESS! Official models downloaded")
    elif result == "partial":
        print(f"\n⚠️  Partial success - trying alternatives")
        try_alternative_downloads()
    else:
        print(f"\n⚠️  Official download failed - trying alternatives")
        try_alternative_downloads()
    
    # Verify what we have
    print(f"\n" + "=" * 60)
    print("Checking final model inventory...")
    
    try:
        endpoint = runpod.Endpoint(ENDPOINT_ID)
        job = endpoint.run({"action": "list_models"})
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(2)
        
        if job.status() == "COMPLETED":
            result = job.output()
            models = result.get('models', [])
            total_size = sum(m.get('size_mb', 0) for m in models) / 1024
            
            print(f"Final model inventory:")
            print(f"  Total models: {len(models)}")
            print(f"  Total storage: {total_size:.1f} GB")
            
            # Check if we have enough for basic functionality
            has_video_model = any('wan' in m['name'].lower() or 'multitalk' in m['name'].lower() 
                                 or 'stable' in m['name'].lower() for m in models)
            has_audio_model = any('wav2vec' in m['name'].lower() for m in models)
            
            if has_video_model and has_audio_model:
                print(f"\n✅ Ready for MultiTalk implementation!")
                print("Next: Implement full video generation pipeline")
            else:
                print(f"\n⚠️  Missing critical models")
                print("Will implement with available models and iterate")
                
    except Exception as e:
        print(f"Error checking inventory: {e}")
    
    print(f"\nNext step: Implement complete MultiTalk inference pipeline")