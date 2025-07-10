#!/usr/bin/env python3
"""
Download the critical missing models for full MultiTalk functionality
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

def download_critical_models():
    """Download the essential models for MultiTalk."""
    
    print("Downloading Critical MultiTalk Models")
    print("=" * 60)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Define critical models to download
    critical_models = [
        {
            "name": "Wan2.1-I2V-14B-480P-GGUF",
            "repo_id": "city96/Wan2.1-I2V-14B-480P-gguf",
            "files": ["Wan2.1-I2V-14B-480P_Q4_K_M.gguf"],
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
    
    print(f"Models to download: {len(critical_models)}")
    for model in critical_models:
        print(f"  - {model['name']}")
    
    # Submit download job
    job_input = {
        "action": "download_models",
        "models": critical_models
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"\nDownload job submitted: {job.job_id}")
        print("This will download ~12GB of models...")
        
        # Monitor progress with longer timeout for large downloads
        start_time = time.time()
        last_status = None
        
        while True:
            status = job.status()
            if status != last_status:
                elapsed = time.time() - start_time
                print(f"[{elapsed/60:.1f}min] Status: {status}")
                last_status = status
                
            if status not in ["IN_QUEUE", "IN_PROGRESS"]:
                break
                
            if elapsed > 2400:  # 40 minute timeout for large downloads
                print("Download taking longer than expected...")
                print("This is normal for large models. Check RunPod dashboard.")
                break
                
            time.sleep(30)  # Check every 30 seconds
        
        # Get final result
        if job.status() == "COMPLETED":
            output = job.output()
            
            if isinstance(output, dict) and output.get("success"):
                print(f"\n✅ Models downloaded successfully!")
                
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
                    print("\nErrors encountered:")
                    for error in output["errors"]:
                        print(f"  ❌ {error}")
                
                return True
            else:
                print(f"\n❌ Download failed")
                if isinstance(output, dict):
                    print(f"Errors: {output.get('errors', [])}")
                return False
        else:
            print(f"\n❌ Job failed with status: {job.status()}")
            try:
                print(f"Error: {job.output()}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"\nError: {e}")
        return False

def verify_models():
    """Verify all required models are now available."""
    
    print(f"\n" + "=" * 60)
    print("Verifying model availability...")
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    try:
        job = endpoint.run({"action": "list_models"})
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(2)
        
        if job.status() == "COMPLETED":
            result = job.output()
            models = result.get('models', [])
            total = result.get('total', 0)
            
            print(f"\nTotal models on volume: {total}")
            
            # Check for critical models
            model_names = [m['name'].lower() for m in models]
            
            required_checks = {
                "Wan2.1": any('wan2.1' in name for name in model_names),
                "Chinese Wav2Vec2": any('chinese' in name for name in model_names),
                "Kokoro": any('kokoro' in name for name in model_names),
                "Audio Processing": any('wav2vec2' in name for name in model_names),
                "Face Enhancement": any('gfp' in name for name in model_names)
            }
            
            print("\nCritical model availability:")
            all_present = True
            for model_type, present in required_checks.items():
                status = "✅" if present else "❌"
                print(f"  {status} {model_type}")
                if not present:
                    all_present = False
            
            total_size = sum(m.get('size_mb', 0) for m in models)
            print(f"\nTotal storage used: {total_size/1024:.2f} GB")
            
            return all_present
        else:
            print(f"❌ Failed to verify models: {job.output()}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Download critical models
    success = download_critical_models()
    
    if success:
        # Verify everything is in place
        all_models_present = verify_models()
        
        if all_models_present:
            print(f"\n🎉 SUCCESS!")
            print("✅ All critical models downloaded")
            print("✅ Ready to implement full MultiTalk pipeline")
        else:
            print(f"\n⚠️  Some models still missing")
            print("Check download errors and retry if needed")
    else:
        print(f"\n❌ Download failed")
        print("Check network connectivity and RunPod status")
    
    print(f"\nNext step: Implement complete MultiTalk inference pipeline")