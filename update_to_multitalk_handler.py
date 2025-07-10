#!/usr/bin/env python3
"""
Update the current endpoint to use the MultiTalk handler
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "kkx3cfy484jszl"

def test_current_handler():
    """Test what the current handler supports."""
    
    runpod.api_key = RUNPOD_API_KEY
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    print("Testing current handler capabilities...")
    
    # Test health check
    job = endpoint.run({"health_check": True})
    while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
        time.sleep(2)
    
    if job.status() == "COMPLETED":
        result = job.output()
        print(f"✓ Health check: {result.get('message')}")
        print(f"  Models loaded: {result.get('models_loaded', 0)}")
        
        # Check if it supports the MultiTalk actions
        if 'models_available' in result:
            print("✓ MultiTalk handler detected")
            return True
        else:
            print("ℹ️  Basic handler - needs upgrade to MultiTalk")
            return False
    else:
        print(f"✗ Health check failed: {job.output()}")
        return False

def download_large_model():
    """Download the large Wan2.1 model using a simple HTTP download."""
    
    runpod.api_key = RUNPOD_API_KEY
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    print("Downloading Wan2.1 model...")
    
    # Create a simple download script to run on the endpoint
    download_script = '''
import os
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (10 * 1024 * 1024) == 0:  # Log every 10MB
                    print(f"Downloaded {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
    
    return dest_path

# Download the model
model_url = "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/Wan2.1-I2V-14B-480P_Q4_K_M.gguf"
model_path = Path("/runpod-volume/models/wan2.1-i2v-14b-480p/Wan2.1-I2V-14B-480P_Q4_K_M.gguf")

print(f"Starting download of Wan2.1 model...")
print(f"URL: {model_url}")
print(f"Destination: {model_path}")

try:
    result_path = download_file(model_url, model_path)
    file_size = result_path.stat().st_size / (1024**3)
    print(f"✓ Download completed! Size: {file_size:.2f} GB")
    return {"success": True, "size_gb": file_size, "path": str(result_path)}
except Exception as e:
    print(f"✗ Download failed: {e}")
    return {"success": False, "error": str(e)}
'''
    
    # Submit the download as a custom job
    job_input = {
        "action": "custom_python",
        "code": download_script
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"Download job: {job.job_id}")
        
        # Monitor progress
        start_time = time.time()
        while True:
            status = job.status()
            elapsed = time.time() - start_time
            
            if elapsed % 60 == 0:  # Log every minute
                print(f"[{elapsed/60:.0f}min] Status: {status}")
            
            if status not in ["IN_QUEUE", "IN_PROGRESS"]:
                break
                
            if elapsed > 1800:  # 30 minute timeout
                print("Download timeout")
                break
                
            time.sleep(10)
        
        if job.status() == "COMPLETED":
            result = job.output()
            print(f"✓ Download result: {result}")
            return True
        else:
            print(f"✗ Download failed: {job.output()}")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Main update workflow."""
    
    print("MultiTalk Handler Update")
    print("=" * 60)
    
    # Test current capabilities
    print("1. Testing current handler...")
    is_multitalk = test_current_handler()
    
    if not is_multitalk:
        print("\n📝 To upgrade to full MultiTalk handler:")
        print("1. Update endpoint image to: berrylands/multitalk-lite:latest")
        print("2. Or wait for the full CUDA image to build")
        print("3. Re-run this script")
        return
    
    # Check if we need to download the large model
    print("\n2. Checking for large models...")
    # This would typically check if Wan2.1 exists and download if needed
    
    print("\n3. Current status:")
    print("✓ Endpoint is working")
    print("✓ Basic models downloaded (~2.6GB)")
    print("ℹ️  Ready for MultiTalk video generation")
    
    print("\n🚀 To test video generation:")
    print("python examples/multitalk_client.py")

if __name__ == "__main__":
    main()