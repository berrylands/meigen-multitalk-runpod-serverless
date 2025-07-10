#!/usr/bin/env python3
"""
Download the large Wan2.1 GGUF model to RunPod network storage
This script should be run on a RunPod GPU pod with the network volume mounted
"""

import os
import sys
import time
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return dest_path

def download_wan21_models():
    """Download Wan2.1 models from HuggingFace."""
    
    model_base = Path("/runpod-volume/models")
    model_base.mkdir(parents=True, exist_ok=True)
    
    # Models to download
    models = [
        {
            "name": "Wan2.1-I2V-14B-480P GGUF Q4",
            "url": "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/Wan2.1-I2V-14B-480P_Q4_K_M.gguf",
            "path": model_base / "wan2.1-i2v-14b-480p" / "Wan2.1-I2V-14B-480P_Q4_K_M.gguf",
            "size_gb": 11.2
        },
        {
            "name": "Face Detection Model",
            "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "path": model_base / "face_detection" / "detection_Resnet50_Final.pth",
            "size_gb": 0.1
        },
        {
            "name": "Face Parsing Model", 
            "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
            "path": model_base / "face_parsing" / "parsing_parsenet.pth",
            "size_gb": 0.1
        }
    ]
    
    print("Wan2.1 Model Downloader for RunPod")
    print("=" * 60)
    print(f"Target directory: {model_base}")
    print(f"Models to download: {len(models)}")
    
    total_size = sum(m['size_gb'] for m in models)
    print(f"Total download size: {total_size:.1f} GB")
    print()
    
    # Check available space
    import shutil
    stat = shutil.disk_usage(model_base)
    free_gb = stat.free / (1024**3)
    print(f"Available space: {free_gb:.1f} GB")
    
    if free_gb < total_size * 1.2:  # 20% buffer
        print(f"WARNING: Not enough space! Need {total_size * 1.2:.1f} GB")
        return
    
    # Download each model
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Downloading {model['name']}...")
        print(f"Size: {model['size_gb']:.1f} GB")
        print(f"URL: {model['url']}")
        
        try:
            start_time = time.time()
            download_file(model['url'], model['path'])
            elapsed = time.time() - start_time
            
            print(f"✓ Downloaded in {elapsed:.1f} seconds")
            
            # Verify file size
            actual_size = model['path'].stat().st_size / (1024**3)
            print(f"  File size: {actual_size:.2f} GB")
            
        except Exception as e:
            print(f"✗ Failed to download: {e}")
            continue
    
    # List all downloaded models
    print("\n" + "=" * 60)
    print("Downloaded models:")
    
    for model_dir in model_base.iterdir():
        if model_dir.is_dir():
            size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
            size_gb = size / (1024**3)
            file_count = len(list(model_dir.rglob("*")))
            print(f"  - {model_dir.name}: {size_gb:.2f} GB ({file_count} files)")

if __name__ == "__main__":
    # This script should be run on a RunPod GPU pod
    if not os.path.exists("/runpod-volume"):
        print("ERROR: /runpod-volume not found!")
        print("This script must be run on a RunPod pod with network volume mounted.")
        sys.exit(1)
    
    download_wan21_models()