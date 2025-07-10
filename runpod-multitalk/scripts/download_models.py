#!/usr/bin/env python3
"""
One-time model download script for RunPod network volume.
This should be run on a RunPod instance with the network volume mounted.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import argparse

def download_models(base_path="/runpod-volume/models"):
    """Download all required models to the specified path."""
    
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading models to: {base_path}")
    
    # Model configurations
    models = [
        {
            "name": "Wan2.1-I2V-14B-480P-GGUF",
            "repo_id": "city96/Wan2.1-I2V-14B-480P-gguf",
            "files": ["Wan2.1-I2V-14B-480P_Q4_K_M.gguf"],  # Quantized version ~11GB
            "path": "wan2.1-i2v-14b-480p"
        },
        {
            "name": "MeiGen-MultiTalk",
            "repo_id": "MeiGen-AI/MeiGen-MultiTalk",
            "files": ["multitalk.safetensors"],
            "path": "meigen-multitalk"
        },
        {
            "name": "Chinese-Wav2Vec2-Base",
            "repo_id": "TencentGameMate/chinese-wav2vec2-base",
            "files": None,  # Download all files
            "path": "chinese-wav2vec2-base"
        },
        {
            "name": "Kokoro-82M",
            "repo_id": "hexgrad/Kokoro-82M",
            "files": None,  # Download all files
            "path": "kokoro-82m"
        },
        {
            "name": "Wan2.1-VAE",
            "repo_id": "Wan-AI/Wan2.1-I2V-14B-480P",
            "files": ["Wan2.1_VAE.pth"],
            "path": "wan2.1-vae"
        }
    ]
    
    # Download each model
    for model in models:
        print(f"\n{'='*60}")
        print(f"Downloading: {model['name']}")
        print(f"Repository: {model['repo_id']}")
        
        model_path = base_path / model['path']
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if model['files']:
                # Download specific files
                for file in model['files']:
                    print(f"  Downloading file: {file}")
                    hf_hub_download(
                        repo_id=model['repo_id'],
                        filename=file,
                        local_dir=model_path,
                        resume_download=True
                    )
            else:
                # Download entire repository
                print(f"  Downloading entire repository...")
                snapshot_download(
                    repo_id=model['repo_id'],
                    local_dir=model_path,
                    resume_download=True
                )
            
            print(f"✓ Successfully downloaded {model['name']}")
            
        except Exception as e:
            print(f"✗ Error downloading {model['name']}: {str(e)}")
            sys.exit(1)
    
    # Verify downloads
    print(f"\n{'='*60}")
    print("Verifying downloads...")
    
    total_size = 0
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path) / (1024**3)  # Convert to GB
            total_size += size
            if size > 0.1:  # Only show files larger than 100MB
                print(f"  {file}: {size:.2f} GB")
    
    print(f"\nTotal storage used: {total_size:.2f} GB")
    
    # Create a marker file to indicate successful download
    marker_file = base_path / "download_complete.txt"
    with open(marker_file, 'w') as f:
        f.write("Model download completed successfully\n")
        f.write(f"Total size: {total_size:.2f} GB\n")
    
    print("\n✓ All models downloaded successfully!")

def main():
    parser = argparse.ArgumentParser(description="Download MultiTalk models to RunPod network volume")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/runpod-volume/models",
        help="Base path for model storage (default: /runpod-volume/models)"
    )
    
    args = parser.parse_args()
    
    # Check if running on RunPod
    if not os.path.exists("/runpod-volume") and args.model_path == "/runpod-volume/models":
        print("WARNING: /runpod-volume not found. Are you running this on RunPod?")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    download_models(args.model_path)

if __name__ == "__main__":
    main()