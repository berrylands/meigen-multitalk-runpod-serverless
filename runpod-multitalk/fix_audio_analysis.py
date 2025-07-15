#!/usr/bin/env python3
"""
Quick fix script to download missing src/audio_analysis modules
"""

import os
import requests

def download_file(url, local_path):
    """Download a file from URL to local path"""
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        response = requests.get(url)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded: {local_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {local_path}: {e}")
        return False

def main():
    base_url = "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main"
    
    # Files needed to fix the immediate issue
    files = [
        "src/audio_analysis/wav2vec2.py",
        "src/audio_analysis/torch_utils.py",
    ]
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/audio_analysis/__init__.py",
    ]
    
    print("Fixing missing audio_analysis modules...")
    
    # Download files
    for file_path in files:
        url = f"{base_url}/{file_path}"
        download_file(url, file_path)
    
    # Create __init__.py files
    for init_file in init_files:
        os.makedirs(os.path.dirname(init_file), exist_ok=True)
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("")
            print(f"✓ Created: {init_file}")
    
    print("\nAudio analysis module fix complete!")

if __name__ == "__main__":
    main()