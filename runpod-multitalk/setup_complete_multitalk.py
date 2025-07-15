#!/usr/bin/env python3
"""
Complete setup script for MeiGen MultiTalk
Downloads ALL required files from the official repository
"""

import os
import requests
import sys
from pathlib import Path

def download_file(url, local_path):
    """Download a file from URL to local path"""
    try:
        # Create directory if it doesn't exist
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

def setup_multitalk():
    """Download all required files for MultiTalk"""
    
    base_url = "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main"
    
    # Define all files to download with their paths
    files_to_download = [
        # Root files
        "generate_multitalk.py",
        "requirements.txt",
        
        # src/audio_analysis directory
        "src/audio_analysis/wav2vec2.py",
        "src/audio_analysis/torch_utils.py",
        
        # src directory
        "src/utils.py",
        
        # src/vram_management directory
        "src/vram_management/__init__.py",
        "src/vram_management/layers.py",
        
        # wan package - main files
        "wan/__init__.py",
        "wan/first_last_frame2video.py",
        "wan/image2video.py",
        "wan/multitalk.py",
        "wan/text2video.py",
        "wan/vace.py",
        "wan/wan_lora.py",
        
        # wan/configs
        "wan/configs/__init__.py",
        "wan/configs/shared_config.py",
        "wan/configs/wan_i2v_14B.py",
        "wan/configs/wan_multitalk_14B.py",
        "wan/configs/wan_t2v_14B.py",
        "wan/configs/wan_t2v_1_3B.py",
        
        # wan/distributed
        "wan/distributed/__init__.py",
        "wan/distributed/fsdp.py",
        
        # wan/modules
        "wan/modules/__init__.py",
        "wan/modules/clip.py",
        "wan/modules/multitalk_model.py",
        "wan/modules/t5.py",
        "wan/modules/vae.py",
        
        # wan/utils
        "wan/utils/__init__.py",
        "wan/utils/fm_solvers.py",
        "wan/utils/fm_solvers_unipc.py",
        "wan/utils/multitalk_utils.py",
        "wan/utils/prompt_extend.py",
        "wan/utils/qwen_vl_utils.py",
        "wan/utils/utils.py",
        "wan/utils/vace_processor.py",
        
        # kokoro package
        "kokoro/__init__.py",
        "kokoro/__main__.py",
        "kokoro/custom_stft.py",
        "kokoro/istftnet.py",
        "kokoro/model.py",
        "kokoro/modules.py",
        "kokoro/pipeline.py",
    ]
    
    # Create __init__.py files for packages that might not have them
    init_files = [
        "src/__init__.py",
        "src/audio_analysis/__init__.py",
        "src/vram_management/__init__.py",
        "wan/modules/__init__.py",
        "wan/distributed/__init__.py",
    ]
    
    print("Setting up MeiGen MultiTalk...")
    print(f"Downloading {len(files_to_download)} files...")
    
    success_count = 0
    fail_count = 0
    
    # Download all files
    for file_path in files_to_download:
        url = f"{base_url}/{file_path}"
        if download_file(url, file_path):
            success_count += 1
        else:
            fail_count += 1
    
    # Create empty __init__.py files where needed
    print("\nCreating __init__.py files...")
    for init_file in init_files:
        os.makedirs(os.path.dirname(init_file), exist_ok=True)
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("")
            print(f"✓ Created: {init_file}")
    
    # The vram_management files are already included in the main download list
    
    print(f"\nDownload complete!")
    print(f"✓ Successfully downloaded: {success_count} files")
    if fail_count > 0:
        print(f"✗ Failed downloads: {fail_count} files")
    
    # Install requirements
    print("\nInstalling Python requirements...")
    if os.path.exists("requirements.txt"):
        os.system(f"{sys.executable} -m pip install -r requirements.txt")
        
        # Install additional dependencies found in code but not in requirements.txt
        additional_deps = [
            "einops",
            "soundfile",
            "librosa",
            "Pillow",
            "safetensors",
            "torchvision",
            "loguru"
        ]
        
        print("\nInstalling additional dependencies...")
        for dep in additional_deps:
            os.system(f"{sys.executable} -m pip install {dep}")
    
    print("\nSetup complete!")
    print("\nNote: You still need to download the model weights separately.")
    print("Model weights are not included in the GitHub repository.")

if __name__ == "__main__":
    setup_multitalk()