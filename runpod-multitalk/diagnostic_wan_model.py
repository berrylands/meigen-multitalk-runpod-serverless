#!/usr/bin/env python3
"""
Diagnostic script to examine Wan2.1 model structure on RunPod
This will help us understand the exact file structure to fix validation
"""

import os
import json
from pathlib import Path

def examine_model_structure():
    """Examine the actual structure of the Wan2.1 model on RunPod."""
    model_base = Path("/runpod-volume/models")
    wan_path = model_base / "wan2.1-i2v-14b-480p"
    
    print("=" * 80)
    print("DIAGNOSTIC: Examining Wan2.1 Model Structure")
    print("=" * 80)
    
    if not wan_path.exists():
        print(f"❌ Wan2.1 path does not exist: {wan_path}")
        return
    
    print(f"✅ Wan2.1 path exists: {wan_path}")
    
    # Recursively examine all files and directories
    def examine_directory(path, level=0):
        indent = "  " * level
        try:
            items = list(path.iterdir())
            print(f"{indent}{path.name}/ ({len(items)} items)")
            
            for item in sorted(items):
                if item.is_dir():
                    if level < 3:  # Limit depth to avoid too much output
                        examine_directory(item, level + 1)
                    else:
                        sub_items = len(list(item.iterdir())) if item.exists() else 0
                        print(f"{indent}  {item.name}/ ({sub_items} items)")
                else:
                    size = item.stat().st_size if item.exists() else 0
                    size_mb = size / (1024 * 1024)
                    if size_mb > 1:
                        print(f"{indent}  {item.name} ({size_mb:.1f}MB)")
                    else:
                        print(f"{indent}  {item.name} ({size} bytes)")
        except Exception as e:
            print(f"{indent}❌ Error examining {path}: {e}")
    
    examine_directory(wan_path)
    
    # Look for specific file patterns that might indicate model format
    print("\n" + "=" * 80)
    print("CHECKING FOR COMMON MODEL FILE PATTERNS")
    print("=" * 80)
    
    patterns_to_check = [
        "*.json",
        "*.safetensors", 
        "*.bin",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*model*",
        "*config*",
        "*index*"
    ]
    
    for pattern in patterns_to_check:
        matches = list(wan_path.rglob(pattern))
        if matches:
            print(f"\n📁 Files matching '{pattern}':")
            for match in sorted(matches)[:10]:  # Limit to first 10 matches
                rel_path = match.relative_to(wan_path)
                size = match.stat().st_size if match.exists() else 0
                size_mb = size / (1024 * 1024)
                if size_mb > 1:
                    print(f"  {rel_path} ({size_mb:.1f}MB)")
                else:
                    print(f"  {rel_path} ({size} bytes)")
    
    # Check for specific expected files from different model formats
    print("\n" + "=" * 80)
    print("CHECKING FOR EXPECTED MODEL FORMAT FILES")
    print("=" * 80)
    
    expected_files = [
        # Diffusers format
        "model_index.json",
        "scheduler/scheduler_config.json", 
        "vae/config.json",
        "unet/config.json",
        "text_encoder/config.json",
        
        # SafeTensors format
        "model.safetensors",
        "model.safetensors.index.json",
        
        # PyTorch format
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        
        # Config files
        "config.json",
        "generation_config.json"
    ]
    
    found_files = []
    for expected_file in expected_files:
        file_path = wan_path / expected_file
        if file_path.exists():
            size = file_path.stat().st_size
            size_mb = size / (1024 * 1024)
            found_files.append(expected_file)
            if size_mb > 1:
                print(f"✅ {expected_file} ({size_mb:.1f}MB)")
            else:
                print(f"✅ {expected_file} ({size} bytes)")
    
    if not found_files:
        print("❌ No expected model format files found")
    
    # Try to read any JSON config files to understand the model format
    print("\n" + "=" * 80) 
    print("EXAMINING JSON CONFIG FILES")
    print("=" * 80)
    
    json_files = list(wan_path.rglob("*.json"))
    for json_file in json_files[:5]:  # Limit to first 5 JSON files
        try:
            rel_path = json_file.relative_to(wan_path)
            print(f"\n📄 {rel_path}:")
            with open(json_file, 'r') as f:
                config = json.load(f)
            
            # Print key information
            for key in ['_class_name', 'model_type', '_diffusers_version', 'architectures']:
                if key in config:
                    print(f"  {key}: {config[key]}")
            
            if len(config) < 20:  # If it's a small config, print it all
                print(f"  Content: {json.dumps(config, indent=2)[:500]}...")
                
        except Exception as e:
            print(f"  ❌ Error reading {rel_path}: {e}")

if __name__ == "__main__":
    examine_model_structure()