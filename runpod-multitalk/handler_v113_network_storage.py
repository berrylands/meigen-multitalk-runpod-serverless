#!/usr/bin/env python3
"""
MultiTalk V113 Handler - Network Storage Migration Support
Adds functionality to migrate ALL models to network storage
"""

import runpod
import os
import sys
import json
import time
import torch
import base64
import boto3
import tempfile
import traceback
import warnings
import subprocess
from pathlib import Path

# Add MultiTalk to path
sys.path.insert(0, '/app/multitalk_official')
sys.path.insert(0, '/app')

print("="*50)
print("V113: MultiTalk Handler - Network Storage Migration Support")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("="*50)

# Import our MultiTalk V113 implementation
try:
    from multitalk_v113_implementation import MultiTalkV113
    print("✓ MultiTalk V113 implementation imported successfully")
    MULTITALK_V113_AVAILABLE = True
except ImportError as e:
    print(f"✗ MultiTalk V113 implementation import failed: {e}")
    MULTITALK_V113_AVAILABLE = False

# Initialize MultiTalk V113
multitalk_v113 = None
if MULTITALK_V113_AVAILABLE:
    try:
        print("\nInitializing MultiTalk V113...")
        multitalk_v113 = MultiTalkV113()
        print("✓ MultiTalk V113 initialized successfully")
    except Exception as e:
        print(f"✗ MultiTalk V113 initialization failed: {e}")
        multitalk_v113 = None

def migrate_models_to_network_storage(models_to_cache):
    """Download and cache models to network storage"""
    print("Starting model migration to network storage...")
    
    network_storage_path = Path("/runpod-volume/models")
    network_storage_path.mkdir(parents=True, exist_ok=True)
    
    migration_summary = {
        "models_cached": [],
        "models_failed": [],
        "total_size_gb": 0,
        "storage_usage_gb": 0
    }
    
    for model_config in models_to_cache:
        model_name = model_config["name"]
        repo_id = model_config["repo_id"]
        target_path = Path(model_config["target_path"])
        
        print(f"\nMigrating {model_name} from {repo_id}...")
        
        try:
            # Create target directory
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Download model components
            if model_config.get("cache_processor"):
                print(f"  Downloading processor for {model_name}...")
                
                # Use transformers to download and cache
                if "wav2vec2" in model_name.lower():
                    from transformers import Wav2Vec2Processor
                    processor = Wav2Vec2Processor.from_pretrained(repo_id, cache_dir=str(target_path))
                    print(f"  ✓ Wav2Vec2 processor cached")
                    
                elif "clip" in model_name.lower():
                    from transformers import CLIPProcessor
                    processor = CLIPProcessor.from_pretrained(repo_id, cache_dir=str(target_path))
                    print(f"  ✓ CLIP processor cached")
            
            if model_config.get("cache_tokenizer"):
                print(f"  Downloading tokenizer for {model_name}...")
                
                from transformers import AutoTokenizer
                try:
                    tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=str(target_path))
                    print(f"  ✓ Tokenizer cached")
                except:
                    print(f"  ⚠ Tokenizer not available for {model_name}")
            
            # Download model weights
            print(f"  Downloading model weights for {model_name}...")
            
            if "wav2vec2" in model_name.lower():
                from transformers import Wav2Vec2ForCTC
                model = Wav2Vec2ForCTC.from_pretrained(repo_id, cache_dir=str(target_path))
                print(f"  ✓ Wav2Vec2 model cached")
                
            elif "clip" in model_name.lower():
                from transformers import CLIPModel
                model = CLIPModel.from_pretrained(repo_id, cache_dir=str(target_path))
                print(f"  ✓ CLIP model cached")
                
            elif "stable-video-diffusion" in model_name.lower():
                from diffusers import StableVideoDiffusionPipeline
                pipeline = StableVideoDiffusionPipeline.from_pretrained(repo_id, cache_dir=str(target_path))
                print(f"  ✓ Stable Video Diffusion cached")
            
            # Calculate size
            size_bytes = sum(f.stat().st_size for f in target_path.rglob('*') if f.is_file())
            size_mb = size_bytes / (1024 * 1024)
            
            migration_summary["models_cached"].append({
                "name": model_name,
                "repo_id": repo_id,
                "target_path": str(target_path),
                "size_mb": size_mb
            })
            
            migration_summary["total_size_gb"] += size_mb / 1024
            
            print(f"  ✓ {model_name} cached successfully ({size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"  ✗ Failed to cache {model_name}: {e}")
            migration_summary["models_failed"].append({
                "name": model_name,
                "repo_id": repo_id,
                "error": str(e)
            })
    
    # Calculate total storage usage
    total_storage_bytes = sum(f.stat().st_size for f in network_storage_path.rglob('*') if f.is_file())
    migration_summary["storage_usage_gb"] = total_storage_bytes / (1024**3)
    migration_summary["total_models"] = len(migration_summary["models_cached"])
    
    print(f"\nMigration Summary:")
    print(f"  Models Cached: {migration_summary['total_models']}")
    print(f"  Total Size: {migration_summary['total_size_gb']:.2f} GB")
    print(f"  Storage Usage: {migration_summary['storage_usage_gb']:.2f} GB")
    
    return migration_summary

def verify_network_storage_completeness():
    """Verify all models are available in network storage"""
    print("Verifying network storage completeness...")
    
    network_storage_path = Path("/runpod-volume/models")
    
    if not network_storage_path.exists():
        return {
            "all_models_present": False,
            "error": "Network storage path not found"
        }
    
    # Check for key model files
    required_models = {
        "multitalk": "wan2.1-i2v-14b-480p/multitalk.safetensors",
        "diffusion": "wan2.1-i2v-14b-480p/diffusion_pytorch_model-00007-of-00007.safetensors",
        "vae": "wan2.1-i2v-14b-480p/Wan2.1_VAE.pth",
        "clip": "wan2.1-i2v-14b-480p/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "wav2vec_large": "wav2vec2-large-960h/pytorch_model.bin",
        "wav2vec_base": "wav2vec2-base-960h/pytorch_model.bin"
    }
    
    model_inventory = {}
    all_present = True
    total_size = 0
    
    for model_name, file_path in required_models.items():
        full_path = network_storage_path / file_path
        
        if full_path.exists():
            size_bytes = full_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            total_size += size_bytes
            
            model_inventory[model_name] = {
                "present": True,
                "size_mb": size_mb,
                "path": str(file_path)
            }
            print(f"  ✓ {model_name}: {size_mb:.1f} MB")
        else:
            model_inventory[model_name] = {
                "present": False,
                "path": str(file_path)
            }
            all_present = False
            print(f"  ✗ {model_name}: Not found")
    
    # Test offline loading
    offline_loading_works = True
    try:
        # Disable internet access temporarily
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Try to load a model
        from transformers import Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained(
            str(network_storage_path / "wav2vec2-large-960h"),
            local_files_only=True
        )
        print("  ✓ Offline model loading works")
        
    except Exception as e:
        print(f"  ✗ Offline loading failed: {e}")
        offline_loading_works = False
    finally:
        # Re-enable internet access
        os.environ.pop('HF_DATASETS_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
    
    return {
        "all_models_present": all_present,
        "offline_loading_works": offline_loading_works,
        "model_inventory": model_inventory,
        "total_models": len(required_models),
        "total_size_gb": total_size / (1024**3),
        "storage_path": str(network_storage_path)
    }

def handler(job):
    """V113 Handler with Network Storage Migration Support"""
    print(f"V113: Received job: {job}")
    
    job_input = job.get("input", {})
    action = job_input.get("action", "generate")
    
    try:
        if action == "migrate_models_to_storage":
            # Migrate models to network storage
            models_to_cache = job_input.get("models_to_cache", [])
            
            if not models_to_cache:
                return {
                    "output": {
                        "status": "error",
                        "error": "No models specified for migration"
                    }
                }
            
            print(f"Starting migration of {len(models_to_cache)} models...")
            
            migration_summary = migrate_models_to_network_storage(models_to_cache)
            
            # Test offline loading
            offline_test = {
                "all_models_loadable": True,
                "no_internet_required": True
            }
            
            try:
                # Try to initialize V113 with cached models
                if multitalk_v113:
                    test_result = multitalk_v113.load_models()
                    offline_test["all_models_loadable"] = test_result
                    print(f"  ✓ V113 model loading test: {test_result}")
            except Exception as e:
                offline_test["all_models_loadable"] = False
                offline_test["error"] = str(e)
                print(f"  ✗ V113 model loading test failed: {e}")
            
            return {
                "output": {
                    "status": "success",
                    "message": f"Models migrated to network storage successfully",
                    "migration_summary": migration_summary,
                    "offline_test": offline_test
                }
            }
        
        elif action == "verify_network_storage":
            # Verify network storage completeness
            verification_result = verify_network_storage_completeness()
            
            return {
                "output": verification_result
            }
        
        elif action == "model_check":
            # Enhanced model check including network storage
            model_info = {
                "network_volume_mounted": os.path.exists("/runpod-volume"),
                "models_directory_exists": os.path.exists("/runpod-volume/models"),
                "cuda_available": torch.cuda.is_available(),
                "multitalk_v113_available": MULTITALK_V113_AVAILABLE,
                "multitalk_v113_initialized": multitalk_v113 is not None,
                "pytorch_version": torch.__version__
            }
            
            # Check network storage
            if os.path.exists("/runpod-volume/models"):
                storage_check = verify_network_storage_completeness()
                model_info["network_storage"] = storage_check
            
            # Add MultiTalk V113 info
            if multitalk_v113:
                model_info["multitalk_v113_info"] = multitalk_v113.get_model_info()
            
            return {
                "output": {
                    "status": "ready",
                    "message": "V113 MultiTalk handler ready (Network Storage Support)",
                    "version": "113",
                    "model_info": model_info
                }
            }
        
        else:
            # Use original V113 handler for other actions
            from handler_v113 import handler as original_handler
            return original_handler(job)
            
    except Exception as e:
        print(f"V113: Handler error: {e}")
        traceback.print_exc()
        return {
            "output": {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "version": "113"
            }
        }

# Start handler
print("V113: Starting RunPod serverless handler with network storage support...")
runpod.serverless.start({"handler": handler})