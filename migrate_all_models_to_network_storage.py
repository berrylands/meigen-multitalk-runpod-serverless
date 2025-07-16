#!/usr/bin/env python3
"""
Migrate ALL models to RunPod network storage
Ensures complete offline operation with no runtime downloads
"""

import os
import time
import json
import subprocess
import sys

# Install dependencies if needed
subprocess.run([sys.executable, "-m", "pip", "install", "runpod", "python-dotenv"], capture_output=True)

import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "cs0uznjognle22"  # V113 endpoint with network storage

def migrate_models_to_network_storage():
    """Download and cache all models to network storage"""
    print("=" * 80)
    print("MIGRATING ALL MODELS TO RUNPOD NETWORK STORAGE")
    print("=" * 80)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Complete model migration job
    migration_job = {
        "action": "migrate_models_to_storage",
        "models_to_cache": [
            {
                "name": "wav2vec2-large-960h",
                "repo_id": "facebook/wav2vec2-large-960h",
                "target_path": "/runpod-volume/models/wav2vec2-large-960h",
                "cache_processor": True,
                "cache_tokenizer": True
            },
            {
                "name": "wav2vec2-base-960h", 
                "repo_id": "facebook/wav2vec2-base-960h",
                "target_path": "/runpod-volume/models/wav2vec2-base-960h",
                "cache_processor": True,
                "cache_tokenizer": True
            },
            {
                "name": "clip-vit-large-patch14",
                "repo_id": "openai/clip-vit-large-patch14", 
                "target_path": "/runpod-volume/models/clip-vit-large-patch14",
                "cache_processor": True,
                "cache_tokenizer": True
            },
            {
                "name": "stable-video-diffusion-img2vid-xt",
                "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt",
                "target_path": "/runpod-volume/models/stable-video-diffusion",
                "cache_processor": True,
                "cache_tokenizer": True
            },
            {
                "name": "diffusers-cache",
                "repo_id": "diffusers",
                "target_path": "/runpod-volume/models/diffusers-cache",
                "cache_all_components": True
            }
        ],
        "verify_existing_models": True,
        "create_model_manifest": True,
        "test_offline_loading": True
    }
    
    print(f"Submitting model migration job...")
    print(f"This will download and cache ALL models to network storage")
    print(f"Expected duration: 30-60 minutes for complete migration")
    
    try:
        job = endpoint.run(migration_job)
        print(f"\nMigration job submitted: {job.job_id}")
        
        # Monitor with extended timeout for large downloads
        start_time = time.time()
        last_status = None
        last_update = time.time()
        
        while True:
            try:
                status = job.status()
                current_time = time.time()
                elapsed = current_time - start_time
                
                if status != last_status or (current_time - last_update) > 300:  # Update every 5 minutes
                    print(f"[{elapsed/60:.1f}min] Status: {status}")
                    last_status = status
                    last_update = current_time
                    
                    # Try to get partial results
                    if status == "IN_PROGRESS":
                        try:
                            partial_output = job.output()
                            if partial_output and "progress" in str(partial_output):
                                print(f"  Progress: {partial_output}")
                        except:
                            pass
                
                if status not in ["IN_QUEUE", "IN_PROGRESS"]:
                    break
                    
                if elapsed > 7200:  # 2 hour timeout
                    print("Migration taking longer than expected...")
                    print("Check RunPod dashboard for detailed progress")
                    break
                    
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Status check error: {e}")
                time.sleep(60)
                continue
        
        # Get final results
        final_status = job.status()
        print(f"\nFinal Status: {final_status}")
        
        if final_status == "COMPLETED":
            try:
                output = job.output()
                
                if isinstance(output, dict) and "output" in output:
                    result = output["output"]
                    
                    if result.get("status") == "success":
                        print(f"\n✅ Model migration successful!")
                        
                        # Show migration summary
                        if "migration_summary" in result:
                            summary = result["migration_summary"]
                            print(f"\nMigration Summary:")
                            print(f"  Total Models Cached: {summary.get('total_models', 0)}")
                            print(f"  Total Size: {summary.get('total_size_gb', 0):.2f} GB")
                            print(f"  Network Storage Usage: {summary.get('storage_usage_gb', 0):.2f} GB")
                            
                            # Show per-model results
                            if "models_cached" in summary:
                                print(f"\nModels Successfully Cached:")
                                for model in summary["models_cached"]:
                                    print(f"  ✅ {model['name']} ({model.get('size_mb', 0):.1f} MB)")
                            
                            if "models_failed" in summary:
                                print(f"\nFailed Models:")
                                for model in summary["models_failed"]:
                                    print(f"  ❌ {model['name']}: {model.get('error', 'Unknown error')}")
                        
                        # Show offline test results
                        if "offline_test" in result:
                            offline_test = result["offline_test"]
                            print(f"\nOffline Loading Test:")
                            print(f"  All Models Loadable: {offline_test.get('all_models_loadable', False)}")
                            print(f"  No Internet Required: {offline_test.get('no_internet_required', False)}")
                        
                        return True
                    else:
                        print(f"\n❌ Migration failed: {result.get('error', 'Unknown error')}")
                        return False
                else:
                    print(f"\nUnexpected output format: {output}")
                    return False
                    
            except Exception as e:
                print(f"Error getting results: {e}")
                return False
        else:
            print(f"\n❌ Migration job failed: {final_status}")
            try:
                error_output = job.output()
                print(f"Error details: {error_output}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"\nMigration error: {e}")
        return False

def verify_network_storage_completeness():
    """Verify all models are in network storage"""
    print("\n" + "=" * 80)
    print("VERIFYING NETWORK STORAGE COMPLETENESS")
    print("=" * 80)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    verification_job = {
        "action": "verify_network_storage",
        "check_models": [
            "wav2vec2-large-960h",
            "wav2vec2-base-960h", 
            "clip-vit-large-patch14",
            "stable-video-diffusion",
            "wan2.1-i2v-14b-480p",
            "multitalk.safetensors"
        ],
        "test_offline_loading": True,
        "generate_manifest": True
    }
    
    try:
        job = endpoint.run(verification_job)
        print(f"Verification job: {job.job_id}")
        
        # Wait for completion
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            print(f"Status: {job.status()}")
            time.sleep(10)
        
        if job.status() == "COMPLETED":
            output = job.output()
            
            if isinstance(output, dict) and "output" in output:
                result = output["output"]
                
                print(f"\n📊 Network Storage Verification:")
                print(f"  Storage Path: {result.get('storage_path', '/runpod-volume/models')}")
                print(f"  Total Models: {result.get('total_models', 0)}")
                print(f"  Total Size: {result.get('total_size_gb', 0):.2f} GB")
                print(f"  All Models Present: {result.get('all_models_present', False)}")
                print(f"  Offline Loading Works: {result.get('offline_loading_works', False)}")
                
                # Show model inventory
                if "model_inventory" in result:
                    inventory = result["model_inventory"]
                    print(f"\nModel Inventory:")
                    for model_name, info in inventory.items():
                        status = "✅" if info.get("present", False) else "❌"
                        size = info.get("size_mb", 0)
                        print(f"  {status} {model_name}: {size:.1f} MB")
                
                return result.get("all_models_present", False)
            else:
                print(f"Unexpected verification output: {output}")
                return False
        else:
            print(f"❌ Verification failed: {job.status()}")
            return False
            
    except Exception as e:
        print(f"Verification error: {e}")
        return False

def main():
    print("Model Migration to RunPod Network Storage")
    print("Ensures complete offline operation with no runtime downloads")
    print("=" * 80)
    
    # Step 1: Migrate all models to network storage
    print("\n🔄 Step 1: Migrating models to network storage...")
    migration_success = migrate_models_to_network_storage()
    
    if migration_success:
        print("\n✅ Migration completed successfully!")
        
        # Step 2: Verify completeness
        print("\n🔍 Step 2: Verifying storage completeness...")
        verification_success = verify_network_storage_completeness()
        
        if verification_success:
            print("\n🎉 SUCCESS: All models are now stored in network storage!")
            print("✅ V113 will operate completely offline")
            print("✅ No runtime downloads required")
            print("✅ Faster cold starts")
            print("✅ More reliable operation")
            
            print("\nNext steps:")
            print("1. Update V113 implementation to use cached models")
            print("2. Test offline operation")
            print("3. Deploy V114 with network storage optimization")
        else:
            print("\n⚠️  Some models may be missing from network storage")
            print("Check verification results and retry if needed")
    else:
        print("\n❌ Migration failed")
        print("Check error logs and retry")

if __name__ == "__main__":
    main()