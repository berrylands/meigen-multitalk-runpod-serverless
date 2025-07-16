#!/usr/bin/env python3
"""
Cache Missing Components to Network Storage
Adds processors, tokenizers, and pipeline components for complete offline operation
"""

import subprocess
import json
import os
import time

def cache_missing_components():
    """Submit job to cache missing processors and tokenizers"""
    
    api_key = os.getenv("RUNPOD_API_KEY")
    endpoint_id = "zu0ik6c8yukyl6"  # Original endpoint with network storage
    
    # Job to cache missing components
    cache_job = {
        "input": {
            "action": "cache_missing_components",
            "components_to_cache": [
                {
                    "name": "wav2vec2-processor",
                    "repo_id": "facebook/wav2vec2-large-960h",
                    "cache_path": "/runpod-volume/models/wav2vec2-large-960h",
                    "component_type": "processor"
                },
                {
                    "name": "wav2vec2-tokenizer", 
                    "repo_id": "facebook/wav2vec2-large-960h",
                    "cache_path": "/runpod-volume/models/wav2vec2-large-960h",
                    "component_type": "tokenizer"
                },
                {
                    "name": "clip-processor",
                    "repo_id": "openai/clip-vit-large-patch14",
                    "cache_path": "/runpod-volume/models/clip-components",
                    "component_type": "processor"
                },
                {
                    "name": "clip-tokenizer",
                    "repo_id": "openai/clip-vit-large-patch14", 
                    "cache_path": "/runpod-volume/models/clip-components",
                    "component_type": "tokenizer"
                },
                {
                    "name": "diffusers-vae",
                    "repo_id": "stabilityai/sd-vae-ft-mse",
                    "cache_path": "/runpod-volume/models/diffusers-cache",
                    "component_type": "vae"
                },
                {
                    "name": "diffusers-scheduler",
                    "repo_id": "stabilityai/stable-diffusion-2-1",
                    "cache_path": "/runpod-volume/models/diffusers-cache",
                    "component_type": "scheduler"
                }
            ],
            "verify_offline_loading": True,
            "create_loading_manifest": True
        }
    }
    
    print("=" * 70)
    print("CACHING MISSING COMPONENTS FOR OFFLINE OPERATION")
    print("=" * 70)
    print(f"Components to cache: {len(cache_job['input']['components_to_cache'])}")
    print("This will enable complete offline operation with no runtime downloads")
    print("=" * 70)
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    curl_cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(cache_job)
    ]
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                print(f"✅ Caching job submitted: {job_id}")
                print("⏱️  Expected duration: 10-20 minutes")
                
                # Poll for results with extended timeout
                for i in range(1800):  # 30 minutes
                    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
                    
                    status_cmd = [
                        "curl", "-H", f"Authorization: Bearer {api_key}",
                        status_url
                    ]
                    
                    status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                    
                    if status_result.returncode == 0:
                        status_data = json.loads(status_result.stdout)
                        status = status_data.get("status")
                        
                        if i % 60 == 0:  # Print every minute
                            print(f"[{i//60}min] Status: {status}")
                        
                        if status == "COMPLETED":
                            output = status_data.get("output", {})
                            
                            print(f"\n🎉 CACHING COMPLETED!")
                            print("=" * 70)
                            
                            if "output" in output:
                                result_data = output["output"]
                                
                                if result_data.get("status") == "success":
                                    print("✅ All components cached successfully!")
                                    
                                    # Show caching summary
                                    if "caching_summary" in result_data:
                                        summary = result_data["caching_summary"]
                                        print(f"\nCaching Summary:")
                                        print(f"  Components Cached: {summary.get('components_cached', 0)}")
                                        print(f"  Total Size Added: {summary.get('total_size_mb', 0):.1f} MB")
                                        print(f"  New Storage Usage: {summary.get('storage_usage_gb', 0):.2f} GB")
                                        
                                        # Show per-component results
                                        if "cached_components" in summary:
                                            print(f"\nCached Components:")
                                            for comp in summary["cached_components"]:
                                                print(f"  ✅ {comp['name']}: {comp.get('size_mb', 0):.1f} MB")
                                        
                                        if "failed_components" in summary:
                                            print(f"\nFailed Components:")
                                            for comp in summary["failed_components"]:
                                                print(f"  ❌ {comp['name']}: {comp.get('error', 'Unknown error')}")
                                    
                                    # Show offline test results
                                    if "offline_test" in result_data:
                                        offline_test = result_data["offline_test"]
                                        print(f"\n📊 Offline Operation Test:")
                                        print(f"  All Components Load: {offline_test.get('all_components_load', False)}")
                                        print(f"  No Internet Required: {offline_test.get('no_internet_required', False)}")
                                        print(f"  Inference Ready: {offline_test.get('inference_ready', False)}")
                                        
                                        if offline_test.get('no_internet_required'):
                                            print(f"\n🎯 SUCCESS: Complete offline operation achieved!")
                                            print(f"✅ No runtime downloads from HuggingFace")
                                            print(f"✅ Faster cold starts")
                                            print(f"✅ More reliable operation")
                                        else:
                                            print(f"\n⚠️  Some components still require internet access")
                                    
                                    return True
                                else:
                                    print(f"❌ Caching failed: {result_data.get('error', 'Unknown error')}")
                                    if "details" in result_data:
                                        print(f"Details: {json.dumps(result_data['details'], indent=2)}")
                                    return False
                            else:
                                print(f"Unexpected output format: {output}")
                                return False
                        
                        elif status == "FAILED":
                            print(f"❌ Caching job failed: {status_data}")
                            return False
                        
                        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                            time.sleep(1)
                            continue
                        else:
                            print(f"⚠️  Unknown status: {status}")
                            return False
                    else:
                        print(f"❌ Status check failed: {status_result.stderr}")
                        return False
                
                print("⏱️  Timeout waiting for caching completion")
                return False
            else:
                print(f"❌ No job ID in response: {response}")
                return False
        else:
            print(f"❌ Curl failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def verify_complete_offline_operation():
    """Verify system can operate completely offline"""
    
    print(f"\n" + "=" * 70)
    print("VERIFYING COMPLETE OFFLINE OPERATION")
    print("=" * 70)
    
    api_key = os.getenv("RUNPOD_API_KEY")
    endpoint_id = "zu0ik6c8yukyl6"
    
    # Test offline operation
    offline_test = {
        "input": {
            "action": "test_offline_operation",
            "disable_internet": True,
            "test_model_loading": True,
            "test_inference": True
        }
    }
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    curl_cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(offline_test)
    ]
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                print(f"Offline test job: {job_id}")
                
                # Wait for completion
                for i in range(300):  # 5 minutes
                    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
                    
                    status_cmd = [
                        "curl", "-H", f"Authorization: Bearer {api_key}",
                        status_url
                    ]
                    
                    status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                    
                    if status_result.returncode == 0:
                        status_data = json.loads(status_result.stdout)
                        status = status_data.get("status")
                        
                        if i % 30 == 0:
                            print(f"[{i//30 * 30}s] Status: {status}")
                        
                        if status == "COMPLETED":
                            output = status_data.get("output", {})
                            
                            if "output" in output:
                                result_data = output["output"]
                                
                                print(f"\n📊 Offline Operation Test Results:")
                                print(f"  Model Loading: {result_data.get('model_loading_success', False)}")
                                print(f"  Inference Works: {result_data.get('inference_success', False)}")
                                print(f"  No Downloads: {result_data.get('no_downloads_detected', False)}")
                                print(f"  Complete Offline: {result_data.get('complete_offline_operation', False)}")
                                
                                if result_data.get('complete_offline_operation'):
                                    print(f"\n🎉 COMPLETE OFFLINE OPERATION ACHIEVED!")
                                    print(f"✅ All models load from network storage")
                                    print(f"✅ No internet connectivity required")
                                    print(f"✅ Ready for production deployment")
                                    return True
                                else:
                                    print(f"\n⚠️  Offline operation not complete")
                                    print(f"Some components still require internet access")
                                    return False
                            else:
                                print(f"Unexpected test output: {output}")
                                return False
                        
                        elif status == "FAILED":
                            print(f"❌ Offline test failed: {status_data}")
                            return False
                        
                        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                            time.sleep(1)
                            continue
                        else:
                            print(f"⚠️  Unknown status: {status}")
                            return False
                    else:
                        print(f"❌ Status check failed: {status_result.stderr}")
                        return False
                
                print("⏱️  Timeout waiting for offline test")
                return False
            else:
                print(f"❌ No job ID in response: {response}")
                return False
        else:
            print(f"❌ Curl failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Network Storage Component Caching")
    print("Enabling complete offline operation for MultiTalk V113")
    print("=" * 70)
    
    # Step 1: Cache missing components
    print("\n🔄 Step 1: Caching missing processors and tokenizers...")
    cache_success = cache_missing_components()
    
    if cache_success:
        print("\n✅ Component caching completed!")
        
        # Step 2: Verify offline operation
        print("\n🔍 Step 2: Verifying complete offline operation...")
        offline_success = verify_complete_offline_operation()
        
        if offline_success:
            print(f"\n🎯 MISSION ACCOMPLISHED!")
            print(f"✅ All models stored in RunPod network storage")
            print(f"✅ Complete offline operation achieved")
            print(f"✅ No runtime downloads from HuggingFace")
            print(f"✅ Faster, more reliable operation")
            
            print(f"\nNext steps:")
            print(f"1. Create V114 with offline-optimized implementation")
            print(f"2. Deploy and test performance improvements")
            print(f"3. Update documentation")
        else:
            print(f"\n⚠️  Offline operation verification failed")
            print(f"Some components may still require internet access")
    else:
        print(f"\n❌ Component caching failed")
        print(f"Check error logs and retry if needed")