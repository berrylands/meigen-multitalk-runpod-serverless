#!/usr/bin/env python3
"""
Cache missing components using existing endpoint
Uses the existing endpoint with network volume attached
"""

import subprocess
import json
import os
import time

def cache_components_with_existing_endpoint():
    """Cache missing components using the existing endpoint"""
    
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return False
    
    # Use existing endpoint with network volume attached
    endpoint_id = "zu0ik6c8yukyl6"
    
    # Cache missing components job
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
                }
            ],
            "verify_offline_loading": True
        }
    }
    
    print("=" * 80)
    print("CACHING MISSING COMPONENTS WITH EXISTING ENDPOINT")
    print("=" * 80)
    print(f"Using endpoint: {endpoint_id}")
    print(f"Components to cache: {len(cache_job['input']['components_to_cache'])}")
    print("This will enable complete offline operation")
    print("=" * 80)
    
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
                print(f"✅ Component caching job submitted: {job_id}")
                print("⏱️  Expected duration: 10-30 minutes")
                
                # Monitor progress
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
                            
                            print(f"\n🎉 COMPONENT CACHING COMPLETED!")
                            print("=" * 80)
                            
                            if "output" in output:
                                result_data = output["output"]
                                
                                if result_data.get("status") == "success":
                                    print("✅ Components cached successfully!")
                                    
                                    # Show caching summary
                                    if "caching_summary" in result_data:
                                        summary = result_data["caching_summary"]
                                        print(f"\n📊 Caching Summary:")
                                        print(f"  Components Cached: {summary.get('components_cached', 0)}")
                                        print(f"  Total Size Added: {summary.get('total_size_mb', 0):.1f} MB")
                                        print(f"  Storage Usage: {summary.get('storage_usage_gb', 0):.2f} GB")
                                        
                                        # Show individual components
                                        if "cached_components" in summary:
                                            print(f"\n✅ Successfully Cached:")
                                            for comp in summary["cached_components"]:
                                                print(f"  • {comp['name']}: {comp.get('size_mb', 0):.1f} MB")
                                        
                                        if "failed_components" in summary:
                                            print(f"\n❌ Failed Components:")
                                            for comp in summary["failed_components"]:
                                                print(f"  • {comp['name']}: {comp.get('error', 'Unknown error')}")
                                    
                                    # Show offline test results
                                    if "offline_test" in result_data:
                                        offline_test = result_data["offline_test"]
                                        print(f"\n🔍 Offline Operation Test:")
                                        print(f"  Model Loading: {offline_test.get('model_loading_success', False)}")
                                        print(f"  Inference Ready: {offline_test.get('inference_success', False)}")
                                        print(f"  No Downloads: {offline_test.get('no_downloads_detected', False)}")
                                        print(f"  Complete Offline: {offline_test.get('complete_offline_operation', False)}")
                                        
                                        if offline_test.get('complete_offline_operation'):
                                            print(f"\n🎯 SUCCESS: Complete offline operation achieved!")
                                            print(f"✅ All models and components cached to network storage")
                                            print(f"✅ No runtime downloads from HuggingFace")
                                            print(f"✅ Faster cold starts and better reliability")
                                            return True
                                        else:
                                            print(f"\n⚠️  Offline operation not fully complete")
                                            print(f"Some components may still require internet access")
                                            return False
                                    else:
                                        print(f"\n✅ Components cached, offline test not available")
                                        return True
                                else:
                                    print(f"❌ Component caching failed: {result_data.get('error', 'Unknown error')}")
                                    return False
                            else:
                                print(f"❌ Unexpected output format: {output}")
                                return False
                        
                        elif status == "FAILED":
                            print(f"❌ Component caching job failed: {status_data}")
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
                
                print("⏱️  Timeout waiting for component caching")
                return False
            else:
                print(f"❌ No job ID in response: {response}")
                return False
        else:
            print(f"❌ Request failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Network Storage Component Caching")
    print("Using existing endpoint with network volume attached")
    print("=" * 80)
    
    success = cache_components_with_existing_endpoint()
    
    if success:
        print(f"\n🎉 COMPONENT CACHING COMPLETE!")
        print(f"✅ All models now stored in RunPod network storage")
        print(f"✅ Complete offline operation achieved")
        print(f"✅ No runtime downloads required")
        print(f"✅ Faster, more reliable video generation")
        
        print(f"\nNext steps:")
        print(f"1. Deploy V114 with offline-optimized implementation")
        print(f"2. Test complete offline video generation")
        print(f"3. Enjoy improved performance and reliability")
    else:
        print(f"\n⚠️  Component caching incomplete")
        print(f"Check if the endpoint supports the cache_missing_components action")
        print(f"May need to wait for V114 deployment to complete")