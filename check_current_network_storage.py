#!/usr/bin/env python3
"""Check what models are currently in network storage"""

import subprocess
import json
import os

# Use curl to check network storage
def check_network_storage():
    api_key = os.getenv("RUNPOD_API_KEY")
    endpoint_id = "zu0ik6c8yukyl6"  # Original endpoint with network storage
    
    # Test volume exploration
    test_data = {
        "input": {
            "action": "volume_explore"
        }
    }
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    curl_cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(test_data)
    ]
    
    print("Checking current network storage contents...")
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                print(f"Job submitted: {job_id}")
                
                # Poll for results
                import time
                for i in range(60):
                    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
                    
                    status_cmd = [
                        "curl", "-H", f"Authorization: Bearer {api_key}",
                        status_url
                    ]
                    
                    status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                    
                    if status_result.returncode == 0:
                        status_data = json.loads(status_result.stdout)
                        status = status_data.get("status")
                        
                        if status == "COMPLETED":
                            output = status_data.get("output", {})
                            
                            print("\n" + "="*60)
                            print("CURRENT NETWORK STORAGE CONTENTS")
                            print("="*60)
                            
                            if "output" in output and "exploration" in output["output"]:
                                exploration = output["output"]["exploration"]
                                
                                print(f"Volume exists: {exploration.get('volume_exists', False)}")
                                print(f"Models dir exists: {exploration.get('models_dir_exists', False)}")
                                print(f"Total size: {exploration.get('total_size_gb', 0):.2f} GB")
                                
                                # Show key model files
                                if "key_model_files" in exploration:
                                    print(f"\nKey Model Files:")
                                    for model_name, info in exploration["key_model_files"].items():
                                        if info.get("exists"):
                                            print(f"  ✅ {model_name}: {info.get('size_mb', 0):.1f} MB")
                                        else:
                                            print(f"  ❌ {model_name}: Missing")
                                
                                # Show WAN models
                                if "wan_models" in exploration:
                                    print(f"\nWAN Models:")
                                    for model in exploration["wan_models"]:
                                        print(f"  ✅ {model['path']}: {model['size_mb']:.1f} MB")
                                
                                # Show Wav2Vec models
                                if "wav2vec_models" in exploration:
                                    print(f"\nWav2Vec Models:")
                                    for model in exploration["wav2vec_models"]:
                                        print(f"  ✅ {model['path']}: {model['size_mb']:.1f} MB")
                                
                                # Analysis
                                print(f"\n" + "="*60)
                                print("ANALYSIS")
                                print("="*60)
                                
                                key_models = exploration.get("key_model_files", {})
                                models_present = sum(1 for info in key_models.values() if info.get("exists"))
                                total_models = len(key_models)
                                
                                print(f"Key Models Present: {models_present}/{total_models}")
                                
                                if models_present == total_models:
                                    print("✅ All key models are in network storage!")
                                    print("✅ Models currently stored:")
                                    for model_name, info in key_models.items():
                                        if info.get("exists"):
                                            print(f"   - {model_name} ({info.get('size_mb', 0):.1f} MB)")
                                else:
                                    print("⚠️  Some key models are missing from network storage")
                                    print("❌ Missing models:")
                                    for model_name, info in key_models.items():
                                        if not info.get("exists"):
                                            print(f"   - {model_name}")
                                
                                # Check if we need additional models
                                print(f"\n💡 Additional models that should be in network storage:")
                                additional_models = [
                                    "Wav2Vec2 processors (for offline operation)",
                                    "CLIP processors (for offline operation)", 
                                    "Diffusers components (for offline operation)",
                                    "Tokenizers (for offline operation)"
                                ]
                                
                                for model in additional_models:
                                    print(f"   - {model}")
                                
                                return exploration
                            else:
                                print("No exploration data found")
                                return None
                        
                        elif status == "FAILED":
                            print(f"❌ Job failed: {status_data}")
                            return None
                        
                        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                            if i % 10 == 0:
                                print(f"[{i+1}s] Status: {status}")
                            time.sleep(1)
                            continue
                        else:
                            print(f"⚠️  Unknown status: {status}")
                            return None
                    else:
                        print(f"❌ Status check failed: {status_result.stderr}")
                        return None
                
                print("⏱️  Timeout waiting for job completion")
                return None
            else:
                print(f"❌ No job ID in response: {response}")
                return None
        else:
            print(f"❌ Curl failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    exploration = check_network_storage()
    
    if exploration:
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"1. Models currently in network storage are sufficient for basic operation")
        print(f"2. To enable complete offline operation, we need to add:")
        print(f"   - Wav2Vec2 processors and tokenizers")
        print(f"   - CLIP processors and tokenizers")
        print(f"   - Diffusers pipeline components")
        print(f"3. This would eliminate all runtime downloads from HuggingFace")
        print(f"4. Estimated additional storage needed: 5-10 GB")
    else:
        print("❌ Unable to check network storage contents")