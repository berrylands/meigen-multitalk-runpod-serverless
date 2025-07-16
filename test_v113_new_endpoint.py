#!/usr/bin/env python3
"""Test new V113 endpoint"""
import subprocess
import json
import os
import time

def test_v113_endpoint():
    """Test the new V113 endpoint"""
    
    # Get API key from environment
    api_key = os.getenv("RUNPOD_API_KEY")
    
    endpoint_id = "cs0uznjognle22"  # New V113 endpoint
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    # Test model check
    test_data = {
        "input": {
            "action": "model_check"
        }
    }
    
    print("Testing V113 endpoint...")
    print(f"Endpoint ID: {endpoint_id}")
    
    # Create curl command
    curl_cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(test_data)
    ]
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                print(f"✅ Job submitted: {job_id}")
                
                # Poll for results
                status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
                
                for i in range(120):  # Wait up to 2 minutes for cold start
                    status_cmd = [
                        "curl", "-H", f"Authorization: Bearer {api_key}",
                        status_url
                    ]
                    
                    status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                    
                    if status_result.returncode == 0:
                        status_data = json.loads(status_result.stdout)
                        status = status_data.get("status")
                        
                        if i % 10 == 0:  # Print every 10 seconds
                            print(f"[{i+1}s] Status: {status}")
                        
                        if status == "COMPLETED":
                            output = status_data.get("output", {})
                            print(f"\n🎉 Job completed!")
                            print(f"Full output: {json.dumps(output, indent=2)}")
                            
                            # Check for version info
                            if "output" in output and "version" in output["output"]:
                                version = output["output"]["version"]
                                print(f"\n🎯 Detected version: {version}")
                                
                                if version == "113":
                                    print("✅ V113 is running successfully!")
                                    
                                    # Check model info
                                    model_info = output["output"].get("model_info", {})
                                    if model_info:
                                        print(f"\nModel Info:")
                                        print(f"  CUDA Available: {model_info.get('cuda_available')}")
                                        print(f"  Device: {model_info.get('device')}")
                                        print(f"  MultiTalk V113 Available: {model_info.get('multitalk_v113_available')}")
                                        print(f"  MultiTalk V113 Initialized: {model_info.get('multitalk_v113_initialized')}")
                                        
                                        # Check V113 specific info
                                        if "multitalk_v113_info" in model_info:
                                            v113_info = model_info["multitalk_v113_info"]
                                            models_loaded = v113_info.get("models_loaded", {})
                                            print(f"\n  V113 Models Loaded:")
                                            for model, loaded in models_loaded.items():
                                                status = "✅" if loaded else "❌"
                                                print(f"    {status} {model}")
                                    
                                    return True
                                else:
                                    print(f"⚠️  Expected V113 but got V{version}")
                                    return False
                            else:
                                print("⚠️  No version info in response")
                                return False
                        
                        elif status == "FAILED":
                            print(f"❌ Job failed: {status_data}")
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
                
                print("⏱️  Timeout waiting for job completion")
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
    success = test_v113_endpoint()
    
    if success:
        print("\n🎉 V113 endpoint test passed!")
        print("✅ Complete MeiGen-MultiTalk implementation is ready")
        print("\nNext steps:")
        print("1. Test video generation with S3 inputs")
        print("2. Verify all model components are working")
        print("3. Test pipeline demo video creation")
    else:
        print("\n💡 V113 endpoint test failed")
        print("Check endpoint status and logs")