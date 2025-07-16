#!/usr/bin/env python3
"""
Test Current Endpoint
Check what version is currently running on the endpoint
"""

import subprocess
import json
import os
import time

def test_current_endpoint():
    """Test the current endpoint to see what version is running"""
    
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        print("Please set it with: export RUNPOD_API_KEY=your_api_key")
        return False
    
    endpoint_id = "zu0ik6c8yukyl6"
    
    print("🔍 Testing current endpoint...")
    print(f"📡 Endpoint ID: {endpoint_id}")
    
    # Test with health check
    health_job = {
        "input": {
            "health_check": True
        }
    }
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    curl_cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(health_job)
    ]
    
    try:
        print("🚀 Submitting health check...")
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                print(f"✅ Health check job submitted: {job_id}")
                
                # Wait for completion
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
                            
                            if "output" in output:
                                health_data = output["output"]
                                
                                print(f"\\n📊 CURRENT ENDPOINT STATUS:")
                                print(f"=" * 50)
                                print(f"✅ Status: {health_data.get('status', 'Unknown')}")
                                print(f"🔧 Version: {health_data.get('version', 'Unknown')}")
                                print(f"🎯 Implementation: {health_data.get('implementation', 'Unknown')}")
                                print(f"📦 Message: {health_data.get('message', 'No message')}")
                                
                                # Check for V115 specific fields
                                if health_data.get('version') == 'V115':
                                    print(f"🚀 MultiTalk Available: {health_data.get('multitalk_available', 'Unknown')}")
                                    print(f"📦 MultiTalk Loaded: {health_data.get('multitalk_loaded', 'Unknown')}")
                                    print(f"☁️  S3 Available: {health_data.get('s3_available', 'Unknown')}")
                                    print(f"🎮 CUDA Available: {health_data.get('cuda_available', 'Unknown')}")
                                    
                                    model_info = health_data.get('model_info')
                                    if model_info:
                                        print(f"\\n📋 Model Information:")
                                        print(f"   Device: {model_info.get('device', 'Unknown')}")
                                        available = model_info.get('models_available', {})
                                        loaded = model_info.get('models_loaded', {})
                                        print(f"   Available: {sum(1 for v in available.values() if v) if available else 0}/5")
                                        print(f"   Loaded: {sum(1 for v in loaded.values() if v) if loaded else 0}/3")
                                        
                                        if available:
                                            print(f"   Models Available:")
                                            for model, avail in available.items():
                                                status = "✅" if avail else "❌"
                                                print(f"     {status} {model}")
                                        
                                        if loaded:
                                            print(f"   Components Loaded:")
                                            for comp, load in loaded.items():
                                                status = "✅" if load else "❌"
                                                print(f"     {status} {comp}")
                                else:
                                    print(f"\\n📋 Additional Fields:")
                                    for key, value in health_data.items():
                                        if key not in ['status', 'version', 'implementation', 'message']:
                                            print(f"   {key}: {value}")
                                
                                return True
                            else:
                                print(f"❌ No output in health check response")
                                return False
                        
                        elif status == "FAILED":
                            print(f"❌ Health check failed")
                            error_output = status_data.get("output", {})
                            if error_output:
                                print(f"💥 Error: {error_output}")
                            return False
                        
                        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                            if i % 10 == 0:
                                print(f"[{i}s] Waiting for health check...")
                            time.sleep(1)
                            continue
                        
                        else:
                            print(f"❓ Unknown status: {status}")
                            time.sleep(1)
                            continue
                    
                    else:
                        print(f"❌ Status check failed: {status_result.stderr}")
                        return False
                
                print(f"⏱️  Health check timeout")
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

def main():
    print("=" * 80)
    print("CURRENT ENDPOINT TEST")
    print("=" * 80)
    print("Testing current endpoint to see what version is running")
    print("=" * 80)
    
    success = test_current_endpoint()
    
    print("\\n" + "=" * 80)
    print("ENDPOINT TEST RESULTS")
    print("=" * 80)
    
    if success:
        print("✅ SUCCESS: Current endpoint is responding")
        print("🔧 Check the version and implementation details above")
        
        print("\\n🚀 Next Steps:")
        print("1. If version is not V115, deploy V115 implementation")
        print("2. If version is V115, test video generation")
        print("3. If version is V112 or older, update to V115")
        
    else:
        print("❌ FAILED: Current endpoint is not responding")
        print("🔧 Troubleshooting:")
        print("1. Check if RUNPOD_API_KEY is set correctly")
        print("2. Verify endpoint ID is correct")
        print("3. Check if endpoint is running")
        print("4. Check RunPod dashboard for endpoint status")

if __name__ == "__main__":
    main()