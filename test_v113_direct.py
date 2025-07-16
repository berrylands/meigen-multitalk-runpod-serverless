#!/usr/bin/env python3
"""Direct test of V113 using curl"""
import subprocess
import json
import os
import time

def test_with_curl():
    """Test the endpoint directly using curl"""
    
    # Get API key from environment
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set in environment")
        return False
    
    endpoint_id = "zu0ik6c8yukyl6"
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    # Test model check
    test_data = {
        "input": {
            "action": "model_check"
        }
    }
    
    print("Testing V113 model check via curl...")
    
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
                
                for i in range(60):  # Wait up to 60 seconds
                    status_cmd = [
                        "curl", "-H", f"Authorization: Bearer {api_key}",
                        status_url
                    ]
                    
                    status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                    
                    if status_result.returncode == 0:
                        status_data = json.loads(status_result.stdout)
                        status = status_data.get("status")
                        
                        print(f"[{i+1}s] Status: {status}")
                        
                        if status == "COMPLETED":
                            output = status_data.get("output", {})
                            
                            # Check for version info
                            if "output" in output and "version" in output["output"]:
                                version = output["output"]["version"]
                                print(f"🎉 Detected version: {version}")
                                
                                if version == "113":
                                    print("✅ V113 is running!")
                                    return True
                                else:
                                    print(f"⚠️  Expected V113 but got V{version}")
                                    return False
                            else:
                                print("⚠️  No version info in response")
                                print(f"Response: {json.dumps(output, indent=2)}")
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
    success = test_with_curl()
    
    if success:
        print("\n🎉 V113 test passed!")
        print("Next: Test video generation")
    else:
        print("\n💡 V113 may not be deployed yet")
        print("Check RunPod dashboard to update template")