#!/usr/bin/env python3
"""
Diagnose Video Generation Issues
Check why the V112 handler is failing to generate videos
"""

import subprocess
import json
import os
import time

def get_job_status(job_id):
    """Get detailed status of a job"""
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return None
    
    endpoint_id = "zu0ik6c8yukyl6"
    
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    
    status_cmd = [
        "curl", "-H", f"Authorization: Bearer {api_key}",
        status_url
    ]
    
    try:
        status_result = subprocess.run(status_cmd, capture_output=True, text=True)
        
        if status_result.returncode == 0:
            return json.loads(status_result.stdout)
        else:
            print(f"❌ Status check failed: {status_result.stderr}")
            return None
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return None

def run_simple_test():
    """Run a simple test to see what's happening"""
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return False
    
    endpoint_id = "zu0ik6c8yukyl6"
    
    print("🔍 Running simple video generation test...")
    
    # Test with basic generation
    test_job = {
        "input": {
            "action": "generate",
            "audio_1": "1.wav",
            "condition_image": "multi1.png",
            "prompt": "A person talking",
            "dry_run": False  # Force actual generation
        }
    }
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    curl_cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(test_job)
    ]
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                print(f"✅ Job submitted: {job_id}")
                
                # Monitor for completion
                print("⏱️  Monitoring job...")
                
                for i in range(300):  # 5 minutes max
                    status_data = get_job_status(job_id)
                    
                    if status_data:
                        status = status_data.get("status")
                        
                        if status == "COMPLETED":
                            print(f"✅ Job completed!")
                            
                            # Show detailed output
                            output = status_data.get("output", {})
                            if "output" in output:
                                result_data = output["output"]
                                print(f"📊 Result status: {result_data.get('status')}")
                                
                                if result_data.get("status") == "error":
                                    print(f"❌ Error: {result_data.get('error')}")
                                    details = result_data.get("details", {})
                                    print(f"📋 Details:")
                                    for key, value in details.items():
                                        print(f"  {key}: {value}")
                                    
                                    # Show full output for debugging
                                    print(f"\\n🔍 Full output:")
                                    print(json.dumps(output, indent=2))
                                    
                                elif result_data.get("status") == "completed":
                                    print(f"✅ Video generated successfully!")
                                    video_url = result_data.get("video_url")
                                    if video_url:
                                        print(f"🔗 Video URL: {video_url}")
                                    
                            return True
                        
                        elif status == "FAILED":
                            print(f"❌ Job failed!")
                            # Show error details
                            error_output = status_data.get("output", {})
                            if error_output:
                                print(f"💥 Error details:")
                                print(json.dumps(error_output, indent=2))
                            return False
                        
                        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                            if i % 10 == 0:  # Update every 10 seconds
                                print(f"[{i}s] Status: {status}")
                            time.sleep(1)
                            continue
                        
                        else:
                            print(f"❓ Unknown status: {status}")
                            time.sleep(1)
                            continue
                    
                    else:
                        print(f"❌ Failed to get job status")
                        return False
                
                print(f"⏱️  Job monitoring timeout")
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

def check_model_loading():
    """Check if models are loading correctly"""
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return False
    
    endpoint_id = "zu0ik6c8yukyl6"
    
    print("🔍 Checking model loading...")
    
    # Test model loading
    test_job = {
        "input": {
            "action": "load_models"
        }
    }
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    curl_cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(test_job)
    ]
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                print(f"✅ Model loading job submitted: {job_id}")
                
                # Wait for completion
                for i in range(120):  # 2 minutes max
                    status_data = get_job_status(job_id)
                    
                    if status_data:
                        status = status_data.get("status")
                        
                        if status == "COMPLETED":
                            output = status_data.get("output", {})
                            if "output" in output:
                                result_data = output["output"]
                                
                                print(f"📊 Model loading result:")
                                print(f"  Success: {result_data.get('success')}")
                                print(f"  Models loaded: {result_data.get('models_loaded')}")
                                if result_data.get('error'):
                                    print(f"  Error: {result_data.get('error')}")
                                
                                available_models = result_data.get('available_models', [])
                                print(f"  Available models: {available_models}")
                                
                            return True
                        
                        elif status == "FAILED":
                            print(f"❌ Model loading failed!")
                            return False
                        
                        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                            if i % 10 == 0:
                                print(f"[{i}s] Loading models...")
                            time.sleep(1)
                            continue
                    
                    else:
                        print(f"❌ Failed to get job status")
                        return False
                
                print(f"⏱️  Model loading timeout")
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
    print("VIDEO GENERATION DIAGNOSIS")
    print("=" * 80)
    print("Investigating why V112 handler fails to generate videos")
    print("=" * 80)
    
    # Step 1: Check model loading
    print("\\n🔧 Step 1: Testing model loading...")
    model_success = check_model_loading()
    
    # Step 2: Test video generation
    print("\\n🎬 Step 2: Testing video generation...")
    video_success = run_simple_test()
    
    # Results
    print("\\n" + "=" * 80)
    print("DIAGNOSIS RESULTS")
    print("=" * 80)
    
    if model_success and video_success:
        print("✅ DIAGNOSIS: System is working correctly")
        print("   - Models load successfully")
        print("   - Video generation completes")
    elif model_success and not video_success:
        print("⚠️  DIAGNOSIS: Model loading works, but video generation fails")
        print("   - Models load successfully")
        print("   - Video generation has implementation issues")
        print("   - Check V111/V112 implementation for bugs")
    elif not model_success and video_success:
        print("⚠️  DIAGNOSIS: Unexpected - video works without model loading")
        print("   - This suggests caching or fallback mechanisms")
    else:
        print("❌ DIAGNOSIS: System has fundamental issues")
        print("   - Model loading fails")
        print("   - Video generation fails")
        print("   - System needs debugging")
    
    print("\\n💡 Recommendation:")
    if not video_success:
        print("   - Check V111 implementation for video file creation bugs")
        print("   - Verify FFmpeg/moviepy dependencies are working")
        print("   - Check file permissions in container")
        print("   - Add more detailed logging to video generation")

if __name__ == "__main__":
    main()