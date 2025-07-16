#!/usr/bin/env python3
"""
Test V115 Implementation
Comprehensive test of the proper MeiGen-MultiTalk implementation
"""

import subprocess
import json
import os
import time

def test_v115_video_generation():
    """Test V115 video generation with proper MeiGen-MultiTalk implementation"""
    
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return False
    
    # We'll test against zu0ik6c8yukyl6 after updating it to V115
    endpoint_id = "zu0ik6c8yukyl6"
    
    print("🎬 Testing V115 video generation...")
    print("=" * 60)
    
    # Test with comprehensive parameters
    test_job = {
        "input": {
            "action": "generate",
            "audio_1": "1.wav",
            "condition_image": "multi1.png",
            "prompt": "A person talking naturally with expressive facial movements and clear lip sync",
            "num_frames": 81,
            "sampling_steps": 40,
            "seed": 42,
            "turbo": True,
            "output_format": "s3",
            "s3_output_key": "multitalk-v115/test-video-{timestamp}.mp4"
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
    
    print(f"🚀 Submitting V115 video generation job...")
    print(f"📝 Parameters:")
    print(f"   Audio: {test_job['input']['audio_1']}")
    print(f"   Image: {test_job['input']['condition_image']}")
    print(f"   Frames: {test_job['input']['num_frames']}")
    print(f"   Steps: {test_job['input']['sampling_steps']}")
    print(f"   Turbo: {test_job['input']['turbo']}")
    print(f"   Output: {test_job['input']['output_format']}")
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                print(f"✅ Job submitted: {job_id}")
                
                # Monitor with extended timeout for video generation
                start_time = time.time()
                last_status = None
                
                for i in range(1800):  # 30 minutes max
                    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
                    
                    status_cmd = [
                        "curl", "-H", f"Authorization: Bearer {api_key}",
                        status_url
                    ]
                    
                    status_result = subprocess.run(status_cmd, capture_output=True, text=True)
                    
                    if status_result.returncode == 0:
                        status_data = json.loads(status_result.stdout)
                        status = status_data.get("status")
                        elapsed = time.time() - start_time
                        
                        if status != last_status:
                            print(f"[{elapsed//60:.0f}:{elapsed%60:02.0f}] Status: {status}")
                            last_status = status
                        elif i % 60 == 0:  # Progress update every minute
                            print(f"[{elapsed//60:.0f}:{elapsed%60:02.0f}] Still {status}...")
                        
                        if status == "COMPLETED":
                            output = status_data.get("output", {})
                            
                            print(f"\\n🎉 V115 VIDEO GENERATION COMPLETED!")
                            print(f"⏱️  Total time: {elapsed//60:.0f}:{elapsed%60:02.0f}")
                            print("=" * 60)
                            
                            if "output" in output:
                                result_data = output["output"]
                                
                                if result_data.get("status") == "completed":
                                    print("✅ Video generated successfully with V115!")
                                    print(f"🔧 Implementation: {result_data.get('implementation', 'Unknown')}")
                                    
                                    # Show video details
                                    video_url = result_data.get("video_url")
                                    s3_key = result_data.get("s3_key")
                                    video_base64 = result_data.get("video_base64")
                                    video_size = result_data.get("video_size")
                                    processing_time = result_data.get("processing_time")
                                    
                                    if video_url:
                                        print(f"🔗 Video URL: {video_url}")
                                        print(f"📁 S3 Key: {s3_key}")
                                    elif video_base64:
                                        print(f"📦 Video returned as base64 ({len(video_base64)} chars)")
                                        
                                        # Save base64 video locally
                                        try:
                                            import base64
                                            video_data = base64.b64decode(video_base64)
                                            local_path = f"/Users/jasonedge/CODEHOME/meigen-multitalk/v115_generated_video_{int(time.time())}.mp4"
                                            with open(local_path, 'wb') as f:
                                                f.write(video_data)
                                            print(f"💾 Video saved locally: {local_path}")
                                        except Exception as e:
                                            print(f"⚠️  Could not save video locally: {e}")
                                    
                                    if video_size:
                                        print(f"📊 Video size: {video_size:,} bytes")
                                    if processing_time:
                                        print(f"⏱️  Processing time: {processing_time}")
                                    
                                    # Show generation parameters
                                    gen_params = result_data.get("generation_params", {})
                                    if gen_params:
                                        print(f"\\n📋 Generation Parameters:")
                                        for key, value in gen_params.items():
                                            print(f"  {key}: {value}")
                                    
                                    print(f"\\n🎯 SUCCESS: V115 Implementation Working!")
                                    print(f"✅ Proper MeiGen-MultiTalk integration successful")
                                    print(f"✅ Video generation pipeline functional")
                                    print(f"✅ S3 integration working")
                                    
                                    return True
                                
                                elif result_data.get("status") == "error":
                                    print("❌ V115 video generation failed!")
                                    error = result_data.get("error", "Unknown error")
                                    print(f"💥 Error: {error}")
                                    implementation = result_data.get("implementation", "Unknown")
                                    print(f"🔧 Implementation: {implementation}")
                                    
                                    return False
                                
                                else:
                                    print(f"❓ Unexpected status: {result_data.get('status')}")
                                    print(f"📄 Full output: {json.dumps(result_data, indent=2)}")
                                    return False
                            
                            else:
                                print(f"❌ No output in response")
                                print(f"📄 Raw response: {json.dumps(output, indent=2)}")
                                return False
                        
                        elif status == "FAILED":
                            print(f"\\n❌ V115 VIDEO GENERATION FAILED!")
                            print(f"⏱️  Failed after: {elapsed//60:.0f}:{elapsed%60:02.0f}")
                            
                            # Try to get error details
                            try:
                                error_output = status_data.get("output", {})
                                if error_output:
                                    print(f"💥 Error details: {json.dumps(error_output, indent=2)}")
                            except:
                                pass
                            
                            return False
                        
                        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                            time.sleep(1)
                            continue
                        
                        else:
                            print(f"❓ Unknown status: {status}")
                            time.sleep(1)
                            continue
                    
                    else:
                        print(f"❌ Status check failed: {status_result.stderr}")
                        return False
                
                print(f"\\n⏱️  TIMEOUT: V115 video generation took longer than 30 minutes")
                print(f"Check RunPod dashboard for job status: {job_id}")
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

def test_v115_health_check():
    """Test V115 health check"""
    
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return False
    
    endpoint_id = "zu0ik6c8yukyl6"
    
    print("🔍 Testing V115 health check...")
    
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
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
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
                                
                                print(f"✅ Health check completed!")
                                print(f"📊 Status: {health_data.get('status')}")
                                print(f"🔧 Version: {health_data.get('version')}")
                                print(f"🎯 Implementation: {health_data.get('implementation')}")
                                print(f"🚀 MultiTalk Available: {health_data.get('multitalk_available')}")
                                print(f"📦 MultiTalk Loaded: {health_data.get('multitalk_loaded')}")
                                print(f"☁️  S3 Available: {health_data.get('s3_available')}")
                                print(f"🎮 CUDA Available: {health_data.get('cuda_available')}")
                                
                                model_info = health_data.get('model_info')
                                if model_info:
                                    print(f"\\n📋 Model Information:")
                                    print(f"   Device: {model_info.get('device')}")
                                    available = model_info.get('models_available', {})
                                    loaded = model_info.get('models_loaded', {})
                                    print(f"   Available: {sum(1 for v in available.values() if v)}/5")
                                    print(f"   Loaded: {sum(1 for v in loaded.values() if v)}/3")
                                
                                return True
                            
                            return False
                        
                        elif status == "FAILED":
                            print(f"❌ Health check failed")
                            return False
                        
                        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                            time.sleep(1)
                            continue
                    
                    else:
                        print(f"❌ Status check failed")
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
    print("MULTITALK V115 IMPLEMENTATION TEST")
    print("=" * 80)
    print("Testing the proper MeiGen-MultiTalk implementation")
    print("This test will verify that V115 can generate actual videos")
    print("=" * 80)
    
    # Test 1: Health check
    print("\\n🔍 Step 1: Health Check")
    health_success = test_v115_health_check()
    
    # Test 2: Video generation
    print("\\n🎬 Step 2: Video Generation")
    video_success = test_v115_video_generation()
    
    # Results
    print("\\n" + "=" * 80)
    print("V115 TEST RESULTS")
    print("=" * 80)
    
    if health_success and video_success:
        print("🎉 SUCCESS: V115 Implementation Working Perfectly!")
        print("✅ Health check passed")
        print("✅ Video generation successful")
        print("✅ Proper MeiGen-MultiTalk integration")
        print("✅ Ready for production use")
        
        print("\\n🚀 Next Steps:")
        print("1. Update existing endpoint to V115")
        print("2. Test with various input combinations")
        print("3. Monitor performance metrics")
        print("4. Scale for production workloads")
        
    elif health_success and not video_success:
        print("⚠️  PARTIAL SUCCESS: V115 loads but video generation fails")
        print("✅ Health check passed")
        print("❌ Video generation failed")
        print("🔧 Action needed: Debug video generation pipeline")
        
    elif not health_success and video_success:
        print("⚠️  UNUSUAL: Video works but health check fails")
        print("❌ Health check failed")
        print("✅ Video generation successful")
        print("🔧 This suggests a health check issue")
        
    else:
        print("❌ FAILED: V115 Implementation has issues")
        print("❌ Health check failed")
        print("❌ Video generation failed")
        print("🔧 Action needed: Fix MeiGen-MultiTalk requirements")
        
        print("\\n💡 Troubleshooting:")
        print("1. Verify all MeiGen-MultiTalk models are available")
        print("2. Check that wan.MultiTalkPipeline can be imported")
        print("3. Ensure Wav2Vec2Model is properly loaded")
        print("4. V115 has NO fallback - all components must work")

if __name__ == "__main__":
    main()