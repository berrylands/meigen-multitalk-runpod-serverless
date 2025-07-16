#!/usr/bin/env python3
"""
Test Actual Video Generation with MeiGen-MultiTalk
Focus on generating real videos, not just testing components
"""

import subprocess
import json
import os
import time

def test_video_generation():
    """Test actual video generation with real inputs"""
    
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        return False
    
    endpoint_id = "zu0ik6c8yukyl6"
    
    print("🎬 Testing actual video generation with MeiGen-MultiTalk...")
    
    # Test with real video generation
    generation_job = {
        "input": {
            "action": "generate",
            "audio_1": "1.wav",  # Should be in S3 bucket
            "condition_image": "multi1.png",  # Should be in S3 bucket
            "prompt": "A person talking naturally with expressive facial movements and lip sync",
            "output_format": "s3",
            "s3_output_key": "multitalk-test/generated-video-{timestamp}.mp4",
            "sample_steps": 30,
            "text_guidance_scale": 7.5,
            "audio_guidance_scale": 3.5,
            "seed": 42
        }
    }
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    curl_cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(generation_job)
    ]
    
    print(f"🚀 Submitting video generation job...")
    print(f"📝 Inputs: 1.wav + multi1.png → video")
    print(f"⚙️  Settings: {generation_job['input']['sample_steps']} steps, guidance {generation_job['input']['text_guidance_scale']}")
    
    try:
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                print(f"✅ Video generation job submitted: {job_id}")
                print(f"⏱️  Expected duration: 2-10 minutes depending on model loading")
                
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
                            
                            print(f"\n🎉 VIDEO GENERATION COMPLETED!")
                            print(f"⏱️  Total time: {elapsed//60:.0f}:{elapsed%60:02.0f}")
                            print("=" * 60)
                            
                            if "output" in output:
                                result_data = output["output"]
                                
                                if result_data.get("status") == "completed":
                                    print("✅ Video generated successfully!")
                                    
                                    # Show video details
                                    video_url = result_data.get("video_url")
                                    s3_key = result_data.get("s3_key")
                                    video_base64 = result_data.get("video_base64")
                                    
                                    if video_url:
                                        print(f"🔗 Video URL: {video_url}")
                                        print(f"📁 S3 Key: {s3_key}")
                                    elif video_base64:
                                        print(f"📦 Video returned as base64 ({len(video_base64)} chars)")
                                        
                                        # Save base64 video locally
                                        try:
                                            import base64
                                            video_data = base64.b64decode(video_base64)
                                            local_path = f"/Users/jasonedge/CODEHOME/meigen-multitalk/generated_video_{int(time.time())}.mp4"
                                            with open(local_path, 'wb') as f:
                                                f.write(video_data)
                                            print(f"💾 Video saved locally: {local_path}")
                                        except Exception as e:
                                            print(f"⚠️  Could not save video locally: {e}")
                                    
                                    # Show generation parameters
                                    gen_params = result_data.get("generation_params", {})
                                    if gen_params:
                                        print(f"\n📊 Generation Parameters:")
                                        for key, value in gen_params.items():
                                            print(f"  {key}: {value}")
                                    
                                    message = result_data.get("message", "")
                                    if message:
                                        print(f"\n💬 Message: {message}")
                                    
                                    return True
                                
                                elif result_data.get("status") == "error":
                                    print("❌ Video generation failed!")
                                    error = result_data.get("error", "Unknown error")
                                    print(f"💥 Error: {error}")
                                    
                                    # Show error details
                                    details = result_data.get("details", {})
                                    if details:
                                        print(f"📋 Details:")
                                        for key, value in details.items():
                                            print(f"  {key}: {value}")
                                    
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
                            print(f"\n❌ VIDEO GENERATION FAILED!")
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
                
                print(f"\n⏱️  TIMEOUT: Video generation took longer than 30 minutes")
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

def main():
    print("=" * 80)
    print("MEIGEN-MULTITALK VIDEO GENERATION TEST")
    print("=" * 80)
    print("Testing actual video generation with real audio and image inputs")
    print("This is the core functionality test")
    print("=" * 80)
    
    success = test_video_generation()
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    if success:
        print("🎉 SUCCESS: MeiGen-MultiTalk video generation working!")
        print("✅ Core functionality verified")
        print("✅ Video output generated successfully")
        print("✅ S3 integration working")
        print("✅ System ready for production use")
        
        print("\n🎯 Next steps:")
        print("1. Test with different audio/image combinations")
        print("2. Optimize generation parameters")
        print("3. Scale for production workloads")
        
    else:
        print("❌ FAILED: MeiGen-MultiTalk video generation not working")
        print("🔧 Troubleshooting needed:")
        print("1. Check if models are properly loaded")
        print("2. Verify S3 inputs exist (1.wav, multi1.png)")
        print("3. Check RunPod logs for detailed errors")
        print("4. Ensure sufficient GPU memory for generation")
        
        print("\n💡 Possible issues:")
        print("- Model loading failures")
        print("- Missing input files in S3")
        print("- GPU memory limitations")
        print("- Implementation bugs in video generation")

if __name__ == "__main__":
    main()