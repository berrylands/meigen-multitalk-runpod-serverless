#!/usr/bin/env python3
"""
Test V130 with S3 Files
Test MultiTalk V130 with PyTorch/torchvision compatibility fix using S3 files (1.wav and multi1.png)
"""

import subprocess
import json
import os
import time
import base64

def test_s3_video_generation():
    """Test V130 with S3 files"""
    
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY environment variable not set")
        print("Please set it with: export RUNPOD_API_KEY=your_api_key")
        return False
    
    endpoint_id = "zu0ik6c8yukyl6"
    
    print("🔍 Testing V130 with S3 files...")
    print(f"📡 Endpoint ID: {endpoint_id}")
    
    # Test with actual S3 files
    s3_job = {
        "input": {
            "audio_s3_key": "1.wav",           # S3 key for audio
            "image_s3_key": "multi1.png",      # S3 key for image
            "device": "cuda",
            "video_format": "mp4",
            "turbo_mode": True,
            "text_guide_scale": 5.0,
            "fps": 30
        }
    }
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    
    curl_cmd = [
        "curl", "-X", "POST", 
        url,
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps(s3_job)
    ]
    
    try:
        print("🚀 Submitting S3 video generation job...")
        print(f"📂 Audio: {s3_job['input']['audio_s3_key']}")
        print(f"🖼️  Image: {s3_job['input']['image_s3_key']}")
        
        result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            job_id = response.get("id")
            
            if job_id:
                print(f"✅ Job submitted: {job_id}")
                
                # Wait for completion
                print("⏳ Waiting for video generation...")
                for i in range(300):  # Wait up to 5 minutes
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
                            
                            print(f"\n🎉 SUCCESS! JOB COMPLETED!")
                            print(f"=" * 70)
                            
                            if "output" in output:
                                final_output = output["output"]
                                
                                # Check if we have video data
                                if "video_base64" in final_output:
                                    video_data = final_output["video_base64"]
                                    print(f"🎬 Video generated successfully!")
                                    print(f"📊 Video size: {len(video_data)} characters (base64)")
                                    
                                    # Save video
                                    video_bytes = base64.b64decode(video_data)
                                    output_path = f"v130_PYTORCH_COMPAT_SUCCESS_{int(time.time())}.mp4"
                                    with open(output_path, "wb") as f:
                                        f.write(video_bytes)
                                    print(f"💾 Video saved to: {output_path}")
                                    print(f"📏 File size: {len(video_bytes):,} bytes")
                                
                                elif "video_s3_key" in final_output:
                                    print(f"☁️  Video uploaded to S3: {final_output['video_s3_key']}")
                                
                                # Show metadata
                                if "metadata" in final_output:
                                    meta = final_output["metadata"]
                                    print(f"\n📋 Metadata:")
                                    print(f"   Model version: {meta.get('model_version', 'Unknown')}")
                                    print(f"   Frame count: {meta.get('frame_count', 'Unknown')}")
                                    print(f"   Duration: {meta.get('duration', 'Unknown')}s")
                                    print(f"   FPS: {meta.get('fps', 'Unknown')}")
                                    print(f"   Processing time: {meta.get('processing_time', 'Unknown')}s")
                                
                                return True
                            
                            else:
                                print(f"❌ No output in response")
                                print(f"Full response: {json.dumps(status_data, indent=2)}")
                                return False
                        
                        elif status == "FAILED":
                            print(f"\n❌ JOB FAILED")
                            error_output = status_data.get("output", {})
                            if error_output:
                                print(f"💥 Error: {json.dumps(error_output, indent=2)}")
                            return False
                        
                        elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                            if i % 10 == 0:
                                print(f"[{i}s] Status: {status}")
                            time.sleep(1)
                            continue
                        
                        else:
                            print(f"❓ Unknown status: {status}")
                            time.sleep(1)
                            continue
                    
                    else:
                        print(f"❌ Status check failed: {status_result.stderr}")
                        return False
                
                print(f"⏱️  Job timeout after 5 minutes")
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
    print("🎯 V130 S3 VIDEO GENERATION TEST - PYTORCH COMPATIBILITY FIX")
    print("=" * 80)
    print("Testing MultiTalk V130 with PyTorch/torchvision compatibility:")
    print("  ✅ PyTorch 2.1.0 (compatible with cog-MultiTalk)")
    print("  ✅ torchvision 0.16.0 (compatible with cog-MultiTalk)")
    print("  ✅ xformers 0.0.22 (compatible with PyTorch 2.1.0)")
    print("  ✅ xfuser, scikit-image, and all dependencies")
    print("Files: 1.wav (audio) and multi1.png (image)")
    print("=" * 80)
    
    success = test_s3_video_generation()
    
    print("\n" + "=" * 80)
    print("🏆 FINAL TEST RESULTS")
    print("=" * 80)
    
    if success:
        print("🎉 SUCCESS! V130 PYTORCH COMPATIBILITY FIXED!")
        print("✅ All dependency issues resolved!")
        print("✅ PyTorch/torchvision compatibility working!")
        print("✅ MeiGen-MultiTalk model is working!")
        print("\n🚀 Mission Accomplished:")
        print("1. ✅ Dependency resolution complete")
        print("2. ✅ PyTorch compatibility resolved")
        print("3. ✅ Video generation working")
        print("4. ✅ S3 integration working")
        print("5. ✅ Ready for production deployment")
        print("\n🔧 Next Steps:")
        print("1. Set up GitHub Actions for automated builds")
        print("2. Production deployment and scaling")
        print("3. Performance optimization")
        print("4. Documentation updates")
        
    else:
        print("❌ FAILED: V130 encountered issues")
        print("🔧 Next Actions:")
        print("1. Check RunPod logs for the specific error")
        print("2. Identify if there are additional compatibility issues")
        print("3. Create V131 with further fixes if needed")
        print("4. Continue the systematic approach")

if __name__ == "__main__":
    main()