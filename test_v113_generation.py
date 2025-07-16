#!/usr/bin/env python3
"""Test V113 MultiTalk Generation with S3"""
import os
import time
import json
import subprocess
import sys

# Install dependencies if needed
subprocess.run([sys.executable, "-m", "pip", "install", "runpod", "python-dotenv"], capture_output=True)

import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

def test_model_check():
    """Test model availability check"""
    print("\n" + "="*60)
    print("Testing V113 Model Check")
    print("="*60)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    job_input = {
        "action": "model_check"
    }
    
    try:
        print(f"Submitting model check job...")
        job = endpoint.run(job_input)
        print(f"Job ID: {job.job_id}")
        
        # Wait for completion
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            print(f"Status: {job.status()}")
            time.sleep(2)
        
        if job.status() == "COMPLETED":
            result = job.output()
            print("\n✅ Model check completed!")
            
            if isinstance(result, dict) and "output" in result:
                output = result["output"]
                model_info = output.get("model_info", {})
                
                print(f"\nVersion: {output.get('version')}")
                print(f"Status: {output.get('status')}")
                print(f"CUDA Available: {model_info.get('cuda_available')}")
                print(f"Device: {model_info.get('device')}")
                print(f"PyTorch Version: {model_info.get('pytorch_version')}")
                print(f"xfuser Available: {model_info.get('xfuser_available')}")
                print(f"MultiTalk V113 Available: {model_info.get('multitalk_v113_available')}")
                print(f"MultiTalk V113 Initialized: {model_info.get('multitalk_v113_initialized')}")
                
                # Check V113 specific info
                if "multitalk_v113_info" in model_info:
                    v113_info = model_info["multitalk_v113_info"]
                    models_loaded = v113_info.get("models_loaded", {})
                    print(f"\nV113 Models Loaded:")
                    for model, loaded in models_loaded.items():
                        status = "✅" if loaded else "❌"
                        print(f"  {status} {model}")
                
                return True
            else:
                print(f"Unexpected output: {result}")
                return False
        else:
            print(f"❌ Job failed: {job.status()}")
            print(f"Output: {job.output()}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_video_generation():
    """Test actual video generation with S3"""
    print("\n" + "="*60)
    print("Testing V113 Video Generation")
    print("="*60)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Use simple filenames that will be downloaded from S3
    job_input = {
        "action": "generate",
        "audio_1": "1.wav",  # Will download from S3
        "condition_image": "multi1.png",  # Will download from S3
        "prompt": "A person talking naturally with expressive facial movements",
        "output_format": "s3",
        "s3_output_key": "multitalk-v113/output-{timestamp}.mp4",
        "sample_steps": 30,
        "text_guidance_scale": 7.5,
        "audio_guidance_scale": 3.5,
        "seed": 42
    }
    
    try:
        print(f"Submitting video generation job...")
        print(f"Input: {json.dumps(job_input, indent=2)}")
        
        job = endpoint.run(job_input)
        print(f"\nJob ID: {job.job_id}")
        
        # Monitor progress
        start_time = time.time()
        last_status = None
        
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            current_status = job.status()
            elapsed = time.time() - start_time
            
            if current_status != last_status:
                print(f"[{elapsed:.1f}s] Status: {current_status}")
                last_status = current_status
            
            if elapsed > 300:  # 5 minute timeout
                print("⏱️  Generation taking longer than expected...")
                break
                
            time.sleep(5)
        
        final_status = job.status()
        elapsed = time.time() - start_time
        print(f"\n[{elapsed:.1f}s] Final status: {final_status}")
        
        if final_status == "COMPLETED":
            result = job.output()
            
            if isinstance(result, dict) and "output" in result:
                output = result["output"]
                
                if output.get("status") == "completed":
                    print("\n✅ Video generation successful!")
                    print(f"Video URL: {output.get('video_url')}")
                    print(f"S3 Key: {output.get('s3_key')}")
                    print(f"Message: {output.get('message')}")
                    
                    # Show generation parameters
                    params = output.get('generation_params', {})
                    if params:
                        print(f"\nGeneration Parameters:")
                        for key, value in params.items():
                            print(f"  {key}: {value}")
                    
                    return True
                elif output.get("status") == "error":
                    print(f"\n❌ Generation failed: {output.get('error')}")
                    if "details" in output:
                        print(f"Details: {json.dumps(output['details'], indent=2)}")
                    return False
                else:
                    print(f"\nUnexpected status: {output.get('status')}")
                    print(f"Output: {json.dumps(output, indent=2)}")
                    return False
            else:
                print(f"Unexpected result format: {result}")
                return False
        else:
            print(f"❌ Job failed with status: {final_status}")
            try:
                error_output = job.output()
                print(f"Error output: {json.dumps(error_output, indent=2)}")
            except:
                pass
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("MultiTalk V113 Test Suite")
    print("Complete MeiGen-MultiTalk Implementation")
    print("="*60)
    
    # First check models
    print("\n📋 Phase 1: Checking models...")
    model_check_success = test_model_check()
    
    if not model_check_success:
        print("\n⚠️  Model check failed, but continuing with generation test...")
    
    # Test generation
    print("\n🎬 Phase 2: Testing video generation...")
    generation_success = test_video_generation()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Model Check: {'✅ PASSED' if model_check_success else '❌ FAILED'}")
    print(f"Video Generation: {'✅ PASSED' if generation_success else '❌ FAILED'}")
    
    if generation_success:
        print("\n🎉 V113 is working! Complete MeiGen-MultiTalk pipeline operational!")
        print("\nNext steps:")
        print("1. Check the generated video in your S3 bucket")
        print("2. Test with different audio/image inputs")
        print("3. Fine-tune generation parameters")
    else:
        print("\n💡 Troubleshooting tips:")
        print("1. Ensure V113 Docker image is deployed")
        print("2. Check S3 bucket has 1.wav and multi1.png")
        print("3. Verify AWS credentials in RunPod secrets")
        print("4. Check RunPod logs for detailed errors")

if __name__ == "__main__":
    main()