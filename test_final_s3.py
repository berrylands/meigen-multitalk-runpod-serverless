#!/usr/bin/env python3
"""
Test the final S3 + PyTorch implementation
"""

import runpod
import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_final_implementation():
    """Test the complete implementation"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🎯 Final MultiTalk S3 + PyTorch Test")
    print("=" * 60)
    print("Image: berrylands/multitalk-pytorch:latest")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Health Check...")
    job = endpoint.run({"action": "health"})
    result = job.output(timeout=30)
    
    if result:
        print(f"✅ Health check passed")
        print(f"   S3: {result.get('s3_integration', {})}")
        print(f"   Models: {result.get('models_status', {})}")
    
    # Test 2: S3 input with real video generation
    print("\n2. Testing S3 input (using just filename)...")
    
    job_input = {
        "action": "generate",
        "audio": "1.wav",  # Just the filename!
        "duration": 5.0,
        "width": 480,
        "height": 480,
        "fps": 30
    }
    
    print(f"Input: {json.dumps(job_input, indent=2)}")
    
    try:
        job = endpoint.run(job_input)
        print(f"Job ID: {job.job_id}")
        
        # Wait for result
        start_time = time.time()
        last_status = None
        
        while True:
            status = job.status()
            elapsed = time.time() - start_time
            
            if status != last_status:
                print(f"[{elapsed:.1f}s] Status: {status}")
                last_status = status
            
            if status in ["COMPLETED", "FAILED"]:
                break
                
            if elapsed > 300:  # 5 minutes
                print("Timeout")
                break
                
            time.sleep(2)
        
        result = job.output()
        
        if status == "COMPLETED":
            print("\n✅ SUCCESS!")
            
            if isinstance(result, dict):
                # Check for real video generation
                if 'processing_note' in str(result):
                    if "Test implementation" in str(result):
                        print("⚠️  Still using dummy implementation")
                        print("   Need to update RunPod to use: berrylands/multitalk-pytorch:latest")
                    else:
                        print("✅ Real video generation!")
                
                print("\nOutput details:")
                for key, value in result.items():
                    if key != 'video' and not isinstance(value, dict):
                        print(f"  {key}: {value}")
                    elif key == 'video' and isinstance(value, str):
                        print(f"  video: {len(value)} chars")
                        
        else:
            print(f"\n❌ Failed: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: S3 output
    print("\n3. Testing S3 output...")
    
    job_input = {
        "action": "generate",
        "audio": "2.wav",  # Test with the other file
        "duration": 3.0,
        "output_format": "s3",
        "s3_output_key": f"outputs/final_test_{int(time.time())}.mp4"
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"Job ID: {job.job_id}")
        
        result = job.output(timeout=120)
        
        if job.status() == "COMPLETED":
            print("✅ S3 output successful!")
            if isinstance(result, dict) and 'video' in result:
                print(f"   Video saved to: {result['video']}")
        else:
            print(f"❌ Failed: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("\n📝 Summary:")
    print("1. S3 works with just filenames (e.g., '1.wav')")
    print("2. Full PyTorch image includes all ML dependencies")
    print("3. Update RunPod to: berrylands/multitalk-pytorch:latest")
    print("4. This will enable real video generation, not dummy output")

if __name__ == "__main__":
    test_final_implementation()