#!/usr/bin/env python3
"""
Test S3 integration with the MultiTalk endpoint
This script helps debug S3-related issues.
"""

import os
import sys
import time
import runpod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

# Configuration
ENDPOINT_ID = "kkx3cfy484jszl"

def test_s3_integration():
    """Test S3 integration status and functionality."""
    print("Testing S3 Integration")
    print("=" * 50)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # First, check health to see S3 status
    print("\n1. Checking endpoint health and S3 status...")
    try:
        job = endpoint.run({"health_check": True})
        print(f"Job ID: {job.job_id}")
        
        # Wait for completion
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            print(f"Status: {job.status()}")
            time.sleep(2)
        
        if job.status() == "COMPLETED":
            result = job.output()
            print("\nHealth Check Results:")
            print(f"Status: {result.get('status')}")
            print(f"Version: {result.get('version')}")
            
            # Check S3 integration
            s3_info = result.get('s3_integration', {})
            print(f"\nS3 Integration Status:")
            print(f"  Available: {s3_info.get('available', False)}")
            print(f"  Enabled: {s3_info.get('enabled', False)}")
            print(f"  Default Bucket: {s3_info.get('default_bucket', 'None')}")
            print(f"  Can Write: {s3_info.get('can_write', False)}")
            
            if 'error' in s3_info:
                print(f"  Error: {s3_info['error']}")
            
            return s3_info.get('enabled', False)
        else:
            print(f"Health check failed: {job.status()}")
            print(f"Output: {job.output()}")
            return False
            
    except Exception as e:
        print(f"Error running health check: {e}")
        return False


def test_s3_audio(s3_url):
    """Test processing an S3 audio URL."""
    print(f"\n2. Testing S3 audio processing...")
    print(f"S3 URL: {s3_url}")
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    try:
        # Submit job with S3 URL
        job_input = {
            "action": "generate",
            "audio": s3_url,
            "duration": 5.0,
            "fps": 30,
            "width": 512,
            "height": 512
        }
        
        print(f"\nSubmitting job with S3 audio URL...")
        job = endpoint.run(job_input)
        print(f"Job ID: {job.job_id}")
        
        # Wait for completion
        start_time = time.time()
        last_status = None
        
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            status = job.status()
            if status != last_status:
                print(f"[{time.time() - start_time:.1f}s] Status: {status}")
                last_status = status
            time.sleep(2)
            
            if time.time() - start_time > 120:  # 2 minute timeout
                print("Timeout!")
                break
        
        # Check result
        final_status = job.status()
        print(f"\nFinal status: {final_status}")
        
        output = job.output()
        print(f"\nJob output:")
        
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "video" and isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: <base64 data, {len(value)} chars>")
                elif key == "traceback":
                    print(f"  {key}:")
                    print("    " + "\n    ".join(value.split("\n")))
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {output}")
        
        return final_status == "COMPLETED"
        
    except Exception as e:
        print(f"Error testing S3 audio: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base64_with_s3_output():
    """Test base64 input with S3 output."""
    print(f"\n3. Testing base64 input with S3 output...")
    
    # Create simple test audio
    import numpy as np
    import base64
    
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    try:
        job_input = {
            "action": "generate",
            "audio": audio_b64,
            "duration": duration,
            "output_format": "s3",
            "s3_output_key": f"test/multitalk_test_{int(time.time())}.mp4"
        }
        
        print(f"Submitting job with base64 audio, requesting S3 output...")
        job = endpoint.run(job_input)
        print(f"Job ID: {job.job_id}")
        
        # Wait for completion
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(2)
        
        output = job.output()
        if job.status() == "COMPLETED" and output.get("success"):
            print(f"\nSuccess! Output type: {output.get('output_type')}")
            if output.get('output_type') == 's3_url':
                print(f"S3 URL: {output.get('video')}")
                print(f"S3 Key: {output.get('s3_key')}")
            else:
                print("Output was base64 (S3 might not be configured)")
        else:
            print(f"\nFailed: {output}")
            
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main test function."""
    print("MultiTalk S3 Integration Test")
    print("=" * 50)
    
    # Test 1: Health check
    s3_enabled = test_s3_integration()
    
    if not s3_enabled:
        print("\n⚠️  S3 integration is not enabled on the endpoint!")
        print("\nPossible causes:")
        print("1. boto3 is not installed in the container")
        print("2. S3 handler file is not included in the Docker image")
        print("3. AWS credentials are not configured in RunPod")
        print("\nTo fix:")
        print("1. Rebuild the Docker image with the latest Dockerfile.complete")
        print("2. Ensure AWS credentials are set in RunPod endpoint environment")
        print("3. Push the new image and update the endpoint")
        return
    
    # Test 2: S3 audio processing
    if len(sys.argv) > 1 and sys.argv[1].startswith('s3://'):
        s3_url = sys.argv[1]
        test_s3_audio(s3_url)
    else:
        print("\n⚠️  No S3 URL provided for testing")
        print("Usage: python test_s3_integration.py s3://bucket/audio.wav")
    
    # Test 3: S3 output
    test_base64_with_s3_output()


if __name__ == "__main__":
    main()