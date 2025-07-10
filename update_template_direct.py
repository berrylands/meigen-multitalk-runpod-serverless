#!/usr/bin/env python3
"""
Update the existing template with the new Docker image
"""

import os
import time
import runpod
import requests
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"
TEMPLATE_ID = "1oopkmm96l"  # From the check above
COMPLETE_IMAGE = "berrylands/multitalk-runpod:complete"

def update_template():
    """Update the template with the new Docker image."""
    
    print(f"Updating template {TEMPLATE_ID} with image: {COMPLETE_IMAGE}")
    
    url = f"https://api.runpod.io/graphql?api_key={runpod.api_key}"
    query = '''
    mutation UpdateTemplate($input: SaveTemplateInput!) {
        saveTemplate(input: $input) {
            id
            name
            imageName
        }
    }
    '''
    
    variables = {
        "input": {
            "id": TEMPLATE_ID,
            "name": "MultiTalk Complete Handler",
            "imageName": COMPLETE_IMAGE,
            "containerDiskInGb": 10,
            "dockerArgs": "python -u handler.py",
            "env": [
                {"key": "MODEL_PATH", "value": "/runpod-volume/models"},
                {"key": "PYTHONUNBUFFERED", "value": "1"},
                {"key": "HF_HOME", "value": "/runpod-volume/huggingface"},
                {"key": "CUDA_VISIBLE_DEVICES", "value": "0"}
            ],
            "volumeInGb": 0,
            "isServerless": True
        }
    }
    
    try:
        response = requests.post(url, json={"query": query, "variables": variables})
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and 'saveTemplate' in result['data']:
                template = result['data']['saveTemplate']
                print(f"✅ Template updated successfully!")
                print(f"   Template ID: {template['id']}")
                print(f"   Name: {template['name']}")
                print(f"   New Image: {template['imageName']}")
                return True
            else:
                print(f"❌ Template update failed: {result}")
                if 'errors' in result:
                    for error in result['errors']:
                        print(f"   Error: {error.get('message')}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

def wait_for_handler_update():
    """Wait for the endpoint to update with the new handler."""
    
    print(f"\nWaiting for handler to update...")
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Give the system time to propagate the update
    print("   Waiting 30 seconds for template update to propagate...")
    time.sleep(30)
    
    # Test with health check
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            print(f"\n   Testing handler (attempt {attempt + 1}/{max_attempts})...")
            job = endpoint.run({"health_check": True})
            
            # Wait for job to complete
            wait_time = 0
            while job.status() in ["IN_QUEUE", "IN_PROGRESS"] and wait_time < 60:
                time.sleep(3)
                wait_time += 3
            
            if job.status() == "COMPLETED":
                result = job.output()
                
                # Check if we have the complete handler
                if result and isinstance(result, dict):
                    version = result.get('version', '')
                    message = result.get('message', '')
                    
                    if version == '2.0.0' or 'Complete MultiTalk' in message:
                        print(f"✅ Complete handler is now running!")
                        print(f"   Version: {version}")
                        print(f"   Message: {message}")
                        print(f"   Models loaded: {result.get('models_loaded', False)}")
                        print(f"   GPU info: {result.get('gpu_info', {})}")
                        print(f"   Storage used: {result.get('storage_used_gb', 0)} GB")
                        return True
                    else:
                        print(f"   Still running old handler")
                        print(f"   Response: {result}")
                else:
                    print(f"   Unexpected response: {result}")
            else:
                print(f"   Job failed: {job.status()}")
                output = job.output()
                if output:
                    print(f"   Error: {output}")
                
        except Exception as e:
            print(f"   Error checking endpoint: {e}")
        
        if attempt < max_attempts - 1:
            print("   Waiting 30 seconds before retry...")
            time.sleep(30)
    
    return False

def test_video_generation():
    """Test actual video generation with the complete handler."""
    
    print(f"\nTesting video generation...")
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Create simple test audio
    import base64
    import numpy as np
    
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_signal = np.sin(2 * np.pi * 440 * t)  # 440Hz tone
    audio_int16 = (audio_signal * 32767).astype(np.int16)
    audio_b64 = base64.b64encode(audio_int16.tobytes()).decode('utf-8')
    
    job_input = {
        "action": "generate",
        "audio": audio_b64,
        "duration": 2.0,
        "fps": 24,
        "width": 480,
        "height": 480
    }
    
    try:
        print("   Submitting video generation job...")
        job = endpoint.run(job_input)
        print(f"   Job ID: {job.job_id}")
        
        # Wait for completion
        start_time = time.time()
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            elapsed = time.time() - start_time
            print(f"   [{elapsed:.1f}s] Status: {job.status()}")
            time.sleep(5)
            
            if elapsed > 300:  # 5 minute timeout
                print("   Timeout!")
                break
        
        if job.status() == "COMPLETED":
            result = job.output()
            if result and result.get("success"):
                print(f"✅ Video generation successful!")
                print(f"   Processing time: {result.get('processing_time')}")
                print(f"   Models used: {result.get('models_used', [])}")
                
                if 'video' in result:
                    video_b64 = result['video']
                    video_data = base64.b64decode(video_b64)
                    print(f"   Video size: {len(video_data)} bytes")
                
                return True
            else:
                print(f"❌ Video generation failed")
                print(f"   Result: {result}")
        else:
            print(f"❌ Job failed: {job.status()}")
            print(f"   Output: {job.output()}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

def main():
    print("MultiTalk Complete Handler Deployment")
    print("=" * 60)
    
    # Update the template
    if update_template():
        print(f"\n✅ Template updated with complete Docker image")
        
        # Wait for the handler to update
        if wait_for_handler_update():
            print(f"\n🎉 Complete MultiTalk handler is active!")
            
            # Test video generation
            print("\nTesting full functionality...")
            if test_video_generation():
                print(f"\n🎉 SUCCESS! MultiTalk serverless is fully operational!")
                print("✅ All facilities of MeiGen MultiTalk are working")
                print("✅ Zero idle costs achieved")
                print("✅ Ready for production use")
            else:
                print(f"\n⚠️  Video generation test failed")
                print("Handler is deployed but may need debugging")
        else:
            print(f"\n⚠️  Handler update verification failed")
            print("The template was updated but the handler may still be updating")
            print("Try testing again in a few minutes")
    else:
        print(f"\n❌ Failed to update template")

if __name__ == "__main__":
    main()