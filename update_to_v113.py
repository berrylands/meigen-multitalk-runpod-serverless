#!/usr/bin/env python3
"""Update RunPod endpoint to V113"""
import os
import subprocess
import sys

def update_endpoint():
    print("="*60)
    print("Updating RunPod Endpoint to V113")
    print("="*60)
    
    # Install runpod if needed
    subprocess.run([sys.executable, "-m", "pip", "install", "runpod", "python-dotenv"], capture_output=True)
    
    import runpod
    from dotenv import load_dotenv
    
    load_dotenv()
    runpod.api_key = os.getenv("RUNPOD_API_KEY")
    
    ENDPOINT_ID = "kkx3cfy484jszl"
    DOCKER_IMAGE = "multitalk/multitalk-runpod:v113"
    
    print(f"Endpoint ID: {ENDPOINT_ID}")
    print(f"Docker Image: {DOCKER_IMAGE}")
    
    try:
        endpoint = runpod.Endpoint(ENDPOINT_ID)
        
        # Update the template
        result = endpoint.update(
            container_image=DOCKER_IMAGE
        )
        
        print(f"\n✅ Successfully updated endpoint to V113!")
        print(f"Docker image: {DOCKER_IMAGE}")
        print(f"Endpoint will restart with new image...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error updating endpoint: {e}")
        return False

if __name__ == "__main__":
    success = update_endpoint()
    if success:
        print("\nNext steps:")
        print("1. Wait for endpoint to restart (~2-3 minutes)")
        print("2. Test with: python3 test_v113_generation.py")
    else:
        print("\nPlease check your RUNPOD_API_KEY and try again")