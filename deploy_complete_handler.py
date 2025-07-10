#!/usr/bin/env python3
"""
Deploy the complete MultiTalk handler by uploading the code directly to the endpoint
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

def deploy_complete_handler():
    """Deploy the complete handler by updating the current container."""
    
    print("Deploying Complete MultiTalk Handler")
    print("=" * 60)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Read the complete handler code
    with open("runpod-multitalk/complete_multitalk_handler.py", "r") as f:
        handler_code = f.read()
    
    # First check if we can access the endpoint
    try:
        print("Testing endpoint connectivity...")
        test_job = endpoint.run({"health_check": True})
        
        # Wait briefly for response
        wait_time = 0
        while test_job.status() in ["IN_QUEUE", "IN_PROGRESS"] and wait_time < 30:
            time.sleep(2)
            wait_time += 2
        
        if test_job.status() == "COMPLETED":
            print("✅ Endpoint is accessible")
        else:
            print(f"⚠️  Endpoint response: {test_job.status()}")
    
    except Exception as e:
        print(f"❌ Endpoint test failed: {e}")
        return False
    
    # Deploy by writing the handler code to a file in the container
    print(f"\nDeploying complete handler code ({len(handler_code)} characters)...")
    
    job_input = {
        "action": "deploy_code",
        "handler_code": handler_code,
        "target_file": "/app/handler.py"
    }
    
    try:
        job = endpoint.run(job_input)
        print(f"Deployment job: {job.job_id}")
        
        # Monitor with timeout
        start_time = time.time()
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Deploying...")
            
            if elapsed > 120:  # 2 minute timeout
                print("Deployment taking longer than expected")
                break
                
            time.sleep(5)
        
        if job.status() == "COMPLETED":
            result = job.output()
            print(f"✅ Deployment result: {result}")
            return True
        else:
            print(f"❌ Deployment failed: {job.output()}")
            return False
            
    except Exception as e:
        print(f"Error during deployment: {e}")
        return False

def test_complete_handler():
    """Test the newly deployed complete handler."""
    
    print(f"\n" + "=" * 60)
    print("Testing Complete Handler")
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Test health check with complete handler features
    print("\n1. Testing complete health check...")
    try:
        job = endpoint.run({"health_check": True})
        
        wait_time = 0
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"] and wait_time < 30:
            time.sleep(2)
            wait_time += 2
        
        if job.status() == "COMPLETED":
            result = job.output()
            print(f"✅ Complete handler health check:")
            print(f"   Version: {result.get('version', 'Unknown')}")
            print(f"   Models available: {result.get('models_available', {})}")
            print(f"   GPU: {result.get('gpu_info', {})}")
            print(f"   Storage: {result.get('storage_used_gb', 0)} GB")
        else:
            print(f"❌ Health check failed: {job.output()}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test model loading
    print("\n2. Testing model loading...")
    try:
        job = endpoint.run({"action": "load_models"})
        
        wait_time = 0
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"] and wait_time < 60:
            time.sleep(3)
            wait_time += 3
        
        if job.status() == "COMPLETED":
            result = job.output()
            print(f"✅ Model loading: {result.get('success', False)}")
            if result.get('available_models'):
                print(f"   Available: {', '.join(result['available_models'])}")
        else:
            print(f"❌ Model loading failed: {job.output()}")
            
    except Exception as e:
        print(f"❌ Model loading error: {e}")
    
    return True

def main():
    print("Complete MultiTalk Handler Deployment")
    print("Using direct code deployment method")
    
    # Deploy the complete handler
    success = deploy_complete_handler()
    
    if success:
        print(f"\n🎉 Handler deployed successfully!")
        
        # Test the deployment
        test_success = test_complete_handler()
        
        if test_success:
            print(f"\n✅ Complete MultiTalk handler is operational!")
            print("Ready for full video generation testing")
        else:
            print(f"\n⚠️  Handler deployed but testing had issues")
            
    else:
        print(f"\n❌ Deployment failed")
        print("Will attempt alternative deployment method")
    
    return success

if __name__ == "__main__":
    main()