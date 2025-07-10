#!/usr/bin/env python3
"""
Quick test to see if the handler has been updated
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

def quick_test():
    """Quick health check to see handler version."""
    
    print("Quick Handler Test")
    print("=" * 40)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    try:
        print("Sending health check...")
        job = endpoint.run({"health_check": True})
        
        # Wait for completion
        wait_time = 0
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"] and wait_time < 60:
            print(f"   Status: {job.status()}")
            time.sleep(3)
            wait_time += 3
        
        print(f"\nFinal status: {job.status()}")
        
        if job.status() == "COMPLETED":
            result = job.output()
            
            # Check what we got
            if result and isinstance(result, dict):
                # Check for version 2.0.0 indicators
                version = result.get('version', 'Unknown')
                message = result.get('message', '')
                models_loaded = result.get('models_loaded', False)
                
                print(f"\nHandler Response:")
                print(f"   Version: {version}")
                print(f"   Message: {message}")
                print(f"   Models loaded: {models_loaded}")
                
                if 'models_available' in result:
                    print(f"   Models available: {result['models_available']}")
                
                if 'gpu_info' in result:
                    print(f"   GPU: {result['gpu_info']}")
                
                if 'storage_used_gb' in result:
                    print(f"   Storage used: {result['storage_used_gb']} GB")
                
                # Determine if it's the complete handler
                if version == '2.0.0' or 'Complete MultiTalk' in message:
                    print(f"\n✅ COMPLETE HANDLER IS RUNNING!")
                    return True
                else:
                    print(f"\n❌ Still running old handler")
                    return False
            else:
                print(f"\nUnexpected response format: {result}")
        else:
            print(f"Job failed: {job.output()}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    return False

if __name__ == "__main__":
    if quick_test():
        print("\n🎉 MultiTalk Complete Handler is deployed!")
        print("Ready for full video generation testing")
    else:
        print("\n⚠️  Handler not yet updated")
        print("Template update may still be propagating")