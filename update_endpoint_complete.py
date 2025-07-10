#!/usr/bin/env python3
"""
Update the endpoint to use the complete MultiTalk image
"""

import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"
COMPLETE_IMAGE = "berrylands/multitalk-runpod:complete"

def update_endpoint_image():
    """Update the endpoint to use the complete image."""
    
    print("Updating Endpoint to Complete MultiTalk Image")
    print("=" * 60)
    
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Update the endpoint configuration
    print(f"Updating endpoint to use: {COMPLETE_IMAGE}")
    
    try:
        # Use the API to update the endpoint
        url = f"https://api.runpod.io/graphql?api_key={runpod.api_key}"
        query = '''
        mutation SaveEndpoint($id: String!, $imageName: String!) {
            saveEndpoint(
                input: {
                    id: $id
                    imageName: $imageName
                }
            ) {
                id
                name
                imageName
            }
        }
        '''
        
        variables = {
            "id": ENDPOINT_ID,
            "imageName": COMPLETE_IMAGE
        }
        
        import requests
        response = requests.post(url, json={
            "query": query,
            "variables": variables
        })
        
        if response.status_code == 200:
            result = response.json()
            if 'errors' in result:
                print(f"❌ GraphQL errors: {result['errors']}")
                return False
            else:
                print(f"✅ Endpoint updated successfully!")
                print(f"   New image: {COMPLETE_IMAGE}")
                return True
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating endpoint: {e}")
        return False

def wait_for_endpoint_ready():
    """Wait for the endpoint to be ready with the new image."""
    
    print(f"\nWaiting for endpoint to update...")
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Give it time to update
    time.sleep(10)
    
    # Test with health check
    max_attempts = 20
    for attempt in range(max_attempts):
        try:
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
                    if version == '2.0.0' or 'Complete' in result.get('message', ''):
                        print(f"✅ Complete handler is running!")
                        print(f"   Version: {version}")
                        print(f"   Models available: {result.get('models_available', {})}")
                        return True
                    else:
                        print(f"   Still running old handler (attempt {attempt + 1}/{max_attempts})")
                else:
                    print(f"   Unexpected response: {result}")
            else:
                print(f"   Job failed: {job.status()}")
                
        except Exception as e:
            print(f"   Error checking endpoint: {e}")
        
        if attempt < max_attempts - 1:
            print("   Waiting 30 seconds before retry...")
            time.sleep(30)
    
    return False

def main():
    print("Complete MultiTalk Deployment Update")
    
    # Update the endpoint
    if update_endpoint_image():
        print(f"\n✅ Image update initiated")
        
        # Wait for it to be ready
        if wait_for_endpoint_ready():
            print(f"\n🎉 Complete MultiTalk handler deployed!")
            print("Ready for full video generation testing")
            return True
        else:
            print(f"\n⚠️  Handler update may still be in progress")
            print("Try testing again in a few minutes")
            return False
    else:
        print(f"\n❌ Failed to update endpoint")
        return False

if __name__ == "__main__":
    main()