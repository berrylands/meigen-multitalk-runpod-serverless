#!/usr/bin/env python3
"""
Update the endpoint by updating its template with the new Docker image
"""

import os
import time
import runpod
import requests
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"
COMPLETE_IMAGE = "berrylands/multitalk-runpod:complete"

def get_endpoint_info():
    """Get the endpoint's current template ID."""
    
    print("Getting endpoint information...")
    
    url = f"https://api.runpod.io/graphql?api_key={runpod.api_key}"
    query = '''
    query GetEndpoint($id: String!) {
        endpoint(id: $id) {
            id
            name
            templateId
        }
    }
    '''
    
    variables = {"id": ENDPOINT_ID}
    
    try:
        response = requests.post(url, json={"query": query, "variables": variables})
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and 'endpoint' in result['data']:
                endpoint = result['data']['endpoint']
                print(f"✅ Endpoint found: {endpoint['name']}")
                print(f"   Template ID: {endpoint.get('templateId', 'None')}")
                return endpoint.get('templateId')
            else:
                print(f"❌ No endpoint data: {result}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

def create_new_template():
    """Create a new template with the complete Docker image."""
    
    print(f"\nCreating new template with image: {COMPLETE_IMAGE}")
    
    url = f"https://api.runpod.io/graphql?api_key={runpod.api_key}"
    query = '''
    mutation CreateTemplate($input: SaveTemplateInput!) {
        saveTemplate(input: $input) {
            id
            name
            imageName
            isServerless
        }
    }
    '''
    
    template_name = f"MultiTalk Complete {int(time.time())}"
    
    variables = {
        "input": {
            "name": template_name,
            "imageName": COMPLETE_IMAGE,
            "containerDiskInGb": 10,
            "dockerArgs": "python -u handler.py",
            "env": [
                {"key": "MODEL_PATH", "value": "/runpod-volume/models"},
                {"key": "PYTHONUNBUFFERED", "value": "1"},
                {"key": "HF_HOME", "value": "/runpod-volume/huggingface"}
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
                print(f"✅ Template created successfully!")
                print(f"   Template ID: {template['id']}")
                print(f"   Name: {template['name']}")
                print(f"   Image: {template['imageName']}")
                return template['id']
            else:
                print(f"❌ Template creation failed: {result}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

def update_endpoint_template(template_id):
    """Update the endpoint to use the new template."""
    
    print(f"\nUpdating endpoint to use template: {template_id}")
    
    url = f"https://api.runpod.io/graphql?api_key={runpod.api_key}"
    query = '''
    mutation UpdateEndpoint($input: EndpointInput!) {
        saveEndpoint(input: $input) {
            id
            name
            templateId
        }
    }
    '''
    
    variables = {
        "input": {
            "id": ENDPOINT_ID,
            "name": "MultiTalk Complete",
            "templateId": template_id,
            "gpuIds": "ADA_24",  # RTX 4090
            "workersMax": 3,
            "workersMin": 0,
            "idleTimeout": 5
        }
    }
    
    try:
        response = requests.post(url, json={"query": query, "variables": variables})
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and 'saveEndpoint' in result['data']:
                endpoint = result['data']['saveEndpoint']
                print(f"✅ Endpoint updated successfully!")
                print(f"   Endpoint ID: {endpoint['id']}")
                print(f"   Name: {endpoint['name']}")
                print(f"   Template ID: {endpoint['templateId']}")
                return True
            else:
                print(f"❌ Endpoint update failed: {result}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return False

def wait_for_complete_handler():
    """Wait for the endpoint to be ready with the complete handler."""
    
    print(f"\nWaiting for complete handler to be ready...")
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    # Give it time to update
    time.sleep(10)
    
    # Test with health check
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            print(f"   Attempt {attempt + 1}/{max_attempts}...")
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
                    if version == '2.0.0':
                        print(f"✅ Complete handler is running!")
                        print(f"   Version: {version}")
                        print(f"   Message: {result.get('message', '')}")
                        print(f"   Models loaded: {result.get('models_loaded', False)}")
                        print(f"   GPU: {result.get('gpu_info', {})}")
                        return True
                    else:
                        print(f"   Still running old handler (version: {version})")
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
    print("Complete MultiTalk Deployment via Template Update")
    print("=" * 60)
    
    # Get current template
    template_id = get_endpoint_info()
    
    # Create new template with complete image
    new_template_id = create_new_template()
    
    if new_template_id:
        # Update endpoint to use new template
        if update_endpoint_template(new_template_id):
            # Wait for it to be ready
            if wait_for_complete_handler():
                print(f"\n🎉 Complete MultiTalk handler deployed successfully!")
                print("Ready for full video generation testing")
                return True
            else:
                print(f"\n⚠️  Handler may still be updating")
                print("Try testing again in a few minutes")
                return False
        else:
            print(f"\n❌ Failed to update endpoint")
            return False
    else:
        print(f"\n❌ Failed to create new template")
        return False

if __name__ == "__main__":
    main()