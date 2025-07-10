#!/usr/bin/env python3
"""
Create a simple test endpoint using direct API calls
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_API_BASE = "https://api.runpod.io/graphql"

def make_graphql_request(query, variables=None):
    """Make a GraphQL request to RunPod API."""
    url = f"{RUNPOD_API_BASE}?api_key={RUNPOD_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        print(response.text)
        return None
    
    data = response.json()
    if "errors" in data:
        print(f"GraphQL errors: {data['errors']}")
        return None
    
    return data.get("data", {})

def create_template():
    """Create a template for our MultiTalk endpoint."""
    
    mutation = """
    mutation CreateTemplate($input: CreateTemplateInput!) {
        createTemplate(input: $input) {
            id
            name
            dockerImage
        }
    }
    """
    
    variables = {
        "input": {
            "name": "multitalk-test-template",
            "dockerImage": "runpod/pytorch:2.2.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            "dockerStartCmd": 'bash -c "pip install runpod && python -c \\"import runpod; import os; def handler(job): return {\'status\': \'healthy\', \'message\': \'Test handler working\', \'volume_exists\': os.path.exists(\'/runpod-volume\')}; runpod.serverless.start({\'handler\': handler})\\"',
            "containerDiskInGb": 10,
            "env": [
                {"key": "MODEL_PATH", "value": "/runpod-volume/models"},
                {"key": "RUNPOD_DEBUG_LEVEL", "value": "INFO"}
            ],
            "volumeInGb": 0,
            "isServerless": True
        }
    }
    
    result = make_graphql_request(mutation, variables)
    if result:
        template = result.get("createTemplate", {})
        print(f"✓ Template created: {template['name']} ({template['id']})")
        return template
    return None

def create_endpoint_from_template(template_id):
    """Create an endpoint from a template."""
    
    mutation = """
    mutation CreateEndpoint($input: CreateEndpointInput!) {
        createEndpoint(input: $input) {
            id
            name
        }
    }
    """
    
    variables = {
        "input": {
            "name": "multitalk-test-v1",
            "templateId": template_id,
            "networkVolumeId": "pth5bf7dey",  # Your network volume
            "locations": "US",
            "idleTimeout": 60,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 1,
            "workersMin": 0,
            "workersMax": 1,
            "gpuIds": "NVIDIA GeForce RTX 4090"
        }
    }
    
    result = make_graphql_request(mutation, variables)
    if result:
        endpoint = result.get("createEndpoint", {})
        print(f"✓ Endpoint created: {endpoint['name']} ({endpoint['id']})")
        return endpoint
    return None

def test_endpoint(endpoint_id):
    """Test the endpoint."""
    import time
    
    # Wait a bit for startup
    print("Waiting for endpoint to start...")
    time.sleep(30)
    
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    
    test_data = {"input": {"health_check": True}}
    
    print(f"Testing endpoint...")
    try:
        response = requests.post(url, json=test_data, headers=headers, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Test successful!")
            print(f"Response: {result}")
            return True
        else:
            print(f"✗ Test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False

def main():
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY not found")
        sys.exit(1)
    
    print("Creating RunPod Test Endpoint")
    print("=" * 50)
    
    # Create template
    template = create_template()
    if not template:
        print("Failed to create template")
        sys.exit(1)
    
    # Create endpoint
    endpoint = create_endpoint_from_template(template['id'])
    if not endpoint:
        print("Failed to create endpoint")
        sys.exit(1)
    
    # Test endpoint
    if test_endpoint(endpoint['id']):
        print("\n" + "=" * 50)
        print("🎉 Success! Basic endpoint is working!")
        print(f"Endpoint ID: {endpoint['id']}")
        print(f"Test URL: https://api.runpod.ai/v2/{endpoint['id']}/runsync")
    else:
        print("Endpoint created but test failed - check RunPod logs")

if __name__ == "__main__":
    main()