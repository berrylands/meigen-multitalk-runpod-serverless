#!/usr/bin/env python3
"""
Update the original endpoint zu0ik6c8yukyl6 to use V121 template
This endpoint already has the network volume attached
"""

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
ENDPOINT_ID = "zu0ik6c8yukyl6"  # Original endpoint with network volume
TEMPLATE_ID = "slyfdvoag8"  # V121 template
API_KEY = os.environ.get('RUNPOD_API_KEY')

def update_endpoint_template():
    """Update the original endpoint to use V121 template."""
    
    # RunPod GraphQL API endpoint
    url = "https://api.runpod.ai/graphql"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # GraphQL mutation to update endpoint
    mutation = """
    mutation updateEndpoint($input: UpdateEndpointInput!) {
        updateEndpoint(input: $input) {
            id
            name
            templateId
        }
    }
    """
    
    variables = {
        "input": {
            "id": ENDPOINT_ID,
            "templateId": TEMPLATE_ID
        }
    }
    
    payload = {
        "query": mutation,
        "variables": variables
    }
    
    print(f"Updating endpoint {ENDPOINT_ID} to use V121 template {TEMPLATE_ID}")
    
    response = requests.post(url, headers=headers, json=payload)
    
    print(f"Response status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    
    if response.status_code == 200 and not result.get('errors'):
        print("✅ Endpoint updated successfully!")
        return True
    else:
        print("❌ Failed to update endpoint")
        if 'errors' in result:
            for error in result['errors']:
                print(f"Error: {error.get('message', 'Unknown error')}")
        return False

if __name__ == "__main__":
    print("Updating original endpoint to V121...")
    success = update_endpoint_template()
    
    if success:
        print("\n🎯 Ready to test V121 on original endpoint with network volume!")
        print("The endpoint now has:")
        print("- V121 mock xfuser implementation")
        print("- Network volume with all models attached")
        print("- Proper environment variables configured")
    else:
        print("\n⚠️ Manual update may be needed in RunPod console")