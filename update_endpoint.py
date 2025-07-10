#!/usr/bin/env python3
"""
Update the existing RunPod endpoint configuration
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = "pdz7evo425qwmz"

def update_endpoint():
    """Update endpoint configuration."""
    
    url = "https://api.runpod.ai/graphql"
    
    # GraphQL mutation to update endpoint
    mutation = """
    mutation updateEndpoint($endpointId: String!, $input: EndpointUpdateInput!) {
        updateEndpoint(endpointId: $endpointId, input: $input) {
            id
            name
            gpuIds
            idleTimeout
        }
    }
    """
    
    variables = {
        "endpointId": ENDPOINT_ID,
        "input": {
            "gpuIds": "NVIDIA GeForce RTX 4090",
            "idleTimeout": 60,
            "workersStandby": 0,  # Set to 0 to avoid keeping idle workers
            "env": [
                {"key": "MODEL_PATH", "value": "/runpod-volume/models"},
                {"key": "RUNPOD_DEBUG_LEVEL", "value": "DEBUG"}
            ]
        }
    }
    
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        url,
        json={"query": mutation, "variables": variables},
        headers=headers
    )
    
    if response.status_code == 200:
        result = response.json()
        if "errors" in result:
            print(f"GraphQL Errors: {result['errors']}")
        else:
            endpoint_data = result.get("data", {}).get("updateEndpoint", {})
            print(f"✓ Endpoint updated successfully!")
            print(f"  ID: {endpoint_data.get('id')}")
            print(f"  Name: {endpoint_data.get('name')}")
            print(f"  GPU IDs: {endpoint_data.get('gpuIds')}")
            print(f"  Idle Timeout: {endpoint_data.get('idleTimeout')}s")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print(f"Updating endpoint {ENDPOINT_ID} to use RTX 4090...")
    update_endpoint()