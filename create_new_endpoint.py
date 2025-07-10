#!/usr/bin/env python3
"""
Create a new RunPod endpoint with correct configuration
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

def create_endpoint():
    """Create a new serverless endpoint."""
    
    url = "https://api.runpod.ai/graphql"
    
    # GraphQL mutation to create endpoint
    mutation = """
    mutation createEndpoint($input: EndpointInput!) {
        saveEndpoint(input: $input) {
            id
            name
            templateId
        }
    }
    """
    
    variables = {
        "input": {
            "name": "multitalk-test-v2",
            "templateId": None,  # No template, direct configuration
            "gpuIds": "NVIDIA GeForce RTX 4090",
            "networkVolumeId": "pth5bf7dey",  # Your existing volume
            "locations": "US-NC-1",
            "idleTimeout": 60,
            "scalerType": "QUEUE_DELAY", 
            "scalerValue": 4,
            "workersMin": 0,
            "workersMax": 1,
            "workersStandby": 0,
            "containerDiskInGb": 5,
            "volumeInGb": 0,
            "volumeMountPath": "/runpod-volume",
            "env": [
                {"key": "MODEL_PATH", "value": "/runpod-volume/models"},
                {"key": "RUNPOD_DEBUG_LEVEL", "value": "DEBUG"}
            ],
            "dockerArgs": "berrylands/multitalk-test:latest"
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
            endpoint_data = result.get("data", {}).get("saveEndpoint", {})
            print(f"✓ Endpoint created successfully!")
            print(f"  ID: {endpoint_data.get('id')}")
            print(f"  Name: {endpoint_data.get('name')}")
            print(f"\nIMPORTANT: Update the ENDPOINT_ID in your test scripts to: {endpoint_data.get('id')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("Creating new RunPod endpoint with RTX 4090...")
    create_endpoint()