#!/usr/bin/env python3
"""
Deploy to RunPod Serverless
"""

import os
import sys
import json
import time
import argparse
import requests

RUNPOD_API_BASE = "https://api.runpod.ai/graphql"

def create_endpoint(api_key, endpoint_name, docker_image):
    """Create or update a RunPod serverless endpoint."""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # First, check if endpoint already exists
    query_check = """
    query {
        myself {
            serverlessEndpoints {
                id
                name
            }
        }
    }
    """
    
    response = requests.post(
        RUNPOD_API_BASE,
        json={"query": query_check},
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Error checking endpoints: {response.text}")
        sys.exit(1)
    
    data = response.json()
    endpoints = data.get("data", {}).get("myself", {}).get("serverlessEndpoints", [])
    
    # Look for existing endpoint
    endpoint_id = None
    for endpoint in endpoints:
        if endpoint["name"] == endpoint_name:
            endpoint_id = endpoint["id"]
            print(f"Found existing endpoint: {endpoint_id}")
            break
    
    if endpoint_id:
        # Update existing endpoint
        mutation = """
        mutation updateEndpoint($input: UpdateEndpointInput!) {
            updateEndpoint(input: $input) {
                id
                name
            }
        }
        """
        
        variables = {
            "input": {
                "id": endpoint_id,
                "dockerImage": docker_image,
                "gpuType": "RTX 4090",
                "minWorkers": 0,
                "maxWorkers": 3,
                "idleTimeout": 60,
                "scalerType": "QUEUE_DEPTH",
                "scalerValue": 1,
                "containerDiskSize": 10,
                "envs": [
                    {"key": "MODEL_PATH", "value": "/runpod-volume/models"},
                    {"key": "RUNPOD_DEBUG_LEVEL", "value": "INFO"}
                ]
            }
        }
    else:
        # Create new endpoint
        mutation = """
        mutation createEndpoint($input: CreateEndpointInput!) {
            createEndpoint(input: $input) {
                id
                name
            }
        }
        """
        
        variables = {
            "input": {
                "name": endpoint_name,
                "dockerImage": docker_image,
                "gpuType": "RTX 4090",
                "minWorkers": 0,
                "maxWorkers": 3,
                "idleTimeout": 60,
                "scalerType": "QUEUE_DEPTH",
                "scalerValue": 1,
                "containerDiskSize": 10,
                "volumeMountPath": "/runpod-volume",
                "envs": [
                    {"key": "MODEL_PATH", "value": "/runpod-volume/models"},
                    {"key": "RUNPOD_DEBUG_LEVEL", "value": "INFO"}
                ]
            }
        }
    
    response = requests.post(
        RUNPOD_API_BASE,
        json={"query": mutation, "variables": variables},
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"Error creating/updating endpoint: {response.text}")
        sys.exit(1)
    
    data = response.json()
    
    if "errors" in data:
        print(f"GraphQL errors: {data['errors']}")
        sys.exit(1)
    
    if endpoint_id:
        result = data["data"]["updateEndpoint"]
    else:
        result = data["data"]["createEndpoint"]
    
    endpoint_id = result["id"]
    print(f"Endpoint deployed successfully: {endpoint_id}")
    
    # Save endpoint ID for later steps
    with open(os.environ.get("GITHUB_OUTPUT", "output.txt"), "a") as f:
        f.write(f"endpoint-id={endpoint_id}\n")
    
    return endpoint_id

def wait_for_endpoint_ready(api_key, endpoint_id, timeout=300):
    """Wait for endpoint to be ready."""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    query = """
    query getEndpoint($id: String!) {
        endpoint(id: $id) {
            id
            name
            status
            readyWorkerCount
        }
    }
    """
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.post(
            RUNPOD_API_BASE,
            json={"query": query, "variables": {"id": endpoint_id}},
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            endpoint = data.get("data", {}).get("endpoint", {})
            status = endpoint.get("status", "")
            ready_workers = endpoint.get("readyWorkerCount", 0)
            
            print(f"Endpoint status: {status}, Ready workers: {ready_workers}")
            
            if status == "READY" and ready_workers > 0:
                print("Endpoint is ready!")
                return True
        
        time.sleep(10)
    
    print("Timeout waiting for endpoint to be ready")
    return False

def main():
    parser = argparse.ArgumentParser(description="Deploy to RunPod Serverless")
    parser.add_argument("--image", required=True, help="Docker image to deploy")
    parser.add_argument("--endpoint-name", required=True, help="Name for the endpoint")
    parser.add_argument("--wait", action="store_true", help="Wait for endpoint to be ready")
    
    args = parser.parse_args()
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY environment variable not set")
        sys.exit(1)
    
    # Deploy endpoint
    endpoint_id = create_endpoint(api_key, args.endpoint_name, args.image)
    
    # Wait for ready if requested
    if args.wait:
        if not wait_for_endpoint_ready(api_key, endpoint_id):
            sys.exit(1)

if __name__ == "__main__":
    main()