#!/usr/bin/env python3
"""
RunPod API utilities
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load environment variables
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

def check_account():
    """Check account information."""
    query = """
    query {
        myself {
            id
            email
            currentSpendPerHr
            referralEarnings
            spendLimit
        }
    }
    """
    
    result = make_graphql_request(query)
    if result:
        user = result.get("myself", {})
        print("Account Information:")
        print(f"  ID: {user.get('id', 'N/A')}")
        print(f"  Email: {user.get('email', 'N/A')}")
        print(f"  Current spend/hr: ${user.get('currentSpendPerHr', 0):.4f}")
        print(f"  Spend limit: ${user.get('spendLimit', 0):.2f}")
    return result

def list_network_volumes():
    """List network storage volumes."""
    query = """
    query {
        myself {
            networkVolumes {
                id
                name
                size
                dataCenterId
            }
        }
    }
    """
    
    result = make_graphql_request(query)
    volumes = []
    if result:
        volumes = result.get("myself", {}).get("networkVolumes", [])
        print(f"\nNetwork Volumes ({len(volumes)} found):")
        for vol in volumes:
            print(f"  - {vol['name']} ({vol['size']}GB)")
            print(f"    ID: {vol['id']}")
            print(f"    Data Center: {vol['dataCenterId']}")
    return volumes

def create_network_volume(name, size_gb, data_center_id):
    """Create a network volume."""
    mutation = """
    mutation CreateNetworkVolume($input: CreateNetworkVolumeInput!) {
        createNetworkVolume(input: $input) {
            id
            name
            size
            dataCenterId
        }
    }
    """
    
    variables = {
        "input": {
            "name": name,
            "size": size_gb,
            "dataCenterId": data_center_id
        }
    }
    
    result = make_graphql_request(mutation, variables)
    if result:
        volume = result.get("createNetworkVolume", {})
        print(f"Created volume: {volume['name']} ({volume['size']}GB)")
        return volume
    return None

def list_serverless_endpoints():
    """List serverless endpoints."""
    query = """
    query {
        myself {
            endpoints {
                id
                name
                templateId
                workersMin
                workersMax
                gpuIds
            }
        }
    }
    """
    
    result = make_graphql_request(query)
    endpoints = []
    if result:
        endpoints = result.get("myself", {}).get("endpoints", [])
        print(f"\nServerless Endpoints ({len(endpoints)} found):")
        for ep in endpoints:
            print(f"  - {ep['name']}")
            print(f"    ID: {ep['id']}")
            print(f"    Workers: {ep.get('workersMin', 0)}-{ep.get('workersMax', 0)}")
    return endpoints

def get_gpu_types():
    """Get available GPU types."""
    query = """
    query {
        gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
        }
    }
    """
    
    result = make_graphql_request(query)
    if result:
        gpu_types = result.get("gpuTypes", [])
        print("\nAvailable GPU Types:")
        for gpu in gpu_types:
            if gpu.get('secureCloud') or gpu.get('communityCloud'):
                print(f"  - {gpu['displayName']} ({gpu['id']})")
                print(f"    Memory: {gpu['memoryInGb']}GB")
                print(f"    Available in: ", end="")
                locations = []
                if gpu.get('secureCloud'):
                    locations.append("Secure Cloud")
                if gpu.get('communityCloud'):
                    locations.append("Community Cloud")
                print(", ".join(locations))
        return gpu_types
    return []

def get_data_centers():
    """Get available data centers."""
    query = """
    query {
        datacenters {
            id
            name
            location
        }
    }
    """
    
    result = make_graphql_request(query)
    if result:
        datacenters = result.get("datacenters", [])
        print("\nAvailable Data Centers:")
        for dc in datacenters:
            print(f"  - {dc['name']} ({dc['id']})")
            print(f"    Location: {dc['location']}")
        return datacenters
    return []

if __name__ == "__main__":
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY not found in environment")
        sys.exit(1)
    
    print("RunPod API Test")
    print("=" * 60)
    
    # Check account
    check_account()
    
    # Get available resources
    get_data_centers()
    get_gpu_types()
    
    # List existing resources
    list_network_volumes()
    list_serverless_endpoints()