#!/usr/bin/env python3
"""
Check RunPod account status and resources
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    response = requests.post(RUNPOD_API_BASE, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        print(response.text)
        return None
    
    data = response.json()
    if "errors" in data:
        print(f"GraphQL errors: {data['errors']}")
        return None
    
    return data.get("data", {})

def check_account_info():
    """Check account information."""
    query = """
    query {
        myself {
            id
            email
            currentSpendPerHr
            machineQuota
            referralEarnings
            spendLimit
            balance
        }
    }
    """
    
    result = make_graphql_request(query)
    if result:
        user = result.get("myself", {})
        print("Account Information:")
        print(f"  Email: {user.get('email', 'N/A')}")
        print(f"  Balance: ${user.get('balance', 0):.2f}")
        print(f"  Current spend/hr: ${user.get('currentSpendPerHr', 0):.4f}")
        print(f"  Spend limit: ${user.get('spendLimit', 0):.2f}")
        print(f"  Machine quota: {user.get('machineQuota', 0)}")
    return result

def check_network_volumes():
    """Check network storage volumes."""
    query = """
    query {
        myself {
            networkVolumes {
                id
                name
                size
                dataCenterId
                mountPath
            }
        }
    }
    """
    
    volumes = []
    result = make_graphql_request(query)
    if result:
        volumes = result.get("myself", {}).get("networkVolumes", [])
        print(f"\nNetwork Volumes ({len(volumes)} found):")
        for vol in volumes:
            print(f"  - {vol['name']}")
            print(f"    ID: {vol['id']}")
            print(f"    Size: {vol['size']}GB")
            print(f"    Data Center: {vol['dataCenterId']}")
            print(f"    Mount Path: {vol.get('mountPath', 'N/A')}")
    return volumes

def check_serverless_endpoints():
    """Check existing serverless endpoints."""
    query = """
    query {
        myself {
            serverlessEndpoints {
                id
                name
                status
                dockerImage
                gpuType
                minWorkers
                maxWorkers
                readyWorkerCount
                queuedJobCount
            }
        }
    }
    """
    
    result = make_graphql_request(query)
    if result:
        endpoints = result.get("myself", {}).get("serverlessEndpoints", [])
        print(f"\nServerless Endpoints ({len(endpoints)} found):")
        for ep in endpoints:
            print(f"  - {ep['name']}")
            print(f"    ID: {ep['id']}")
            print(f"    Status: {ep['status']}")
            print(f"    Image: {ep['dockerImage']}")
            print(f"    GPU: {ep['gpuType']}")
            print(f"    Workers: {ep['minWorkers']}-{ep['maxWorkers']} (ready: {ep['readyWorkerCount']})")
            print(f"    Queued jobs: {ep['queuedJobCount']}")
    return endpoints

def check_available_gpus():
    """Check available GPU types."""
    query = """
    query availableGpuTypes {
        gpuTypes {
            id
            displayName
            memoryInGb
        }
    }
    """
    
    result = make_graphql_request(query)
    if result:
        gpu_types = result.get("gpuTypes", [])
        print(f"\nAvailable GPU Types:")
        for gpu in gpu_types:
            print(f"  - {gpu['displayName']} ({gpu['id']})")
            print(f"    Memory: {gpu['memoryInGb']}GB")
    return result

def main():
    if not RUNPOD_API_KEY:
        print("Error: RUNPOD_API_KEY not found in environment")
        sys.exit(1)
    
    print("RunPod Account Status Check")
    print("=" * 60)
    
    # Check account
    account = check_account_info()
    
    # Check network volumes
    volumes = check_network_volumes()
    
    # Check endpoints
    endpoints = check_serverless_endpoints()
    
    # Check available GPUs
    gpus = check_available_gpus()
    
    print("\n" + "=" * 60)
    print("Status check complete!")
    
    # Return summary
    return {
        "has_balance": account and account.get("myself", {}).get("balance", 0) > 0,
        "has_network_volume": len(volumes) > 0,
        "has_endpoints": len(endpoints) > 0,
        "volumes": volumes,
        "endpoints": endpoints
    }

if __name__ == "__main__":
    main()