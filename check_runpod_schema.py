#!/usr/bin/env python3
"""
Check RunPod GraphQL schema and current endpoint details
"""

import os
import runpod
import requests
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "kkx3cfy484jszl"

def check_my_endpoints():
    """List all my endpoints to understand the structure."""
    
    print("Checking my endpoints...")
    
    url = f"https://api.runpod.io/graphql?api_key={runpod.api_key}"
    query = '''
    query {
        myself {
            serverlessDiscount {
                discountFactor
            }
            endpoints {
                id
                name
                templateId
                gpuIds
                idleTimeout
                locations
                networkVolumeId
                scalerType
                scalerValue
                workersMax
                workersMin
            }
        }
    }
    '''
    
    try:
        response = requests.post(url, json={"query": query})
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and 'myself' in result['data']:
                endpoints = result['data']['myself'].get('endpoints', [])
                print(f"✅ Found {len(endpoints)} endpoints")
                
                for ep in endpoints:
                    print(f"\n   Endpoint: {ep['name']} (ID: {ep['id']})")
                    print(f"   Template ID: {ep.get('templateId')}")
                    print(f"   GPU: {ep.get('gpuIds')}")
                    print(f"   Network Volume: {ep.get('networkVolumeId')}")
                    
                    if ep['id'] == ENDPOINT_ID:
                        return ep
            else:
                print(f"❌ No data: {result}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return None

def check_templates():
    """List all my templates."""
    
    print("\n\nChecking my templates...")
    
    url = f"https://api.runpod.io/graphql?api_key={runpod.api_key}"
    query = '''
    query {
        myself {
            serverlessTemplates {
                id
                name
                imageName
                dockerArgs
                containerDiskInGb
                volumeInGb
                env {
                    key
                    value
                }
            }
        }
    }
    '''
    
    try:
        response = requests.post(url, json={"query": query})
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and 'myself' in result['data']:
                templates = result['data']['myself'].get('serverlessTemplates', [])
                print(f"✅ Found {len(templates)} templates")
                
                for template in templates:
                    print(f"\n   Template: {template['name']} (ID: {template['id']})")
                    print(f"   Image: {template.get('imageName')}")
                    print(f"   Command: {template.get('dockerArgs')}")
                    
                return templates
            else:
                print(f"❌ No data: {result}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return []

def main():
    print("RunPod Configuration Check")
    print("=" * 60)
    
    endpoint = check_my_endpoints()
    templates = check_templates()
    
    if endpoint:
        print(f"\n✅ Current endpoint configuration:")
        print(f"   ID: {endpoint['id']}")
        print(f"   Name: {endpoint['name']}")
        print(f"   Template ID: {endpoint.get('templateId')}")
        
        # Find the template being used
        template_id = endpoint.get('templateId')
        if template_id and templates:
            for template in templates:
                if template['id'] == template_id:
                    print(f"\n   Using template: {template['name']}")
                    print(f"   Current image: {template.get('imageName')}")
                    print(f"   This is what needs to be updated to: {COMPLETE_IMAGE}")
                    break

if __name__ == "__main__":
    main()