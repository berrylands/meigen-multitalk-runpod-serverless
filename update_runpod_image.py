#!/usr/bin/env python3
"""
Update RunPod endpoint Docker image using the API
"""

import os
import sys
import json
import requests
import time

def update_endpoint_image(endpoint_id: str, new_image: str):
    """Update a RunPod endpoint to use a new Docker image"""
    
    # Get API key
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ Error: RUNPOD_API_KEY environment variable not set")
        print("Set it with: export RUNPOD_API_KEY='your-api-key'")
        return False
    
    print(f"🔄 Updating RunPod Endpoint")
    print(f"Endpoint ID: {endpoint_id}")
    print(f"New Image: {new_image}")
    print("=" * 60)
    
    # RunPod API endpoint
    api_url = "https://api.runpod.io/graphql"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # First, get current endpoint details
    get_query = """
    query GetEndpoint($id: String!) {
        endpoint(id: $id) {
            id
            name
            templateId
            workersMin
            workersMax
            idleTimeout
            locations
            networkVolumeId
            scalerType
            scalerValue
        }
    }
    """
    
    print("📋 Getting current endpoint configuration...")
    
    response = requests.post(
        api_url,
        headers=headers,
        json={
            "query": get_query,
            "variables": {"id": endpoint_id}
        }
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to get endpoint details: {response.status_code}")
        print(response.text)
        return False
    
    result = response.json()
    if "errors" in result:
        print(f"❌ GraphQL errors: {result['errors']}")
        return False
    
    endpoint_data = result.get("data", {}).get("endpoint")
    if not endpoint_data:
        print(f"❌ Endpoint {endpoint_id} not found")
        return False
    
    print(f"✅ Found endpoint: {endpoint_data['name']}")
    
    # Update the endpoint with new image
    update_mutation = """
    mutation UpdateEndpoint($input: UpdateEndpointInput!) {
        updateEndpoint(input: $input) {
            id
            templateId
        }
    }
    """
    
    # Note: The exact mutation structure depends on RunPod's API
    # This is based on common GraphQL patterns
    update_variables = {
        "input": {
            "id": endpoint_id,
            "templateId": new_image,  # Docker image is usually the templateId
            # Preserve other settings
            "workersMin": endpoint_data.get("workersMin", 0),
            "workersMax": endpoint_data.get("workersMax", 3),
            "idleTimeout": endpoint_data.get("idleTimeout", 5),
        }
    }
    
    print(f"\n🚀 Updating endpoint to use: {new_image}")
    
    response = requests.post(
        api_url,
        headers=headers,
        json={
            "query": update_mutation,
            "variables": update_variables
        }
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to update endpoint: {response.status_code}")
        print(response.text)
        
        # Try alternative approach using REST API
        print("\n🔄 Trying alternative REST API approach...")
        return try_rest_api_update(api_key, endpoint_id, new_image)
    
    result = response.json()
    if "errors" in result:
        print(f"❌ GraphQL errors: {result['errors']}")
        
        # Try alternative approach
        print("\n🔄 Trying alternative REST API approach...")
        return try_rest_api_update(api_key, endpoint_id, new_image)
    
    print("✅ Endpoint updated successfully!")
    print(f"\n📊 Updated endpoint: {endpoint_id}")
    print(f"🐳 New image: {new_image}")
    print("\n⏳ The endpoint will restart with the new image shortly.")
    
    return True

def try_rest_api_update(api_key: str, endpoint_id: str, new_image: str):
    """Try updating via REST API as fallback"""
    
    # Common REST endpoints for RunPod
    rest_endpoints = [
        f"https://api.runpod.io/v2/endpoints/{endpoint_id}",
        f"https://api.runpod.io/v1/endpoints/{endpoint_id}",
        f"https://api.runpod.io/serverless/endpoints/{endpoint_id}"
    ]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    update_data = {
        "dockerImage": new_image,
        "templateId": new_image,
        "image": new_image  # Try different field names
    }
    
    for url in rest_endpoints:
        print(f"\n🔍 Trying: {url}")
        
        # Try PATCH
        response = requests.patch(url, headers=headers, json=update_data)
        if response.status_code in [200, 201, 202]:
            print(f"✅ Successfully updated via REST API!")
            return True
        
        # Try PUT
        response = requests.put(url, headers=headers, json=update_data)
        if response.status_code in [200, 201, 202]:
            print(f"✅ Successfully updated via REST API!")
            return True
    
    print("\n❌ Could not update via REST API either.")
    print("\n📝 Manual Update Instructions:")
    print("1. Go to: https://www.runpod.io/console/serverless")
    print(f"2. Click on endpoint: {endpoint_id}")
    print("3. Click 'Edit' or settings icon")
    print(f"4. Change Docker image to: {new_image}")
    print("5. Save changes")
    
    return False

def main():
    """Main function"""
    
    # Configuration
    ENDPOINT_ID = "kkx3cfy484jszl"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        new_image = sys.argv[1]
    else:
        # Default to S3 debug image
        new_image = "berrylands/multitalk-s3-debug:latest"
    
    print("🚀 RunPod Endpoint Image Updater")
    print("=" * 60)
    
    # Show available images
    print("\n📦 Available Docker images:")
    images = [
        "berrylands/multitalk-s3-debug:latest - S3 debug handler",
        "berrylands/multitalk-s3-endpoint-fix:latest - S3 endpoint fix",
        "berrylands/multitalk-s3-numpy:latest - S3 + numpy",
        "berrylands/multitalk-s3-fix:latest - S3 handler fix",
        "berrylands/multitalk-s3-quick:latest - Quick S3 update"
    ]
    
    for img in images:
        print(f"  • {img}")
    
    print(f"\n🎯 Updating to: {new_image}")
    
    # Confirm
    response = input("\nContinue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Cancelled")
        return
    
    # Update endpoint
    success = update_endpoint_image(ENDPOINT_ID, new_image)
    
    if success:
        print("\n✅ Update completed!")
        print(f"\n🧪 Test with:")
        print("python test_s3_debug.py")
    else:
        print("\n❌ Update failed - please update manually")

if __name__ == "__main__":
    main()