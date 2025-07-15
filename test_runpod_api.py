#!/usr/bin/env python3
"""
Test RunPod API access and endpoint information
"""

import os
import runpod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api():
    """Test RunPod API access"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not found")
        return
    
    print("🔍 Testing RunPod API Access")
    print("=" * 60)
    
    # Set API key
    runpod.api_key = api_key
    
    try:
        # Test endpoint access
        endpoint_id = "kkx3cfy484jszl"
        endpoint = runpod.Endpoint(endpoint_id)
        
        print(f"✅ Successfully connected to endpoint: {endpoint_id}")
        
        # Try to get endpoint info
        print("\n📋 Endpoint Information:")
        print(f"Endpoint ID: {endpoint.endpoint_id}")
        
        # Check if we can access any properties
        if hasattr(endpoint, 'status'):
            print(f"Status: {endpoint.status}")
        
        # Try health check
        print("\n🏥 Running health check...")
        try:
            result = endpoint.health()
            print(f"Health: {result}")
        except Exception as e:
            print(f"Health check error: {e}")
        
        # Check for any update methods
        print("\n🔍 Available methods:")
        methods = [m for m in dir(endpoint) if not m.startswith('_')]
        for method in methods:
            print(f"  - {method}")
        
        # Note about updating
        print("\n📝 Note: The RunPod Python SDK doesn't provide direct endpoint update methods.")
        print("Endpoint configuration updates must be done through the web interface:")
        print("https://www.runpod.io/console/serverless")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()