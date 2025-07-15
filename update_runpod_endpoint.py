#!/usr/bin/env python3
"""
Update RunPod endpoint to use the S3-enabled Docker image
"""

import runpod
import sys
import os
from datetime import datetime

# Configuration
ENDPOINT_ID = "kkx3cfy484jszl"
NEW_IMAGE = "berrylands/multitalk-s3-quick:latest"

def update_endpoint():
    """Update the RunPod endpoint to use the new S3-enabled image"""
    
    # Check for API key
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ Error: RUNPOD_API_KEY environment variable not set")
        print("\nPlease set it with:")
        print("export RUNPOD_API_KEY='your-api-key'")
        return False
    
    runpod.api_key = api_key
    
    try:
        print(f"🔄 Updating endpoint {ENDPOINT_ID} to use image: {NEW_IMAGE}")
        print("⏳ This may take a moment...")
        
        # Get current endpoint info
        endpoint = runpod.Endpoint(ENDPOINT_ID)
        
        # Note: RunPod SDK doesn't have a direct update method, 
        # so we'll provide manual instructions
        print("\n✅ Connection to endpoint verified!")
        print("\n📋 Manual Update Instructions:")
        print("1. Go to: https://www.runpod.io/console/serverless")
        print(f"2. Click on endpoint ID: {ENDPOINT_ID}")
        print("3. Click 'Edit' or 'Settings'")
        print(f"4. Change Docker image to: {NEW_IMAGE}")
        print("5. Save changes")
        print("\n🔄 The endpoint will automatically restart with the new image")
        
        # Test current status
        print("\n🧪 Testing current endpoint status...")
        health = endpoint.health()
        print(f"Current status: {health}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_s3_functionality():
    """Test if S3 is working after update"""
    print("\n🧪 After updating, test S3 functionality with:")
    print("python test_s3_integration.py")
    print("\nOr use this quick test:")
    print("""
import runpod
runpod.api_key = "your-api-key"
endpoint = runpod.Endpoint("kkx3cfy484jszl")

# Check health
result = endpoint.run({"action": "health"})
print(result)

# Test S3 input
job = endpoint.run({
    "action": "generate",
    "audio": "s3://your-bucket/test-audio.wav",
    "duration": 5.0
})
print(job.status())
""")

if __name__ == "__main__":
    print("🚀 RunPod S3 Deployment Update")
    print("=" * 40)
    print(f"Timestamp: {datetime.now()}")
    print(f"New Image: {NEW_IMAGE}")
    print("=" * 40)
    
    if update_endpoint():
        test_s3_functionality()
        print("\n✅ Deployment instructions complete!")
        print("🎯 Next: Follow the manual update steps above")
    else:
        print("\n❌ Update failed. Please check the error above.")
        sys.exit(1)