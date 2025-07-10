#!/usr/bin/env python3
"""
Create a new RunPod serverless endpoint using the correct API
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

def create_endpoint():
    """Create endpoint using RunPod API."""
    
    print("Creating new RunPod endpoint...")
    print("=" * 60)
    
    # RunPod uses GraphQL for endpoint creation
    url = "https://api.runpod.ai/graphql"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}"
    }
    
    # GraphQL mutation for creating a serverless endpoint
    mutation = """
    mutation CreateEndpoint {
        saveEndpoint(input: {
            name: "multitalk-test-v2",
            templateId: null,
            gpuIds: "ADA_24",
            networkVolumeId: "pth5bf7dey",
            locations: "US-NC-1",
            idleTimeout: 60,
            scalerType: "QUEUE_DELAY",
            scalerValue: 4,
            workersMin: 0,
            workersMax: 1,
            workersStandby: 0,
            flashBootEnabled: false,
            gpuCount: 1,
            volumeMountPath: "/runpod-volume",
            containerDiskInGb: 10,
            containerRegistryAuthId: null,
            env: [
                {key: "MODEL_PATH", value: "/runpod-volume/models"},
                {key: "RUNPOD_DEBUG_LEVEL", value: "DEBUG"},
                {key: "PYTHONUNBUFFERED", value: "1"}
            ],
            dockerArgs: "berrylands/multitalk-test:latest python -u handler.py"
        }) {
            id
            name
            templateId
        }
    }
    """
    
    try:
        response = requests.post(
            url,
            json={"query": mutation},
            headers=headers
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if "errors" in result:
                print(f"✗ GraphQL errors: {result['errors']}")
                return None
                
            if "data" in result and result["data"]:
                endpoint = result["data"].get("saveEndpoint", {})
                endpoint_id = endpoint.get("id")
                
                if endpoint_id:
                    print(f"✓ Endpoint created successfully!")
                    print(f"  ID: {endpoint_id}")
                    print(f"  Name: {endpoint.get('name')}")
                    
                    # Save endpoint ID
                    with open(".new_endpoint_id", "w") as f:
                        f.write(endpoint_id)
                    
                    return endpoint_id
                else:
                    print("✗ No endpoint ID in response")
                    print(f"Response: {json.dumps(result, indent=2)}")
            else:
                print("✗ No data in response")
                print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"✗ HTTP error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"✗ Exception: {e}")
        
    return None

def test_new_endpoint(endpoint_id):
    """Create test script for new endpoint."""
    
    test_script = f'''#!/usr/bin/env python3
"""Test the new endpoint"""
import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "{endpoint_id}"

print(f"Testing new endpoint: {{ENDPOINT_ID}}")
print("=" * 60)

endpoint = runpod.Endpoint(ENDPOINT_ID)

# Check health
print("\\nChecking endpoint health...")
try:
    health = endpoint.health()
    print(f"Health: {{health}}")
except Exception as e:
    print(f"Health check error: {{e}}")

# Submit test job
print("\\nSubmitting test job...")
try:
    job = endpoint.run({{"health_check": True}})
    print(f"Job ID: {{job.job_id}}")
    
    # Monitor status
    start_time = time.time()
    last_status = None
    
    while True:
        status = job.status()
        if status != last_status:
            elapsed = time.time() - start_time
            print(f"[{{elapsed:.1f}}s] Status: {{status}}")
            last_status = status
            
        if status not in ["IN_QUEUE", "IN_PROGRESS"]:
            break
            
        if time.time() - start_time > 180:  # 3 minute timeout
            print("Timeout waiting for job completion")
            break
            
        time.sleep(5)
    
    # Get final result
    if job.status() == "COMPLETED":
        print(f"\\n✓ Job completed successfully!")
        print(f"Output: {{job.output()}}")
    else:
        print(f"\\n✗ Job failed with status: {{job.status()}}")
        try:
            print(f"Output: {{job.output()}}")
        except:
            pass
            
except Exception as e:
    print(f"Error: {{e}}")

print("\\n" + "=" * 60)
'''
    
    with open("test_new_endpoint.py", "w") as f:
        f.write(test_script)
    os.chmod("test_new_endpoint.py", 0o755)
    
    print(f"\nTest script created: test_new_endpoint.py")
    print("Run it with: python test_new_endpoint.py")

if __name__ == "__main__":
    endpoint_id = create_endpoint()
    
    if endpoint_id:
        print("\n" + "=" * 60)
        print("SUCCESS! New endpoint created without template")
        print(f"Endpoint ID: {endpoint_id}")
        print("\nThis endpoint:")
        print("- Uses your Docker image directly")
        print("- Has no template interference")
        print("- Should start workers properly")
        
        test_new_endpoint(endpoint_id)
    else:
        print("\n" + "=" * 60)
        print("Failed to create endpoint")
        print("Please check the RunPod dashboard and create manually")