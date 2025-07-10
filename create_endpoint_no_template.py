#!/usr/bin/env python3
"""
Create a new RunPod endpoint without using a template
"""

import os
import runpod
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
runpod.api_key = RUNPOD_API_KEY

def create_new_endpoint():
    """Create a new serverless endpoint without template."""
    
    print("Creating new RunPod endpoint without template...")
    print("=" * 60)
    
    # Use the direct API since SDK might have limitations
    import requests
    
    url = "https://api.runpod.io/v2/endpoints"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Endpoint configuration
    endpoint_data = {
        "name": "multitalk-test-v2",
        "imageName": "berrylands/multitalk-test:latest",
        "gpuIds": "ADA_24",  # RTX 4090
        "networkVolumeId": "pth5bf7dey",
        "locations": ["US-NC-1"],
        "idleTimeout": 60,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "workersMin": 0,
        "workersMax": 1,
        "gpuCount": 1,
        "containerDiskInGb": 10,
        "volumeMountPath": "/runpod-volume",
        "env": [
            {"name": "MODEL_PATH", "value": "/runpod-volume/models"},
            {"name": "RUNPOD_DEBUG_LEVEL", "value": "DEBUG"},
            {"name": "PYTHONUNBUFFERED", "value": "1"}
        ],
        "handlerPath": "handler.handler"  # Python module path
    }
    
    try:
        # Create the endpoint
        response = requests.post(url, json=endpoint_data, headers=headers)
        
        if response.status_code in [200, 201]:
            endpoint = response.json()
            print(f"✓ Endpoint created successfully!")
            print(f"  Response: {endpoint}")
            
            endpoint_id = endpoint.get('id') or endpoint.get('endpointId')
            if endpoint_id:
                print(f"\nIMPORTANT: Save this endpoint ID: {endpoint_id}")
                
                # Write the new endpoint ID to a file
                with open(".new_endpoint_id", "w") as f:
                    f.write(endpoint_id)
                    
                return endpoint_id
            else:
                print("Warning: Could not extract endpoint ID from response")
                return None
        else:
            print(f"✗ Error creating endpoint: {response.status_code}")
            print(f"  Response: {response.text}")
            return None
        
    except Exception as e:
        print(f"✗ Error creating endpoint: {e}")
        return None

if __name__ == "__main__":
    new_endpoint_id = create_new_endpoint()
    
    if new_endpoint_id:
        print("\n" + "=" * 60)
        print("Next steps:")
        print("1. Update your test scripts with the new endpoint ID")
        print("2. Run: python test_new_endpoint.py")
        print("3. Monitor the endpoint in the dashboard")
        
        # Create a test script for the new endpoint
        test_script = f'''#!/usr/bin/env python3
"""Test the new endpoint"""
import os
import time
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

ENDPOINT_ID = "{new_endpoint_id}"

print(f"Testing new endpoint: {{ENDPOINT_ID}}")
print("=" * 60)

endpoint = runpod.Endpoint(ENDPOINT_ID)

# Health check
print("\\n1. Health check:")
try:
    health = endpoint.health()
    print(f"   Health: {{health}}")
except Exception as e:
    print(f"   Error: {{e}}")

# Submit test job
print("\\n2. Submitting test job:")
try:
    job = endpoint.run({{"health_check": True}})
    print(f"   Job ID: {{job.job_id}}")
    
    # Wait for result
    start_time = time.time()
    while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
        elapsed = time.time() - start_time
        print(f"   [{{elapsed:.1f}}s] Status: {{job.status()}}")
        time.sleep(5)
        if elapsed > 120:
            print("   Timeout!")
            break
    
    print(f"\\n   Final status: {{job.status()}}")
    if job.status() == "COMPLETED":
        print(f"   Output: {{job.output()}}")
        
except Exception as e:
    print(f"   Error: {{e}}")
'''
        
        with open("test_new_endpoint.py", "w") as f:
            f.write(test_script)
        os.chmod("test_new_endpoint.py", 0o755)