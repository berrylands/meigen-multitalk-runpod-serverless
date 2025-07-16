#!/usr/bin/env python3
"""Quick test to see if V113 is deployed"""
import json
import time

# Use the MCP runpod tools to test the endpoint
def test_endpoint():
    print("Testing current endpoint for V113...")
    
    # Test model check
    job_input = {
        "action": "model_check"
    }
    
    print(f"Testing with input: {json.dumps(job_input)}")
    
    # Write to file for manual testing
    with open("/Users/jasonedge/CODEHOME/meigen-multitalk/test_input.json", "w") as f:
        json.dump(job_input, f, indent=2)
    
    print("Test input written to test_input.json")
    print("Now testing via MCP...")

if __name__ == "__main__":
    test_endpoint()