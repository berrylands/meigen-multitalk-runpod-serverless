#!/usr/bin/env python3
"""
Direct execution script for building V102 on RunPod pod
This bypasses SSH and uses HTTP to trigger the build
"""

import requests
import time

# Pod details
POD_ID = "3u4ec0uirgphll"
POD_IP = "94.101.98.96"
POD_HTTP_PORT = "8888"  # HTTP port

# Build command
BUILD_COMMAND = """
cd /workspace && \
curl -sSL https://raw.githubusercontent.com/berrylands/meigen-multitalk-runpod-serverless/master/build_v102_single_command.sh | bash
"""

print(f"Executing build on pod {POD_ID}")
print(f"Pod URL: http://{POD_IP}:{POD_HTTP_PORT}")

# Try to trigger build via HTTP endpoint if available
try:
    response = requests.post(
        f"http://{POD_IP}:{POD_HTTP_PORT}/execute",
        json={"command": BUILD_COMMAND},
        timeout=10
    )
    print(f"Response: {response.status_code}")
    print(response.text)
except Exception as e:
    print(f"HTTP execution failed: {e}")
    print("\nPlease manually execute the following in the RunPod web terminal:")
    print("-" * 60)
    print(BUILD_COMMAND)
    print("-" * 60)