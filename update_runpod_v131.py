#!/usr/bin/env python3
"""
Update RunPod Template to V131
"""

import os
import subprocess
import json

# RunPod API configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
TEMPLATE_ID = "5y1gyg4n78kqwz"

if not RUNPOD_API_KEY:
    print("❌ Error: RUNPOD_API_KEY not set")
    exit(1)

print("🚀 Updating RunPod template to V131...")
print(f"📋 Template ID: {TEMPLATE_ID}")

# Update template using RunPod API
update_data = {
    "templateId": TEMPLATE_ID,
    "name": "MeiGen MultiTalk V131",
    "imageName": "berrylands/multitalk-runpod:v131",
    "dockerArgs": "",
    "containerDiskInGb": 20,
    "volumeInGb": 0,
    "volumeMountPath": "/runpod-volume",
    "env": [
        {"key": "MODEL_PATH", "value": "/runpod-volume/models"},
        {"key": "HF_HOME", "value": "/runpod-volume/huggingface"},
        {"key": "PYTHONUNBUFFERED", "value": "1"}
    ]
}

# Make the API call
cmd = [
    "curl", "-X", "POST",
    "https://api.runpod.io/graphql",
    "-H", f"Authorization: Bearer {RUNPOD_API_KEY}",
    "-H", "Content-Type: application/json",
    "-d", json.dumps({
        "query": """
            mutation UpdateTemplate($input: UpdateTemplateInput!) {
                updateTemplate(input: $input) {
                    id
                    name
                    imageName
                }
            }
        """,
        "variables": {
            "input": update_data
        }
    })
]

print("\n📡 Updating template...")
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    try:
        response = json.loads(result.stdout)
        if "data" in response and response["data"]["updateTemplate"]:
            print("✅ Template updated successfully!")
            print(f"🐳 New image: berrylands/multitalk-runpod:v131")
            print("\n🎯 V131 improvements:")
            print("  - PyTorch 2.1.0 base image (avoiding NumPy conflicts)")
            print("  - NumPy 1.26.4 (fixed for Numba compatibility)")
            print("  - Numba 0.59.1 installed")
            print("  - All dependencies working")
            print("\n🧪 Ready to test with test_v131_quick.py")
        else:
            print(f"❌ Failed to update template: {response}")
    except Exception as e:
        print(f"❌ Error parsing response: {e}")
        print(f"Response: {result.stdout}")
else:
    print(f"❌ API call failed: {result.stderr}")