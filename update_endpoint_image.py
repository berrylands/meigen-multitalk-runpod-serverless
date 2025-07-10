#!/usr/bin/env python3
"""
Instructions to update endpoint image
"""

ENDPOINT_ID = "kkx3cfy484jszl"
NEW_IMAGE = "berrylands/multitalk-download:latest"

print(f"""
UPDATE ENDPOINT IMAGE
{'=' * 60}

Please update your endpoint to use the new image with download support:

1. Go to: https://www.runpod.io/console/serverless
2. Click on your endpoint (ID: {ENDPOINT_ID})
3. Click "Edit" or settings icon
4. Update the Container Image to: {NEW_IMAGE}
5. Keep all other settings the same
6. Click "Save" or "Update"

The new image includes:
- Model download functionality
- HuggingFace Hub integration
- List models action
- Same health check functionality

Once updated, I'll test the download functionality.
""")