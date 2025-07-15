#!/usr/bin/env python3

"""
Script to update RunPod template after manual build
"""

import os
import glob
import re

# Find latest version
dockerfile_files = glob.glob('runpod-multitalk/Dockerfile.v*')
version_files = [f for f in dockerfile_files if re.match(r'.*Dockerfile\.v\d+$', f)]
if not version_files:
    print("No versioned Dockerfiles found")
    exit(1)

latest_file = sorted(version_files, key=lambda x: int(re.search(r'v(\d+)$', x).group(1)))[-1]
latest_version = re.search(r'v(\d+)$', latest_file).group(1)

print(f"Latest version: {latest_version}")
print(f"Image: berrylands/multitalk-runpod:v{latest_version}")

# Instructions for manual template update
print("\n=== Manual Template Update Instructions ===")
print("1. Go to RunPod console -> Templates")
print("2. Find template 'multitalk-v95-proper-implementation' (or latest)")
print("3. Click Edit")
print(f"4. Change Image Name to: berrylands/multitalk-runpod:v{latest_version}")
print(f"5. Change Name to: multitalk-v{latest_version}-manual-build")
print(f"6. Update README to: MultiTalk V{latest_version} - Manual build with xfuser")
print("7. Save template")
print("8. Update your endpoint to use the new template")

print(f"\nOr use this command to update via API:")
print(f"# Update template with new image")
print(f"# Template ID: joospbpdol")
print(f"# New image: berrylands/multitalk-runpod:v{latest_version}")