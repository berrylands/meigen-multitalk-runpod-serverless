#!/usr/bin/env python3
"""
Update RunPod template to V108
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

def update_template():
    """Update RunPod template to V108"""
    
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    # Template details
    template_id = "joospbpdol"
    new_image = "berrylands/multitalk-runpod:v108"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Get current template
    print("🔍 Getting current template...")
    response = requests.get(
        f"https://api.runpod.ai/graphql",
        headers=headers,
        json={
            "query": """
            query GetTemplate($templateId: String!) {
                template(id: $templateId) {
                    id
                    name
                    imageName
                    containerDiskInGb
                    volumeInGb
                    volumeMountPath
                    env {
                        key
                        value
                    }
                    ports
                    readme
                    isPublic
                    dockerEntrypoint
                    dockerStartCmd
                }
            }
            """,
            "variables": {"templateId": template_id}
        }
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to get template: {response.status_code}")
        return
    
    data = response.json()
    if "errors" in data:
        print(f"❌ GraphQL errors: {data['errors']}")
        return
    
    template = data["data"]["template"]
    if not template:
        print(f"❌ Template {template_id} not found")
        return
    
    print(f"📋 Current template: {template['name']}")
    print(f"🐳 Current image: {template['imageName']}")
    
    # Update template
    print(f"🔄 Updating template to {new_image}...")
    
    response = requests.post(
        f"https://api.runpod.ai/graphql",
        headers=headers,
        json={
            "query": """
            mutation UpdateTemplate(
                $templateId: String!,
                $name: String!,
                $imageName: String!,
                $containerDiskInGb: Int,
                $volumeInGb: Int,
                $volumeMountPath: String,
                $env: [EnvironmentVariableInput!],
                $ports: String,
                $readme: String,
                $isPublic: Boolean,
                $dockerEntrypoint: String,
                $dockerStartCmd: String
            ) {
                updateTemplate(
                    templateId: $templateId,
                    name: $name,
                    imageName: $imageName,
                    containerDiskInGb: $containerDiskInGb,
                    volumeInGb: $volumeInGb,
                    volumeMountPath: $volumeMountPath,
                    env: $env,
                    ports: $ports,
                    readme: $readme,
                    isPublic: $isPublic,
                    dockerEntrypoint: $dockerEntrypoint,
                    dockerStartCmd: $dockerStartCmd
                ) {
                    id
                    name
                    imageName
                }
            }
            """,
            "variables": {
                "templateId": template_id,
                "name": template["name"].replace("v91", "v108") if "v91" in template["name"] else template["name"],
                "imageName": new_image,
                "containerDiskInGb": template["containerDiskInGb"],
                "volumeInGb": template["volumeInGb"],
                "volumeMountPath": template["volumeMountPath"],
                "env": template["env"],
                "ports": template["ports"],
                "readme": template["readme"],
                "isPublic": template["isPublic"],
                "dockerEntrypoint": template["dockerEntrypoint"],
                "dockerStartCmd": template["dockerStartCmd"]
            }
        }
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to update template: {response.status_code}")
        print(response.text)
        return
    
    data = response.json()
    if "errors" in data:
        print(f"❌ GraphQL errors: {data['errors']}")
        return
    
    updated_template = data["data"]["updateTemplate"]
    print(f"✅ Template updated successfully!")
    print(f"📋 Name: {updated_template['name']}")
    print(f"🐳 Image: {updated_template['imageName']}")
    
    print("\n🚀 Template is now ready for testing!")
    print("Run: ./test_v108_models.sh")

if __name__ == "__main__":
    update_template()