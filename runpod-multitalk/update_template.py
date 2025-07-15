#!/usr/bin/env python3
"""
Update RunPod template with new image version
"""
import os
import sys
import runpod

# Template configuration
TEMPLATE_ID = "joospbpdol"
TEMPLATE_NAME = "multitalk-v80-auto-update"

def update_template(version):
    """Update the RunPod template with a new image version"""
    
    # Initialize RunPod client
    runpod.api_key = os.environ.get("RUNPOD_API_KEY", "CKRTDIOF0IGFFSI4A11KTVP569QQAKQ4NK091965")
    
    image_name = f"berrylands/multitalk-runpod:{version}"
    
    print(f"🔄 Updating template {TEMPLATE_ID} to use image: {image_name}")
    
    try:
        # Update template
        result = runpod.update_template(
            template_id=TEMPLATE_ID,
            name=TEMPLATE_NAME,
            image_name=image_name,
            container_disk_in_gb=50,
            volume_mount_path="/runpod-volume",
            env={
                "AWS_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_AWS_ACCESS_KEY_ID }}",
                "AWS_REGION": "{{ RUNPOD_SECRET_AWS_REGION }}",
                "AWS_S3_BUCKET_NAME": "{{ RUNPOD_SECRET_AWS_S3_BUCKET_NAME }}",
                "AWS_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_AWS_SECRET_ACCESS_KEY }}",
                "BUCKET_ENDPOINT_URL": "{{ RUNPOD_SECRET_BUCKET_ENDPOINT_URL }}",
                "HF_HOME": "/runpod-volume/huggingface",
                "TRANSFORMERS_CACHE": "/runpod-volume/huggingface",
                "MODEL_PATH": "/runpod-volume/models"
            },
            is_serverless=True,
            readme=f"MultiTalk {version} - Auto-update template with fixed xfuser and cache permissions"
        )
        
        print(f"✅ Template updated successfully!")
        print(f"📋 Template ID: {TEMPLATE_ID}")
        print(f"🎯 New image: {image_name}")
        
        # Update endpoint to use new template
        print("\n🔄 Updating endpoint to use new template...")
        # This would update the endpoint, but we need the endpoint ID
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update template: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_template.py <version>")
        print("Example: python update_template.py v80")
        sys.exit(1)
    
    version = sys.argv[1]
    success = update_template(version)
    sys.exit(0 if success else 1)