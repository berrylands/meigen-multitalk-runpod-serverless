#!/usr/bin/env python3
"""
Force RunPod to Use Latest Docker Image
Multiple strategies to ensure RunPod pulls the latest version
"""

import os
import time
import subprocess
from datetime import datetime

def main():
    print("Force RunPod to Use Latest Docker Image")
    print("=" * 60)
    
    print("\n🎯 Strategy 1: Use Unique Version Tags (RECOMMENDED)")
    print("-" * 50)
    print("Instead of using 'latest' or reusing version numbers, use unique tags:")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_tag = f"v2.1.0-{timestamp}"
    
    print(f"\nExample unique tag: berrylands/multitalk-complete:{unique_tag}")
    print("\nBuild command:")
    print(f"docker buildx build --platform linux/amd64 \\")
    print(f"  -t berrylands/multitalk-complete:{unique_tag} \\")
    print(f"  -f Dockerfile.complete --push .")
    
    print("\n✅ Benefits:")
    print("- RunPod MUST pull the new image (tag doesn't exist in cache)")
    print("- You can track exactly which version is running")
    print("- Easy rollback to previous versions")
    
    print("\n" + "=" * 60)
    print("\n🎯 Strategy 2: Use SHA256 Digest (MOST RELIABLE)")
    print("-" * 50)
    print("After pushing, get the exact image digest and use that:")
    
    print("\n1. Push your image:")
    print("   docker push berrylands/multitalk-complete:v2.1.0")
    
    print("\n2. Get the digest:")
    print("   docker inspect berrylands/multitalk-complete:v2.1.0 --format='{{.RepoDigests}}'")
    
    print("\n3. Use the digest in RunPod:")
    print("   berrylands/multitalk-complete@sha256:abc123...")
    
    print("\n✅ Benefits:")
    print("- Absolutely guarantees the exact image version")
    print("- No possibility of cache issues")
    
    print("\n" + "=" * 60)
    print("\n🎯 Strategy 3: Force RunPod Cache Refresh")
    print("-" * 50)
    print("If you must reuse a tag, force RunPod to refresh:")
    
    print("\n1. Change to a temporary tag:")
    print("   - Update endpoint to use 'berrylands/multitalk-complete:temp'")
    print("   - Wait for it to fail (tag doesn't exist)")
    
    print("\n2. Update back to your desired tag:")
    print("   - Change back to 'berrylands/multitalk-complete:v2.1.0'")
    print("   - RunPod will be forced to pull fresh")
    
    print("\n" + "=" * 60)
    print("\n🎯 Strategy 4: Delete and Recreate Endpoint")
    print("-" * 50)
    print("Nuclear option - completely fresh start:")
    
    print("\n1. Note your current endpoint settings")
    print("2. Delete the endpoint")
    print("3. Create a new endpoint with the same settings")
    print("4. Use your new image tag")
    
    print("\n⚠️  Note: This will change your endpoint ID")
    
    print("\n" + "=" * 60)
    print("\n📋 Quick Build & Deploy Script")
    print("-" * 50)
    
    print("\nSave this as 'deploy.sh':")
    print("""
#!/bin/bash
# deploy.sh - Build and deploy with unique tag

# Generate unique tag
TAG="v2.1.0-$(date +%Y%m%d-%H%M%S)"
IMAGE="berrylands/multitalk-complete:$TAG"

echo "Building image: $IMAGE"

# Build and push
cd runpod-multitalk
docker buildx build --platform linux/amd64 -t $IMAGE -f Dockerfile.complete --push .

echo ""
echo "✅ Image pushed: $IMAGE"
echo ""
echo "Now update your RunPod endpoint to use:"
echo "  $IMAGE"
echo ""
echo "This guarantees RunPod will use the latest version!"
""")
    
    print("\n" + "=" * 60)
    print("\n🔍 Verify Current Running Version")
    print("-" * 50)
    
    print("\nTo check what version is actually running:")
    print("\n1. Add version info to your health check:")
    print("   - Include build timestamp in handler")
    print("   - Return Docker image tag in response")
    
    print("\n2. Example health check response:")
    print("""
{
  "status": "healthy",
  "version": "2.1.0",
  "build_time": "2024-01-10 15:30:00",
  "image_tag": "v2.1.0-20240110-153000",
  "handler_checksum": "abc123..."
}
""")
    
    choice = input("\nWould you like to build with a unique timestamp tag now? (y/n): ")
    
    if choice.lower() == 'y':
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        tag = f"v2.1.0-{timestamp}"
        image = f"berrylands/multitalk-complete:{tag}"
        
        print(f"\nBuilding with tag: {tag}")
        
        cmd = [
            "docker", "buildx", "build",
            "--platform", "linux/amd64",
            "-t", image,
            "-f", "runpod-multitalk/Dockerfile.complete",
            "--push",
            "runpod-multitalk"
        ]
        
        print("\nCommand:")
        print(" ".join(cmd))
        
        try:
            subprocess.run(cmd, check=True)
            print(f"\n✅ Success! Image pushed: {image}")
            print(f"\n📋 Update your RunPod endpoint to use: {image}")
            
            # Save the tag for reference
            with open("last_deployed_tag.txt", "w") as f:
                f.write(f"{image}\n")
                f.write(f"Deployed at: {datetime.now()}\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Build failed: {e}")
        except KeyboardInterrupt:
            print("\n\nBuild cancelled.")


if __name__ == "__main__":
    main()