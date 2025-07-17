#!/usr/bin/env python3
"""
Monitor V131 Build Status
"""

import subprocess
import time
import json
import sys

def check_docker_image():
    """Check if V131 image is available on Docker Hub"""
    cmd = [
        "curl", "-s",
        "https://hub.docker.com/v2/repositories/berrylands/multitalk-runpod/tags/v131"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and "v131" in result.stdout:
            data = json.loads(result.stdout)
            if "last_updated" in data:
                return True, data['last_updated']
        return False, None
    except:
        return False, None

def main():
    print("🔍 Monitoring V131 Build Status...")
    print("=" * 60)
    
    start_time = time.time()
    check_count = 0
    
    while True:
        check_count += 1
        elapsed = int(time.time() - start_time)
        
        print(f"\r[{elapsed}s] Check #{check_count}: ", end="", flush=True)
        
        ready, last_updated = check_docker_image()
        
        if ready:
            print(f"\n\n✅ V131 is ready on Docker Hub!")
            print(f"📅 Last updated: {last_updated}")
            print(f"🐳 Image: berrylands/multitalk-runpod:v131")
            print("\n🎯 Next steps:")
            print("1. Update RunPod template to use v131")
            print("2. Run test_v131_quick.py to verify NumPy fix")
            break
        else:
            print("⏳ Still building...", end="", flush=True)
            time.sleep(30)  # Check every 30 seconds
            
            if elapsed > 1800:  # 30 minute timeout
                print("\n\n⚠️  Build is taking longer than expected.")
                print("Check: https://github.com/berrylands/meigen-multitalk-runpod-serverless/actions")
                break

if __name__ == "__main__":
    main()