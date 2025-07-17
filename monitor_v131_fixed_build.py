#!/usr/bin/env python3
"""Monitor V131-fixed build status"""
import subprocess
import time
import json

print("🔍 Monitoring V131-fixed Build Status...")
print("=" * 60)
print("GitHub Actions: https://github.com/berrylands/meigen-multitalk-runpod-serverless/actions")
print("=" * 60)

start_time = time.time()
check_count = 0

while True:
    check_count += 1
    elapsed = int(time.time() - start_time)
    
    # Check Docker Hub for v131-fixed
    cmd = [
        "curl", "-s",
        "https://hub.docker.com/v2/repositories/berrylands/multitalk-runpod/tags/v131-fixed"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and "v131-fixed" in result.stdout:
        try:
            data = json.loads(result.stdout)
            if "last_updated" in data:
                print(f"\n\n✅ V131-fixed is ready on Docker Hub!")
                print(f"📅 Last updated: {data['last_updated']}")
                print(f"🐳 Image: berrylands/multitalk-runpod:v131")
                print("\n🎯 Next steps:")
                print("1. Update RunPod template to use v131")
                print("2. Test with test_v131_fixed.py")
                break
        except:
            pass
    
    # Also check for v131 tag (without -fixed)
    cmd2 = [
        "curl", "-s",
        "https://hub.docker.com/v2/repositories/berrylands/multitalk-runpod/tags/v131"
    ]
    
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    if result2.returncode == 0 and "v131" in result2.stdout and "v131-fixed" not in result2.stdout:
        try:
            data = json.loads(result2.stdout)
            if "last_updated" in data:
                # Check if it's newer than 5 minutes ago
                print(f"\n\n✅ V131 is ready on Docker Hub!")
                print(f"📅 Last updated: {data['last_updated']}")
                print(f"🐳 Image: berrylands/multitalk-runpod:v131")
                print("\n🎯 Next steps:")
                print("1. Template already set to v131")
                print("2. Test with existing jobs or submit new one")
                break
        except:
            pass
    
    print(f"\r[{elapsed}s] Check #{check_count}: Build in progress...", end="", flush=True)
    time.sleep(30)
    
    if elapsed > 1200:  # 20 minute timeout
        print("\n\n⏰ Timeout after 20 minutes")
        print("Check GitHub Actions for build status")
        break