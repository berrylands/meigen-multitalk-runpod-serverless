#!/usr/bin/env python3
"""Monitor V131 build progress"""
import subprocess
import time
import json
import os
from datetime import datetime

def check_docker_hub():
    """Check if V131 is available on Docker Hub"""
    cmd = ["curl", "-s", "https://hub.docker.com/v2/repositories/berrylands/multitalk-runpod/tags/v131"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and "last_updated" in result.stdout:
        try:
            data = json.loads(result.stdout)
            return True, data.get('last_updated', 'Unknown')
        except:
            pass
    return False, None

def check_local_build():
    """Check local build status"""
    log_file = "runpod-multitalk/v131_build.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            content = f.read()
            if "Build successful!" in content:
                return "completed"
            elif "Build failed" in content:
                return "failed"
            else:
                return "running"
    return "not_started"

print("🔍 Monitoring V131 Build...")
print("=" * 40)

for i in range(10):  # Check 10 times over 5 minutes
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Check Docker Hub
    available, last_updated = check_docker_hub()
    
    if available:
        print(f"\n✅ [{timestamp}] V131 is available on Docker Hub!")
        print(f"📅 Last updated: {last_updated}")
        print(f"🐳 Image: berrylands/multitalk-runpod:v131")
        print("\n🎯 Ready to test!")
        exit(0)
    
    # Check local build
    local_status = check_local_build()
    
    print(f"[{timestamp}] Docker Hub: ⏳ Building | Local: {local_status}")
    
    if local_status == "completed":
        print(f"✅ Local build completed! Checking Docker Hub...")
        time.sleep(5)
        continue
    elif local_status == "failed":
        print(f"❌ Local build failed!")
        if os.path.exists("runpod-multitalk/v131_build.log"):
            print("Last few lines of build log:")
            with open("runpod-multitalk/v131_build.log", 'r') as f:
                lines = f.readlines()
                for line in lines[-5:]:
                    print(f"  {line.strip()}")
        break
    
    time.sleep(30)  # Wait 30 seconds

print("\n⏰ Monitoring complete")