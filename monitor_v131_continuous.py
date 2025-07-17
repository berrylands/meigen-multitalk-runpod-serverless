#!/usr/bin/env python3
"""Continuously monitor V131 build progress"""
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

def main():
    print("🔍 Continuous V131 Build Monitor")
    print("=" * 50)
    start_time = time.time()
    
    while True:
        elapsed = int(time.time() - start_time)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Check Docker Hub
        available, last_updated = check_docker_hub()
        
        if available:
            print(f"\n✅ [{timestamp}] V131 is available on Docker Hub!")
            print(f"📅 Last updated: {last_updated}")
            print(f"🐳 Image: berrylands/multitalk-runpod:v131")
            print("\n🎯 Ready to test!")
            print("Run: python test_v131_fixed.py")
            break
        
        # Check local build
        local_status = check_local_build()
        status_emoji = {
            "not_started": "⏸️",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌"
        }
        
        print(f"\r[{timestamp}] Docker Hub: ⏳ Not ready | Local Build: {status_emoji.get(local_status, '❓')} {local_status.title()}", end="", flush=True)
        
        if local_status == "completed":
            print(f"\n✅ [{timestamp}] Local build completed!")
            # Check Docker Hub again in case it's now available
            time.sleep(5)
            continue
        elif local_status == "failed":
            print(f"\n❌ [{timestamp}] Local build failed!")
            print("Check runpod-multitalk/v131_build.log for details")
            break
        
        time.sleep(30)  # Check every 30 seconds
        
        # Timeout after 30 minutes
        if elapsed > 1800:
            print(f"\n⏰ [{timestamp}] Timeout after 30 minutes")
            break

if __name__ == "__main__":
    main()