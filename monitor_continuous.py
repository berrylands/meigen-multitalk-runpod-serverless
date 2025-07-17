#!/usr/bin/env python3
"""Continuous monitoring of V131 build"""
import subprocess
import json
import time
from datetime import datetime

def check_build_status():
    """Check current build status"""
    try:
        result = subprocess.run(
            ["gh", "run", "list", "--limit", "1", "--json", "status,conclusion,displayTitle,createdAt,url"],
            capture_output=True,
            text=True,
            cwd="/Users/jasonedge/CODEHOME/meigen-multitalk"
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if data:
                return data[0]
        return None
    except Exception as e:
        return None

def check_docker_hub():
    """Check if V131 is available on Docker Hub"""
    try:
        result = subprocess.run(
            ["curl", "-s", "https://hub.docker.com/v2/repositories/berrylands/multitalk-runpod/tags/v131"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and "last_updated" in result.stdout:
            data = json.loads(result.stdout)
            return True, data.get('last_updated', 'Unknown')
        return False, None
    except:
        return False, None

def main():
    print("🔍 Continuous V131 Build Monitoring")
    print("=" * 50)
    
    start_time = time.time()
    last_status = None
    check_count = 0
    
    while True:
        check_count += 1
        elapsed = int(time.time() - start_time)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Get build status
        run_info = check_build_status()
        
        if run_info:
            status = run_info.get('status', 'unknown')
            conclusion = run_info.get('conclusion', '')
            url = run_info.get('url', '')
            
            # Show status update
            if status != last_status:
                print(f"\n[{timestamp}] Status changed: {last_status or 'unknown'} → {status}")
                if status == "in_progress":
                    print(f"🔗 Monitor at: {url}")
                last_status = status
            else:
                print(f"\r[{timestamp}] Check #{check_count}: {status} (elapsed: {elapsed//60}m{elapsed%60}s)", end="", flush=True)
            
            # Check if completed
            if status == "completed":
                print(f"\n\n🎯 Build completed!")
                
                if conclusion == "success":
                    print("✅ Build succeeded!")
                    
                    # Check Docker Hub
                    print("🔍 Checking Docker Hub availability...")
                    for i in range(3):  # Try 3 times
                        available, last_updated = check_docker_hub()
                        if available:
                            print(f"✅ V131 is available on Docker Hub!")
                            print(f"📅 Last updated: {last_updated}")
                            print(f"🐳 Image: berrylands/multitalk-runpod:v131")
                            print("\n🚀 Ready to test!")
                            return
                        else:
                            print(f"⏳ Attempt {i+1}/3: V131 not yet visible on Docker Hub")
                            if i < 2:
                                time.sleep(10)
                    
                    print("⚠️  V131 may take a few more minutes to appear on Docker Hub")
                    
                elif conclusion == "failure":
                    print("❌ Build failed!")
                    print(f"🔗 Check logs: {url}")
                    
                break
        else:
            print(f"\r[{timestamp}] Unable to get build status", end="", flush=True)
        
        time.sleep(30)  # Check every 30 seconds
        
        # Timeout after 25 minutes
        if elapsed > 1500:
            print(f"\n⏰ Timeout after 25 minutes")
            print("Check GitHub Actions manually if needed")
            break

if __name__ == "__main__":
    main()