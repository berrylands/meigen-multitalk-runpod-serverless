#!/usr/bin/env python3
"""Monitor the new V131 build"""
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
    print("🔍 Monitoring New V131 Build (Fresh Runner)")
    print("=" * 55)
    
    check_count = 0
    last_status = None
    
    while True:
        check_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Get build status
        run_info = check_build_status()
        
        if run_info:
            status = run_info.get('status', 'unknown')
            conclusion = run_info.get('conclusion', '')
            url = run_info.get('url', '')
            
            # Show status
            if status != last_status:
                print(f"\n[{timestamp}] Status: {last_status or 'unknown'} → {status}")
                last_status = status
            else:
                print(f"\r[{timestamp}] Check #{check_count}: {status}", end="", flush=True)
            
            # Check if completed
            if status == "completed":
                print(f"\n\n🎯 Build completed with result: {conclusion}")
                
                if conclusion == "success":
                    print("✅ Build succeeded!")
                    
                    # Check Docker Hub
                    print("🔍 Checking Docker Hub...")
                    for attempt in range(5):  # Try 5 times
                        available, last_updated = check_docker_hub()
                        if available:
                            print(f"✅ V131 is available on Docker Hub!")
                            print(f"📅 Last updated: {last_updated}")
                            print(f"🐳 Image: berrylands/multitalk-runpod:v131")
                            print("\n🚀 SUCCESS! Ready to test V131!")
                            return
                        else:
                            print(f"⏳ Attempt {attempt+1}/5: Waiting for Docker Hub...")
                            time.sleep(15)
                    
                    print("⚠️  V131 build succeeded but not yet visible on Docker Hub")
                    print("It may take a few more minutes to propagate")
                    
                elif conclusion == "failure":
                    print("❌ Build failed again!")
                    print(f"🔗 Check logs: {url}")
                    
                return
        else:
            print(f"\r[{timestamp}] Unable to get build status", end="", flush=True)
        
        time.sleep(60)  # Check every minute
        
        # Stop after 20 minutes
        if check_count > 20:
            print(f"\n⏰ Stopping after 20 checks")
            break

if __name__ == "__main__":
    main()