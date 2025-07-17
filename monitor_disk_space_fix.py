#!/usr/bin/env python3
"""Monitor V131 build with disk space fix"""
import subprocess
import json
import time
from datetime import datetime

def get_latest_run():
    """Get the latest GitHub Actions run"""
    try:
        result = subprocess.run(
            ["gh", "run", "list", "--limit", "1", "--json", "status,conclusion,displayTitle,createdAt,url,databaseId"],
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
    print("🔍 V131 Build Monitor - With Disk Space Fix")
    print("=" * 60)
    print("🛠️  Applied fix: maximize-build-space action")
    print("💾 Expected free space: 25-29GB+ (vs previous ~14GB)")
    print("🎯 Should resolve 'No space left on device' errors")
    print("=" * 60)
    
    check_count = 0
    last_status = None
    start_time = time.time()
    
    while True:
        check_count += 1
        elapsed = int(time.time() - start_time)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Get build status
        run_info = get_latest_run()
        
        if run_info:
            status = run_info.get('status', 'unknown')
            conclusion = run_info.get('conclusion', '')
            url = run_info.get('url', '')
            
            # Show status update
            if status != last_status:
                print(f"\n[{timestamp}] Status: {last_status or 'unknown'} → {status}")
                if status == "in_progress":
                    print(f"🔗 Monitor: {url}")
                last_status = status
            else:
                print(f"\r[{timestamp}] Check #{check_count}: {status} (runtime: {elapsed//60}m{elapsed%60}s)", end="", flush=True)
            
            # Check if completed
            if status == "completed":
                print(f"\n\n🎯 Build completed with result: {conclusion}")
                
                if conclusion == "success":
                    print("✅ SUCCESS! Disk space fix worked!")
                    print("🎉 V131 build completed successfully!")
                    
                    # Check Docker Hub
                    print("\n🔍 Checking Docker Hub...")
                    for attempt in range(5):
                        available, last_updated = check_docker_hub()
                        if available:
                            print(f"✅ V131 is available on Docker Hub!")
                            print(f"📅 Last updated: {last_updated}")
                            print(f"🐳 Image: berrylands/multitalk-runpod:v131")
                            print("\n🚀 FINAL SUCCESS! Ready to test V131!")
                            print("Run: python test_v131_fixed.py")
                            return True
                        else:
                            print(f"⏳ Attempt {attempt+1}/5: Waiting for Docker Hub...")
                            time.sleep(15)
                    
                    print("⚠️  Build succeeded but V131 not yet visible on Docker Hub")
                    return True
                    
                elif conclusion == "failure":
                    print("❌ Build failed even with disk space fix!")
                    print(f"🔗 Check logs: {url}")
                    return False
                    
                break
        else:
            print(f"\r[{timestamp}] Unable to get build status", end="", flush=True)
        
        time.sleep(30)  # Check every 30 seconds
        
        # Timeout after 25 minutes
        if elapsed > 1500:
            print(f"\n⏰ Timeout after 25 minutes")
            return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 V131 DEPLOYMENT COMPLETE!")
    else:
        print("\n❌ V131 deployment failed - check logs for issues")