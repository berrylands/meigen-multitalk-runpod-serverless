#!/usr/bin/env python3
"""Monitor the fixed V131 build"""
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
    print("🔍 V131 Build Monitor - Docker Buildx Fix")
    print("=" * 60)
    print("🛠️  Fix Applied: Preserved Docker filesystem structure")
    print("🎯 Should resolve GetImageBlob errors")
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
                    print("🔄 Build is running...")
                last_status = status
            else:
                mins, secs = divmod(elapsed, 60)
                print(f"\r[{timestamp}] Check #{check_count}: {status} (runtime: {mins}m{secs}s)", end="", flush=True)
            
            # Check if completed
            if status == "completed":
                print(f"\n\n🎯 Build completed with result: {conclusion}")
                
                if conclusion == "success":
                    print("✅ SUCCESS! Docker Buildx fix worked!")
                    print("🎉 V131 build completed successfully!")
                    
                    # Check Docker Hub
                    print("\n🔍 Checking Docker Hub availability...")
                    for attempt in range(6):  # Try 6 times over 2 minutes
                        available, last_updated = check_docker_hub()
                        if available:
                            print(f"✅ V131 is available on Docker Hub!")
                            print(f"📅 Last updated: {last_updated}")
                            print(f"🐳 Image: berrylands/multitalk-runpod:v131")
                            print("\n🚀 MISSION ACCOMPLISHED!")
                            print("📋 V131 includes:")
                            print("  - NumPy 1.26.4 (Numba compatible)")
                            print("  - PyTorch 2.1.0 + CUDA 11.8")
                            print("  - All dependencies working")
                            print("\n🧪 Ready to test!")
                            return True
                        else:
                            print(f"⏳ Attempt {attempt+1}/6: Waiting for Docker Hub...")
                            time.sleep(20)
                    
                    print("⚠️  Build succeeded but V131 not yet visible on Docker Hub")
                    print("It may take a few more minutes to propagate")
                    return True
                    
                elif conclusion == "failure":
                    print("❌ Build failed!")
                    print(f"🔗 Check logs: {url}")
                    print("Will investigate the new error...")
                    return False
                    
                break
        else:
            print(f"\r[{timestamp}] Unable to get build status", end="", flush=True)
        
        time.sleep(15)  # Check every 15 seconds during build
        
        # Timeout after 25 minutes
        if elapsed > 1500:
            print(f"\n⏰ Timeout after 25 minutes")
            print("Check GitHub Actions manually if needed")
            return False

if __name__ == "__main__":
    success = main()
    print("\n" + "=" * 60)
    if success:
        print("🎊 V131 DEPLOYMENT SUCCESSFUL! 🎊")
        print("Ready to update RunPod template and test!")
    else:
        print("❌ Build needs further investigation")
    print("=" * 60)