#!/usr/bin/env python3
"""Monitor GitHub Actions build progress"""
import subprocess
import json
import time
from datetime import datetime

def get_latest_run_status():
    """Get the status of the latest GitHub Actions run"""
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
        print(f"Error getting run status: {e}")
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
    print("🔍 GitHub Actions Build Monitor")
    print("=" * 50)
    
    start_time = time.time()
    last_status = None
    
    while True:
        elapsed = int(time.time() - start_time)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Get GitHub Actions status
        run_info = get_latest_run_status()
        
        if run_info:
            status = run_info.get('status', 'unknown')
            conclusion = run_info.get('conclusion', '')
            url = run_info.get('url', '')
            
            # Print status update if changed
            if status != last_status:
                print(f"\n[{timestamp}] Status: {last_status or 'unknown'} → {status}")
                if url:
                    print(f"🔗 Monitor at: {url}")
                last_status = status
            else:
                print(f"\r[{timestamp}] Status: {status}", end="", flush=True)
            
            # Check if build completed
            if status == "completed":
                print(f"\n\n🎯 Build completed with conclusion: {conclusion}")
                
                if conclusion == "success":
                    print("✅ Build succeeded!")
                    # Check Docker Hub
                    print("🔍 Checking Docker Hub...")
                    available, last_updated = check_docker_hub()
                    
                    if available:
                        print(f"✅ V131 is available on Docker Hub!")
                        print(f"📅 Last updated: {last_updated}")
                        print(f"🐳 Image: berrylands/multitalk-runpod:v131")
                        print("\n🚀 Ready to test!")
                    else:
                        print("⏳ V131 not yet visible on Docker Hub (may take a moment)")
                        
                elif conclusion == "failure":
                    print("❌ Build failed!")
                    print(f"🔗 Check logs at: {url}")
                    
                break
            
        else:
            print(f"\r[{timestamp}] Unable to get run status", end="", flush=True)
        
        time.sleep(10)  # Check every 10 seconds
        
        # Timeout after 20 minutes
        if elapsed > 1200:
            print(f"\n⏰ Timeout after 20 minutes")
            break

if __name__ == "__main__":
    main()