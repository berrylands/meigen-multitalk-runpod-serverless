#!/usr/bin/env python3
"""Check GitHub Actions build status"""
import subprocess
import json
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
        print(f"Error: {e}")
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
    print("🔍 V131 Build Status Check")
    print("=" * 40)
    
    # Check GitHub Actions
    run_info = check_build_status()
    if run_info:
        status = run_info.get('status', 'unknown')
        conclusion = run_info.get('conclusion', '')
        url = run_info.get('url', '')
        created_at = run_info.get('createdAt', '')
        
        print(f"📊 GitHub Actions Status: {status}")
        if conclusion:
            print(f"📋 Conclusion: {conclusion}")
        print(f"🔗 Monitor: {url}")
        print(f"⏰ Started: {created_at}")
        
        if status == "completed":
            if conclusion == "success":
                print("\n✅ Build completed successfully!")
                
                # Check Docker Hub
                print("🔍 Checking Docker Hub...")
                available, last_updated = check_docker_hub()
                
                if available:
                    print(f"✅ V131 is available on Docker Hub!")
                    print(f"📅 Last updated: {last_updated}")
                    print(f"🐳 Image: berrylands/multitalk-runpod:v131")
                    print("\n🚀 Ready to test with: python test_v131_fixed.py")
                else:
                    print("⏳ V131 not yet visible on Docker Hub")
                    
            elif conclusion == "failure":
                print("\n❌ Build failed!")
                print("Check the logs in the GitHub Actions URL above")
                
        elif status == "in_progress":
            print("\n🔄 Build is still running...")
            print("Estimated time: 10-15 minutes for Docker builds")
    else:
        print("❌ Unable to get build status")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    main()