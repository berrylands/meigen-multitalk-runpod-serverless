#!/usr/bin/env python3
"""
Monitor V121 job status on RunPod
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
ENDPOINT_ID = "8wsqldxzg8bj7f"
JOB_ID = "sync-04af8d28-13b5-4adf-a33d-0de70c327580-e1"  # From last run
API_KEY = os.environ.get('RUNPOD_API_KEY')
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

def monitor_job():
    """Monitor the V121 job status."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Monitoring V121 job: {JOB_ID}")
    print("=" * 80)
    
    for attempt in range(30):  # Monitor for up to 5 minutes
        try:
            response = requests.get(
                f"{BASE_URL}/status/{JOB_ID}",
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                status = result.get('status', 'UNKNOWN')
                
                print(f"[{attempt+1:2d}] Status: {status}")
                
                if status == 'COMPLETED':
                    print("\n✅ Job completed!")
                    output = result.get('output', {})
                    
                    if 'error' in output:
                        print(f"❌ Error: {output['error']}")
                        if 'traceback' in output:
                            print(f"Traceback:\n{output['traceback']}")
                    else:
                        print("✅ Success!")
                        if 'video_url' in output:
                            print(f"🎥 Video URL: {output['video_url']}")
                        
                        # Show the full output
                        print(f"\nFull output:")
                        print(json.dumps(output, indent=2))
                    
                    return True
                    
                elif status == 'FAILED':
                    print("\n❌ Job failed!")
                    error = result.get('error', 'Unknown error')
                    print(f"Error: {error}")
                    return False
                    
                elif status in ['IN_PROGRESS', 'IN_QUEUE']:
                    print(f"    Still processing... (delay: {result.get('delayTime', 0)}ms)")
                    time.sleep(10)
                    
                else:
                    print(f"    Unknown status: {status}")
                    time.sleep(10)
                    
            else:
                print(f"[{attempt+1:2d}] Status check failed: {response.status_code}")
                time.sleep(10)
                
        except Exception as e:
            print(f"[{attempt+1:2d}] Error: {str(e)}")
            time.sleep(10)
    
    print("\n⏰ Monitoring timeout reached")
    return False

if __name__ == "__main__":
    print("V121 Job Monitor")
    print("Checking for mock xfuser implementation results...")
    
    success = monitor_job()
    
    if success:
        print("\n🎯 V121 monitoring complete!")
    else:
        print("\n⚠️ Check RunPod logs for V121 details")