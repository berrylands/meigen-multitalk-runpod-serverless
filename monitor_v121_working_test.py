#!/usr/bin/env python3
"""
Monitor V121 Working Implementation test
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
ENDPOINT_ID = "zu0ik6c8yukyl6"
JOB_ID = "sync-ac8df5bd-1400-449d-9e49-164b1dc92ddb-e2"
API_KEY = os.environ.get('RUNPOD_API_KEY')
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

def monitor_job():
    """Monitor the V121 working implementation test."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"Monitoring V121 Working Implementation test: {JOB_ID}")
    print("Testing with S3 files: 1.wav and multi1.png")
    print("=" * 80)
    
    for attempt in range(60):  # Monitor for up to 10 minutes
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
                    elif output.get('success'):
                        print("✅ Video generation successful!")
                        if 'video_url' in output:
                            print(f"🎥 Video URL: {output['video_url']}")
                        if 'generation_info' in output:
                            gen_info = output['generation_info']
                            print(f"📊 Generation info:")
                            print(f"   - Frames: {gen_info.get('num_frames')}")
                            print(f"   - Steps: {gen_info.get('sampling_steps')}")
                            print(f"   - Turbo: {gen_info.get('turbo_mode')}")
                            print(f"   - Implementation: {gen_info.get('implementation')}")
                    else:
                        print("⚠️ Completed but no success flag")
                        
                    print(f"\nFull output:")
                    print(json.dumps(output, indent=2))
                    return True
                    
                elif status == 'FAILED':
                    print("\n❌ Job failed!")
                    error = result.get('error', 'Unknown error')
                    print(f"Error: {error}")
                    return False
                    
                elif status in ['IN_PROGRESS', 'IN_QUEUE']:
                    delay = result.get('delayTime', 0)
                    worker = result.get('workerId', 'N/A')
                    print(f"    Processing... (delay: {delay}ms, worker: {worker})")
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
    print("V121 Working Implementation Test Monitor")
    print("Based on proven cog-MultiTalk code")
    
    success = monitor_job()
    
    if success:
        print("\n🎯 V121 Working Implementation test complete!")
        print("Check the video URL above for results")
    else:
        print("\n⚠️ Check RunPod logs for details")
        print("This implementation should work with the proven cog-MultiTalk approach")