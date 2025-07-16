#!/usr/bin/env python3
"""
Test V121 on RunPod
Mock xfuser implementation to bypass import errors
"""

import requests
import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# RunPod endpoint configuration
ENDPOINT_ID = "zu0ik6c8yukyl6"  # Original endpoint now using V121 image
API_KEY = os.environ.get('RUNPOD_API_KEY')
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

def test_v121_runpod():
    """Test V121 with mock xfuser on RunPod."""
    print("=" * 80)
    print("Testing MultiTalk V121 on RunPod")
    print("Version: V121 with mock xfuser to bypass import errors")
    print("=" * 80)
    
    # Test payload - empty input to use default test files
    payload = {
        "input": {}
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"🚀 Sending request to RunPod...")
    print(f"Endpoint: {BASE_URL}/runsync")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Send request
        response = requests.post(
            f"{BASE_URL}/runsync",
            headers=headers,
            json=payload,
            timeout=300  # 5 minute timeout
        )
        
        print(f"\n📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Request successful!")
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Handle queued or in-progress status
            if result.get('status') in ['IN_QUEUE', 'IN_PROGRESS']:
                print("⏳ Request queued, waiting for completion...")
                # Wait a bit for processing
                time.sleep(10)
                
                # Try to get status
                status_response = requests.get(
                    f"{BASE_URL}/status/{result['id']}",
                    headers=headers,
                    timeout=30
                )
                
                if status_response.status_code == 200:
                    status_result = status_response.json()
                    print(f"📊 Status: {json.dumps(status_result, indent=2)}")
                    
                    if status_result.get('status') == 'COMPLETED':
                        output = status_result.get('output', {})
                        if 'error' in output:
                            print(f"❌ Handler error: {output['error']}")
                            if 'traceback' in output:
                                print(f"Traceback:\n{output['traceback']}")
                        else:
                            print(f"✅ Generation successful!")
                            if 'video_url' in output:
                                print(f"🎥 Video URL: {output['video_url']}")
                        return True
                    else:
                        print(f"⚠️ Still processing: {status_result.get('status')}")
                        return False
                else:
                    print(f"❌ Status check failed: {status_response.status_code}")
                    return False
            
            # Check for expected V121 indicators
            elif 'output' in result:
                output = result['output']
                if 'error' in output:
                    print(f"❌ Handler error: {output['error']}")
                    if 'traceback' in output:
                        print(f"Traceback:\n{output['traceback']}")
                    return False
                else:
                    print(f"✅ Generation successful!")
                    if 'video_url' in output:
                        print(f"🎥 Video URL: {output['video_url']}")
                    return True
            else:
                print(f"⚠️ Unexpected response format: {result}")
                return False
                
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

def check_v121_logs():
    """Instructions for checking V121 logs."""
    print("\n" + "=" * 80)
    print("V121 LOG MONITORING")
    print("=" * 80)
    print("Monitor RunPod logs for these V121 indicators:")
    print()
    print("✅ SUCCESS INDICATORS:")
    print("  - 'xfuser 0.4.0-mock (mock) installed'")
    print("  - 'xfuser distributed mock OK'")
    print("  - 'V121' in handler startup messages")
    print("  - No 'ModuleNotFoundError: No module named xfuser' errors")
    print("  - MeiGen-MultiTalk imports succeed")
    print()
    print("❌ FAILURE INDICATORS:")
    print("  - Import errors related to xfuser")
    print("  - MeiGen-MultiTalk still failing to load")
    print("  - Handler crashes during startup")
    print()
    print("📊 V121 GOALS:")
    print("  - Bypass xfuser import errors with mock implementation")
    print("  - Allow MeiGen-MultiTalk to load and show actual errors")
    print("  - Progress beyond import failures to real implementation issues")

if __name__ == "__main__":
    print("MultiTalk V121 RunPod Test")
    print("Mock xfuser implementation")
    
    # Check if API key is set
    if not API_KEY:
        print("❌ RUNPOD_API_KEY not found in environment or .env file")
        exit(1)
    
    print(f"🔑 Using API key: {API_KEY[:8]}...")
    print(f"🎯 Endpoint ID: {ENDPOINT_ID}")
    
    # Run test
    success = test_v121_runpod()
    
    # Show log monitoring instructions
    check_v121_logs()
    
    print(f"\n🎯 V121 Test Result: {'SUCCESS' if success else 'NEEDS INVESTIGATION'}")
    
    if not success:
        print("Check RunPod logs to debug V121 issues.")
        print("V121 specifically targets xfuser import errors with mock implementation.")