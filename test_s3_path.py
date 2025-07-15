#!/usr/bin/env python3
"""
Test S3 with different path formats
"""

import runpod
import os
from dotenv import load_dotenv

load_dotenv()

def test_s3_paths():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 Testing Different S3 Path Formats")
    print("=" * 60)
    
    # Test different path formats
    test_cases = [
        {
            "name": "Simple filename",
            "audio": "1.wav",
            "image": "multi1.png"
        },
        {
            "name": "With subfolder",
            "audio": "audio/1.wav",
            "image": "images/multi1.png"
        },
        {
            "name": "Full S3 URL",
            "audio": "s3://760572149-framepack/1.wav",
            "image": "s3://760572149-framepack/multi1.png"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test['name']}")
        print(f"   Audio: {test['audio']}")
        print(f"   Image: {test['image']}")
        
        job = endpoint.run({
            "action": "generate",
            "audio": test['audio'],
            "reference_image": test['image'],
            "duration": 2.0,
            "width": 128,
            "height": 128,
            "fps": 10
        })
        
        result = job.output(timeout=60)
        
        if result:
            if result.get('success'):
                params = result.get('parameters', {})
                print(f"   ✅ Success! Audio size: {params.get('audio_size', 0):,} bytes")
            else:
                error = result.get('error', 'Unknown error')
                print(f"   ❌ Failed: {error}")
                if 'not found' in str(error).lower():
                    print("      → File not at this path")
        
        # Only run first test if it succeeds to save time
        if i == 1 and result and result.get('success'):
            print("\n✅ First format worked! Your files are in the bucket root.")
            break
    
    print("\n" + "=" * 60)
    print("\n💡 Run 'python check_s3_files.py' to see where your files actually are.")

if __name__ == "__main__":
    test_s3_paths()