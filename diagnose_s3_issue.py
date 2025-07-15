#!/usr/bin/env python3
"""
Diagnose S3 access issue
"""

import runpod
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def diagnose_s3():
    """Diagnose S3 access issue"""
    
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("❌ RUNPOD_API_KEY not set")
        return
    
    runpod.api_key = api_key
    endpoint = runpod.Endpoint("kkx3cfy484jszl")
    
    print("🔍 S3 Diagnostic Tool")
    print("=" * 60)
    
    # Get the actual URL from the user
    print("\n📋 Please provide the exact URL you get when you:")
    print("1. Go to S3 Console")
    print("2. Select your file '1.wav'")
    print("3. Click 'Copy URL'")
    print("\nThe URL should look something like:")
    print("https://bucket-name.s3.region.amazonaws.com/path/to/file.wav")
    print("\nOr if you click 'Open', copy the URL from your browser")
    
    print("\n" + "=" * 60)
    
    # Test what we know works
    print("\n✅ What we know works:")
    print("1. S3 credentials are properly configured")
    print("2. Can write to bucket: s3://760572149-framepack/test/")
    print("3. Can read files we write")
    
    print("\n❌ What's not working:")
    print("1. Reading s3://760572149-framepack/1.wav")
    
    print("\n🤔 Possible reasons:")
    
    print("\n1. **File is in a subfolder**")
    print("   If your file is actually at:")
    print("   - s3://760572149-framepack/audio/1.wav")
    print("   - s3://760572149-framepack/inputs/1.wav")
    print("   - s3://760572149-framepack/some-folder/1.wav")
    
    print("\n2. **Different file name or extension**")
    print("   - File might be '1.WAV' (uppercase)")
    print("   - File might be '01.wav' or 'audio1.wav'")
    print("   - File might be '.wav' without the '1'")
    
    print("\n3. **Bucket name issue**")
    print("   - Is '760572149-framepack' the exact bucket name?")
    print("   - No typos or extra characters?")
    
    print("\n4. **Permissions**")
    print("   - The file might have different permissions than the test/ folder")
    print("   - The IAM user might not have GetObject permission for that specific file")
    
    print("\n💡 Quick Test:")
    print("Try uploading a test file to the root of your bucket:")
    print("\naws s3 cp test.wav s3://760572149-framepack/test-audio.wav")
    print("\nThen test with: s3://760572149-framepack/test-audio.wav")
    
    print("\n📝 To find your file:")
    print("aws s3 ls s3://760572149-framepack/ --recursive | grep -i wav")

if __name__ == "__main__":
    diagnose_s3()