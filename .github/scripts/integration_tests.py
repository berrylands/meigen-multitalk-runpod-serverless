#!/usr/bin/env python3
"""
Integration tests for RunPod deployment
"""

import os
import sys
import json
import base64
import time
from io import BytesIO
from PIL import Image
import numpy as np
import requests

class IntegrationTests:
    def __init__(self, endpoint_id, api_key):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def create_test_assets(self):
        """Create test image and audio."""
        # Create test image
        img = Image.new('RGB', (640, 480))
        pixels = np.array(img)
        
        # Add gradient
        for i in range(480):
            pixels[i, :] = [int(255 * i / 480), 100, 100]
        
        img = Image.fromarray(pixels)
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create test audio (simple sine wave)
        import wave
        
        sample_rate = 16000
        duration = 1.5
        frequency = 440
        
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        buffer = BytesIO()
        with wave.open(buffer, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_data.tobytes())
        
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return img_base64, audio_base64
    
    def test_single_person_generation(self):
        """Test single person video generation."""
        print("\n=== Testing Single Person Generation ===")
        
        img_base64, audio_base64 = self.create_test_assets()
        
        request_data = {
            "input": {
                "reference_image": img_base64,
                "audio_1": audio_base64,
                "prompt": "A person giving a presentation",
                "num_frames": 30,
                "seed": 123,
                "turbo": True,
                "sampling_steps": 5
            }
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/runsync",
            json=request_data,
            headers=self.headers,
            timeout=180
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if "error" not in result:
                print(f"✓ Single person generation successful ({elapsed:.2f}s)")
                return True
            else:
                print(f"✗ Error: {result['error']}")
        else:
            print(f"✗ HTTP {response.status_code}: {response.text}")
        
        return False
    
    def test_multi_person_generation(self):
        """Test multi-person video generation."""
        print("\n=== Testing Multi-Person Generation ===")
        
        img_base64, audio1_base64 = self.create_test_assets()
        
        # Create second audio with different frequency
        import wave
        buffer = BytesIO()
        sample_rate = 16000
        duration = 1.5
        frequency = 880  # Different frequency
        
        num_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, num_samples)
        audio_data = np.sin(2 * np.pi * frequency * t)
        audio_data = (audio_data * 32767).astype(np.int16)
        
        with wave.open(buffer, 'w') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_data.tobytes())
        
        buffer.seek(0)
        audio2_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        request_data = {
            "input": {
                "reference_image": img_base64,
                "audio_1": audio1_base64,
                "audio_2": audio2_base64,
                "prompt": "Two people having a conversation about technology",
                "num_frames": 40,
                "seed": 456,
                "turbo": True,
                "sampling_steps": 8
            }
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/runsync",
            json=request_data,
            headers=self.headers,
            timeout=180
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if "error" not in result:
                print(f"✓ Multi-person generation successful ({elapsed:.2f}s)")
                return True
            else:
                print(f"✗ Error: {result['error']}")
        else:
            print(f"✗ HTTP {response.status_code}: {response.text}")
        
        return False
    
    def test_async_generation(self):
        """Test async job submission and polling."""
        print("\n=== Testing Async Generation ===")
        
        img_base64, audio_base64 = self.create_test_assets()
        
        request_data = {
            "input": {
                "reference_image": img_base64,
                "audio_1": audio_base64,
                "prompt": "A tutorial video",
                "num_frames": 25,
                "turbo": True
            }
        }
        
        # Submit async job
        response = requests.post(
            f"{self.base_url}/run",
            json=request_data,
            headers=self.headers
        )
        
        if response.status_code != 200:
            print(f"✗ Failed to submit job: {response.text}")
            return False
        
        job_id = response.json()["id"]
        print(f"Job submitted: {job_id}")
        
        # Poll for completion
        start_time = time.time()
        timeout = 180
        
        while time.time() - start_time < timeout:
            status_response = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers
            )
            
            if status_response.status_code == 200:
                status = status_response.json()
                
                if status["status"] == "COMPLETED":
                    elapsed = time.time() - start_time
                    print(f"✓ Async generation completed ({elapsed:.2f}s)")
                    return True
                elif status["status"] == "FAILED":
                    print(f"✗ Job failed: {status.get('error', 'Unknown error')}")
                    return False
                else:
                    print(f"Status: {status['status']}...")
            
            time.sleep(5)
        
        print("✗ Timeout waiting for job completion")
        return False
    
    def test_error_handling(self):
        """Test error handling with invalid inputs."""
        print("\n=== Testing Error Handling ===")
        
        test_cases = [
            {
                "name": "Missing reference image",
                "input": {
                    "audio_1": "invalid_base64",
                    "prompt": "Test"
                },
                "expected_error": "reference_image is required"
            },
            {
                "name": "Invalid base64",
                "input": {
                    "reference_image": "not_valid_base64!@#",
                    "audio_1": "also_invalid",
                    "prompt": "Test"
                },
                "expected_error": "Failed to decode base64"
            }
        ]
        
        passed = 0
        for test in test_cases:
            response = requests.post(
                f"{self.base_url}/runsync",
                json={"input": test["input"]},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    print(f"✓ {test['name']}: Got expected error")
                    passed += 1
                else:
                    print(f"✗ {test['name']}: Expected error but got success")
            else:
                print(f"✓ {test['name']}: Got error response")
                passed += 1
        
        return passed == len(test_cases)
    
    def run_all_tests(self):
        """Run all integration tests."""
        print(f"\nRunning integration tests for endpoint: {self.endpoint_id}")
        print("="*60)
        
        tests = [
            ("Health Check", lambda: self.test_health_check()),
            ("Single Person Generation", self.test_single_person_generation),
            ("Multi-Person Generation", self.test_multi_person_generation),
            ("Async Generation", self.test_async_generation),
            ("Error Handling", self.test_error_handling)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
            except Exception as e:
                print(f"✗ {test_name} failed with exception: {e}")
        
        print("\n" + "="*60)
        print(f"Integration Tests: {passed}/{total} passed")
        
        return passed == total
    
    def test_health_check(self):
        """Test health check endpoint."""
        print("\n=== Testing Health Check ===")
        
        request_data = {
            "input": {
                "health_check": True
            }
        }
        
        response = requests.post(
            f"{self.base_url}/runsync",
            json=request_data,
            headers=self.headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "healthy":
                print("✓ Health check passed")
                print(f"  GPU: {result.get('gpu_name', 'Unknown')}")
                print(f"  Models exist: {result.get('models_directory_exists', False)}")
                return True
        
        print("✗ Health check failed")
        return False

def main():
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    
    if not api_key or not endpoint_id:
        print("ERROR: RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set")
        sys.exit(1)
    
    tester = IntegrationTests(endpoint_id, api_key)
    
    if tester.run_all_tests():
        print("\n✓ All integration tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some integration tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()