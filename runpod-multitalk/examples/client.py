#!/usr/bin/env python3
"""
Example client for RunPod MultiTalk API
"""

import os
import sys
import time
import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any

class MultiTalkClient:
    def __init__(self, api_key: str, endpoint_id: str):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def generate_video_sync(
        self,
        reference_image: str,
        audio_1: str,
        audio_2: Optional[str] = None,
        prompt: str = "Two people having a conversation",
        num_frames: int = 100,
        seed: int = 42,
        turbo: bool = False,
        sampling_steps: int = 20,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate video synchronously (blocking).
        Returns path to saved video file.
        """
        # Prepare input data
        input_data = self._prepare_input(
            reference_image, audio_1, audio_2,
            prompt, num_frames, seed, turbo, sampling_steps
        )
        
        # Send request
        print("Sending request to RunPod...")
        response = requests.post(
            f"{self.base_url}/runsync",
            json={"input": input_data},
            headers=self.headers,
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        if "error" in result:
            raise Exception(f"Generation error: {result['error']}")
        
        # Save video
        if output_path is None:
            output_path = f"output_{int(time.time())}.mp4"
        
        if "video_url" in result['output']:
            # Download from URL
            print(f"Downloading video from: {result['output']['video_url']}")
            video_response = requests.get(result['output']['video_url'])
            with open(output_path, 'wb') as f:
                f.write(video_response.content)
        else:
            # Decode base64
            print("Decoding base64 video...")
            video_data = base64.b64decode(result['output']['video_base64'])
            with open(output_path, 'wb') as f:
                f.write(video_data)
        
        print(f"Video saved to: {output_path}")
        return output_path
    
    def generate_video_async(
        self,
        reference_image: str,
        audio_1: str,
        audio_2: Optional[str] = None,
        prompt: str = "Two people having a conversation",
        num_frames: int = 100,
        seed: int = 42,
        turbo: bool = False,
        sampling_steps: int = 20
    ) -> str:
        """
        Generate video asynchronously (non-blocking).
        Returns job ID for status checking.
        """
        # Prepare input data
        input_data = self._prepare_input(
            reference_image, audio_1, audio_2,
            prompt, num_frames, seed, turbo, sampling_steps
        )
        
        # Send request
        print("Sending async request to RunPod...")
        response = requests.post(
            f"{self.base_url}/run",
            json={"input": input_data},
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        result = response.json()
        job_id = result['id']
        
        print(f"Job submitted: {job_id}")
        return job_id
    
    def check_status(self, job_id: str) -> Dict[str, Any]:
        """Check status of async job."""
        response = requests.get(
            f"{self.base_url}/status/{job_id}",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def wait_for_completion(self, job_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for async job to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.check_status(job_id)
            
            if status['status'] == 'COMPLETED':
                return status
            elif status['status'] == 'FAILED':
                raise Exception(f"Job failed: {status.get('error', 'Unknown error')}")
            
            print(f"Status: {status['status']}...")
            time.sleep(5)
        
        raise Exception(f"Timeout waiting for job {job_id}")
    
    def _prepare_input(
        self,
        reference_image: str,
        audio_1: str,
        audio_2: Optional[str],
        prompt: str,
        num_frames: int,
        seed: int,
        turbo: bool,
        sampling_steps: int
    ) -> Dict[str, Any]:
        """Prepare input data for API request."""
        input_data = {
            "reference_image": self._load_file(reference_image),
            "audio_1": self._load_file(audio_1),
            "prompt": prompt,
            "num_frames": num_frames,
            "seed": seed,
            "turbo": turbo,
            "sampling_steps": sampling_steps
        }
        
        if audio_2:
            input_data["audio_2"] = self._load_file(audio_2)
        
        return input_data
    
    def _load_file(self, file_path: str) -> str:
        """Load file as base64 or return URL if already a URL."""
        if file_path.startswith('http'):
            return file_path
        
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')


def main():
    # Load environment variables
    api_key = os.environ.get('RUNPOD_API_KEY')
    endpoint_id = os.environ.get('RUNPOD_ENDPOINT_ID')
    
    if not api_key or not endpoint_id:
        print("Please set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables")
        sys.exit(1)
    
    # Create client
    client = MultiTalkClient(api_key, endpoint_id)
    
    # Example usage
    try:
        # Synchronous generation
        output_path = client.generate_video_sync(
            reference_image="example_image.jpg",
            audio_1="audio1.wav",
            audio_2="audio2.wav",  # Optional
            prompt="Two friends discussing their weekend plans",
            num_frames=100,
            turbo=True
        )
        print(f"Success! Video saved to: {output_path}")
        
        # Asynchronous generation example
        # job_id = client.generate_video_async(
        #     reference_image="example_image.jpg",
        #     audio_1="audio1.wav",
        #     prompt="A business meeting"
        # )
        # 
        # result = client.wait_for_completion(job_id)
        # print(f"Job completed: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()