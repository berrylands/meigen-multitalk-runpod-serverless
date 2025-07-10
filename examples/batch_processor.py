#!/usr/bin/env python3
"""
Batch Video Processor for MultiTalk
Process multiple audio files in parallel using the MultiTalk serverless endpoint.
"""

import os
import sys
import time
import runpod
import base64
import wave
import numpy as np
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

# Configuration
ENDPOINT_ID = "kkx3cfy484jszl"
MAX_WORKERS = 3  # Maximum parallel jobs

def load_audio(filename):
    """Load and prepare audio for processing."""
    try:
        with wave.open(filename, 'rb') as wav:
            frames = wav.readframes(wav.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)
            sample_rate = wav.getframerate()
            duration = len(audio) / sample_rate
            
            # Convert to mono if stereo
            if wav.getnchannels() == 2:
                audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                ratio = 16000 / sample_rate
                new_length = int(len(audio) * ratio)
                x_old = np.linspace(0, len(audio)-1, len(audio))
                x_new = np.linspace(0, len(audio)-1, new_length)
                audio = np.interp(x_new, x_old, audio).astype(np.int16)
                duration = len(audio) / 16000
            
            return audio, duration
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None

def process_single_file(audio_file, output_dir):
    """Process a single audio file."""
    print(f"\n📁 Processing: {audio_file}")
    
    # Load audio
    audio_data, duration = load_audio(audio_file)
    if audio_data is None:
        return audio_file, False, "Failed to load audio"
    
    # Prepare output filename
    base_name = Path(audio_file).stem
    output_file = output_dir / f"{base_name}_video.mp4"
    
    # Encode audio
    audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
    
    # Submit job
    endpoint = runpod.Endpoint(ENDPOINT_ID)
    
    try:
        job = endpoint.run({
            "action": "generate",
            "audio": audio_b64,
            "duration": duration,
            "fps": 30,
            "width": 512,
            "height": 512
        })
        
        print(f"   Job ID: {job.job_id}")
        
        # Wait for completion
        start_time = time.time()
        while job.status() in ["IN_QUEUE", "IN_PROGRESS"]:
            time.sleep(3)
            if time.time() - start_time > 300:  # 5 minute timeout
                return audio_file, False, "Timeout"
        
        # Check result
        if job.status() == "COMPLETED":
            result = job.output()
            if result and result.get("success"):
                # Save video
                video_data = base64.b64decode(result["video"])
                with open(output_file, "wb") as f:
                    f.write(video_data)
                
                processing_time = time.time() - start_time
                return audio_file, True, f"Success ({processing_time:.1f}s)"
            else:
                return audio_file, False, result.get("error", "Unknown error")
        else:
            return audio_file, False, f"Job failed: {job.status()}"
            
    except Exception as e:
        return audio_file, False, str(e)

def process_directory(input_dir, output_dir, pattern="*.wav"):
    """Process all audio files in a directory."""
    
    # Setup paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find audio files
    audio_files = list(input_path.glob(pattern))
    
    if not audio_files:
        print(f"No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    print(f"Output directory: {output_path}")
    print(f"Using {MAX_WORKERS} parallel workers\n")
    
    # Process files in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_file, f, output_path): f 
            for f in audio_files
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            file, success, message = future.result()
            results.append((file, success, message))
            
            status = "✅" if success else "❌"
            print(f"{status} {Path(file).name}: {message}")
    
    # Summary
    print("\n" + "=" * 50)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 50)
    
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"Total files: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for file, success, message in results:
            if not success:
                print(f"  - {Path(file).name}: {message}")

def main():
    """Main function."""
    print("MultiTalk Batch Video Processor")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python batch_processor.py <input_directory> [output_directory] [pattern]")
        print("\nExamples:")
        print("  python batch_processor.py ./audio_files")
        print("  python batch_processor.py ./audio_files ./videos")
        print("  python batch_processor.py ./audio_files ./videos '*.mp3'")
        
        # Demo mode
        print("\nNo directory specified. Creating demo...")
        
        # Create demo directory
        demo_dir = Path("demo_audio")
        demo_dir.mkdir(exist_ok=True)
        
        # Create a test audio file
        print(f"Creating test audio in {demo_dir}/")
        
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        audio_int16 = (audio * 32767).astype(np.int16)
        
        test_file = demo_dir / "test_audio.wav"
        with wave.open(str(test_file), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())
        
        print(f"Created: {test_file}")
        
        # Process demo
        process_directory(demo_dir, demo_dir / "videos")
        
    else:
        # Process user-specified directory
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else f"{input_dir}_videos"
        pattern = sys.argv[3] if len(sys.argv) > 3 else "*.wav"
        
        process_directory(input_dir, output_dir, pattern)

if __name__ == "__main__":
    main()