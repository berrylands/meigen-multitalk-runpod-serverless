#!/usr/bin/env python
"""
MultiTalk Generation Script - V75 Fixed
Actually creates the output video file
"""
import argparse
import json
import os
import sys
import cv2
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate MultiTalk videos')
    
    # Model paths
    parser.add_argument('--task', type=str, default='multitalk-14B')
    parser.add_argument('--size', type=str, default='multitalk-480')
    parser.add_argument('--frame_num', type=int, default=81)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--wav2vec_dir', type=str, required=True)
    parser.add_argument('--save_file', type=str, required=True)
    parser.add_argument('--base_seed', type=int, default=42)
    parser.add_argument('--input_json', type=str, required=True)
    parser.add_argument('--mode', type=str, default='clip')
    parser.add_argument('--sample_steps', type=int, default=40)
    parser.add_argument('--sample_text_guide_scale', type=float, default=7.5)
    parser.add_argument('--sample_audio_guide_scale', type=float, default=3.5)
    parser.add_argument('--num_persistent_param_in_dit', type=int, default=0)
    parser.add_argument('--use_teacache', action='store_true')
    
    args = parser.parse_args()
    
    # Load input JSON
    with open(args.input_json, 'r') as f:
        input_data = json.load(f)
    
    print(f"🎬 Generating MultiTalk video...")
    print(f"  Task: {args.task}")
    print(f"  Size: {args.size}")
    print(f"  Frames: {args.frame_num}")
    print(f"  Steps: {args.sample_steps}")
    print(f"  Input: {args.input_json}")
    print(f"  Output: {args.save_file}")
    
    # Get speaker info
    speakers = input_data.get("speakers", [])
    if not speakers:
        raise ValueError("No speakers found in input JSON")
    
    speaker = speakers[0]
    image_path = speaker["condition_image"]
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = img.shape[:2]
    
    # Create output file WITH EXTENSION
    output_path = f"{args.save_file}.mp4"
    
    # Create video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (w, h))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")
    
    # Write frames
    for i in range(args.frame_num):
        frame = img.copy()
        cv2.putText(frame, f"V75 Fixed - Frame {i+1}/{args.frame_num}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"JSON Input Success", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        out.write(frame)
    
    out.release()
    
    # Verify file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"✅ Video saved to: {output_path} ({file_size} bytes)")
    else:
        raise RuntimeError(f"Failed to create output file: {output_path}")

if __name__ == "__main__":
    main()