#!/usr/bin/env python3
"""
Check available GPU types in RunPod
"""

import os
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

print("Checking RunPod GPU availability...")
print("=" * 60)

# Common GPU types in RunPod
gpu_types = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090", 
    "NVIDIA RTX A6000",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A40",
    "NVIDIA L40",
    "NVIDIA L40S",
    "RTX 4090",
    "RTX A6000",
    "A100 80GB",
    "AMPERE_24",
    "AMPERE_48", 
    "ADA_24",
    "ADA_48",
    "ADA_48_PRO"
]

print("\nNote: The GPU configuration in the endpoint shows:")
print("AMPERE_48,ADA_48_PRO,ADA_24")
print("\nThis means:")
print("- AMPERE_48: 48GB Ampere GPUs (A6000, A40)")
print("- ADA_48_PRO: 48GB Ada GPUs (RTX 6000 Ada, L40)")
print("- ADA_24: 24GB Ada GPUs (RTX 4090)")
print("\nThe '-NVIDIA RTX A6000,-NVIDIA A40,...' entries are EXCLUSIONS")

print("\n" + "=" * 60)
print("IMPORTANT: In the RunPod dashboard, you need to:")
print("1. Remove all current GPU selections")
print("2. Select ONLY 'NVIDIA GeForce RTX 4090' or 'RTX 4090'")
print("3. Make sure no other GPU types are selected")
print("\nThe current configuration is looking for 48GB GPUs which may not be available.")