#!/usr/bin/env python3
"""
Minimal handler to test basic RunPod functionality
"""
import runpod
import os
import sys

print(f"[STARTUP] Python version: {sys.version}", flush=True)
print(f"[STARTUP] Working directory: {os.getcwd()}", flush=True)
print(f"[STARTUP] RunPod module location: {runpod.__file__}", flush=True)

def handler(job):
    """Minimal handler that just returns success"""
    print(f"[HANDLER] Received job: {job}", flush=True)
    
    job_input = job.get("input", {})
    action = job_input.get("action", "echo")
    
    if action == "health":
        return {
            "status": "healthy",
            "handler": "minimal",
            "python_version": sys.version,
            "working_dir": os.getcwd()
        }
    
    return {
        "status": "completed",
        "message": "Minimal handler working",
        "input_received": job_input
    }

if __name__ == "__main__":
    print("[MAIN] Starting RunPod serverless handler...", flush=True)
    try:
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        print(f"[ERROR] Failed to start: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)