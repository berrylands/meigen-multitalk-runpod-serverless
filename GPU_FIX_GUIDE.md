# GPU Configuration Fix Guide

## Current Problem

Your endpoint is configured to look for multiple GPU types:
- `AMPERE_48` - 48GB GPUs like A6000, A40
- `ADA_48_PRO` - 48GB GPUs like RTX 6000 Ada, L40
- `ADA_24` - 24GB GPUs (includes RTX 4090)

This is causing jobs to get stuck because RunPod is trying to find GPUs that match ALL these criteria.

## Solution

In the RunPod Dashboard:

1. Go to: https://www.runpod.io/console/serverless
2. Click on your endpoint "meigen-multitalk -fb"
3. Click "Edit" or the settings icon
4. In the GPU selection:
   - **UNCHECK/REMOVE** all currently selected GPU types
   - **SELECT ONLY** one of these:
     - "NVIDIA GeForce RTX 4090"
     - "RTX 4090" 
     - "24GB" (if available as an option)
5. Make sure **ONLY ONE** GPU type is selected
6. Save the changes

## Alternative: Create New Endpoint

If the template is preventing proper GPU selection, you may need to:

1. Delete the current endpoint
2. Create a new one WITHOUT using a template
3. Configure it manually with:
   - Container: `berrylands/multitalk-test:latest`
   - GPU: RTX 4090 only
   - Network Volume: Select your existing volume
   - Idle Timeout: 60 seconds

## Verification

After fixing, run:
```bash
python diagnose_endpoint.py
```

The `gpuIds` field should show something like:
- Just `NVIDIA GeForce RTX 4090` 
- Or just `RTX_4090`
- Or just `ADA_24`

NOT multiple GPU types separated by commas.