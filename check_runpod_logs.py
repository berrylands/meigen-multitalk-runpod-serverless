#!/usr/bin/env python3
"""
Instructions for checking RunPod logs
"""

print("""
IMPORTANT: Check these in the RunPod Dashboard

1. Go to: https://www.runpod.io/console/serverless
2. Click on "meigen-multitalk -fb" endpoint
3. Check these tabs:

   a) LOGS tab:
      - Look for any error messages
      - Check if Docker image is being pulled
      - Look for "exec format error" (would mean architecture issue persists)
      - Look for "image not found" errors

   b) WORKERS tab:
      - See if any workers are starting
      - Check worker status (initializing, running, failed)
      - Look at worker logs if available

   c) METRICS tab:
      - Check if there are queue depth issues
      - See if workers are being requested

4. Common issues to look for:
   - "No available GPU" - RTX 4090 not available in region
   - "Failed to pull image" - Docker Hub issue
   - "exec format error" - Architecture mismatch (should be fixed)
   - "Template override" - Template forcing different settings

5. Also check:
   - Is the container image set to: berrylands/multitalk-test:latest
   - Is the network volume properly attached
   - Are there any region-specific issues

Please check these and let me know what you find!
""")