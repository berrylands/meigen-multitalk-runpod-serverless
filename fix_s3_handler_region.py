#!/usr/bin/env python3
"""
Create a fixed S3 handler that explicitly uses eu-west-1
"""

# Show the fix needed
print("""
🔧 S3 HANDLER FIX NEEDED

The issue might be that even though AWS_REGION is set to eu-west-1 in RunPod,
the S3 handler might not be using it correctly due to initialization order.

Here's a quick fix to force eu-west-1:

In s3_handler.py, change line 29 from:
    region = os.environ.get('AWS_REGION', 'us-east-1')

To:
    region = os.environ.get('AWS_REGION', 'eu-west-1')

Or even better, force it:
    region = 'eu-west-1'  # Your bucket is ALWAYS in eu-west-1

This ensures the S3 client always uses the correct region.
""")