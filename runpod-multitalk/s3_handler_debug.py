"""
Debug wrapper for S3 handler to add comprehensive logging
"""
import os
import sys
import logging

# Add the app directory to path
sys.path.insert(0, '/app')

# Import the debug module
from s3_debug import debug_s3_environment, debug_s3_bucket_access, debug_s3_download

logger = logging.getLogger(__name__)

# Run comprehensive S3 debugging at import time
logger.error("STARTING COMPREHENSIVE S3 DEBUG...")
s3_client = debug_s3_environment()

if s3_client:
    bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', '760572149-framepack')
    debug_s3_bucket_access(s3_client, bucket_name)

logger.error("S3 DEBUG COMPLETE - CONTINUING WITH NORMAL IMPORT...")

# Now import the original s3_handler
import s3_handler

# Patch the download method to add debug info
original_download = s3_handler.S3Handler.download_from_s3

def debug_download_from_s3(self, s3_url):
    """Download with comprehensive debugging"""
    logger.error(f"DOWNLOAD REQUEST: {s3_url}")
    
    try:
        bucket, key = self.parse_s3_url(s3_url)
        logger.error(f"PARSED: bucket={bucket}, key={key}")
        
        # Use our debug function
        if hasattr(self, 's3_client') and self.s3_client:
            data = debug_s3_download(self.s3_client, bucket, key)
            logger.error(f"DOWNLOAD SUCCESS: {len(data)} bytes")
            return data
        else:
            logger.error("ERROR: No S3 client available")
            raise RuntimeError("S3 client not initialized")
            
    except Exception as e:
        logger.error(f"DOWNLOAD FAILED: {e}")
        raise

# Replace the method
s3_handler.S3Handler.download_from_s3 = debug_download_from_s3

# Export everything from s3_handler
from s3_handler import *