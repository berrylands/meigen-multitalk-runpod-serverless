#!/usr/bin/env python3
"""
Debug wrapper that adds logging before importing the main handler
"""
import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Log environment variables
logger.info("=== ENVIRONMENT DEBUG ===")
logger.info(f"AWS_REGION={os.environ.get('AWS_REGION', 'NOT_SET')}")
logger.info(f"AWS_DEFAULT_REGION={os.environ.get('AWS_DEFAULT_REGION', 'NOT_SET')}")
logger.info(f"AWS_S3_BUCKET_NAME={os.environ.get('AWS_S3_BUCKET_NAME', 'NOT_SET')}")
logger.info(f"BUCKET_ENDPOINT_URL={os.environ.get('BUCKET_ENDPOINT_URL', 'NOT_SET')}")
logger.info(f"AWS_ACCESS_KEY_ID exists: {'AWS_ACCESS_KEY_ID' in os.environ}")
logger.info(f"AWS_SECRET_ACCESS_KEY exists: {'AWS_SECRET_ACCESS_KEY' in os.environ}")
logger.info("=== END ENVIRONMENT DEBUG ===")

# Import and run the main handler
sys.path.insert(0, '/app')
from handler import *

# If this is the main module, run the handler
if __name__ == "__main__":
    # The handler's main code will execute when imported
    pass