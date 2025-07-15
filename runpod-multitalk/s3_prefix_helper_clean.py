import logging

logger = logging.getLogger(__name__)

def try_with_prefixes(s3_client, bucket, key):
    """Just try the key as-is, no prefixes"""
    logger.info(f"[S3_HELPER] Checking for file: {key}")
    # Just return the key as-is, no prefix manipulation
    return key