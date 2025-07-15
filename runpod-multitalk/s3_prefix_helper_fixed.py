import logging

logger = logging.getLogger(__name__)

def try_with_prefixes(s3_client, bucket, key):
    """Try to find a file with common prefixes"""
    logger.info(f"[PREFIX_HELPER] try_with_prefixes called for key: {key}")
    prefixes = ["", "comfy_outputs/", f"comfy_outputs/{bucket}/"]
    
    for prefix in prefixes:
        test_key = f"{prefix}{key}" if prefix else key
        try:
            s3_client.head_object(Bucket=bucket, Key=test_key)
            if prefix:
                logger.info(f"[S3_PREFIX] Found file with prefix: {test_key}")
            return test_key
        except Exception as e:
            logger.debug(f"[PREFIX_HELPER] Not found at {test_key}")
            continue
    
    logger.warning(f"[PREFIX_HELPER] File not found with any prefix: {key}")
    return key  # Return original if not found