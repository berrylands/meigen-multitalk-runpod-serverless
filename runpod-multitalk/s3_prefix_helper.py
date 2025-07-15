import logging

logger = logging.getLogger(__name__)

def try_with_prefixes(s3_client, bucket, key):
    """Try to find a file with common prefixes"""
    prefixes = ["", "comfy_outputs/", f"comfy_outputs/{bucket}/"]
    
    for prefix in prefixes:
        test_key = f"{prefix}{key}" if prefix else key
        try:
            s3_client.head_object(Bucket=bucket, Key=test_key)
            if prefix:
                logger.info(f"[S3_PREFIX] Found file with prefix: {test_key}")
            return test_key
        except:
            continue
    
    return key  # Return original if not found