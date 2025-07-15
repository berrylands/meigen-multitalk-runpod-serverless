import os

# Common prefixes to check
COMMON_PREFIXES = [
    "",
    "comfy_outputs/",
    f"comfy_outputs/{os.environ.get('AWS_S3_BUCKET_NAME', '')}/"
]

def find_with_prefix(s3_handler, bucket, key):
    """Try to find file with various prefixes"""
    for prefix in COMMON_PREFIXES:
        full_key = f"{prefix}{key}" if prefix else key
        try:
            s3_handler.s3_client.head_object(Bucket=bucket, Key=full_key)
            return full_key
        except:
            continue
    return None