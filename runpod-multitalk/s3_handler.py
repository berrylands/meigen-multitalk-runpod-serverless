"""
S3 Integration Handler for MultiTalk
Handles S3 uploads/downloads for audio, images, and video files.
"""

import os
import re
import time
import boto3
import logging
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

class S3Handler:
    """Handles S3 operations for binary data."""
    
    def __init__(self):
        """Initialize S3 client with RunPod environment variables."""
        self.enabled = False
        self.s3_client = None
        self.default_bucket = None
        
        # Get credentials from environment
        access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        region = os.environ.get('AWS_REGION', 'us-east-1')
        self.default_bucket = os.environ.get('AWS_S3_BUCKET_NAME')
        endpoint_url = os.environ.get('BUCKET_ENDPOINT_URL')
        
        # Only use endpoint_url if it's not empty
        if endpoint_url is not None and endpoint_url.strip() == '':
            endpoint_url = None
        
        if access_key and secret_key:
            try:
                # Create S3 client
                client_config = {
                    's3': {
                        'aws_access_key_id': access_key,
                        'aws_secret_access_key': secret_key,
                        'region_name': region
                    }
                }
                
                # Only add endpoint_url if it exists and is not empty
                if endpoint_url:
                    self.s3_client = boto3.client(
                        's3',
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key,
                        region_name=region,
                        endpoint_url=endpoint_url
                    )
                else:
                    self.s3_client = boto3.client(
                        's3',
                        aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key,
                        region_name=region
                    )
                
                self.enabled = True
                logger.info(f"S3 integration enabled. Default bucket: {self.default_bucket}, Region: {region}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                self.enabled = False
        else:
            logger.info("S3 credentials not found. S3 integration disabled.")
    
    def is_s3_url(self, url: str) -> bool:
        """Check if the given string is an S3 URL."""
        if not isinstance(url, str):
            return False
        
        # Check for s3:// protocol
        if url.startswith('s3://'):
            return True
        
        # Check for HTTPS S3 URLs
        if url.startswith('https://'):
            parsed = urlparse(url)
            # Common S3 URL patterns
            if '.s3.' in parsed.netloc or '.s3-' in parsed.netloc:
                return True
            if parsed.netloc.endswith('.amazonaws.com'):
                return True
        
        return False
    
    def parse_s3_url(self, url: str) -> Tuple[str, str]:
        """Parse S3 URL and return (bucket, key)."""
        if url.startswith('s3://'):
            # s3://bucket/key format
            parts = url[5:].split('/', 1)
            if len(parts) == 2:
                return parts[0], parts[1]
            else:
                raise ValueError(f"Invalid S3 URL format: {url}")
        
        elif url.startswith('https://'):
            parsed = urlparse(url)
            
            # Virtual-hosted-style URL: https://bucket.s3.region.amazonaws.com/key
            if '.s3.' in parsed.netloc or '.s3-' in parsed.netloc:
                bucket = parsed.netloc.split('.')[0]
                key = parsed.path.lstrip('/')
                return bucket, key
            
            # Path-style URL: https://s3.region.amazonaws.com/bucket/key
            elif parsed.netloc.endswith('.amazonaws.com'):
                path_parts = parsed.path.lstrip('/').split('/', 1)
                if len(path_parts) == 2:
                    return path_parts[0], path_parts[1]
                else:
                    raise ValueError(f"Invalid S3 URL format: {url}")
        
        raise ValueError(f"Not a valid S3 URL: {url}")
    
    def download_from_s3(self, s3_url: str) -> bytes:
        """Download binary data from S3."""
        if not self.enabled:
            raise RuntimeError("S3 integration is not enabled. Missing AWS credentials.")
        
        bucket, key = self.parse_s3_url(s3_url)
        
        try:
            logger.info(f"Downloading from S3: s3://{bucket}/{key}")
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = response['Body'].read()
            logger.info(f"Downloaded {len(data)} bytes from S3")
            return data
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{key}")
            elif error_code == 'AccessDenied':
                raise PermissionError(f"Access denied to S3 object: s3://{bucket}/{key}")
            else:
                raise RuntimeError(f"S3 download failed: {e}")
    
    def upload_to_s3(
        self, 
        data: bytes, 
        key: str, 
        bucket: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        """Upload binary data to S3 and return the S3 URL."""
        if not self.enabled:
            raise RuntimeError("S3 integration is not enabled. Missing AWS credentials.")
        
        bucket = bucket or self.default_bucket
        if not bucket:
            raise ValueError("No S3 bucket specified and no default bucket configured")
        
        # Add timestamp to key if it doesn't have one
        if not re.search(r'\d{8}_\d{6}', key):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(key)
            key = f"{base}_{timestamp}{ext}"
        
        try:
            logger.info(f"Uploading to S3: s3://{bucket}/{key}")
            
            # Prepare upload parameters
            upload_params = {
                'Bucket': bucket,
                'Key': key,
                'Body': data
            }
            
            # Add content type if specified
            if content_type:
                upload_params['ContentType'] = content_type
            else:
                # Auto-detect content type based on extension
                ext = os.path.splitext(key)[1].lower()
                content_types = {
                    '.mp4': 'video/mp4',
                    '.avi': 'video/x-msvideo',
                    '.mov': 'video/quicktime',
                    '.wav': 'audio/wav',
                    '.mp3': 'audio/mpeg',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png'
                }
                if ext in content_types:
                    upload_params['ContentType'] = content_types[ext]
            
            # Upload to S3
            self.s3_client.put_object(**upload_params)
            
            # Return S3 URL
            s3_url = f"s3://{bucket}/{key}"
            logger.info(f"Uploaded {len(data)} bytes to {s3_url}")
            return s3_url
            
        except ClientError as e:
            raise RuntimeError(f"S3 upload failed: {e}")
    
    def generate_presigned_url(
        self, 
        s3_url: str, 
        expiration: int = 3600
    ) -> str:
        """Generate a presigned URL for S3 object access."""
        if not self.enabled:
            raise RuntimeError("S3 integration is not enabled. Missing AWS credentials.")
        
        bucket, key = self.parse_s3_url(s3_url)
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            raise RuntimeError(f"Failed to generate presigned URL: {e}")
    
    def check_s3_access(self) -> Dict[str, Any]:
        """Check S3 access and return status information."""
        status = {
            "enabled": self.enabled,
            "default_bucket": self.default_bucket,
            "has_credentials": bool(os.environ.get('AWS_ACCESS_KEY_ID')),
            "region": os.environ.get('AWS_REGION', 'us-east-1'),
            "endpoint_url": os.environ.get('BUCKET_ENDPOINT_URL'),
            "can_list_bucket": False,
            "can_write": False
        }
        
        if self.enabled and self.default_bucket:
            try:
                # Try to list objects (just to check access)
                self.s3_client.list_objects_v2(
                    Bucket=self.default_bucket,
                    MaxKeys=1
                )
                status["can_list_bucket"] = True
                
                # Try to write a test object
                test_key = "test/multitalk_access_test.txt"
                self.s3_client.put_object(
                    Bucket=self.default_bucket,
                    Key=test_key,
                    Body=b"MultiTalk S3 access test"
                )
                self.s3_client.delete_object(
                    Bucket=self.default_bucket,
                    Key=test_key
                )
                status["can_write"] = True
                
            except Exception as e:
                status["error"] = str(e)
        
        return status


# Global S3 handler instance
s3_handler = S3Handler()


def download_input(input_data: str) -> bytes:
    """Download input data from S3 or decode from base64."""
    if s3_handler.is_s3_url(input_data):
        return s3_handler.download_from_s3(input_data)
    else:
        # Assume base64 encoded
        import base64
        return base64.b64decode(input_data)


def prepare_output(
    data: bytes, 
    output_format: str = "base64",
    s3_key: Optional[str] = None,
    content_type: Optional[str] = None
) -> str:
    """Prepare output data based on requested format."""
    if output_format == "s3" and s3_handler.enabled:
        if not s3_key:
            raise ValueError("s3_key must be specified when output_format is 's3'")
        return s3_handler.upload_to_s3(data, s3_key, content_type=content_type)
    else:
        # Default to base64
        import base64
        return base64.b64encode(data).decode('utf-8')