"""
Comprehensive S3 debugging module
"""
import os
import boto3
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

def debug_s3_environment():
    """Debug S3 environment and credentials"""
    logger.error("=" * 60)
    logger.error("S3 DEBUG ENVIRONMENT CHECK")
    logger.error("=" * 60)
    
    # Environment variables
    env_vars = [
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN',
        'AWS_REGION', 'AWS_DEFAULT_REGION', 'AWS_S3_BUCKET_NAME',
        'BUCKET_ENDPOINT_URL', 'AWS_PROFILE', 'AWS_CONFIG_FILE'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'NOT_SET')
        if 'SECRET' in var or 'TOKEN' in var:
            if value != 'NOT_SET' and len(value) > 10:
                value = value[:10] + "..." + value[-4:]
            elif value != 'NOT_SET':
                value = "***"
        logger.error(f"ENV: {var}={value}")
    
    logger.error("-" * 60)
    
    # Test S3 client creation
    try:
        s3 = boto3.client('s3')
        logger.error(f"S3 CLIENT: Created successfully")
        logger.error(f"S3 CLIENT: Region = {s3.meta.region_name}")
        
        # Test credentials
        try:
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            logger.error(f"STS IDENTITY: Account={identity.get('Account', 'Unknown')}")
            logger.error(f"STS IDENTITY: UserId={identity.get('UserId', 'Unknown')}")
            logger.error(f"STS IDENTITY: Arn={identity.get('Arn', 'Unknown')}")
        except Exception as e:
            logger.error(f"STS ERROR: {e}")
            
    except Exception as e:
        logger.error(f"S3 CLIENT ERROR: {e}")
        return None
    
    logger.error("-" * 60)
    return s3

def debug_s3_bucket_access(s3_client, bucket_name):
    """Debug bucket access and permissions"""
    logger.error(f"TESTING BUCKET ACCESS: {bucket_name}")
    
    # Test bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.error(f"BUCKET: {bucket_name} exists and is accessible")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"BUCKET ERROR: {error_code} - {e}")
        return False
    
    # Test list permissions
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
        if 'Contents' in response:
            logger.error(f"BUCKET CONTENTS: Found {len(response['Contents'])} objects")
            for obj in response['Contents']:
                logger.error(f"  - {obj['Key']} ({obj['Size']} bytes, {obj['LastModified']})")
        else:
            logger.error("BUCKET CONTENTS: No objects found")
    except Exception as e:
        logger.error(f"LIST ERROR: {e}")
    
    # Test specific file
    test_files = ['1.wav', 'multi1.png']
    for filename in test_files:
        try:
            response = s3_client.head_object(Bucket=bucket_name, Key=filename)
            logger.error(f"FILE CHECK: {filename} EXISTS - {response['ContentLength']} bytes")
        except ClientError as e:
            logger.error(f"FILE CHECK: {filename} NOT FOUND - {e.response['Error']['Code']}")
        except Exception as e:
            logger.error(f"FILE CHECK: {filename} ERROR - {e}")
    
    logger.error("=" * 60)
    return True

def debug_s3_download(s3_client, bucket, key):
    """Debug a specific S3 download attempt"""
    logger.error(f"DOWNLOAD DEBUG: s3://{bucket}/{key}")
    
    try:
        # First check if file exists
        head_response = s3_client.head_object(Bucket=bucket, Key=key)
        logger.error(f"HEAD OBJECT: SUCCESS - {head_response['ContentLength']} bytes")
        logger.error(f"HEAD OBJECT: LastModified = {head_response['LastModified']}")
        logger.error(f"HEAD OBJECT: ETag = {head_response.get('ETag', 'None')}")
        
        # Try to download
        get_response = s3_client.get_object(Bucket=bucket, Key=key)
        data = get_response['Body'].read()
        logger.error(f"GET OBJECT: SUCCESS - Downloaded {len(data)} bytes")
        return data
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"DOWNLOAD ERROR: {error_code}")
        logger.error(f"DOWNLOAD ERROR: {e}")
        
        # List bucket to see what's available
        try:
            logger.error("LISTING BUCKET TO DEBUG:")
            response = s3_client.list_objects_v2(Bucket=bucket, MaxKeys=20)
            if 'Contents' in response:
                for obj in response['Contents']:
                    similarity = "MATCH!" if obj['Key'] == key else ""
                    logger.error(f"  - {obj['Key']} ({obj['Size']} bytes) {similarity}")
            else:
                logger.error("  No objects in bucket")
        except Exception as le:
            logger.error(f"BUCKET LIST ERROR: {le}")
            
        raise
    except Exception as e:
        logger.error(f"UNEXPECTED DOWNLOAD ERROR: {e}")
        raise