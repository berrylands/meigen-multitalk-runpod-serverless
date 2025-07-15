"""
S3 utility functions for smart file detection
"""

import base64
import logging

logger = logging.getLogger(__name__)

def is_likely_base64(data: str) -> bool:
    """Check if a string is likely base64 encoded."""
    # Check length - filenames are usually short
    if len(data) < 100:
        return False
    
    # Check for file extensions
    if any(data.lower().endswith(ext) for ext in ['.wav', '.mp3', '.jpg', '.jpeg', '.png', '.mp4']):
        return False
    
    # Check for valid base64 characters
    try:
        # Remove whitespace
        cleaned = data.replace('\n', '').replace('\r', '').replace(' ', '')
        
        # Check if all characters are valid base64
        if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in cleaned):
            # Try to decode to verify
            decoded = base64.b64decode(cleaned, validate=True)
            # If successful and has reasonable size, it's likely base64
            return len(decoded) > 100
    except:
        pass
    
    return False

def detect_binary_type(data: bytes) -> str:
    """Detect the type of binary data."""
    if len(data) < 12:
        return "unknown"
    
    # Check for common file signatures
    if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
        return "wav"
    elif data[:3] == b'ID3' or (data[:2] == b'\xff\xfb'):
        return "mp3"
    elif data[:8] == b'\x89PNG\r\n\x1a\n':
        return "png"
    elif data[:2] == b'\xff\xd8':
        return "jpeg"
    elif data[4:12] == b'ftypmp42' or data[4:12] == b'ftypisom':
        return "mp4"
    else:
        return "binary"

def process_input_data(input_data: str, data_type: str, s3_handler=None, default_bucket: str = None) -> bytes:
    """
    Process input data that could be:
    1. Base64 encoded data
    2. S3 URL (s3://bucket/key)
    3. S3 filename (just the key)
    
    Returns binary data.
    """
    # Check if it's an S3 URL
    if input_data.startswith('s3://') or 'amazonaws.com' in input_data:
        if not s3_handler:
            raise ValueError("S3 URL provided but S3 handler not available")
        logger.info(f"Detected S3 URL for {data_type}: {input_data}")
        return s3_handler.download_from_s3(input_data)
    
    # Check if it's likely base64
    if is_likely_base64(input_data):
        logger.info(f"Detected base64 {data_type}")
        try:
            return base64.b64decode(input_data)
        except Exception as e:
            logger.error(f"Failed to decode base64: {e}")
            raise
    
    # Otherwise, treat as S3 filename
    if s3_handler and default_bucket:
        logger.info(f"Treating {data_type} as S3 filename: {input_data}")
        s3_url = f"s3://{default_bucket}/{input_data}"
        try:
            data = s3_handler.download_from_s3(s3_url)
            # Verify it's the expected type
            detected_type = detect_binary_type(data)
            logger.info(f"Downloaded {len(data)} bytes, detected type: {detected_type}")
            return data
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            # Last resort - try base64
            try:
                return base64.b64decode(input_data)
            except:
                raise ValueError(f"Could not process {data_type} as S3 file or base64: {input_data[:100]}")
    
    # Final fallback
    raise ValueError(f"Could not determine how to process {data_type}: {input_data[:100]}")