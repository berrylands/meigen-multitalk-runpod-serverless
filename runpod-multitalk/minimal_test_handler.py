import os
import logging
import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def handler(event):
    """Minimal test handler"""
    try:
        return {
            "success": True,
            "message": "Container is healthy and ready",
            "model": "minimal-test"
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    logger.info("=" * 50)
    logger.info("MINIMAL TEST HANDLER STARTING")
    logger.info("=" * 50)
    
    model_path = os.environ.get('MODEL_PATH', '/runpod-volume/models')
    logger.info(f"Model path: {model_path}")
    logger.info(f"Container is healthy and ready for requests")
    
    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler})

if __name__ == "__main__":
    main()