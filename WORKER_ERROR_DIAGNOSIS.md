# Worker Exit Code 1 - Diagnosis and Solutions

## Issue
The RunPod worker is exiting with code 1, indicating a startup failure.

## Deployed Images
1. `berrylands/multitalk-s3-quick:latest` - Quick S3 update
2. `berrylands/multitalk-s3-fix:latest` - Fixed handler mapping with debug info

## Common Causes of Worker Exit Code 1

### 1. **Import Errors**
- Missing Python packages (though boto3 is installed)
- Circular imports between modules
- Module not found at expected path

### 2. **File Path Issues**
- Handler expects to be at `/app/handler.py`
- S3 handler expects to be at `/app/s3_handler.py`
- Working directory mismatch

### 3. **RunPod Specific Issues**
- Missing RUNPOD_AI_API_KEY environment variable
- Incorrect handler signature
- Startup timeout

### 4. **Python Errors**
- Syntax errors in the handler
- Unicode/encoding issues
- Missing dependencies

## Immediate Actions

### 1. Check Worker Logs
Go to RunPod dashboard:
- https://www.runpod.io/console/serverless
- Click on endpoint `kkx3cfy484jszl`
- Go to "Workers" tab
- Look at the logs for the failed worker

### 2. Try the Fixed Image
Update your endpoint to use: `berrylands/multitalk-s3-fix:latest`

This image includes:
- Verified file paths (`/app/handler.py` and `/app/s3_handler.py`)
- Debug output during build
- Explicit PYTHONUNBUFFERED=1

### 3. Use Debug Handler
If the issue persists, try the minimal handler:

```bash
# Build minimal test image
cd /Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk
docker build -t berrylands/multitalk-minimal:latest -f Dockerfile.minimal .
docker push berrylands/multitalk-minimal:latest
```

Then update RunPod to use `berrylands/multitalk-minimal:latest`

### 4. Check Environment Variables
Ensure these are set in RunPod:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY  
- AWS_REGION
- AWS_S3_BUCKET_NAME

## Debugging Scripts Available

1. **test_worker_error.py** - Diagnoses deployment issues
2. **debug_handler.py** - Detailed startup debugging
3. **minimal_handler.py** - Bare minimum handler for testing

## Next Steps Based on Log Output

### If you see "No module named 'runpod'"
- The base image might be corrupted
- Try rebuilding from python:3.10-slim base

### If you see "No such file or directory"
- Handler file path is wrong
- Check if files are copied to correct location

### If you see "ImportError: cannot import name..."
- There's a code issue in the handler
- Check the specific import that's failing

### If no logs appear
- Container might be failing before Python starts
- Check Dockerfile CMD syntax

## Quick Test Command

After updating the image:
```bash
python test_worker_error.py
```

This will help identify if the issue is with:
- Network connectivity
- API authentication
- Handler execution
- S3 integration