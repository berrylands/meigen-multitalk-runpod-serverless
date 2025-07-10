import runpod
import os
import sys
import json
import time
from pathlib import Path

def download_models(model_list):
    """Download models to the network volume."""
    results = {
        "success": True,
        "downloaded": [],
        "errors": [],
        "model_path": "/runpod-volume/models"
    }
    
    # Create models directory
    model_base_path = Path("/runpod-volume/models")
    model_base_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import huggingface_hub
        from huggingface_hub import snapshot_download, hf_hub_download
        
        for model in model_list:
            try:
                model_name = model.get("name", "Unknown")
                repo_id = model.get("repo_id")
                path = model.get("path", model_name.lower().replace(" ", "_"))
                files = model.get("files", None)
                
                local_dir = model_base_path / path
                
                print(f"Downloading {model_name} from {repo_id}...")
                
                if files:
                    # Download specific files
                    local_dir.mkdir(parents=True, exist_ok=True)
                    for file in files:
                        print(f"  Downloading {file}...")
                        hf_hub_download(
                            repo_id=repo_id,
                            filename=file,
                            local_dir=local_dir,
                            local_dir_use_symlinks=False
                        )
                else:
                    # Download entire repository
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False
                    )
                
                results["downloaded"].append({
                    "name": model_name,
                    "path": str(local_dir),
                    "size": sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
                })
                print(f"✓ Downloaded {model_name}")
                
            except Exception as e:
                error_msg = f"Failed to download {model_name}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"✗ {error_msg}")
        
        if results["errors"]:
            results["success"] = False
            
    except ImportError:
        results["success"] = False
        results["errors"].append("huggingface_hub not installed. Installing...")
        
        # Try to install it
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            results["errors"].append("Installed huggingface_hub. Please retry the download.")
        except:
            results["errors"].append("Failed to install huggingface_hub")
    
    return results

def handler(job):
    try:
        job_input = job.get('input', {})
        
        # Health check
        if job_input.get('health_check'):
            return {
                "status": "healthy",
                "message": "MultiTalk handler with download support is working!",
                "python_version": sys.version,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown'),
                "volume_mounted": os.path.exists('/runpod-volume'),
                "model_path_exists": os.path.exists('/runpod-volume/models'),
                "environment": {
                    "MODEL_PATH": os.environ.get('MODEL_PATH', 'Not set'),
                    "RUNPOD_DEBUG_LEVEL": os.environ.get('RUNPOD_DEBUG_LEVEL', 'Not set')
                }
            }
        
        # Model download action
        if job_input.get('action') == 'download_models':
            models = job_input.get('models', [])
            if not models:
                return {"error": "No models specified for download"}
            
            print(f"Starting download of {len(models)} models...")
            start_time = time.time()
            
            results = download_models(models)
            
            elapsed = time.time() - start_time
            results["download_time"] = f"{elapsed:.1f} seconds"
            
            return results
        
        # List models action
        if job_input.get('action') == 'list_models':
            model_path = Path("/runpod-volume/models")
            if not model_path.exists():
                return {
                    "models": [],
                    "message": "No models directory found"
                }
            
            models = []
            for model_dir in model_path.iterdir():
                if model_dir.is_dir():
                    size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
                    models.append({
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "size_mb": round(size / (1024 * 1024), 2),
                        "files": len(list(model_dir.rglob("*")))
                    })
            
            return {
                "models": models,
                "total": len(models),
                "model_path": str(model_path)
            }
        
        # Default echo response
        return {
            "message": "MultiTalk handler ready!",
            "echo": job_input,
            "supported_actions": ["health_check", "download_models", "list_models"],
            "server_info": {
                "python_version": sys.version,
                "worker_id": os.environ.get('RUNPOD_POD_ID', 'unknown'),
                "volume_available": os.path.exists('/runpod-volume')
            }
        }
        
    except Exception as e:
        return {"error": f"Handler failed: {str(e)}"}

if __name__ == "__main__":
    print("Starting MultiTalk handler with download support...")
    runpod.serverless.start({"handler": handler})