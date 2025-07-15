"""
MultiTalk V63 - Debug Model Inspector
Lists all model files available on RunPod to understand what we're working with
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelInspector:
    """Inspect model files on RunPod volume"""
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        self.model_path = Path(model_path)
        logger.info(f"Initializing Model Inspector for path: {model_path}")
        
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get detailed information about a file"""
        try:
            stat = file_path.stat()
            
            # Get first few bytes to check file type
            magic_bytes = ""
            file_type = "unknown"
            try:
                with open(file_path, 'rb') as f:
                    first_bytes = f.read(64)
                    magic_bytes = first_bytes[:8].hex()
                    
                    # Identify file type by magic bytes
                    if first_bytes.startswith(b'PK'):
                        file_type = "zip/pytorch"
                    elif first_bytes.startswith(b'\x80\x02\x8a\n'):
                        file_type = "pickle/pytorch"
                    elif b'GGUF' in first_bytes:
                        file_type = "gguf"
                    elif first_bytes.startswith(b'{"'):
                        file_type = "json"
                    elif b'safetensors' in first_bytes or first_bytes.startswith(b'\x93NUMPY'):
                        file_type = "safetensors"
            except:
                pass
                
            return {
                "name": file_path.name,
                "path": str(file_path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "extension": file_path.suffix,
                "magic_bytes": magic_bytes,
                "file_type": file_type,
                "is_symlink": file_path.is_symlink()
            }
        except Exception as e:
            return {
                "name": file_path.name,
                "path": str(file_path),
                "error": str(e)
            }
            
    def inspect_directory(self, directory: Path, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
        """Recursively inspect a directory"""
        if current_depth >= max_depth:
            return {"truncated": True}
            
        result = {
            "path": str(directory),
            "files": [],
            "directories": {}
        }
        
        try:
            for item in sorted(directory.iterdir()):
                if item.is_file():
                    # Only include files larger than 1MB or with relevant extensions
                    if item.stat().st_size > 1024 * 1024 or item.suffix in ['.pth', '.pt', '.safetensors', '.gguf', '.bin', '.json', '.yaml', '.txt', '.md']:
                        result["files"].append(self.get_file_info(item))
                elif item.is_dir() and not item.name.startswith('.'):
                    result["directories"][item.name] = self.inspect_directory(item, max_depth, current_depth + 1)
        except Exception as e:
            result["error"] = str(e)
            
        return result
        
    def find_model_files(self) -> Dict[str, List[Dict[str, Any]]]:
        """Find all potential model files"""
        model_extensions = ['.pth', '.pt', '.safetensors', '.gguf', '.bin', '.ckpt', '.pkl']
        config_extensions = ['.json', '.yaml', '.yml', '.txt']
        
        results = {
            "model_files": [],
            "config_files": [],
            "large_files": []  # Files > 100MB
        }
        
        try:
            for root, dirs, files in os.walk(self.model_path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    file_path = Path(root) / file
                    
                    try:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        
                        if file_path.suffix in model_extensions:
                            results["model_files"].append({
                                "path": str(file_path),
                                "name": file,
                                "size_mb": round(size_mb, 2),
                                "parent_dir": file_path.parent.name
                            })
                        elif file_path.suffix in config_extensions and size_mb < 10:  # Config files should be small
                            results["config_files"].append({
                                "path": str(file_path),
                                "name": file,
                                "size_mb": round(size_mb, 2),
                                "parent_dir": file_path.parent.name
                            })
                            
                        if size_mb > 100:  # Track all large files
                            results["large_files"].append({
                                "path": str(file_path),
                                "name": file,
                                "size_mb": round(size_mb, 2),
                                "extension": file_path.suffix
                            })
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Error scanning for model files: {e}")
            
        # Sort by size
        for category in results:
            results[category].sort(key=lambda x: x.get('size_mb', 0), reverse=True)
            
        return results
        
    def check_specific_models(self) -> Dict[str, Any]:
        """Check for specific expected models"""
        expected = {
            "wan2.1_gguf": self.model_path / "wan2.1-i2v-14b-480p" / "Wan2.1-I2V-14B-480P_Q4_K_M.gguf",
            "wan2.1_vae": self.model_path / "wan2.1-vae" / "Wan2.1_VAE.pth",
            "multitalk_safetensors": self.model_path / "meigen-multitalk" / "multitalk.safetensors",
            "wav2vec2": self.model_path / "wav2vec2-base-960h",
            "chinese_wav2vec2": self.model_path / "chinese-wav2vec2-base",
            
            # Check for other possible Wan2.1 files
            "wan2.1_unet": self.model_path / "wan2.1-i2v-14b-480p" / "unet",
            "wan2.1_diffusion": self.model_path / "wan2.1-i2v-14b-480p" / "diffusion_pytorch_model.bin",
            "wan2.1_safetensors": self.model_path / "wan2.1-i2v-14b-480p" / "diffusion_pytorch_model.safetensors",
        }
        
        results = {}
        
        for name, path in expected.items():
            if path.exists():
                if path.is_file():
                    info = self.get_file_info(path)
                    results[name] = {
                        "exists": True,
                        "type": "file",
                        **info
                    }
                else:
                    # It's a directory, list contents
                    contents = []
                    try:
                        for item in path.iterdir():
                            if item.is_file() and item.stat().st_size > 1024 * 1024:  # > 1MB
                                contents.append(self.get_file_info(item))
                    except:
                        pass
                    
                    results[name] = {
                        "exists": True,
                        "type": "directory",
                        "path": str(path),
                        "contents": contents
                    }
            else:
                results[name] = {
                    "exists": False,
                    "path": str(path)
                }
                
        return results
        
    def check_symlinks(self) -> List[Dict[str, str]]:
        """Check for any symlinks that might point to model files"""
        symlinks = []
        
        try:
            for root, dirs, files in os.walk(self.model_path):
                for item in files + dirs:
                    path = Path(root) / item
                    if path.is_symlink():
                        try:
                            target = path.resolve()
                            symlinks.append({
                                "link": str(path),
                                "target": str(target),
                                "target_exists": target.exists()
                            })
                        except:
                            symlinks.append({
                                "link": str(path),
                                "target": "unresolvable",
                                "target_exists": False
                            })
        except Exception as e:
            logger.error(f"Error checking symlinks: {e}")
            
        return symlinks


class MultiTalkV63Pipeline:
    """Debug pipeline that inspects models instead of running inference"""
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        self.model_path = model_path
        self.inspector = ModelInspector(model_path)
        logger.info("MultiTalk V63 Debug Pipeline initialized")
        
    def process_audio_to_video(
        self,
        audio_data: bytes,
        reference_image: bytes,
        prompt: str = "Debug inspection",
        **kwargs
    ) -> Dict[str, Any]:
        """Instead of processing, return model inspection results"""
        
        logger.info("Running model inspection...")
        
        # Run all inspections
        inspection_results = {
            "model_path": self.model_path,
            "model_path_exists": os.path.exists(self.model_path),
            "specific_models": self.inspector.check_specific_models(),
            "all_model_files": self.inspector.find_model_files(),
            "directory_structure": self.inspector.inspect_directory(Path(self.model_path), max_depth=3),
            "symlinks": self.inspector.check_symlinks()
        }
        
        # Create a summary
        summary = {
            "gguf_found": inspection_results["specific_models"].get("wan2.1_gguf", {}).get("exists", False),
            "vae_found": inspection_results["specific_models"].get("wan2.1_vae", {}).get("exists", False),
            "multitalk_found": inspection_results["specific_models"].get("multitalk_safetensors", {}).get("exists", False),
            "total_model_files": len(inspection_results["all_model_files"]["model_files"]),
            "total_large_files": len(inspection_results["all_model_files"]["large_files"]),
            "wan2.1_pytorch_found": any(
                "wan2.1" in str(f.get("path", "")).lower() and 
                f.get("path", "").endswith(('.pth', '.pt', '.safetensors', '.bin'))
                for f in inspection_results["all_model_files"]["model_files"]
            )
        }
        
        # Log key findings
        logger.info(f"Model inspection summary: {json.dumps(summary, indent=2)}")
        
        # Log ALL model files found
        logger.info("\n=== ALL MODEL FILES FOUND ===")
        for f in inspection_results["all_model_files"]["model_files"]:
            logger.info(f"Model: {f['path']} ({f['size_mb']} MB)")
            
        # Log ALL large files
        logger.info("\n=== ALL LARGE FILES (>100MB) ===")
        for f in inspection_results["all_model_files"]["large_files"]:
            logger.info(f"Large file: {f['path']} ({f['size_mb']} MB, ext: {f['extension']})")
            
        # Log specific model checks
        logger.info("\n=== SPECIFIC MODEL CHECKS ===")
        for name, info in inspection_results["specific_models"].items():
            if info["exists"]:
                logger.info(f"{name}: EXISTS at {info.get('path', 'unknown')}")
                if info["type"] == "file":
                    logger.info(f"  Size: {info.get('size_mb', 'unknown')} MB")
                elif info["type"] == "directory" and info.get("contents"):
                    logger.info(f"  Contents: {len(info['contents'])} files")
                    for content in info["contents"][:3]:  # First 3 files
                        logger.info(f"    - {content['name']} ({content['size_mb']} MB)")
            else:
                logger.info(f"{name}: NOT FOUND at {info['path']}")
                
        # Log Wan-specific findings
        wan_files = [f for f in inspection_results["all_model_files"]["model_files"] 
                     if "wan" in str(f.get("path", "")).lower()]
        if wan_files:
            logger.info(f"\n=== WAN MODEL FILES ({len(wan_files)} found) ===")
            for f in wan_files:
                logger.info(f"  - {f['path']} ({f['size_mb']} MB)")
        
        # Return results instead of video
        return {
            "success": True,
            "video_data": json.dumps(inspection_results, indent=2).encode('utf-8'),  # Return as "video" data
            "model": "multitalk-v63-debug-inspector",
            "inspection_summary": summary,
            "message": "Model inspection complete. Check logs for detailed results.",
            "note": "This is debug output, not actual video generation"
        }
        
    def cleanup(self):
        """No cleanup needed for debug"""
        pass