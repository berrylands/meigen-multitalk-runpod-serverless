"""
MultiTalk Debug Implementation
Tests all dependencies and assumptions, reports findings
"""
import os
import sys
import torch
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib.util

# Configure detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MultiTalkDebugPipeline:
    """Debug implementation to test all assumptions"""
    
    def __init__(self, model_path: str = "/runpod-volume/models"):
        self.model_path = Path(model_path)
        self.debug_info = {
            "model_path_exists": False,
            "models_found": {},
            "dependencies": {},
            "imports": {},
            "initialization": {},
            "errors": []
        }
        
        logger.info("=== MULTITALK DEBUG INITIALIZATION ===")
        
        # Run all debug checks
        self._check_model_directory()
        self._check_dependencies()
        self._check_imports()
        self._check_model_files()
        self._attempt_model_loading()
        
        # Report findings
        self._report_findings()
    
    def _check_model_directory(self):
        """Check if model directory exists and list contents"""
        try:
            if self.model_path.exists():
                self.debug_info["model_path_exists"] = True
                logger.info(f"✓ Model path exists: {self.model_path}")
                
                # List all directories
                dirs = [d.name for d in self.model_path.iterdir() if d.is_dir()]
                logger.info(f"Directories found: {dirs}")
                
                # List all files in root
                files = [f.name for f in self.model_path.iterdir() if f.is_file()]
                logger.info(f"Files in root: {files}")
                
                # Check specific model directories
                expected_models = [
                    "Wan2.1-I2V-14B-480P",
                    "chinese-wav2vec2-base",
                    "wav2vec2-base-960h",
                    "MeiGen-MultiTalk",
                    "MultiTalk"
                ]
                
                for model in expected_models:
                    model_dir = self.model_path / model
                    if model_dir.exists():
                        self.debug_info["models_found"][model] = True
                        # List contents
                        contents = list(model_dir.iterdir())[:10]  # First 10 items
                        logger.info(f"✓ {model} exists with {len(list(model_dir.iterdir()))} items")
                        logger.debug(f"  Sample contents: {[c.name for c in contents]}")
                    else:
                        self.debug_info["models_found"][model] = False
                        logger.warning(f"✗ {model} not found")
            else:
                logger.error(f"✗ Model path does not exist: {self.model_path}")
                
        except Exception as e:
            logger.error(f"Error checking model directory: {e}")
            self.debug_info["errors"].append(f"Model directory check: {str(e)}")
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        dependencies = {
            "torch": "import torch",
            "transformers": "import transformers",
            "diffusers": "import diffusers",
            "xformers": "import xformers",
            "einops": "import einops",
            "librosa": "import librosa",
            "soundfile": "import soundfile",
            "cv2": "import cv2",
            "PIL": "from PIL import Image",
            "imageio": "import imageio",
            "omegaconf": "import omegaconf",
            "decord": "import decord"
        }
        
        for name, import_str in dependencies.items():
            try:
                exec(import_str)
                self.debug_info["dependencies"][name] = True
                logger.info(f"✓ {name} available")
            except ImportError as e:
                self.debug_info["dependencies"][name] = False
                logger.warning(f"✗ {name} not available: {e}")
    
    def _check_imports(self):
        """Check if we can import MultiTalk components"""
        # Add potential paths to sys.path
        potential_paths = [
            self.model_path / "MultiTalk",
            self.model_path / "MeiGen-MultiTalk",
            self.model_path,
            Path("/app")
        ]
        
        for path in potential_paths:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
                logger.info(f"Added to sys.path: {path}")
        
        # Try to import key components
        imports_to_test = [
            ("wan", "wan"),
            ("wan.utils.config", "wan.utils.config"),
            ("generate_multitalk", "generate_multitalk"),
            ("multitalk", "multitalk")
        ]
        
        for name, module in imports_to_test:
            try:
                if importlib.util.find_spec(module):
                    imported = importlib.import_module(module)
                    self.debug_info["imports"][name] = True
                    logger.info(f"✓ Can import {name}")
                    
                    # Check attributes
                    if hasattr(imported, "__file__"):
                        logger.debug(f"  Location: {imported.__file__}")
                    if hasattr(imported, "__version__"):
                        logger.debug(f"  Version: {imported.__version__}")
                else:
                    self.debug_info["imports"][name] = False
                    logger.warning(f"✗ Cannot find module {name}")
                    
            except Exception as e:
                self.debug_info["imports"][name] = False
                logger.warning(f"✗ Cannot import {name}: {e}")
    
    def _check_model_files(self):
        """Check for specific model files"""
        # Check for GGUF files
        gguf_files = list(self.model_path.rglob("*.gguf"))
        if gguf_files:
            logger.info(f"Found {len(gguf_files)} GGUF files:")
            for f in gguf_files[:5]:  # Show first 5
                logger.info(f"  - {f.relative_to(self.model_path)}")
        
        # Check for PyTorch checkpoints
        pt_files = list(self.model_path.rglob("*.pt"))
        pth_files = list(self.model_path.rglob("*.pth"))
        bin_files = list(self.model_path.rglob("*.bin"))
        safetensors_files = list(self.model_path.rglob("*.safetensors"))
        
        logger.info(f"Model file counts:")
        logger.info(f"  - .pt files: {len(pt_files)}")
        logger.info(f"  - .pth files: {len(pth_files)}")
        logger.info(f"  - .bin files: {len(bin_files)}")
        logger.info(f"  - .safetensors files: {len(safetensors_files)}")
        
        # Check for config files
        config_files = list(self.model_path.rglob("config.json"))
        logger.info(f"Found {len(config_files)} config.json files")
        
        # Check for specific MultiTalk files
        multitalk_indicators = [
            "generate_multitalk.py",
            "wan.py",
            "multitalk_model.py",
            "audio_condition"
        ]
        
        for indicator in multitalk_indicators:
            found = list(self.model_path.rglob(f"*{indicator}*"))
            if found:
                logger.info(f"✓ Found {indicator}: {len(found)} matches")
                logger.debug(f"  First match: {found[0].relative_to(self.model_path)}")
    
    def _attempt_model_loading(self):
        """Try to load models with detailed error reporting"""
        # 1. Try loading Wav2Vec2
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
            
            wav2vec_paths = [
                self.model_path / "wav2vec2-base-960h",
                self.model_path / "chinese-wav2vec2-base"
            ]
            
            for wav2vec_path in wav2vec_paths:
                if wav2vec_path.exists():
                    try:
                        processor = Wav2Vec2Processor.from_pretrained(
                            str(wav2vec_path),
                            local_files_only=True
                        )
                        model = Wav2Vec2Model.from_pretrained(
                            str(wav2vec_path),
                            local_files_only=True
                        )
                        self.debug_info["initialization"]["wav2vec2"] = True
                        logger.info(f"✓ Successfully loaded Wav2Vec2 from {wav2vec_path.name}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load Wav2Vec2 from {wav2vec_path.name}: {e}")
                        
        except ImportError:
            logger.warning("Transformers not available for Wav2Vec2 loading")
        
        # 2. Try to initialize MultiTalk components
        try:
            # Check if we can run generate_multitalk.py
            generate_script = self.model_path / "MultiTalk" / "generate_multitalk.py"
            if generate_script.exists():
                logger.info(f"✓ generate_multitalk.py found at {generate_script}")
                
                # Check if it's executable
                result = subprocess.run(
                    ["python", str(generate_script), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    logger.info("✓ generate_multitalk.py is executable")
                    logger.debug(f"Help output: {result.stdout[:200]}...")
                else:
                    logger.warning(f"✗ generate_multitalk.py failed: {result.stderr}")
                    
        except Exception as e:
            logger.error(f"Error checking generate_multitalk.py: {e}")
    
    def _report_findings(self):
        """Generate comprehensive report"""
        logger.info("\n" + "="*60)
        logger.info("MULTITALK DEBUG REPORT")
        logger.info("="*60)
        
        # Model directory status
        logger.info(f"\n1. MODEL DIRECTORY:")
        logger.info(f"   Path: {self.model_path}")
        logger.info(f"   Exists: {self.debug_info['model_path_exists']}")
        
        # Models found
        logger.info(f"\n2. MODELS FOUND:")
        for model, found in self.debug_info["models_found"].items():
            status = "✓" if found else "✗"
            logger.info(f"   {status} {model}")
        
        # Dependencies
        logger.info(f"\n3. DEPENDENCIES:")
        for dep, available in self.debug_info["dependencies"].items():
            status = "✓" if available else "✗"
            logger.info(f"   {status} {dep}")
        
        # Imports
        logger.info(f"\n4. MULTITALK IMPORTS:")
        for imp, available in self.debug_info["imports"].items():
            status = "✓" if available else "✗"
            logger.info(f"   {status} {imp}")
        
        # Initialization
        logger.info(f"\n5. MODEL INITIALIZATION:")
        for model, success in self.debug_info["initialization"].items():
            status = "✓" if success else "✗"
            logger.info(f"   {status} {model}")
        
        # Errors
        if self.debug_info["errors"]:
            logger.info(f"\n6. ERRORS ENCOUNTERED:")
            for error in self.debug_info["errors"]:
                logger.info(f"   - {error}")
        
        logger.info("\n" + "="*60)
        
        return self.debug_info
    
    def process_audio_to_video(
        self,
        audio_data: bytes,
        reference_image: bytes,
        prompt: str = "A person talking naturally",
        **kwargs
    ) -> Dict[str, Any]:
        """Process request and return debug information"""
        logger.info("\n=== PROCESSING REQUEST ===")
        logger.info(f"Audio size: {len(audio_data)} bytes")
        logger.info(f"Image size: {len(reference_image)} bytes")
        logger.info(f"Prompt: {prompt}")
        
        # Return debug information
        return {
            "success": True,
            "debug_info": self.debug_info,
            "message": "Debug run completed - check logs for detailed information",
            "video_data": b"",  # Empty video data for debug
            "model": "debug-pipeline"
        }