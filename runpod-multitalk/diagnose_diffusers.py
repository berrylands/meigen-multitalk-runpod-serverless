"""
Diagnose diffusers import issues
"""
import sys
import subprocess

print("=== DIAGNOSING DIFFUSERS ISSUE ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Check pip list
print("\n=== Installed packages ===")
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
print(result.stdout)

# Try importing diffusers with detailed error
print("\n=== Attempting to import diffusers ===")
try:
    import diffusers
    print(f"✓ diffusers imported successfully: {diffusers.__version__}")
    print(f"  Location: {diffusers.__file__}")
except Exception as e:
    print(f"✗ Failed to import diffusers: {type(e).__name__}: {e}")
    
    # Try to understand why
    print("\n=== Detailed error ===")
    import traceback
    traceback.print_exc()

# Check torch version compatibility
print("\n=== Checking torch ===")
try:
    import torch
    print(f"✓ torch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"✗ Failed to import torch: {e}")

# Try individual diffusers imports
print("\n=== Testing individual diffusers imports ===")
components = [
    "diffusers.models",
    "diffusers.pipelines", 
    "diffusers.schedulers",
    "diffusers.models.unet_2d_condition",
    "diffusers.models.autoencoder_kl"
]

for component in components:
    try:
        exec(f"import {component}")
        print(f"✓ {component}")
    except Exception as e:
        print(f"✗ {component}: {type(e).__name__}: {e}")

# Check transformers compatibility
print("\n=== Checking transformers ===")
try:
    import transformers
    print(f"✓ transformers version: {transformers.__version__}")
except Exception as e:
    print(f"✗ Failed to import transformers: {e}")

# Check accelerate
print("\n=== Checking accelerate ===")
try:
    import accelerate
    print(f"✓ accelerate version: {accelerate.__version__}")
except Exception as e:
    print(f"✗ Failed to import accelerate: {e}")

# Check for version conflicts
print("\n=== Checking for conflicts ===")
try:
    # Test if we can create basic diffusers objects
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
    print("✓ Can import core diffusers classes")
except Exception as e:
    print(f"✗ Cannot import core classes: {e}")
    
print("\n=== Diagnosis complete ===")