# Deploy MultiTalk V58 - Device Fix

## What's Ready

1. **Fixed Implementation**: `multitalk_v58_device_fix.py`
   - Fixes "Expected all tensors to be on the same device" error
   - L-RoPE speaker embeddings moved to GPU on initialization
   - Speaker tensor created directly on GPU device

2. **Dockerfile**: `Dockerfile.v58-minimal`
   - Minimal build based on v57
   - Updates handler to use V58 implementation

3. **Build Script**: `build_v58_wait.sh`
   - Waits for Docker to be ready
   - Builds and pushes the image

## Steps to Deploy

1. **Wait for Docker to fully start** (it's recreating its VM disk after cleanup):
   ```bash
   # Check Docker status
   docker info
   ```

2. **Once Docker is ready, build v58**:
   ```bash
   cd /Users/jasonedge/CODEHOME/meigen-multitalk/runpod-multitalk
   ./build_v58_wait.sh
   ```

   Or manually:
   ```bash
   docker build -f Dockerfile.v58-minimal -t berrylands/multitalk-v58:device-fix .
   docker push berrylands/multitalk-v58:device-fix
   ```

3. **Deploy to RunPod**:
   - Image: `berrylands/multitalk-v58:device-fix`
   - This fixes the device mismatch error from v57

## Key Fix in V58

The error in v57 was:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

V58 fixes this by:
```python
# In __init__:
self.label_embeddings = nn.Embedding(8, 768).to(self.device)  # Move to GPU

# In extract_features:
speaker_tensor = torch.tensor([speaker_id], device=self.device, dtype=torch.long)  # Create on GPU
```

This ensures all tensors in the L-RoPE speaker binding operation are on the same device.