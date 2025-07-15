# Known Issues and Solutions for MultiTalk RunPod Deployment

This document tracks recurring issues and their solutions to prevent repeated failures.

## 1. NumPy/SciPy Binary Incompatibility

### Error Pattern
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

### Root Cause
- SciPy compiled against different NumPy version than what's installed
- Occurs when importing scipy.stats -> scipy.spatial -> _ckdtree

### Solution
```dockerfile
# Always include this fix when using official MultiTalk
RUN pip uninstall -y numpy scipy && \
    pip install --no-cache-dir numpy==1.24.3 scipy==1.10.1
```

### Affected Versions
- V74.3: First encountered and fixed
- V76: Recurred because fix wasn't carried forward
- V77: Applied fix proactively

## 2. Mock/Placeholder Code Issues

### Pattern
- Mock scripts reporting success but not creating actual output files
- Placeholder implementations instead of real AI inference

### Detection
```bash
# Check for mock indicators
grep -r "mock\|dummy\|placeholder\|demo" .
# Verify real implementation exists
ls /app/multitalk_official/generate_multitalk.py
```

### Solution
- Always download official implementation from GitHub
- Remove ALL fallback/demo logic
- Fail fast if real implementation missing

## 3. Missing Dependencies/Modules

### Pattern
- Import errors for wan subdirectories (configs, distributed, modules, utils)
- Missing src directory modules (vram_management, audio_analysis)
- Missing kokoro module

### Solution
```bash
# Use comprehensive download script that gets ALL subdirectories
# Don't assume directory structure - download file by file
wget -q -O wan/configs/__init__.py https://...
wget -q -O wan/distributed/__init__.py https://...
# etc.
```

## 4. Output File Discovery

### Pattern
- Script completes successfully but output file not found
- Working directory mismatch between handler and subprocess

### Solution
```python
# Always search multiple locations
potential_outputs = [
    Path(self.multitalk_path) / output_name,
    Path(self.multitalk_path) / "output_video.mp4",
    temp_path / output_name,
    Path("/tmp") / output_name,
    Path.cwd() / output_name
]
# Also check for recently created files
for mp4 in Path(self.multitalk_path).glob("*.mp4"):
    if mp4.stat().st_mtime > start_time:
        potential_outputs.append(mp4)
```

## 5. Docker Build Caching Issues

### Pattern
- Changes not reflected in built image
- Old code running despite updates

### Solution
```bash
# Force rebuild without cache
docker build --no-cache -t image:tag .
# Or bust cache at specific layer
ARG CACHEBUST=1
```

## 6. Missing RunPod SDK

### Error Pattern
```
ModuleNotFoundError: No module named 'runpod'
```

### Root Cause
- Handler imports runpod but we forgot to install it
- Critical dependency for RunPod serverless operation

### Solution
```dockerfile
RUN pip install --no-cache-dir \
    runpod==1.7.3 \  # CRITICAL: Must be first!
    numpy==1.24.3 \
    scipy==1.10.1 \
    # ... other dependencies
```

### Verification
```python
RUN python -c "import runpod; print('✅ RunPod imports')"
```

### Affected Versions
- V79: Forgot to install runpod package

---

## 7. Kokoro/Misaki Module Not Found

### Error Pattern
```
ModuleNotFoundError: No module named 'misaki'
```

### Root Cause
- Official MultiTalk repo includes kokoro module that depends on misaki
- Working Replicate implementation doesn't use kokoro at all

### Solution
- Don't copy kokoro module during setup
- Remove kokoro imports from generate_multitalk.py
- Follow Replicate/Cog implementation approach

### Affected Versions
- V77: First encountered when using real MultiTalk
- V78: Fixed by following Replicate approach