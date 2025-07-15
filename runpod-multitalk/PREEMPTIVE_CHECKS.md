# Preemptive Checks for MultiTalk Deployment

## Before Creating New Version

### 1. Check Previous Issues
```bash
# Review known issues document
cat KNOWN_ISSUES.md

# Check if NumPy/SciPy fix is included
grep -A2 "pip.*numpy.*scipy" Dockerfile

# Verify no mock/placeholder code
grep -r "generate_multitalk_mock\|placeholder\|demo_video" .
```

### 2. Dependency Validation
```python
# Add to Dockerfile for build-time validation
RUN python -c "
import scipy.stats
import numpy as np
print(f'NumPy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')
print('✅ NumPy/SciPy imports successfully')
"
```

### 3. Implementation Completeness
```dockerfile
# Verify all required files exist
RUN test -f /app/multitalk_official/generate_multitalk.py || exit 1
RUN test -d /app/multitalk_official/wan/configs || exit 1
RUN test -d /app/multitalk_official/wan/distributed || exit 1
RUN test -d /app/multitalk_official/wan/modules || exit 1
RUN test -d /app/multitalk_official/wan/utils || exit 1
```

## Pattern Recognition Checklist

Before pushing new version:

- [ ] NumPy/SciPy fix applied? (numpy==1.24.3, scipy==1.10.1)
- [ ] All wan subdirectories downloaded? (configs, distributed, modules, utils)
- [ ] No mock/placeholder code remaining?
- [ ] Output file discovery handles multiple locations?
- [ ] Build-time validation tests included?
- [ ] Checked against KNOWN_ISSUES.md?

## Automated Pre-build Script

```bash
#!/bin/bash
# pre-build-check.sh

echo "Running pre-build checks..."

# Check for NumPy/SciPy fix
if ! grep -q "numpy==1.24.3.*scipy==1.10.1" Dockerfile; then
    echo "❌ ERROR: NumPy/SciPy fix not found in Dockerfile"
    echo "Add: RUN pip uninstall -y numpy scipy && pip install numpy==1.24.3 scipy==1.10.1"
    exit 1
fi

# Check for mock code
if grep -r "generate_multitalk_mock\|MOCK\|dummy\|placeholder" . --include="*.py"; then
    echo "❌ ERROR: Mock/placeholder code detected"
    exit 1
fi

# Check for comprehensive wan module download
for module in configs distributed modules utils; do
    if ! grep -q "wan/$module/__init__.py" setup_official_multitalk*.sh; then
        echo "❌ ERROR: Missing wan/$module in setup script"
        exit 1
    fi
done

echo "✅ Pre-build checks passed"
```

## Version Increment Strategy

When creating new versions:

1. **Copy forward all fixes from previous working version**
2. **Run pre-build-check.sh before building**
3. **Document what changed and why in commit message**
4. **Test locally with known edge cases before deploying**

Example:
```bash
# Start from last known good version
cp Dockerfile.v77-numpy-fix Dockerfile.v78-new-feature

# Apply new changes
vim Dockerfile.v78-new-feature

# Run checks
./pre-build-check.sh

# Build and test
docker build -t test:v78 -f Dockerfile.v78-new-feature .
```