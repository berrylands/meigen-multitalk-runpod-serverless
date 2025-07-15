# Issue Prevention Guide for MultiTalk Deployment

## How to Encourage Pattern Recognition and Preemptive Problem Solving

### 1. Maintain a Living Issues Document

Track every issue encountered with:
- **Error signature** (exact error message)
- **Root cause** 
- **Solution applied**
- **Versions affected**

Example:
```yaml
issue: numpy_scipy_incompatibility
error: "ValueError: numpy.dtype size changed"
cause: "Binary incompatibility between NumPy and SciPy versions"
solution: "pip install numpy==1.24.3 scipy==1.10.1"
versions: [v74.3, v76]
```

### 2. Use Incremental Validation

When creating new versions:
```dockerfile
# V78 Dockerfile
FROM berrylands/multitalk-v77:numpy-fix  # Start from WORKING version

# VALIDATION CHECKPOINT 1: Verify inherited fixes
RUN python -c "import scipy.stats; print('✅ NumPy/SciPy OK')"

# Your new changes here...

# VALIDATION CHECKPOINT 2: Verify new changes didn't break anything
RUN python -c "import scipy.stats; print('✅ Still OK after changes')"
```

### 3. Create Issue-Specific Tests

For each recurring issue, create a test:
```python
# test_known_issues.py
def test_numpy_scipy_compatibility():
    """Prevent regression of NumPy/SciPy incompatibility"""
    try:
        import scipy.stats
        import scipy.spatial
        from scipy.spatial._ckdtree import cKDTree
        print("✅ NumPy/SciPy compatibility OK")
    except ValueError as e:
        if "dtype size changed" in str(e):
            raise RuntimeError(
                "NumPy/SciPy incompatibility detected!\n"
                "Fix: pip install numpy==1.24.3 scipy==1.10.1"
            )
```

### 4. Use Explicit Reminders in Code

```dockerfile
# ⚠️ CRITICAL: NumPy/SciPy Fix Required
# We've hit this issue in v74.3 and v76
# MUST include this or scipy imports will fail
RUN pip uninstall -y numpy scipy && \
    pip install --no-cache-dir numpy==1.24.3 scipy==1.10.1
```

### 5. Version Naming Convention

Use descriptive version names that indicate fixes:
- `v77-numpy-fix` (clearly shows it includes the NumPy fix)
- `v78-numpy-fix-plus-feature` (shows it maintains the fix)

### 6. Pre-flight Checklist

Before deploying ANY new version:
```bash
#!/bin/bash
echo "PRE-DEPLOYMENT CHECKLIST:"
echo "[ ] Ran pre-build-check.sh?"
echo "[ ] Started from last working version?"
echo "[ ] Checked KNOWN_ISSUES.md?"
echo "[ ] Tested NumPy/SciPy imports?"
echo "[ ] Verified no mock code?"
echo "[ ] Documented what changed?"
```

### 7. Fail Fast, Fail Clearly

```python
# Better error handling
try:
    import scipy.stats
except ValueError as e:
    if "dtype size changed" in str(e):
        print("="*60)
        print("KNOWN ISSUE: NumPy/SciPy Binary Incompatibility")
        print("This was fixed in v74.3 and v77")
        print("Solution: pip install numpy==1.24.3 scipy==1.10.1")
        print("="*60)
        raise
```

## Example: How V76 Could Have Been Prevented

1. **Started from v75 (broken) instead of v74.3 (working)**
   - Should have checked which version last worked
   
2. **Didn't carry forward the NumPy fix from v74.3**
   - Pre-build check would have caught this
   
3. **No validation of scipy imports**
   - Build-time test would have failed immediately

## Recommended Workflow

1. **Before starting new version:**
   ```bash
   # Check last working version
   grep -l "scipy==1.10.1" Dockerfile.v* | tail -1
   
   # Copy from working version
   cp Dockerfile.v74-working Dockerfile.v78-new
   ```

2. **During development:**
   ```bash
   # Run validation after each change
   ./pre-build-check.sh
   
   # Test locally first
   docker build -t test:v78 .
   docker run test:v78 python -c "import scipy.stats"
   ```

3. **Before pushing:**
   ```bash
   # Final validation
   ./pre-build-check.sh
   
   # Document in commit
   git commit -m "v78: Add feature X, maintains NumPy fix from v74.3"
   ```

## The Meta-Pattern

Most issues follow this pattern:
1. **Fix applied in version X**
2. **New version Y created from different base**
3. **Fix not carried forward**
4. **Same error recurs**

Break this pattern by:
- Always starting from last WORKING version
- Running automated checks
- Documenting fixes prominently
- Testing known failure modes