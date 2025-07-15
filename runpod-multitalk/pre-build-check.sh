#!/bin/bash
# Pre-build validation script for MultiTalk Docker images
# Run this before building any new version to catch known issues

set -e

echo "============================================"
echo "Running MultiTalk Pre-Build Validation"
echo "============================================"

ERRORS=0
WARNINGS=0

# Function to check and report
check() {
    local condition=$1
    local error_msg=$2
    local fix_msg=$3
    
    if ! eval "$condition"; then
        echo "❌ ERROR: $error_msg"
        echo "   FIX: $fix_msg"
        echo ""
        ((ERRORS++))
        return 1
    fi
    return 0
}

warn() {
    local condition=$1
    local warn_msg=$2
    
    if ! eval "$condition"; then
        echo "⚠️  WARNING: $warn_msg"
        echo ""
        ((WARNINGS++))
    fi
}

# Check 1: NumPy/SciPy compatibility fix
echo "1. Checking NumPy/SciPy compatibility fix..."
check 'grep -q "numpy==1.24.3.*scipy==1.10.1" Dockerfile* 2>/dev/null' \
    "NumPy/SciPy binary compatibility fix not found" \
    "Add to Dockerfile: RUN pip uninstall -y numpy scipy && pip install --no-cache-dir numpy==1.24.3 scipy==1.10.1"

# Check 2: Mock/placeholder code
echo "2. Checking for mock/placeholder code..."
check '! grep -r "generate_multitalk_mock\|generate_multitalk_v[0-9]*\|MOCK\|dummy_video\|placeholder" . --include="*.py" --include="*.sh" 2>/dev/null | grep -v pre-build-check' \
    "Mock or placeholder code detected" \
    "Remove all mock implementations and use only official MultiTalk"

# Check 3: Official implementation download
echo "3. Checking official implementation setup..."
if ls setup_official_multitalk*.sh >/dev/null 2>&1; then
    SETUP_SCRIPT=$(ls setup_official_multitalk*.sh | head -1)
    
    # Check for all required wan subdirectories
    for module in configs distributed modules utils; do
        check "grep -q 'wan/$module' $SETUP_SCRIPT" \
            "Missing wan/$module download in setup script" \
            "Add download for wan/$module/__init__.py and all files"
    done
    
    # Check for generate_multitalk.py download
    check "grep -q 'generate_multitalk.py' $SETUP_SCRIPT" \
        "Missing generate_multitalk.py download" \
        "Add download for official generate_multitalk.py"
else
    echo "❌ ERROR: No setup_official_multitalk*.sh script found"
    ((ERRORS++))
fi

# Check 4: Build-time validation
echo "4. Checking build-time validation..."
warn 'grep -q "python -c.*import scipy" Dockerfile* 2>/dev/null' \
    "No build-time scipy import test found"

# Check 5: Output file discovery
echo "5. Checking output file discovery logic..."
if ls multitalk_v*.py handler*.py 2>/dev/null | head -1 >/dev/null; then
    warn 'grep -q "potential_outputs\|multiple.*locations\|glob.*mp4" multitalk_v*.py handler*.py 2>/dev/null' \
        "Output file discovery may not check multiple locations"
fi

# Check 6: Git clone vs wget approach
echo "6. Checking implementation download method..."
if grep -q "git clone.*MeiGen-AI/MultiTalk" setup_official_multitalk*.sh 2>/dev/null; then
    echo "✅ Using git clone (comprehensive approach)"
elif grep -q "wget.*raw.githubusercontent.com.*MultiTalk" setup_official_multitalk*.sh 2>/dev/null; then
    echo "⚠️  Using wget (ensure ALL files are downloaded)"
    ((WARNINGS++))
fi

# Check 7: xFormers and GCC
echo "7. Checking compiler dependencies..."
check 'grep -q "gcc\|g++\|build-essential" Dockerfile* 2>/dev/null' \
    "GCC/build tools not installed" \
    "Add: RUN apt-get update && apt-get install -y gcc g++ build-essential"

# Check 8: Known issues documentation
echo "8. Checking documentation..."
warn 'test -f KNOWN_ISSUES.md' \
    "KNOWN_ISSUES.md not found - consider creating it to track recurring problems"

echo "============================================"
echo "Validation Summary:"
echo "  Errors: $ERRORS"
echo "  Warnings: $WARNINGS"
echo "============================================"

if [ $ERRORS -gt 0 ]; then
    echo "❌ Build validation FAILED - fix errors before building"
    exit 1
else
    echo "✅ Build validation PASSED"
    if [ $WARNINGS -gt 0 ]; then
        echo "   (with $WARNINGS warnings - consider addressing them)"
    fi
    exit 0
fi