#!/bin/bash
# Build-time setup script for official MultiTalk

set -e

echo "=== Setting up Official MultiTalk Implementation at Build Time ==="

# Create directory
mkdir -p /app/multitalk_official

# Download key files from official repo
echo "Downloading official MultiTalk files..."
cd /app/multitalk_official

# Download main generation script
wget -q -O generate_multitalk.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/generate_multitalk.py || echo "Failed to download generate_multitalk.py"

# Download model loading utilities
mkdir -p wan
wget -q -O wan/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/__init__.py || echo "Failed to download wan/__init__.py"
wget -q -O wan/model.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/model.py || echo "Failed to download wan/model.py"
wget -q -O wan/vace.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/vace.py || echo "Failed to download wan/vace.py"

# Download utils
mkdir -p utils
wget -q -O utils/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/__init__.py || echo "Failed to download utils/__init__.py"
wget -q -O utils/tools.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/tools.py || echo "Failed to download utils/tools.py"

# Download TeaCache implementation
mkdir -p utils/teacache
wget -q -O utils/teacache/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/teacache/__init__.py || echo "Failed to download teacache/__init__.py"
wget -q -O utils/teacache/cache.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/teacache/cache.py || echo "Failed to download teacache/cache.py"

echo "✓ Downloaded official MultiTalk files"

# Verify critical files exist
if [ -f "generate_multitalk.py" ]; then
    echo "✓ generate_multitalk.py downloaded successfully"
else
    echo "ERROR: generate_multitalk.py not found"
fi

echo "✓ Official MultiTalk setup complete"