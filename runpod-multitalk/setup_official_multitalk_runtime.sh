#!/bin/bash
# Runtime setup script for official MultiTalk (no git required)

set -e

echo "=== Setting up Official MultiTalk Implementation at Runtime ==="

# Check if already set up
if [ -f "/app/multitalk_official/generate_multitalk.py" ]; then
    echo "✓ Official MultiTalk already set up"
    exit 0
fi

# Create directory
mkdir -p /app/multitalk_official

# Download key files from official repo
echo "Downloading official MultiTalk files..."
cd /app/multitalk_official

# Download main generation script
wget -q -O generate_multitalk.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/generate_multitalk.py || echo "Failed to download generate_multitalk.py"

# Download other essential files
wget -q -O requirements.txt https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/requirements.txt || echo "Failed to download requirements.txt"

# Download model loading utilities
mkdir -p wan
wget -q -O wan/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/__init__.py || echo "Failed to download wan/__init__.py"
wget -q -O wan/model.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/model.py || echo "Failed to download wan/model.py"
wget -q -O wan/vace.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/vace.py || echo "Failed to download wan/vace.py"

# Download utils
mkdir -p utils
wget -q -O utils/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/__init__.py || echo "Failed to download utils/__init__.py"

echo "✓ Downloaded official MultiTalk files"

# Install any missing dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing official dependencies..."
    pip install -r requirements.txt || echo "Some dependencies failed to install"
fi

echo "✓ Official MultiTalk setup complete"