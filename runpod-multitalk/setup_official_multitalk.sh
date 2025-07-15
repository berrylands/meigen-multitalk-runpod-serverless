#!/bin/bash
# Setup script to download and prepare official MultiTalk implementation

set -e

echo "=== Setting up Official MultiTalk Implementation ==="

# Create directory for official code
mkdir -p /app/multitalk_official

# Clone the official repository
echo "Cloning MeiGen-AI/MultiTalk repository..."
cd /app/multitalk_official
git clone https://github.com/MeiGen-AI/MultiTalk.git .

# Install additional dependencies that official MultiTalk needs
echo "Installing official dependencies..."
pip install flash-attn --no-build-isolation || echo "flash-attn installation failed, continuing..."
pip install diffusers accelerate transformers einops imageio-ffmpeg

# Create a wrapper script that sets up the environment
cat > /app/run_official_multitalk.py << 'EOF'
#!/usr/bin/env python3
"""
Wrapper to run official MultiTalk with proper paths
"""
import sys
import os

# Add MultiTalk directory to Python path
sys.path.insert(0, '/app/multitalk_official')

# Import and run the official script
from generate_multitalk import main

if __name__ == "__main__":
    main()
EOF

chmod +x /app/run_official_multitalk.py

echo "✓ Official MultiTalk setup complete"