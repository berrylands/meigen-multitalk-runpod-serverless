#!/bin/bash
# Simplified setup script for official MultiTalk implementation
# Downloads only the essential files needed for the official implementation

set -e

echo "=== Setting up Official MultiTalk Implementation (Simplified) ==="

# Create the directory where the handler expects to find the code
MULTITALK_DIR="/app/multitalk_official"
mkdir -p "$MULTITALK_DIR"
cd "$MULTITALK_DIR"

echo "Downloading official MultiTalk files to $MULTITALK_DIR..."

# Download main generation script
echo "Downloading generate_multitalk.py..."
if ! curl -s -o generate_multitalk.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/generate_multitalk.py"; then
    echo "ERROR: Failed to download generate_multitalk.py"
    exit 1
fi

if [ ! -f "generate_multitalk.py" ] || [ ! -s "generate_multitalk.py" ]; then
    echo "ERROR: generate_multitalk.py not created or empty"
    exit 1
fi

echo "✓ generate_multitalk.py downloaded successfully"

# Download essential wan files  
echo "Downloading essential wan/ files..."
mkdir -p wan/{configs,distributed,modules,utils}

# Key files for wan directory
curl -s -o wan/__init__.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/__init__.py"
curl -s -o wan/multitalk.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/multitalk.py"
curl -s -o wan/image2video.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/image2video.py"

# Config files
curl -s -o wan/configs/__init__.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/__init__.py"
curl -s -o wan/configs/wan_multitalk_14B.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/wan_multitalk_14B.py"

# Module files  
curl -s -o wan/modules/__init__.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/__init__.py"
curl -s -o wan/modules/multitalk_model.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/multitalk_model.py"

# Utils files
curl -s -o wan/utils/__init__.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/__init__.py"
curl -s -o wan/utils/multitalk_utils.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/multitalk_utils.py"

# Create empty __init__.py files for missing ones
touch wan/distributed/__init__.py

echo "✓ Essential wan/ files downloaded"

# Download utils directory
echo "Downloading utils/ files..."
mkdir -p utils
curl -s -o utils/__init__.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/__init__.py"
curl -s -o utils/tools.py "https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/tools.py"

echo "✓ Utils files downloaded"

# Verify critical files exist
echo "Verifying installation..."
if [ -f "generate_multitalk.py" ] && [ -s "generate_multitalk.py" ]; then
    echo "✅ generate_multitalk.py verified"
else
    echo "❌ generate_multitalk.py missing or empty"
    exit 1
fi

if [ -f "wan/__init__.py" ] && [ -f "wan/multitalk.py" ]; then
    echo "✅ wan/ directory structure verified"
else
    echo "❌ wan/ directory incomplete"
    exit 1
fi

echo "✅ Official MultiTalk setup complete (simplified)"
echo "✅ Files installed to: $MULTITALK_DIR"
echo "✅ Ready for production use"