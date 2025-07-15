#!/bin/bash
# Setup script for MultiTalk V76 - Downloads REAL official implementation
set -e

echo "=========================================="
echo "Setting up MultiTalk V76 - REAL Implementation"
echo "=========================================="

# Base directories
MULTITALK_DIR="/app/multitalk_official"
TEMP_DIR="/tmp/multitalk_download"

# Create directories
echo "📁 Creating directory structure..."
mkdir -p "$MULTITALK_DIR"
mkdir -p "$TEMP_DIR"

# Clone the official MultiTalk repository
echo "📥 Cloning official MultiTalk repository..."
cd "$TEMP_DIR"
git clone https://github.com/MeiGen-AI/MultiTalk.git || {
    echo "❌ Failed to clone MultiTalk repository"
    exit 1
}

# Copy the actual implementation files
echo "📋 Copying official implementation..."
cd MultiTalk

# Copy main scripts
cp -r scripts/generate_multitalk.py "$MULTITALK_DIR/" || {
    echo "⚠️ generate_multitalk.py not found in scripts/, checking root..."
    cp generate_multitalk.py "$MULTITALK_DIR/" 2>/dev/null || {
        echo "❌ Could not find generate_multitalk.py"
        exit 1
    }
}

# Copy wan module (the actual implementation)
cp -r wan "$MULTITALK_DIR/" || {
    echo "❌ Failed to copy wan module"
    exit 1
}

# Copy src modules if they exist
if [ -d "src" ]; then
    cp -r src "$MULTITALK_DIR/"
fi

# Copy any other required modules
for module in kokoro audio_processor utils; do
    if [ -d "$module" ]; then
        cp -r "$module" "$MULTITALK_DIR/"
    fi
done

# Make scripts executable
chmod +x "$MULTITALK_DIR/generate_multitalk.py" 2>/dev/null || true

# Install additional Python dependencies specific to MultiTalk
echo "📦 Installing MultiTalk-specific dependencies..."
pip install einops rotary-embedding-torch tensorboardX omegaconf || {
    echo "⚠️ Some dependencies failed to install, continuing..."
}

# Clean up
rm -rf "$TEMP_DIR"

echo "✅ MultiTalk V76 setup complete!"
echo "📁 Official implementation installed at: $MULTITALK_DIR"
echo "   - generate_multitalk.py (official script)"
echo "   - wan/ (core implementation)"
echo "   - All supporting modules"
echo ""
echo "🎉 Ready to use REAL MultiTalk implementation!"