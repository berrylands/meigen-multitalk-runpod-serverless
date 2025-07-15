#!/bin/bash
# Setup script for MultiTalk V78 - Following Replicate implementation approach
# Based on zsxkib/cog-MultiTalk which works WITHOUT kokoro/misaki
set -e

echo "=========================================="
echo "Setting up MultiTalk V78 - Replicate Approach"
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
echo "📋 Copying official implementation (without kokoro)..."
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

# Copy only necessary modules (NOT kokoro!)
# Based on Replicate implementation which doesn't use kokoro
for module in audio_processor utils; do
    if [ -d "$module" ]; then
        echo "📦 Copying $module..."
        cp -r "$module" "$MULTITALK_DIR/"
    fi
done

# IMPORTANT: Remove any kokoro imports from generate_multitalk.py
echo "🔧 Patching generate_multitalk.py to remove kokoro dependency..."
if grep -q "from kokoro import KPipeline" "$MULTITALK_DIR/generate_multitalk.py"; then
    echo "Found kokoro import, removing it..."
    # Comment out the kokoro import line
    sed -i 's/from kokoro import KPipeline/# from kokoro import KPipeline # Removed - not needed per Replicate implementation/' "$MULTITALK_DIR/generate_multitalk.py"
    
    # Also comment out any KPipeline usage
    sed -i 's/kpipeline = KPipeline/# kpipeline = KPipeline # Removed/' "$MULTITALK_DIR/generate_multitalk.py"
    sed -i 's/kpipeline\./# kpipeline./' "$MULTITALK_DIR/generate_multitalk.py"
fi

# Make scripts executable
chmod +x "$MULTITALK_DIR/generate_multitalk.py" 2>/dev/null || true

# Install additional Python dependencies specific to MultiTalk
echo "📦 Installing MultiTalk-specific dependencies..."
pip install einops rotary-embedding-torch tensorboardX omegaconf || {
    echo "⚠️ Some dependencies failed to install, continuing..."
}

# Clean up
rm -rf "$TEMP_DIR"

echo "✅ MultiTalk V78 setup complete!"
echo "📁 Official implementation installed at: $MULTITALK_DIR"
echo "   - generate_multitalk.py (official script, kokoro removed)"
echo "   - wan/ (core implementation)"
echo "   - NO kokoro module (following Replicate approach)"
echo ""
echo "🎉 Ready to use MultiTalk without misaki dependency!"