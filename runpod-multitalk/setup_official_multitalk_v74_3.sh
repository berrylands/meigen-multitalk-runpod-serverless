#!/bin/bash
# Setup script for official MultiTalk implementation - V74.3
# Downloads official MultiTalk code to the correct location expected by the handler

set -e
set -x  # Debug mode - show all commands

echo "=== Setting up Official MultiTalk Implementation V74.3 ==="

# Create the directory where the handler expects to find the code
MULTITALK_DIR="/app/multitalk_official"
mkdir -p "$MULTITALK_DIR"
cd "$MULTITALK_DIR"

echo "Downloading official MultiTalk files to $MULTITALK_DIR..."

# Download main generation script
echo "Downloading generate_multitalk.py..."
wget --timeout=30 --tries=3 -q -O generate_multitalk.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/generate_multitalk.py || {
    echo "ERROR: Failed to download generate_multitalk.py"
    exit 1
}
if [ ! -f "generate_multitalk.py" ]; then
    echo "ERROR: generate_multitalk.py not created"
    exit 1
fi

# Download main wan directory files
echo "Downloading wan/ directory structure..."
mkdir -p wan
wget -q -O wan/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/__init__.py
wget -q -O wan/first_last_frame2video.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/first_last_frame2video.py
wget -q -O wan/image2video.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/image2video.py
wget -q -O wan/multitalk.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/multitalk.py
wget -q -O wan/text2video.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/text2video.py
wget -q -O wan/vace.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/vace.py
wget -q -O wan/wan_lora.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/wan_lora.py

# Download wan/configs directory
echo "Downloading wan/configs/..."
mkdir -p wan/configs
wget -q -O wan/configs/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/__init__.py
wget -q -O wan/configs/shared_config.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/shared_config.py
wget -q -O wan/configs/wan_i2v_14B.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/wan_i2v_14B.py
wget -q -O wan/configs/wan_multitalk_14B.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/wan_multitalk_14B.py
wget -q -O wan/configs/wan_t2v_14B.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/wan_t2v_14B.py
wget -q -O wan/configs/wan_t2v_1_3B.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/wan_t2v_1_3B.py

# Download wan/distributed directory
echo "Downloading wan/distributed/..."
mkdir -p wan/distributed
wget -q -O wan/distributed/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/distributed/__init__.py
wget -q -O wan/distributed/fsdp.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/distributed/fsdp.py
wget -q -O wan/distributed/xdit_context_parallel.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/distributed/xdit_context_parallel.py

# Download wan/modules directory
echo "Downloading wan/modules/..."
mkdir -p wan/modules
wget -q -O wan/modules/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/__init__.py
wget -q -O wan/modules/attention.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/attention.py
wget -q -O wan/modules/clip.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/clip.py
wget -q -O wan/modules/model.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/model.py
wget -q -O wan/modules/multitalk_model.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/multitalk_model.py
wget -q -O wan/modules/t5.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/t5.py
wget -q -O wan/modules/tokenizers.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/tokenizers.py
wget -q -O wan/modules/vace_model.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/vace_model.py
wget -q -O wan/modules/vae.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/vae.py
wget -q -O wan/modules/xlm_roberta.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/xlm_roberta.py

# Download wan/utils directory
echo "Downloading wan/utils/..."
mkdir -p wan/utils
wget -q -O wan/utils/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/__init__.py
wget -q -O wan/utils/fm_solvers.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/fm_solvers.py
wget -q -O wan/utils/fm_solvers_unipc.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/fm_solvers_unipc.py
wget -q -O wan/utils/multitalk_utils.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/multitalk_utils.py
wget -q -O wan/utils/prompt_extend.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/prompt_extend.py
wget -q -O wan/utils/qwen_vl_utils.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/qwen_vl_utils.py
wget -q -O wan/utils/utils.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/utils.py
wget -q -O wan/utils/vace_processor.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/vace_processor.py

# Download utils directory
echo "Downloading utils/..."
mkdir -p utils
wget -q -O utils/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/__init__.py
wget -q -O utils/tools.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/tools.py

# Download TeaCache implementation
echo "Downloading utils/teacache/..."
mkdir -p utils/teacache
wget -q -O utils/teacache/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/teacache/__init__.py
wget -q -O utils/teacache/cache.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/utils/teacache/cache.py

echo "✓ Downloaded official MultiTalk files"

# Verify critical files exist
echo "Verifying installation..."
MISSING_FILES=0

if [ -f "generate_multitalk.py" ]; then
    echo "✓ generate_multitalk.py downloaded successfully"
else
    echo "ERROR: generate_multitalk.py not found"
    MISSING_FILES=1
fi

if [ -f "wan/configs/__init__.py" ]; then
    echo "✓ wan/configs directory downloaded successfully"
else
    echo "ERROR: wan/configs directory incomplete"
    MISSING_FILES=1
fi

if [ -f "wan/distributed/__init__.py" ]; then
    echo "✓ wan/distributed directory downloaded successfully"
else
    echo "ERROR: wan/distributed directory incomplete"
    MISSING_FILES=1
fi

if [ -f "wan/modules/__init__.py" ]; then
    echo "✓ wan/modules directory downloaded successfully"
else
    echo "ERROR: wan/modules directory incomplete"
    MISSING_FILES=1
fi

if [ -f "wan/utils/__init__.py" ]; then
    echo "✅ wan/utils directory downloaded successfully"
else
    echo "ERROR: wan/utils directory incomplete"
    MISSING_FILES=1
fi

if [ $MISSING_FILES -eq 0 ]; then
    echo "✅ Official MultiTalk setup complete with all subdirectories"
    echo "✅ Files installed to: $MULTITALK_DIR"
    echo "✅ Ready for production use - no fallback logic"
else
    echo "❌ Installation incomplete - some files missing"
    exit 1
fi

# Note: Volume symlink will be created at runtime when the volume is mounted
echo "📝 Official MultiTalk code installed at: $MULTITALK_DIR"
echo "📝 Handler will search for implementation at multiple locations including this path"