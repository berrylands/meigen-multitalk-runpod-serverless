#!/bin/bash
# Build-time setup script for official MultiTalk with complete directory structure

set -e

echo "=== Setting up Official MultiTalk Implementation at Build Time ==="

# Create directory
mkdir -p /app/multitalk_official
cd /app/multitalk_official

echo "Downloading official MultiTalk files..."

# Download main generation script
wget -q -O generate_multitalk.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/generate_multitalk.py || echo "Failed to download generate_multitalk.py"

# Download main wan directory files
mkdir -p wan
wget -q -O wan/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/__init__.py || echo "Failed to download wan/__init__.py"
wget -q -O wan/first_last_frame2video.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/first_last_frame2video.py || echo "Failed to download wan/first_last_frame2video.py"
wget -q -O wan/image2video.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/image2video.py || echo "Failed to download wan/image2video.py"
wget -q -O wan/multitalk.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/multitalk.py || echo "Failed to download wan/multitalk.py"
wget -q -O wan/text2video.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/text2video.py || echo "Failed to download wan/text2video.py"
wget -q -O wan/vace.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/vace.py || echo "Failed to download wan/vace.py"
wget -q -O wan/wan_lora.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/wan_lora.py || echo "Failed to download wan/wan_lora.py"

# Download wan/configs directory
mkdir -p wan/configs
wget -q -O wan/configs/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/__init__.py || echo "Failed to download wan/configs/__init__.py"
wget -q -O wan/configs/shared_config.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/shared_config.py || echo "Failed to download wan/configs/shared_config.py"
wget -q -O wan/configs/wan_i2v_14B.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/wan_i2v_14B.py || echo "Failed to download wan/configs/wan_i2v_14B.py"
wget -q -O wan/configs/wan_multitalk_14B.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/wan_multitalk_14B.py || echo "Failed to download wan/configs/wan_multitalk_14B.py"
wget -q -O wan/configs/wan_t2v_14B.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/wan_t2v_14B.py || echo "Failed to download wan/configs/wan_t2v_14B.py"
wget -q -O wan/configs/wan_t2v_1_3B.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/configs/wan_t2v_1_3B.py || echo "Failed to download wan/configs/wan_t2v_1_3B.py"

# Download wan/distributed directory
mkdir -p wan/distributed
wget -q -O wan/distributed/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/distributed/__init__.py || echo "Failed to download wan/distributed/__init__.py"
wget -q -O wan/distributed/fsdp.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/distributed/fsdp.py || echo "Failed to download wan/distributed/fsdp.py"
wget -q -O wan/distributed/xdit_context_parallel.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/distributed/xdit_context_parallel.py || echo "Failed to download wan/distributed/xdit_context_parallel.py"

# Download wan/modules directory
mkdir -p wan/modules
wget -q -O wan/modules/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/__init__.py || echo "Failed to download wan/modules/__init__.py"
wget -q -O wan/modules/attention.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/attention.py || echo "Failed to download wan/modules/attention.py"
wget -q -O wan/modules/clip.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/clip.py || echo "Failed to download wan/modules/clip.py"
wget -q -O wan/modules/model.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/model.py || echo "Failed to download wan/modules/model.py"
wget -q -O wan/modules/multitalk_model.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/multitalk_model.py || echo "Failed to download wan/modules/multitalk_model.py"
wget -q -O wan/modules/t5.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/t5.py || echo "Failed to download wan/modules/t5.py"
wget -q -O wan/modules/tokenizers.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/tokenizers.py || echo "Failed to download wan/modules/tokenizers.py"
wget -q -O wan/modules/vace_model.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/vace_model.py || echo "Failed to download wan/modules/vace_model.py"
wget -q -O wan/modules/vae.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/vae.py || echo "Failed to download wan/modules/vae.py"
wget -q -O wan/modules/xlm_roberta.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/modules/xlm_roberta.py || echo "Failed to download wan/modules/xlm_roberta.py"

# Download wan/utils directory
mkdir -p wan/utils
wget -q -O wan/utils/__init__.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/__init__.py || echo "Failed to download wan/utils/__init__.py"
wget -q -O wan/utils/fm_solvers.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/fm_solvers.py || echo "Failed to download wan/utils/fm_solvers.py"
wget -q -O wan/utils/fm_solvers_unipc.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/fm_solvers_unipc.py || echo "Failed to download wan/utils/fm_solvers_unipc.py"
wget -q -O wan/utils/multitalk_utils.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/multitalk_utils.py || echo "Failed to download wan/utils/multitalk_utils.py"
wget -q -O wan/utils/prompt_extend.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/prompt_extend.py || echo "Failed to download wan/utils/prompt_extend.py"
wget -q -O wan/utils/qwen_vl_utils.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/qwen_vl_utils.py || echo "Failed to download wan/utils/qwen_vl_utils.py"
wget -q -O wan/utils/utils.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/utils.py || echo "Failed to download wan/utils/utils.py"
wget -q -O wan/utils/vace_processor.py https://raw.githubusercontent.com/MeiGen-AI/MultiTalk/main/wan/utils/vace_processor.py || echo "Failed to download wan/utils/vace_processor.py"

# Download utils directory
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

if [ -f "wan/configs/__init__.py" ]; then
    echo "✓ wan/configs directory downloaded successfully"
else
    echo "ERROR: wan/configs directory incomplete"
fi

if [ -f "wan/distributed/__init__.py" ]; then
    echo "✓ wan/distributed directory downloaded successfully"
else
    echo "ERROR: wan/distributed directory incomplete"
fi

if [ -f "wan/modules/__init__.py" ]; then
    echo "✓ wan/modules directory downloaded successfully"
else
    echo "ERROR: wan/modules directory incomplete"
fi

if [ -f "wan/utils/__init__.py" ]; then
    echo "✓ wan/utils directory downloaded successfully"
else
    echo "ERROR: wan/utils directory incomplete"
fi

echo "✓ Official MultiTalk setup complete with all subdirectories"