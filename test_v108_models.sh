#!/bin/bash
# Test V108 model status and exploration

API_KEY="${RUNPOD_API_KEY}"
ENDPOINT_ID="zu0ik6c8yukyl6"
BASE_URL="https://api.runpod.ai/v2/$ENDPOINT_ID"

if [ -z "$API_KEY" ]; then
    echo "❌ RUNPOD_API_KEY not set"
    exit 1
fi

echo "🚀 Testing V108 model capabilities..."
echo ""

# 1. Check model status
echo "1️⃣ Checking model status..."
RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input": {"action": "model_check"}}' \
    "$BASE_URL/run")

JOB_ID=$(echo $RESPONSE | jq -r .id)
echo "Job ID: $JOB_ID"

if [ "$JOB_ID" = "null" ]; then
    echo "Failed to submit job. Response:"
    echo $RESPONSE | jq .
    exit 1
fi

# Wait for model check
sleep 5

for i in {1..20}; do
    STATUS_RESPONSE=$(curl -s -H "Authorization: Bearer $API_KEY" "$BASE_URL/status/$JOB_ID")
    STATUS=$(echo $STATUS_RESPONSE | jq -r .status)
    
    if [ "$STATUS" = "COMPLETED" ]; then
        echo ""
        echo "✅ Model check completed!"
        echo ""
        
        # Show system info
        echo "🖥️  System Info:"
        echo "  - CUDA: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.cuda_available')"
        echo "  - Device: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.device')"
        echo "  - PyTorch: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.pytorch_version')"
        echo "  - xfuser: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.xfuser_available') ($(echo $STATUS_RESPONSE | jq -r '.output.model_info.xfuser_version'))"
        echo ""
        
        # Show volume info
        echo "💾 Volume Info:"
        echo "  - Volume mounted: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.volume_exploration.volume_exists')"
        echo "  - Models directory: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.volume_exploration.models_dir_exists')"
        echo "  - Total size: $(echo $STATUS_RESPONSE | jq -r '.output.model_info.volume_exploration.total_size_gb') GB"
        echo ""
        
        # Show WAN models if any
        WAN_COUNT=$(echo $STATUS_RESPONSE | jq -r '.output.model_info.volume_exploration.wan_models | length')
        if [ "$WAN_COUNT" -gt 0 ]; then
            echo "🤖 WAN Models Found:"
            echo $STATUS_RESPONSE | jq -r '.output.model_info.volume_exploration.wan_models[] | "  - " + .path'
        else
            echo "⚠️  No WAN models found"
        fi
        echo ""
        
        # Show largest checkpoints
        echo "📁 Largest Checkpoints:"
        echo $STATUS_RESPONSE | jq -r '.output.model_info.volume_exploration.checkpoints | sort_by(.size_mb) | reverse | .[:5][] | "  - " + .path + " (" + (.size_mb | tostring) + " MB)"' 2>/dev/null || echo "  None found"
        
        break
        
    elif [ "$STATUS" = "FAILED" ]; then
        echo ""
        echo "❌ Model check failed:"
        echo $STATUS_RESPONSE | jq .
        break
    else
        echo -ne "Status: $STATUS ($i/20)\r"
        sleep 2
    fi
done

echo ""
echo "2️⃣ Testing generation with sample files..."

# Test generation with known sample files
GEN_RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "input": {
            "action": "generate",
            "audio_1": "kokoro-82m/samples/HEARME.wav",
            "condition_image": "kokoro-82m/eval/ArtificialAnalysis-2025-02-26.jpeg",
            "prompt": "A person talking naturally",
            "output_format": "base64"
        }
    }' \
    "$BASE_URL/run")

GEN_JOB_ID=$(echo $GEN_RESPONSE | jq -r .id)
echo "Generation job ID: $GEN_JOB_ID"

if [ "$GEN_JOB_ID" != "null" ]; then
    echo "✅ Generation job submitted successfully"
    echo "🔄 Check RunPod dashboard for results"
else
    echo "❌ Failed to submit generation job"
fi

echo ""
echo "🔍 For detailed volume exploration, run: ./test_v108_explore.sh"