#!/bin/bash
# Test V110 network volume exploration

API_KEY="${RUNPOD_API_KEY}"
ENDPOINT_ID="zu0ik6c8yukyl6"
BASE_URL="https://api.runpod.ai/v2/$ENDPOINT_ID"

if [ -z "$API_KEY" ]; then
    echo "❌ RUNPOD_API_KEY not set"
    exit 1
fi

echo "🔍 Exploring V110 network volume..."
echo ""

# 1. Submit volume exploration job
echo "1️⃣ Submitting volume exploration job..."
RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"input": {"action": "volume_explore"}}' \
    "$BASE_URL/run")

JOB_ID=$(echo $RESPONSE | jq -r .id)
echo "Job ID: $JOB_ID"

if [ "$JOB_ID" = "null" ]; then
    echo "Failed to submit job. Response:"
    echo $RESPONSE | jq .
    exit 1
fi

# 2. Wait for completion
echo ""
echo "2️⃣ Waiting for exploration to complete..."
sleep 10

for i in {1..30}; do
    STATUS_RESPONSE=$(curl -s -H "Authorization: Bearer $API_KEY" "$BASE_URL/status/$JOB_ID")
    STATUS=$(echo $STATUS_RESPONSE | jq -r .status)
    
    if [ "$STATUS" = "COMPLETED" ]; then
        echo ""
        echo "✅ Volume exploration completed!"
        echo ""
        
        # Save full response to file
        echo $STATUS_RESPONSE | jq . > volume_exploration_v110_result.json
        echo "Full results saved to volume_exploration_v110_result.json"
        
        # Show key findings
        echo "📊 Key Findings:"
        echo "Volume exists: $(echo $STATUS_RESPONSE | jq -r '.output.exploration.volume_exists')"
        echo "Models directory exists: $(echo $STATUS_RESPONSE | jq -r '.output.exploration.models_dir_exists')"
        echo "Total model size: $(echo $STATUS_RESPONSE | jq -r '.output.exploration.total_size_gb') GB"
        echo ""
        
        # Show WAN models
        echo "🤖 WAN Models:"
        echo $STATUS_RESPONSE | jq -r '.output.exploration.wan_models[] | "  - " + .path + " (" + (.size_mb | tostring) + " MB)"' 2>/dev/null || echo "  None found"
        echo ""
        
        # Show checkpoints
        echo "📁 Checkpoints (top 10):"
        echo $STATUS_RESPONSE | jq -r '.output.exploration.checkpoints[:10][] | "  - " + .path + " (" + (.size_mb | tostring) + " MB)"' 2>/dev/null || echo "  None found"
        echo ""
        
        # Show model directories
        echo "📂 Model Directories:"
        echo $STATUS_RESPONSE | jq -r '.output.exploration.model_directories | to_entries[] | "  - " + .key + " (" + (.value.file_count | tostring) + " files)"' 2>/dev/null || echo "  None found"
        echo ""
        
        # Show sample files
        echo "🎵 Sample Audio Files:"
        echo $STATUS_RESPONSE | jq -r '.output.exploration.sample_files.audio[] | "  - " + .path' 2>/dev/null || echo "  None found"
        echo ""
        
        echo "🖼️  Sample Image Files:"
        echo $STATUS_RESPONSE | jq -r '.output.exploration.sample_files.images[] | "  - " + .path' 2>/dev/null || echo "  None found"
        echo ""
        
        # Show directory tree (first level)
        echo "🌳 Directory Tree:"
        echo $STATUS_RESPONSE | jq -r '.output.exploration.directory_tree | keys[] | "  - " + .' 2>/dev/null || echo "  None found"
        
        break
        
    elif [ "$STATUS" = "FAILED" ]; then
        echo ""
        echo "❌ Volume exploration failed:"
        echo $STATUS_RESPONSE | jq .
        break
    else
        echo -ne "Status: $STATUS ($i/30)\r"
        sleep 3
    fi
done

echo ""
echo "📋 To view full results: cat volume_exploration_v110_result.json | jq ."
echo "🔄 To test generation: ./test_v110_generation.sh"