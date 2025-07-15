#!/bin/bash
# Direct test of V105 with xfuser

API_KEY="${RUNPOD_API_KEY}"
ENDPOINT_ID="zu0ik6c8yukyl6"
BASE_URL="https://api.runpod.ai/v2/$ENDPOINT_ID"

echo "🚀 Testing V105 with xfuser verification..."
echo ""

# 1. Submit model check job
echo "1️⃣ Submitting model check job..."
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

# 2. Wait for completion
echo ""
echo "2️⃣ Checking job status..."
sleep 5

for i in {1..30}; do
    STATUS_RESPONSE=$(curl -s -H "Authorization: Bearer $API_KEY" "$BASE_URL/status/$JOB_ID")
    STATUS=$(echo $STATUS_RESPONSE | jq -r .status)
    
    if [ "$STATUS" = "COMPLETED" ]; then
        echo ""
        echo "✅ Job completed successfully!"
        echo ""
        
        # Extract xfuser info
        XFUSER_AVAILABLE=$(echo $STATUS_RESPONSE | jq -r '.output.model_info.xfuser_available')
        XFUSER_VERSION=$(echo $STATUS_RESPONSE | jq -r '.output.model_info.xfuser_version')
        PYTORCH_VERSION=$(echo $STATUS_RESPONSE | jq -r '.output.model_info.pytorch_version')
        
        echo "📊 System Information:"
        echo "  - PyTorch version: $PYTORCH_VERSION"
        echo "  - xfuser available: $XFUSER_AVAILABLE"
        echo "  - xfuser version: $XFUSER_VERSION"
        
        if [ "$XFUSER_AVAILABLE" = "true" ]; then
            echo ""
            echo "🎉 SUCCESS: V105 is running with real xfuser version $XFUSER_VERSION!"
        else
            echo ""
            echo "⚠️  WARNING: xfuser is not available in V105"
        fi
        
        # Show full output
        echo ""
        echo "Full model info:"
        echo $STATUS_RESPONSE | jq '.output.model_info'
        break
        
    elif [ "$STATUS" = "FAILED" ]; then
        echo ""
        echo "❌ Job failed:"
        echo $STATUS_RESPONSE | jq .
        break
    else
        echo -ne "Status: $STATUS ($i/30)\r"
        sleep 2
    fi
done

echo ""